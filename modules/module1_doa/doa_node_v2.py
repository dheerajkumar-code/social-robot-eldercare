#!/usr/bin/env python3
"""
Module 1 — DoA Node (v2)
--------------------------
Direction of Arrival estimation for MAIX R6 (6-channel) mic array.

What this fixes vs doa_node.py (v1):
  - v1: No VAD — publishes angle every 128ms even during silence
  - v2: Energy VAD gate — only processes frames with real audio signal

  - v1: Output range -90° to +90° (front hemisphere only, math.asin limit)
  - v2: Full 0–360° using multi-pair voting with 6-channel array

  - v1: Linear smoothing alpha*last + (1-alpha)*new — wraps wrong at 350°→10°
  - v2: Circular smoothing using atan2 — handles wraparound correctly

  - v1: mic_spacing = MIC_DISTANCE * ch — geometrically invented, not real
  - v2: Configurable mic geometry (linear or hexagonal/MAIX R6 layout)

  - v1: No confidence score — head tracks noise as confidently as speech
  - v2: Publishes /doa_confidence (0.0–1.0) from GCC-PHAT peak SNR

  - v1: No max_tau constraint in GCC-PHAT — aliases possible
  - v2: max_tau = mic_spacing / speed_of_sound (physical upper bound)

  - v1: fftconvolve imported, never used
  - v2: Removed unused import

  - v1: logger.info() every frame = 7 log lines/second
  - v2: Only logs when angle changes >5° or speech starts/stops

Topics published:
  /doa_angle       std_msgs/Float32  — 0–360° (0=front, 90=right, 180=back)
  /doa_confidence  std_msgs/Float32  — 0.0 (noise) to 1.0 (clear source)
  /doa_active      std_msgs/Bool     — True when speech energy detected

Standalone:
  python3 doa_node_v2.py --device 2 --channels 6   (MAIX R6)
  python3 doa_node_v2.py --device 2 --channels 2   (basic 2-mic)

ROS2:
  python3 doa_node_v2.py --ros --device 2 --channels 6
"""

import os
import sys
import time
import math
import argparse
import logging
from collections import deque
from datetime import datetime

import numpy as np
import sounddevice as sd

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32, Bool
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
SAMPLE_RATE     = 16000
FRAME_SIZE      = 2048          # 128ms per frame at 16kHz
SPEED_OF_SOUND  = 343.0         # m/s at ~20°C

# VAD parameters
ENERGY_THRESHOLD = 0.005        # RMS below this = silence (tune per environment)
ONSET_FRAMES     = 2            # consecutive loud frames before marking active
TAIL_FRAMES      = 8            # quiet frames after speech before marking inactive

# Smoothing
SMOOTHING_ALPHA  = 0.6          # 0=no smoothing, 1=never updates
MAX_ANGLE_CHANGE = 60.0         # degrees — spike rejection threshold

# Confidence
MIN_CONFIDENCE   = 0.15         # below this → do not publish angle update
LOG_CHANGE_DEG   = 5.0          # only log when angle changes more than this

# MAIX R6 (6-mic hexagonal array) approximate geometry
# Mics are arranged in a ring, ~46mm radius, 60° apart
# Pair (0,3): axis at 0°/180°   → gives left-right component
# Pair (1,4): axis at 60°/240°  → gives front-right/back-left
# Pair (2,5): axis at 120°/300° → gives front-left/back-right
MAIX_R6_MIC_RADIUS = 0.046      # meters, radius of mic ring
MAIX_R6_PAIRS = [
    (0, 3, 0.0),    # (ch_a, ch_b, pair_axis_degrees)
    (1, 4, 60.0),
    (2, 5, 120.0),
]

LOG_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "data", "logs")
LOG_FILE = os.path.join(LOG_DIR, "doa_angles.log")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE)],
)
logger = logging.getLogger("doa_v2")


# ─────────────────────────────────────────────────────────────
# Signal Processing
# ─────────────────────────────────────────────────────────────

def gcc_phat(sig: np.ndarray, refsig: np.ndarray,
             fs: int = SAMPLE_RATE,
             max_tau: float = None) -> tuple:
    """
    GCC-PHAT cross-correlation.

    Args:
        sig, refsig : 1D float64 arrays of same length
        fs          : sample rate
        max_tau     : maximum physically possible delay (seconds).
                      Constrains search to avoid aliasing artifacts.
                      Set to mic_spacing / speed_of_sound.

    Returns:
        (tau_seconds: float, confidence: float 0-1)
        tau_seconds  — time delay of sig relative to refsig
        confidence   — peak-to-noise ratio of cross-correlation (0=noise, 1=clear)
    """
    sig    = np.asarray(sig,    dtype=np.float64).flatten()
    refsig = np.asarray(refsig, dtype=np.float64).flatten()

    n     = sig.size + refsig.size
    SIG   = np.fft.rfft(sig,    n=n)
    REF   = np.fft.rfft(refsig, n=n)
    R     = SIG * np.conj(REF)

    # PHAT weighting: whiten the cross-spectrum
    denom = np.abs(R)
    denom[denom < np.finfo(float).eps] = np.finfo(float).eps
    R /= denom

    cc        = np.fft.irfft(R, n=n)
    max_shift = n // 2

    # Constrain search to physically possible delays
    if max_tau is not None:
        max_shift = min(max_shift, int(fs * max_tau) + 1)

    # Rearrange so index 0 = zero lag
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    peak_idx  = int(np.argmax(np.abs(cc)))
    tau       = (peak_idx - max_shift) / float(fs)

    # Confidence = peak SNR (normalised 0–1)
    peak_val    = float(np.abs(cc[peak_idx]))
    noise_floor = float(np.mean(np.abs(cc)))
    snr         = (peak_val - noise_floor) / (noise_floor + 1e-8)
    confidence  = float(np.clip(snr / 10.0, 0.0, 1.0))   # scale: snr=10 → conf=1.0

    return tau, confidence


def tau_to_angle_1d(tau: float, mic_spacing: float) -> float:
    """
    Convert TDOA to angle for a linear 2-mic pair.
    Returns degrees in range [-90, +90].
    """
    val = np.clip(tau * SPEED_OF_SOUND / mic_spacing, -1.0, 1.0)
    return math.degrees(math.asin(val))


def multi_pair_to_360(pair_results: list) -> tuple:
    """
    Combine multiple mic-pair TDOA estimates into a single 0–360° angle.

    Args:
        pair_results: list of (tau, confidence, broadside_deg, mic_spacing)
                      broadside_deg = direction the pair is most sensitive to
                                      (perpendicular to the mic baseline)

    Returns:
        (angle_360: float, combined_confidence: float)

    Method (linear least-squares):
        For a source at angle θ, the TDOA for a pair with broadside at β is:
            τ = (d/c) * sin(θ - β)
              = (d/c) * [sin(θ)cos(β) - cos(θ)sin(β)]

        Setting x = sin(θ), y = cos(θ):
            τ*c/d = x*cos(β) - y*sin(β)

        This is linear in (x, y). Solve via least-squares across all pairs,
        then θ = atan2(x, y).

        Advantage over vector averaging: correctly resolves 360° ambiguity.
    """
    if not pair_results:
        return 0.0, 0.0

    A_rows, b_vals, weights = [], [], []

    for tau, conf, broadside_deg, spacing in pair_results:
        if conf < MIN_CONFIDENCE:
            continue
        b_rad = math.radians(broadside_deg)
        # Normalised measurement: sin(θ - β)
        norm_meas = np.clip(tau * SPEED_OF_SOUND / spacing, -1.0, 1.0)
        A_rows.append([math.cos(b_rad), -math.sin(b_rad)])
        b_vals.append(float(norm_meas))
        weights.append(float(conf))

    if not A_rows:
        return 0.0, 0.0

    A = np.array(A_rows)           # (N, 2)
    b = np.array(b_vals)           # (N,)
    w = np.array(weights)          # (N,)

    # Weighted least-squares: minimize Σ wᵢ*(Aᵢx - bᵢ)²
    W  = np.diag(w)
    AW = A.T @ W @ A
    bW = A.T @ W @ b
    try:
        xy = np.linalg.solve(AW, bW)
    except np.linalg.LinAlgError:
        xy, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    x, y        = float(xy[0]), float(xy[1])
    angle_360   = math.degrees(math.atan2(x, y)) % 360.0

    # Confidence from residual quality
    residuals       = A @ xy - b
    residual_rms    = float(np.sqrt(np.mean(residuals ** 2)))
    combined_conf   = float(np.clip(np.mean(w) * (1.0 - residual_rms), 0.0, 1.0))

    return angle_360, combined_conf


def circular_smooth(last_deg: float, new_deg: float, alpha: float) -> float:
    """
    Exponential smoothing for circular (angular) values.
    Handles the 350° → 10° wraparound correctly.

    alpha=0.0 → fully follow new value
    alpha=1.0 → never update (hold last)
    """
    last_rad = math.radians(last_deg)
    new_rad  = math.radians(new_deg)
    # Shortest angular difference (−π to +π)
    diff     = math.atan2(math.sin(new_rad - last_rad),
                          math.cos(new_rad - last_rad))
    smoothed = last_rad + (1.0 - alpha) * diff
    return math.degrees(smoothed) % 360.0


# ─────────────────────────────────────────────────────────────
# VAD State
# ─────────────────────────────────────────────────────────────

class FrameVAD:
    """
    Simple frame-level VAD using RMS energy.
    State: SILENT ↔ ACTIVE
    """
    def __init__(self, threshold=ENERGY_THRESHOLD,
                 onset=ONSET_FRAMES, tail=TAIL_FRAMES):
        self.threshold    = threshold
        self.onset_needed = onset
        self.tail_needed  = tail
        self._onset_count = 0
        self._tail_count  = 0
        self.active       = False

    def update(self, frame: np.ndarray) -> bool:
        """Feed one frame. Returns True if speech is active."""
        rms = float(np.sqrt(np.mean(frame ** 2)))

        if not self.active:
            if rms > self.threshold:
                self._onset_count += 1
                if self._onset_count >= self.onset_needed:
                    self.active       = True
                    self._tail_count  = 0
            else:
                self._onset_count = 0
        else:
            if rms < self.threshold:
                self._tail_count += 1
                if self._tail_count >= self.tail_needed:
                    self.active       = False
                    self._onset_count = 0
            else:
                self._tail_count = 0

        return self.active


# ─────────────────────────────────────────────────────────────
# DoA Engine
# ─────────────────────────────────────────────────────────────

class DoAEngine:
    """
    Core DoA estimation engine.
    Works with 2-channel (−90° to +90°) or 6-channel (0–360°) input.
    """

    def __init__(self,
                 sample_rate    = SAMPLE_RATE,
                 n_channels     = 2,
                 smoothing_alpha= SMOOTHING_ALPHA):

        self.sample_rate     = sample_rate
        self.n_channels      = n_channels
        self.smoothing_alpha = smoothing_alpha

        self.last_angle      = 0.0
        self.last_confidence = 0.0
        self.angle_history   = deque(maxlen=10)
        self.vad             = FrameVAD()

        # Build mic pair configuration
        if n_channels >= 6:
            # MAIX R6 hexagonal pairs
            diameter  = 2 * MAIX_R6_MIC_RADIUS
            self.pairs = [
                (a, b, axis, diameter)   # spacing = diameter for opposite pairs
                for a, b, axis in MAIX_R6_PAIRS
            ]
            logger.info(f"DoA: 6-channel hexagonal mode (MAIX R6), "
                        f"mic_diameter={diameter*100:.1f}cm")
        else:
            # Fallback: single pair, channels 0 and 1
            spacing = MIC_DISTANCE if 'MIC_DISTANCE' in globals() else 0.05
            self.pairs = [(0, 1, 0.0, spacing)]
            logger.info(f"DoA: 2-channel linear mode, spacing={spacing*100:.1f}cm")

    def process_frame(self, frame: np.ndarray):
        """
        Process one audio frame.

        Args:
            frame: (N_samples, N_channels) float32

        Returns:
            (angle_deg, confidence, vad_active)
            or (None, None, False) if frame rejected by VAD or confidence too low
        """
        if frame.ndim == 1:
            return None, None, False
        if frame.shape[1] < 2:
            return None, None, False

        # VAD check on reference channel (ch 0)
        ref_channel = frame[:, 0].astype(np.float64)
        vad_active  = self.vad.update(frame[:, 0])

        if not vad_active:
            return None, None, False

        # Compute GCC-PHAT for each pair
        pair_results = []
        for ch_a, ch_b, axis_deg, spacing in self.pairs:
            if ch_a >= frame.shape[1] or ch_b >= frame.shape[1]:
                continue
            sig    = frame[:, ch_b].astype(np.float64)
            refsig = frame[:, ch_a].astype(np.float64)
            max_tau = spacing / SPEED_OF_SOUND   # physical upper bound
            try:
                tau, conf = gcc_phat(sig, refsig,
                                     fs=self.sample_rate,
                                     max_tau=max_tau)
            except Exception as e:
                logger.debug(f"GCC-PHAT error ({ch_a},{ch_b}): {e}")
                continue
            pair_results.append((tau, conf, axis_deg, spacing))

        if not pair_results:
            return None, None, True

        # Combine pairs into single 360° estimate
        raw_angle, confidence = multi_pair_to_360(pair_results)

        if confidence < MIN_CONFIDENCE:
            return None, confidence, True

        # Spike rejection
        angle_delta = abs(
            math.degrees(math.atan2(
                math.sin(math.radians(raw_angle - self.last_angle)),
                math.cos(math.radians(raw_angle - self.last_angle))
            ))
        )
        if angle_delta > MAX_ANGLE_CHANGE:
            raw_angle = circular_smooth(self.last_angle, raw_angle, alpha=0.85)

        # Circular smoothing
        smoothed = circular_smooth(self.last_angle, raw_angle, self.smoothing_alpha)

        self.last_angle      = smoothed
        self.last_confidence = confidence
        self.angle_history.append(smoothed)

        return smoothed, confidence, True


# ─────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────

MIC_DISTANCE = 0.05   # fallback for 2-channel mode

def run_standalone(device=None, channels=6,
                   samplerate=SAMPLE_RATE, blocksize=FRAME_SIZE):
    engine     = DoAEngine(sample_rate=samplerate, n_channels=channels)
    last_log   = 0.0
    last_angle = None

    print("=" * 50)
    print("  Module 1 — DoA v2 (standalone)")
    print("=" * 50)
    print(f"  Channels : {channels}")
    print(f"  VAD thr  : {ENERGY_THRESHOLD}")
    print(f"  Press Ctrl+C to stop\n")

    def callback(indata, frames, time_info, status):
        nonlocal last_log, last_angle
        if status:
            logger.debug(f"Audio status: {status}")
        try:
            angle, conf, active = engine.process_frame(indata.copy())
            now = time.time()

            if not active:
                print("⚪ Silence    ", end="\r")
                return

            if angle is None:
                print(f"🔵 Speech (low conf={conf:.2f})", end="\r")
                return

            print(f"🔴 DoA: {angle:6.1f}°  conf={conf:.2f}  ", end="\r")

            # Only log if angle changed significantly
            changed = (last_angle is None or
                       abs(angle - last_angle) > LOG_CHANGE_DEG)
            if changed and (now - last_log) > 0.5:
                logger.info(f"[DoA] angle={angle:.1f}°  conf={conf:.2f}")
                last_log   = now
                last_angle = angle

        except Exception as e:
            logger.exception(f"Processing error: {e}")

    try:
        with sd.InputStream(samplerate=samplerate, blocksize=blocksize,
                            channels=channels, dtype="float32",
                            device=device, callback=callback):
            while True:
                time.sleep(0.05)
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.exception(f"Stream failed: {e}")


# ─────────────────────────────────────────────────────────────
# ROS2 Node
# ─────────────────────────────────────────────────────────────

class DoARosNodeV2(Node):
    """
    ROS2 DoA node v2.

    Publishes:
      /doa_angle       Float32 — 0–360° (0=front, 90=right, 180=back, 270=left)
      /doa_confidence  Float32 — 0.0 (noise) to 1.0 (strong clear source)
      /doa_active      Bool    — True when speech energy detected by VAD

    Use /doa_active to gate head movement:
      - False → head stays at last known angle (no twitching)
      - True  → head follows /doa_angle weighted by /doa_confidence
    """

    def __init__(self, device=None, channels=6):
        super().__init__("doa_node_v2")

        # ROS2 parameters
        self.declare_parameter("channels",         channels)
        self.declare_parameter("energy_threshold", ENERGY_THRESHOLD)
        self.declare_parameter("smoothing_alpha",  SMOOTHING_ALPHA)
        self.declare_parameter("min_confidence",   MIN_CONFIDENCE)

        # Publishers
        self.pub_angle = self.create_publisher(Float32, "/doa_angle",      10)
        self.pub_conf  = self.create_publisher(Float32, "/doa_confidence", 10)
        self.pub_act   = self.create_publisher(Bool,    "/doa_active",     10)

        # Engine
        ch = self.get_parameter("channels").value
        self.engine = DoAEngine(
            sample_rate     = SAMPLE_RATE,
            n_channels      = ch,
            smoothing_alpha = self.get_parameter("smoothing_alpha").value,
        )

        self._last_active      = False
        self._last_logged_angle = None

        # Audio stream
        try:
            self._stream = sd.InputStream(
                samplerate = SAMPLE_RATE,
                blocksize  = FRAME_SIZE,
                channels   = ch,
                dtype      = "float32",
                device     = device,
                callback   = self._audio_cb,
            )
            self._stream.start()
        except Exception as e:
            self.get_logger().error(f"Failed to open audio: {e}")
            raise

        self.get_logger().info(
            f"DoA node v2 started | channels={ch} | "
            f"VAD threshold={ENERGY_THRESHOLD} | "
            f"topics: /doa_angle, /doa_confidence, /doa_active"
        )

    def _audio_cb(self, indata, frames, time_info, status):
        if status:
            self.get_logger().warning(f"Audio status: {status}")

        try:
            angle, conf, active = self.engine.process_frame(indata.copy())
        except Exception as e:
            self.get_logger().error(f"DoA error: {e}")
            return

        # Always publish VAD active state (changed or not)
        if active != self._last_active:
            msg_act      = Bool()
            msg_act.data = active
            self.pub_act.publish(msg_act)
            self._last_active = active
            if active:
                self.get_logger().info("Speech detected — tracking direction")
            else:
                self.get_logger().debug("Silence — DoA paused")

        if angle is None or not active:
            return

        # Publish angle
        msg_angle      = Float32()
        msg_angle.data = float(angle)
        self.pub_angle.publish(msg_angle)

        # Publish confidence
        msg_conf      = Float32()
        msg_conf.data = float(conf)
        self.pub_conf.publish(msg_conf)

        # Log only on significant change
        if (self._last_logged_angle is None or
                abs(angle - self._last_logged_angle) > LOG_CHANGE_DEG):
            self.get_logger().info(
                f"DoA: {angle:.1f}°  conf={conf:.2f}"
            )
            self._last_logged_angle = angle

    def destroy_node(self):
        self._stream.stop()
        self._stream.close()
        super().destroy_node()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def list_devices():
    print("\nAvailable audio input devices:\n")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"  {i}: {d['name']} (ch={d['max_input_channels']})")


def parse_args():
    p = argparse.ArgumentParser(description="DoA node v2")
    p.add_argument("--ros",          action="store_true",  help="Run as ROS2 node")
    p.add_argument("--device",       type=int, default=None, help="Audio device index")
    p.add_argument("--channels",     type=int, default=6,    help="Mic channels (2 or 6)")
    p.add_argument("--list-devices", action="store_true",  help="List audio devices")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.ros:
        if not ROS_AVAILABLE:
            logger.error("rclpy not available. Install ROS2 Humble.")
            sys.exit(1)
        rclpy.init()
        node = DoARosNodeV2(device=args.device, channels=args.channels)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            node.get_logger().info("Shutting down DoA node v2")
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        run_standalone(device=args.device, channels=args.channels)


if __name__ == "__main__":
    main()
