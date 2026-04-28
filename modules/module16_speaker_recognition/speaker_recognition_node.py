#!/usr/bin/env python3
"""
Module 16 — speaker_recognition_node.py
==========================================
Real-time speaker recognition ROS2 node with automatic enrollment.

Pipeline:
  Mic → VAD → MFCC → SVM → Confidence Check
                               ↓ known:   publish /speaker_id
                               ↓ unknown: trigger auto-enrollment

Architecture:
  - Audio captured in background thread (non-blocking)
  - Sliding window (2s) with 0.5s step (50ms hop)
  - Energy VAD gates inference (no wasted CPU on silence)
  - SVM inference: <1ms per prediction on Pi5
  - Auto-enrollment runs in separate thread (robot keeps working)
  - Model hot-reload: automatically loads new model after enrollment

Topics Published:
  /speaker_id          std_msgs/String   — speaker name or "unknown"
  /speaker_confidence  std_msgs/Float32  — confidence 0.0–1.0

Topics Subscribed:
  (none required — audio captured directly from mic)

ROS2 Parameters:
  mic_device       : int   — sounddevice device index (default: system default)
  confidence_threshold : float — below this = unknown (default: 0.65)
  unknown_cooldown : float — seconds between enrollment triggers (default: 30)
  use_tts          : bool  — speak greetings aloud (default: True)

Standalone (no ROS2):
  python3 speaker_recognition_node.py --standalone
  python3 speaker_recognition_node.py --standalone --device 2
"""

import os
import sys
import time
import json
import queue
import logging
import threading
import argparse
import numpy as np
import sounddevice as sd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_extractor    import (extract_mfcc, is_speech,
                                   SAMPLE_RATE, FEATURE_DIM)
from train_speaker_model  import load_model, MODEL_PATH
from register_voice_auto  import enroll_new_speaker

# Optional ROS2
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String, Float32
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

# Optional TTS
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "speaker_recognition.log"),
            mode="a"
        ),
    ]
)
logger = logging.getLogger("SpeakerRecog")

# ── Configuration ──
CONFIDENCE_THRESHOLD = 0.65   # below this → unknown speaker
WINDOW_SEC           = 2.0    # audio window for one prediction
STEP_SEC             = 0.5    # step between windows
BLOCK_SIZE           = 1600   # 100ms audio blocks at 16kHz
ENERGY_THRESHOLD     = 0.008  # VAD energy threshold
UNKNOWN_COOLDOWN_S   = 30.0   # min seconds between enrollment triggers
MIN_SPEECH_BLOCKS    = 6      # minimum loud blocks before classifying


# ─────────────────────────────────────────────────────────────
# SVM Inference Engine
# ─────────────────────────────────────────────────────────────

class SpeakerEngine:
    """
    Wraps the trained SVM pipeline for thread-safe inference.
    Supports hot-reload when model is updated (after enrollment).
    """

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path  = model_path
        self._pipeline   = None
        self._speakers   = []
        self._lock       = threading.Lock()
        self._model_mtime = 0.0
        self._load()

    def _load(self):
        """Load or reload model from disk."""
        try:
            payload = load_model(self.model_path)
            with self._lock:
                self._pipeline   = payload["model"]
                self._speakers   = payload["speakers"]
                self._model_mtime = os.path.getmtime(self.model_path)
            logger.info(f"Model loaded: {self._speakers}")
            return True
        except FileNotFoundError:
            logger.warning(
                f"Model not found at {self.model_path}. "
                f"Run train_speaker_model.py first or enroll a speaker."
            )
            return False
        except Exception as e:
            logger.error(f"Model load error: {e}")
            return False

    def check_reload(self):
        """Hot-reload if model file was updated (e.g. after enrollment)."""
        if not os.path.exists(self.model_path):
            return
        try:
            mtime = os.path.getmtime(self.model_path)
            if mtime > self._model_mtime:
                logger.info("Model file updated — reloading...")
                self._load()
        except Exception:
            pass

    def predict(self, audio: np.ndarray) -> tuple:
        """
        Classify a speaker from audio.

        Returns:
            (speaker_name: str, confidence: float)
            speaker_name = "unknown" if confidence < threshold or no model
        """
        with self._lock:
            if self._pipeline is None or not self._speakers:
                return "unknown", 0.0

        feat = extract_mfcc(audio, SAMPLE_RATE)

        if np.all(feat == 0):
            return "unknown", 0.0

        with self._lock:
            try:
                proba  = self._pipeline.predict_proba(feat.reshape(1, -1))[0]
                idx    = int(np.argmax(proba))
                conf   = float(proba[idx])
                name   = self._speakers[idx]
            except Exception as e:
                logger.error(f"Inference error: {e}")
                return "unknown", 0.0

        if conf < CONFIDENCE_THRESHOLD:
            return "unknown", conf

        return name, conf

    @property
    def speakers(self):
        with self._lock:
            return list(self._speakers)

    @property
    def is_loaded(self):
        with self._lock:
            return self._pipeline is not None


# ─────────────────────────────────────────────────────────────
# TTS helper
# ─────────────────────────────────────────────────────────────

_tts_lock = threading.Lock()

def speak_async(text: str):
    """Speak text in a background thread (non-blocking)."""
    print(f"\n🤖 Robot: {text}")

    def _speak():
        with _tts_lock:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", 145)
                engine.say(text)
                engine.runAndWait()
                del engine
            except Exception:
                pass

    t = threading.Thread(target=_speak, daemon=True)
    t.start()


# ─────────────────────────────────────────────────────────────
# Core recognition + enrollment logic
# ─────────────────────────────────────────────────────────────

class SpeakerRecognizer:
    """
    Core recognizer — handles:
      - Sliding window audio buffering
      - VAD gating
      - Inference
      - Unknown speaker detection + enrollment trigger
      - Result callbacks
    """

    def __init__(self,
                 engine:              SpeakerEngine,
                 on_result=None,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD,
                 unknown_cooldown:     float = UNKNOWN_COOLDOWN_S,
                 use_tts:              bool  = True,
                 mic_device=None):
        """
        Args:
            engine               : SpeakerEngine instance
            on_result            : callback(name: str, confidence: float)
            confidence_threshold : below this = unknown
            unknown_cooldown     : min seconds between enrollment triggers
            use_tts              : speak greetings
            mic_device           : sounddevice device index
        """
        self.engine               = engine
        self.on_result            = on_result
        self.threshold            = confidence_threshold
        self.unknown_cooldown     = unknown_cooldown
        self.use_tts              = use_tts
        self.mic_device           = mic_device

        # Audio buffer (sliding window)
        self._window_samples = int(WINDOW_SEC * SAMPLE_RATE)
        self._step_samples   = int(STEP_SEC   * SAMPLE_RATE)
        self._buffer         = np.zeros(self._window_samples, dtype=np.float32)
        self._raw_q          = queue.Queue(maxsize=500)

        # State
        self._running             = False
        self._enrolling           = False
        self._last_unknown_time   = 0.0
        self._last_speaker        = None
        self._speech_block_count  = 0
        self._greeted_speakers    = set()

        # Stats
        self.stats = {
            "total_predictions":   0,
            "known_predictions":   0,
            "unknown_predictions": 0,
            "enrollments":         0,
        }

    # ── Start / Stop ──

    def start(self):
        """Start audio capture and inference threads."""
        self._running = True

        # Audio capture thread
        self._audio_thread = threading.Thread(
            target=self._audio_loop, daemon=True, name="audio_capture"
        )
        self._audio_thread.start()

        # Inference thread
        self._infer_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="inference"
        )
        self._infer_thread.start()

        # Model hot-reload checker (every 5s)
        self._reload_timer = threading.Timer(5.0, self._reload_check_loop)
        self._reload_timer.daemon = True
        self._reload_timer.start()

        logger.info(f"Recognizer started | device={self.mic_device} | "
                    f"threshold={self.threshold}")

    def stop(self):
        """Stop all threads."""
        self._running = False
        logger.info("Recognizer stopped")

    # ── Audio capture ──

    def _audio_loop(self):
        """Capture audio blocks in background thread."""
        def callback(indata, frames, time_info, status):
            if status:
                logger.debug(f"Audio status: {status}")
            block = indata[:, 0].copy() if indata.ndim > 1 else indata.flatten()
            try:
                self._raw_q.put_nowait(block)
            except queue.Full:
                pass

        try:
            with sd.InputStream(
                samplerate = SAMPLE_RATE,
                blocksize  = BLOCK_SIZE,
                channels   = 1,
                dtype      = "float32",
                device     = self.mic_device,
                callback   = callback,
            ):
                logger.info("🎙️  Microphone opened")
                while self._running:
                    time.sleep(0.05)
        except Exception as e:
            logger.error(f"Audio stream error: {e}")

    # ── Inference loop ──

    def _inference_loop(self):
        """
        Drain audio queue, maintain sliding window, classify.
        Runs on its own thread — never blocks audio capture.
        """
        samples_since_last = 0

        while self._running:
            # Drain all available blocks into buffer
            new_samples = []
            while not self._raw_q.empty():
                try:
                    block = self._raw_q.get_nowait()
                    new_samples.append(block)
                except queue.Empty:
                    break

            if not new_samples:
                time.sleep(0.02)
                continue

            chunk = np.concatenate(new_samples)

            # Check VAD on latest chunk
            speech_now = is_speech(chunk, ENERGY_THRESHOLD)
            if speech_now:
                self._speech_block_count += 1
            else:
                self._speech_block_count = max(0, self._speech_block_count - 1)

            # Shift buffer and append new data
            chunk_len = min(len(chunk), self._window_samples)
            self._buffer = np.roll(self._buffer, -chunk_len)
            self._buffer[-chunk_len:] = chunk[-chunk_len:]
            samples_since_last += chunk_len

            # Only run inference when:
            #   1. Enough speech energy detected
            #   2. Step interval has passed
            #   3. Not currently enrolling
            if (self._speech_block_count >= MIN_SPEECH_BLOCKS and
                    samples_since_last >= self._step_samples and
                    not self._enrolling):

                samples_since_last = 0
                self._classify(self._buffer.copy())

            time.sleep(0.01)

    # ── Classification ──

    def _classify(self, audio: np.ndarray):
        """Run inference and handle result."""
        name, conf = self.engine.predict(audio)
        self.stats["total_predictions"] += 1

        if name == "unknown":
            self.stats["unknown_predictions"] += 1
            self._handle_unknown(conf)
        else:
            self.stats["known_predictions"] += 1
            self._handle_known(name, conf)

        # Notify callback
        if self.on_result:
            self.on_result(name, conf)

        logger.info(f"Speaker: {name:15s}  conf={conf:.2f}  "
                    f"energy={is_speech(audio, ENERGY_THRESHOLD)}")

    # ── Known speaker ──

    def _handle_known(self, name: str, conf: float):
        """Handle a recognised speaker."""
        # Greet on first recognition per session
        if name not in self._greeted_speakers and self.use_tts:
            self._greeted_speakers.add(name)
            speak_async(f"Hello, {name}! Good to hear you.")

        self._last_speaker = name

    # ── Unknown speaker ──

    def _handle_unknown(self, conf: float):
        """Trigger enrollment if unknown speaker + cooldown passed."""
        now     = time.time()
        elapsed = now - self._last_unknown_time

        if elapsed < self.unknown_cooldown:
            logger.debug(f"Unknown cooldown active ({elapsed:.0f}s < {self.unknown_cooldown}s)")
            return

        if self._enrolling:
            return

        self._last_unknown_time = now
        logger.info("Unknown speaker detected — triggering enrollment")

        # Run enrollment in background thread (robot stays responsive)
        enroll_thread = threading.Thread(
            target=self._run_enrollment,
            daemon=True,
            name="enrollment"
        )
        enroll_thread.start()

    def _run_enrollment(self):
        """Execute enrollment flow in background thread."""
        self._enrolling = True
        try:
            name = enroll_new_speaker(
                device    = self.mic_device,
                n_samples = 5,
                retrain   = True,
            )
            if name:
                self.stats["enrollments"] += 1
                self._greeted_speakers.add(name)
                self._last_speaker = name
                # Hot-reload the updated model
                self.engine.check_reload()
                logger.info(f"Enrollment complete: {name}")
        except Exception as e:
            logger.error(f"Enrollment error: {e}")
        finally:
            self._enrolling = False

    # ── Hot-reload ──

    def _reload_check_loop(self):
        """Check for model updates every 5 seconds."""
        while self._running:
            self.engine.check_reload()
            time.sleep(5.0)


# ─────────────────────────────────────────────────────────────
# ROS2 Node
# ─────────────────────────────────────────────────────────────

class SpeakerRecognitionNode(Node):
    """
    ROS2 node wrapping SpeakerRecognizer.

    Publishes:
      /speaker_id          String   — speaker name or "unknown"
      /speaker_confidence  Float32  — confidence score 0.0–1.0
    """

    def __init__(self):
        super().__init__("speaker_recognition_node")

        # Parameters
        self.declare_parameter("mic_device",           -1)
        self.declare_parameter("confidence_threshold", CONFIDENCE_THRESHOLD)
        self.declare_parameter("unknown_cooldown",     UNKNOWN_COOLDOWN_S)
        self.declare_parameter("use_tts",              True)

        device    = self.get_parameter("mic_device").value
        threshold = self.get_parameter("confidence_threshold").value
        cooldown  = self.get_parameter("unknown_cooldown").value
        use_tts   = self.get_parameter("use_tts").value

        if device < 0:
            device = None

        # Publishers
        self.pub_id   = self.create_publisher(String,  "/speaker_id",          10)
        self.pub_conf = self.create_publisher(Float32, "/speaker_confidence",  10)

        # Engine + Recognizer
        self.engine = SpeakerEngine()
        self.recognizer = SpeakerRecognizer(
            engine               = self.engine,
            on_result            = self._on_result,
            confidence_threshold = threshold,
            unknown_cooldown     = cooldown,
            use_tts              = use_tts,
            mic_device           = device,
        )
        self.recognizer.start()

        self.get_logger().info(
            f"Speaker recognition node started\n"
            f"  Speakers    : {self.engine.speakers}\n"
            f"  Threshold   : {threshold}\n"
            f"  PUB         : /speaker_id  /speaker_confidence"
        )

    def _on_result(self, name: str, confidence: float):
        """Publish result to ROS2 topics."""
        msg_id       = String()
        msg_id.data  = name
        self.pub_id.publish(msg_id)

        msg_conf      = Float32()
        msg_conf.data = float(confidence)
        self.pub_conf.publish(msg_conf)

        self.get_logger().info(f"Speaker: {name}  conf={confidence:.2f}")

    def destroy_node(self):
        self.recognizer.stop()
        super().destroy_node()


# ─────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────

def run_standalone(device=None, threshold=CONFIDENCE_THRESHOLD):
    """
    Run speaker recognition without ROS2.
    Prints results to terminal in real time.
    """
    print("=" * 60)
    print("  Module 16 — Speaker Recognition (Standalone)")
    print("=" * 60)
    print(f"  Threshold   : {threshold}")
    print(f"  Mic device  : {device or 'system default'}")
    print(f"  Model       : {MODEL_PATH}")
    print("  Press Ctrl+C to stop\n")

    engine = SpeakerEngine()

    if not engine.is_loaded:
        print("\n⚠️  No model found. Starting enrollment...")
        from register_voice_auto import enroll_new_speaker
        name = enroll_new_speaker(device=device, n_samples=5, retrain=True)
        if name:
            engine._load()
        else:
            print("❌ Enrollment failed. Cannot start recognition.")
            return

    last_printed = ""

    def on_result(name, conf):
        nonlocal last_printed
        result = f"🎤  {name:15s}  conf={conf*100:5.1f}%"
        if result != last_printed:
            print(result)
            last_printed = result

    recognizer = SpeakerRecognizer(
        engine               = engine,
        on_result            = on_result,
        confidence_threshold = threshold,
        unknown_cooldown     = UNKNOWN_COOLDOWN_S,
        use_tts              = True,
        mic_device           = device,
    )

    recognizer.start()
    print("✅ Listening... (speak into the microphone)\n")

    try:
        while True:
            time.sleep(0.5)
            # Periodically print stats
    except KeyboardInterrupt:
        pass
    finally:
        recognizer.stop()
        print(f"\n📊 Session stats: {recognizer.stats}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def list_devices():
    print("\nAvailable audio input devices:")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            print(f"  {i}: {d['name']} (ch={d['max_input_channels']})")


def parse_args():
    p = argparse.ArgumentParser(description="Speaker recognition node")
    p.add_argument("--ros",        action="store_true",
                   help="Run as ROS2 node")
    p.add_argument("--standalone", action="store_true",
                   help="Run standalone (no ROS2)")
    p.add_argument("--device",     type=int, default=None,
                   help="Microphone device index")
    p.add_argument("--threshold",  type=float, default=CONFIDENCE_THRESHOLD,
                   help="Confidence threshold (default 0.65)")
    p.add_argument("--list-devices", action="store_true",
                   help="List audio devices")
    p.add_argument("--enroll",     action="store_true",
                   help="Enroll a new speaker and exit")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_devices:
        list_devices()
        return

    if args.enroll:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s [%(levelname)s] %(message)s")
        name = enroll_new_speaker(device=args.device)
        print(f"{'✅ Enrolled: ' + name if name else '❌ Enrollment failed'}")
        return

    if args.ros:
        if not ROS_AVAILABLE:
            print("❌ rclpy not available. Install ROS2 Humble.")
            sys.exit(1)
        rclpy.init()
        node = SpeakerRecognitionNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            node.get_logger().info("Shutting down")
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        run_standalone(device=args.device, threshold=args.threshold)


if __name__ == "__main__":
    main()
