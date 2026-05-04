#!/usr/bin/env python3
"""
Module 10 — head_controller_node.py
======================================
Controls the EZ-Robot head pan/tilt servos via USB serial.

Behaviour:
  1. DOA angle  → pan head toward speaker  (when /doa_active = True)
  2. Emotion    → expressive head gesture  (nod, tilt, centre)
  3. TTS busy   → slight nod while speaking
  4. Idle       → slow return to centre

Topics subscribed:
  /doa_angle       Float32  — 0–360° speaker direction (from M1)
  /doa_confidence  Float32  — 0.0–1.0 DOA quality (from M1)
  /doa_active      Bool     — True when speech detected (from M1)
  /emotion_label   String   — current emotion (from M12)
  /tts_speaking    Bool     — True while robot is speaking (from M4)

Serial protocol to EZ-Robot IoTiny (USB):
  Each command is a newline-terminated ASCII string:
    "PAN:<degrees>\n"    degrees = 0 (left) to 180 (right), 90 = centre
    "TILT:<degrees>\n"   degrees = 60 (up) to 120 (down), 90 = centre
    "RESET\n"            return to (90, 90)

Standalone test (no ROS2):
  python3 head_controller_node.py --test
  python3 head_controller_node.py --port /dev/ttyUSB0

ROS2:
  python3 head_controller_node.py --ros
"""

import os
import sys
import time
import math
import serial
import argparse
import threading
import logging

logger = logging.getLogger("HeadController")

# Optional ROS2
try:
    import rclpy
    from rclpy.node   import Node
    from rclpy.executors import MultiThreadedExecutor
    from std_msgs.msg import String, Bool, Float32
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
DEFAULT_PORT   = "/dev/ttyUSB0"   # EZ-Robot IoTiny USB port
BAUD_RATE      = 115200
PAN_CENTRE     = 90               # degrees (0=left, 180=right)
TILT_CENTRE    = 90               # degrees (60=up, 120=down)
PAN_MIN        = 30
PAN_MAX        = 150
TILT_MIN       = 60
TILT_MAX       = 120
DOA_WEIGHT     = 0.25             # how fast to follow DOA angle (lower = smoother)
RETURN_SPEED   = 0.05             # degrees/tick back to centre when idle

# Emotion → (delta_pan, delta_tilt, hold_seconds)
# Positive pan = turn right, negative = turn left
# Positive tilt = nod down, negative = look up
EMOTION_GESTURES = {
    "sad":       (0,   +8,  1.5),   # slight bow
    "happy":     (0,   -5,  1.0),   # look up slightly
    "fearful":   (0,   +5,  0.8),   # lower gaze (calming)
    "angry":     (0,    0,  0.5),   # stay still (de-escalate)
    "surprised": (-10, -8,  1.0),   # tilt + look toward them
    "neutral":   (0,    0,  0.0),   # no gesture
}


# ─────────────────────────────────────────────────────────────
# Serial Driver
# ─────────────────────────────────────────────────────────────

class HeadSerial:
    """Thread-safe USB serial writer for EZ-Robot IoTiny."""

    def __init__(self, port: str = DEFAULT_PORT, baud: int = BAUD_RATE):
        self._port  = port
        self._baud  = baud
        self._ser   = None
        self._lock  = threading.Lock()
        self._dummy = False          # True if serial unavailable → log only

    def connect(self) -> bool:
        try:
            self._ser = serial.Serial(self._port, self._baud, timeout=1.0)
            time.sleep(0.5)          # let EZ-Robot boot
            logger.info(f"✅ EZ-Robot connected on {self._port}")
            return True
        except Exception as e:
            logger.warning(f"EZ-Robot not found ({e}) — running in DUMMY mode")
            self._dummy = True
            return False

    def send(self, cmd: str):
        """Send a command string (newline appended automatically)."""
        line = cmd.strip() + "\n"
        with self._lock:
            if self._dummy:
                logger.debug(f"[DUMMY] {line.strip()}")
                return
            try:
                self._ser.write(line.encode("utf-8"))
            except Exception as e:
                logger.error(f"Serial write error: {e}")

    def close(self):
        with self._lock:
            if self._ser and self._ser.is_open:
                self._ser.close()


# ─────────────────────────────────────────────────────────────
# Head Controller Logic
# ─────────────────────────────────────────────────────────────

class HeadController:
    """
    Manages current pan/tilt position and accepts movement requests.
    All angle clamping and smoothing is done here.
    """

    def __init__(self, serial_driver: HeadSerial):
        self._serial  = serial_driver
        self._pan     = float(PAN_CENTRE)
        self._tilt    = float(TILT_CENTRE)
        self._lock    = threading.Lock()
        self._gesture_end = 0.0      # timestamp when gesture completes
        self._doa_active  = False
        self._tts_busy    = False

    # ── DOA tracking ──

    def update_doa(self, angle_deg: float, confidence: float, active: bool):
        """
        Track speaker direction.
        DOA gives 0–360°. Map to servo pan 30–150°.
        Only move if confidence > 0.4 and speech is active.
        """
        self._doa_active = active
        if not active or confidence < 0.4:
            return
        if time.time() < self._gesture_end:
            return   # gesture in progress, don't interrupt

        # DOA: 0=front, 90=right, 180=back, 270=left
        # Map front-hemisphere (-90 to +90) onto servo 30–150
        if angle_deg > 180:
            angle_deg -= 360         # convert to -180..+180
        angle_deg = max(-90, min(90, angle_deg))

        target_pan = PAN_CENTRE + (angle_deg / 90.0) * (PAN_MAX - PAN_CENTRE)
        target_pan = max(PAN_MIN, min(PAN_MAX, target_pan))

        with self._lock:
            # Smooth: move DOA_WEIGHT fraction toward target each call
            self._pan += DOA_WEIGHT * (target_pan - self._pan)
            self._send_current()

    # ── Emotion gestures ──

    def play_emotion_gesture(self, emotion: str):
        """Move head to express emotion, then return to centre."""
        if emotion not in EMOTION_GESTURES:
            return
        dp, dt, hold = EMOTION_GESTURES[emotion]
        if hold == 0.0:
            return

        with self._lock:
            target_pan  = max(PAN_MIN,  min(PAN_MAX,  self._pan  + dp))
            target_tilt = max(TILT_MIN, min(TILT_MAX, self._tilt + dt))
            self._pan   = target_pan
            self._tilt  = target_tilt
            self._send_current()
            self._gesture_end = time.time() + hold

        # Return to centre after hold
        def _return():
            time.sleep(hold + 0.1)
            with self._lock:
                self._pan  += (PAN_CENTRE  - self._pan)  * 0.5
                self._tilt += (TILT_CENTRE - self._tilt) * 0.5
                self._send_current()

        threading.Thread(target=_return, daemon=True).start()

    # ── TTS nod ──

    def on_tts_busy(self, busy: bool):
        """Slight nod while robot is speaking."""
        self._tts_busy = busy
        if busy:
            with self._lock:
                self._tilt = min(TILT_MAX, self._tilt + 5)
                self._send_current()
        else:
            with self._lock:
                self._tilt = TILT_CENTRE
                self._send_current()

    # ── Reset ──

    def reset(self):
        with self._lock:
            self._pan  = float(PAN_CENTRE)
            self._tilt = float(TILT_CENTRE)
        self._serial.send("RESET")

    # ── Internal ──

    def _send_current(self):
        """Send current pan/tilt to serial (must be called with lock held)."""
        pan_int  = int(round(self._pan))
        tilt_int = int(round(self._tilt))
        self._serial.send(f"PAN:{pan_int}")
        self._serial.send(f"TILT:{tilt_int}")


# ─────────────────────────────────────────────────────────────
# ROS2 Node
# ─────────────────────────────────────────────────────────────

class HeadControllerNode(Node):
    """
    ROS2 node wrapping HeadController.
    Subscribes to DOA, emotion, and TTS topics.
    """

    def __init__(self, serial_port: str = DEFAULT_PORT):
        super().__init__("head_controller_node")

        self.declare_parameter("serial_port", serial_port)
        port = self.get_parameter("serial_port").value

        self._serial  = HeadSerial(port)
        self._serial.connect()
        self._head    = HeadController(self._serial)

        # DOA subscriptions
        self.create_subscription(Float32, "/doa_angle",
                                 self._cb_doa_angle,      10)
        self.create_subscription(Float32, "/doa_confidence",
                                 self._cb_doa_conf,       10)
        self.create_subscription(Bool,    "/doa_active",
                                 self._cb_doa_active,     10)

        # Emotion
        self.create_subscription(String,  "/emotion_label",
                                 self._cb_emotion,        10)

        # TTS busy flag
        self.create_subscription(Bool,    "/tts_speaking",
                                 self._cb_tts_speaking,   10)

        # Internal state
        self._doa_angle = 0.0
        self._doa_conf  = 0.0
        self._doa_act   = False

        # Reset to centre on startup
        self._head.reset()

        self.get_logger().info(
            f"Head controller node started | port={port}\n"
            f"  SUB: /doa_angle  /doa_confidence  /doa_active\n"
            f"  SUB: /emotion_label  /tts_speaking"
        )

    def _cb_doa_angle(self, msg: Float32):
        self._doa_angle = float(msg.data)
        self._head.update_doa(self._doa_angle, self._doa_conf, self._doa_act)

    def _cb_doa_conf(self, msg: Float32):
        self._doa_conf = float(msg.data)

    def _cb_doa_active(self, msg: Bool):
        self._doa_act = bool(msg.data)
        if not msg.data:
            self._head.reset()

    def _cb_emotion(self, msg: String):
        emotion = msg.data.lower().strip()
        self._head.play_emotion_gesture(emotion)
        self.get_logger().debug(f"Emotion gesture: {emotion}")

    def _cb_tts_speaking(self, msg: Bool):
        self._head.on_tts_busy(bool(msg.data))

    def destroy_node(self):
        self._head.reset()
        self._serial.close()
        super().destroy_node()


# ─────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────

def run_test():
    """Test head movement without ROS2 or real hardware."""
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    print("=== Head Controller Standalone Test ===")

    ser  = HeadSerial("/dev/null")   # dummy port → logs only
    ser._dummy = True
    head = HeadController(ser)

    print("\nT1: DOA tracking (angle=45°, confidence=0.8)")
    head.update_doa(45.0, 0.8, True)
    print(f"  Pan={head._pan:.1f}°  (expect ~{PAN_CENTRE + 45/90*(PAN_MAX-PAN_CENTRE):.1f})")

    print("\nT2: Emotion gesture (sad)")
    head.play_emotion_gesture("sad")
    time.sleep(0.1)
    print(f"  Tilt={head._tilt:.1f}° (expect ~{TILT_CENTRE+8})")

    print("\nT3: TTS nod")
    head.on_tts_busy(True)
    print(f"  Tilt={head._tilt:.1f}° (expect >{TILT_CENTRE})")
    head.on_tts_busy(False)
    print(f"  Tilt={head._tilt:.1f}° (expect ={TILT_CENTRE})")

    print("\nT4: Reset")
    head.reset()
    print(f"  Pan={head._pan:.1f}°  Tilt={head._tilt:.1f}°")
    assert head._pan == PAN_CENTRE and head._tilt == TILT_CENTRE

    print("\n✅ All head controller tests passed")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ros",  action="store_true")
    p.add_argument("--test", action="store_true")
    p.add_argument("--port", type=str, default=DEFAULT_PORT)
    return p.parse_args()


def main(args=None):
    parsed = parse_args()
    if parsed.test:
        run_test()
        return
    if not ROS_AVAILABLE:
        print("❌ rclpy not available")
        sys.exit(1)
    rclpy.init(args=args)
    node = HeadControllerNode(parsed.port)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
