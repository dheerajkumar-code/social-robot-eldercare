#!/usr/bin/env python3
"""
Module 10 — health_bridge_node.py
====================================
Watchdog that monitors all 9 modules and publishes /diagnostics.

Monitors whether each module's output topic is still active.
If any critical module goes silent for > TIMEOUT seconds, publishes
a warning to /system_health and logs the fault.

Topics monitored:
  M1  /doa_angle          → DOA
  M2  /asr_text           → ASR
  M3  /dialog_intent      → Dialog Manager
  M4  /tts_speaking       → TTS
  M6  /reminder_alert     → Reminder System
  M11 /person_detected    → Person Detection
  M12 /emotion_label      → Emotion
  M13 /speaker_id         → Speaker Recognition
  M14 /activity_label     → Activity Detection

Topics published:
  /system_health     std_msgs/String   JSON health report
  /diagnostics       diagnostic_msgs/DiagnosticArray (if available)

Standalone test:
  python3 health_bridge_node.py --test

ROS2:
  python3 health_bridge_node.py --ros
"""

import os
import sys
import time
import json
import argparse
import threading
from datetime import datetime

# Optional ROS2
try:
    import rclpy
    from rclpy.node      import Node
    from rclpy.executors import MultiThreadedExecutor
    from std_msgs.msg    import String, Bool, Float32
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

# Optional diagnostic_msgs
try:
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
    DIAG_AVAILABLE = True
except ImportError:
    DIAG_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
TIMEOUT_CRITICAL = 15.0    # seconds — alert if topic silent
TIMEOUT_WARN     = 10.0    # seconds — warn first
PUBLISH_RATE     = 5.0     # publish health report every 5s

# Module definitions: (module_id, topic, msg_type, is_critical)
MODULES = [
    ("M1  DOA",               "/doa_angle",       "Float32",  False),
    ("M2  ASR",               "/asr_text",        "String",   False),
    ("M3  Dialog",            "/dialog_intent",   "String",   False),
    ("M4  TTS",               "/tts_speaking",    "Bool",     False),
    ("M6  Reminders",         "/reminder_alert",  "String",   False),
    ("M11 Person",            "/person_detected", "Bool",     True),
    ("M12 Emotion",           "/emotion_label",   "String",   False),
    ("M13 Speaker",           "/speaker_id",      "String",   False),
    ("M14 Activity",          "/activity_label",  "String",   True),
]


# ─────────────────────────────────────────────────────────────
# Module Health Tracker
# ─────────────────────────────────────────────────────────────

class ModuleHealth:
    """Tracks the last time a topic was seen."""

    def __init__(self, module_id: str, topic: str, critical: bool):
        self.module_id   = module_id
        self.topic       = topic
        self.critical    = critical
        self.last_seen   = 0.0         # 0 = never seen
        self.msg_count   = 0
        self.status      = "STARTING"  # STARTING / OK / WARN / ERROR / NEVER
        self._lock       = threading.Lock()

    def touch(self):
        """Called whenever a message arrives on this topic."""
        with self._lock:
            self.last_seen = time.time()
            self.msg_count += 1
            self.status = "OK"

    def compute_status(self) -> str:
        """Compute current status based on elapsed time."""
        with self._lock:
            if self.last_seen == 0.0:
                self.status = "NEVER"
                return "NEVER"
            elapsed = time.time() - self.last_seen
            if elapsed > TIMEOUT_CRITICAL:
                self.status = "ERROR"
            elif elapsed > TIMEOUT_WARN:
                self.status = "WARN"
            else:
                self.status = "OK"
            return self.status

    def to_dict(self) -> dict:
        elapsed = (time.time() - self.last_seen) if self.last_seen > 0 else -1
        return {
            "module":    self.module_id,
            "topic":     self.topic,
            "status":    self.compute_status(),
            "msgs":      self.msg_count,
            "silent_s":  round(elapsed, 1) if elapsed >= 0 else None,
            "critical":  self.critical,
        }


# ─────────────────────────────────────────────────────────────
# ROS2 Node
# ─────────────────────────────────────────────────────────────

class HealthBridgeNode(Node):
    """
    Monitors all module topics and publishes a health report.
    """

    def __init__(self):
        super().__init__("health_bridge_node")

        # Build health trackers
        self._trackers = {}
        for (mod_id, topic, msg_type, critical) in MODULES:
            self._trackers[topic] = ModuleHealth(mod_id, topic, critical)

        # Create one subscriber per topic
        self._create_subscribers()

        # Publishers
        self.pub_health = self.create_publisher(String, "/system_health", 10)
        if DIAG_AVAILABLE:
            self.pub_diag = self.create_publisher(
                DiagnosticArray, "/diagnostics", 10
            )
        else:
            self.pub_diag = None

        # Publish timer
        self.create_timer(PUBLISH_RATE, self._publish_health)

        self.get_logger().info(
            f"Health bridge started — monitoring {len(self._trackers)} modules\n"
            f"  Publish rate: every {PUBLISH_RATE}s → /system_health"
        )

    def _create_subscribers(self):
        """Create one generic subscriber per topic."""
        # Map type string → ROS msg class
        type_map = {
            "String":  String,
            "Bool":    Bool,
            "Float32": Float32,
        }

        for (mod_id, topic, msg_type, critical) in MODULES:
            cls = type_map.get(msg_type, String)
            tracker = self._trackers[topic]

            # Use default argument capture to bind tracker
            def make_cb(t):
                def cb(msg):
                    t.touch()
                return cb

            self.create_subscription(cls, topic, make_cb(tracker), 10)

    def _publish_health(self):
        """Build and publish the health report."""
        statuses  = [t.to_dict() for t in self._trackers.values()]
        ok_count  = sum(1 for s in statuses if s["status"] == "OK")
        err_count = sum(1 for s in statuses if s["status"] == "ERROR")
        warn_count= sum(1 for s in statuses if s["status"] == "WARN")

        overall = "OK"
        if err_count > 0:
            overall = "ERROR"
        elif warn_count > 0:
            overall = "WARN"

        report = {
            "timestamp": datetime.now().isoformat(),
            "overall":   overall,
            "ok":        ok_count,
            "warn":      warn_count,
            "error":     err_count,
            "modules":   statuses,
        }

        # Publish JSON health
        msg      = String()
        msg.data = json.dumps(report)
        self.pub_health.publish(msg)

        # Log warnings / errors
        for s in statuses:
            if s["status"] == "ERROR":
                self.get_logger().error(
                    f"❌ {s['module']} SILENT for {s['silent_s']}s | topic={s['topic']}"
                )
            elif s["status"] == "WARN":
                self.get_logger().warning(
                    f"⚠️  {s['module']} slow: {s['silent_s']}s | topic={s['topic']}"
                )
            elif s["status"] == "NEVER":
                self.get_logger().warning(
                    f"⚠️  {s['module']} never published | topic={s['topic']}"
                )

        # Publish diagnostic_msgs if available
        if self.pub_diag and DIAG_AVAILABLE:
            self._publish_diagnostics(statuses)

        self.get_logger().debug(
            f"Health: {ok_count} OK | {warn_count} WARN | {err_count} ERROR"
        )

    def _publish_diagnostics(self, statuses: list):
        """Publish standard ROS2 diagnostic_msgs/DiagnosticArray."""
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()

        level_map = {
            "OK":       DiagnosticStatus.OK,
            "WARN":     DiagnosticStatus.WARN,
            "ERROR":    DiagnosticStatus.ERROR,
            "NEVER":    DiagnosticStatus.WARN,
            "STARTING": DiagnosticStatus.OK,
        }

        for s in statuses:
            ds = DiagnosticStatus()
            ds.name    = s["module"]
            ds.level   = level_map.get(s["status"], DiagnosticStatus.WARN)
            ds.message = s["status"]
            ds.values  = [
                KeyValue(key="topic",    value=s["topic"]),
                KeyValue(key="msgs",     value=str(s["msgs"])),
                KeyValue(key="silent_s", value=str(s["silent_s"])),
            ]
            arr.status.append(ds)

        self.pub_diag.publish(arr)


# ─────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────

def run_test():
    print("=== Health Bridge Standalone Test ===")

    tracker = ModuleHealth("M11 Person", "/person_detected", critical=True)

    # T1: Never seen → status = NEVER
    assert tracker.compute_status() == "NEVER", "Expected NEVER"
    print("T1 Never seen → NEVER ✅")

    # T2: Recent message → OK
    tracker.touch()
    assert tracker.compute_status() == "OK"
    print("T2 Recent message → OK ✅")

    # T3: Simulate timeout
    tracker.last_seen = time.time() - (TIMEOUT_WARN + 1)
    assert tracker.compute_status() == "WARN"
    print("T3 After warn timeout → WARN ✅")

    tracker.last_seen = time.time() - (TIMEOUT_CRITICAL + 1)
    assert tracker.compute_status() == "ERROR"
    print("T4 After critical timeout → ERROR ✅")

    # T5: to_dict contains expected keys
    d = tracker.to_dict()
    assert all(k in d for k in ("module", "topic", "status", "msgs", "silent_s"))
    print("T5 to_dict keys correct ✅")

    print("\n✅ All health bridge tests passed")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ros",  action="store_true")
    p.add_argument("--test", action="store_true")
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
    node = HealthBridgeNode()
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
