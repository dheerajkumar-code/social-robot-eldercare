#!/usr/bin/env python3
"""
Module 6 — notifier.py
------------------------
Notification system for reminder triggers.

Handles three notification channels:
  1. Terminal print (always active)
  2. TTS via pyttsx3 (if installed)
  3. ROS2 topic publish to /tts_speak_urgent (if ROS2 available)

Design:
  - Notifier is injected into Scheduler
  - Scheduler calls notifier.notify(reminder, message)
  - Notifier decides which channels to use
"""

import logging
from datetime import datetime
from reminder_base import BaseReminder, Priority

logger = logging.getLogger("Notifier")


class Notifier:
    """
    Sends reminder notifications through all available channels.
    Falls back gracefully if TTS or ROS2 are unavailable.
    """

    def __init__(self,
                 use_tts:    bool = True,
                 use_ros2:   bool = False,
                 tts_rate:   int  = 140,
                 tts_volume: float = 1.0):
        """
        Args:
            use_tts    : Try to use pyttsx3 for voice output
            use_ros2   : Publish to /tts_speak_urgent topic
            tts_rate   : Words per minute (slower = clearer for elderly)
            tts_volume : Volume 0.0–1.0
        """
        self.use_ros2   = use_ros2
        self.tts_engine = None
        self._ros_pub   = None

        # Try TTS initialisation
        if use_tts:
            try:
                import pyttsx3
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty("rate",   tts_rate)
                self.tts_engine.setProperty("volume", tts_volume)
                logger.info("✅ TTS engine (pyttsx3) ready")
            except Exception as e:
                logger.warning(f"TTS not available: {e} — using terminal only")

        # Try ROS2 publisher initialisation
        if use_ros2:
            try:
                import rclpy
                from rclpy.node import Node
                from std_msgs.msg import String
                # Uses a minimal publisher node
                self._ros_pub_available = True
                logger.info("✅ ROS2 publisher available")
            except Exception as e:
                logger.warning(f"ROS2 not available: {e}")
                self.use_ros2 = False

    # ── Main notification entry point ──

    def notify(self, reminder: BaseReminder, message: str):
        """
        Send notification through all available channels.
        Called by Scheduler when a reminder fires.
        """
        # Always print to terminal
        self._print_terminal(reminder, message)

        # TTS if available
        if self.tts_engine:
            self._speak(message)

        # ROS2 if available
        if self.use_ros2 and self._ros_pub:
            self._publish_ros2(message, reminder.priority)

    # ── Channel implementations ──

    def _print_terminal(self, reminder: BaseReminder, message: str):
        """Rich terminal output with colour coding by priority."""
        now = datetime.now().strftime("%H:%M:%S")

        priority_color = {
            Priority.LOW:    "\033[36m",   # cyan
            Priority.MEDIUM: "\033[33m",   # yellow
            Priority.HIGH:   "\033[91m",   # bright red
            Priority.URGENT: "\033[31;1m", # bold red
        }.get(reminder.priority, "\033[0m")

        reset = "\033[0m"
        bold  = "\033[1m"

        border = "─" * 55
        print(f"\n{priority_color}{border}")
        print(f"  ⏰  REMINDER  [{now}]  [{reminder.priority.name}]")
        print(f"  Type   : {reminder.reminder_type.upper()}")
        print(f"  Name   : {reminder.name}")
        print(f"  {bold}{message}{reset}{priority_color}")
        print(f"{border}{reset}\n")

    def _speak(self, message: str):
        """Speak the message using pyttsx3."""
        try:
            self.tts_engine.say(message)
            self.tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"TTS speak error: {e}")

    def _publish_ros2(self, message: str, priority):
        """Publish to ROS2 TTS topic."""
        try:
            from std_msgs.msg import String
            msg      = String()
            msg.data = message
            # Urgent priority → /tts_speak_urgent (interrupts current speech)
            if priority.value >= Priority.HIGH.value:
                self._ros_pub_urgent.publish(msg)
            else:
                self._ros_pub_normal.publish(msg)
        except Exception as e:
            logger.error(f"ROS2 publish error: {e}")

    def test_notification(self):
        """Send a test notification to verify all channels work."""
        from reminder_types import MedicineReminder, Priority
        test = MedicineReminder(
            name="Test", medicine_name="Test Medicine",
            dosage="1 tablet", trigger_time="00:00"
        )
        self.notify(test, "🔔 This is a test notification. All channels working!")
