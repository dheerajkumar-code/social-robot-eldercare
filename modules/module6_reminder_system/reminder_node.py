#!/usr/bin/env python3
"""
Module 6 — reminder_node.py
------------------------------
ROS2 node wrapper for the reminder system.

Integrates the complete reminder system into the DIAT robot pipeline:
  - Scheduler fires reminders
  - Notifier publishes to /tts_speak (normal) or /tts_speak_urgent (HIGH/URGENT)
  - Publishes /reminder_alert with type info for decision engine
  - Subscribes to /reminder_ack to confirm reminder was heard

Topics Published:
  /tts_speak         std_msgs/String  — reminder text (LOW/MEDIUM priority)
  /tts_speak_urgent  std_msgs/String  — reminder text (HIGH/URGENT priority)
  /reminder_alert    std_msgs/String  — JSON {id, type, name, priority}

Topics Subscribed:
  /reminder_ack      std_msgs/String  — reminder ID acknowledged by user

ROS2 Parameters:
  timezone         — scheduler timezone (default: Asia/Kolkata)
  use_tts          — use local pyttsx3 TTS (default: False, use ROS TTS)
  reminders_json   — path to reminders JSON file to preload

Standalone (no ROS2):
  python3 reminder_node.py  ← runs main.py instead
"""

import os
import sys
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

from reminder_base  import Priority
from reminder_types import (MedicineReminder, FoodReminder,
                             DrinkingReminder, WalkingReminder)
from registry   import ReminderRegistry
from notifier   import Notifier
from scheduler  import ReminderScheduler

logger = logging.getLogger("ReminderNode")


class ReminderNode(Node):
    """
    ROS2 node for the reminder system.
    Runs the scheduler in background and publishes to TTS topics.
    """

    def __init__(self):
        super().__init__("reminder_node")

        # ROS2 parameters
        self.declare_parameter("timezone",       "Asia/Kolkata")
        self.declare_parameter("use_local_tts",  False)
        self.declare_parameter("reminders_json", "")

        tz           = self.get_parameter("timezone").value
        use_local_tts = self.get_parameter("use_local_tts").value
        json_path    = self.get_parameter("reminders_json").value

        # Publishers
        self.pub_tts        = self.create_publisher(String, "/tts_speak",        10)
        self.pub_tts_urgent = self.create_publisher(String, "/tts_speak_urgent", 10)
        self.pub_alert      = self.create_publisher(String, "/reminder_alert",   10)

        # Subscribers
        self.create_subscription(String, "/reminder_ack", self._on_ack, 10)

        # Build reminder system
        self.registry  = ReminderRegistry()
        self.notifier  = Notifier(use_tts=use_local_tts)
        self.scheduler = ReminderScheduler(self.registry, self.notifier, timezone=tz)

        # Override notifier to publish to ROS2 topics
        self.notifier.notify = self._ros_notify

        # Load JSON if provided
        if json_path and os.path.exists(json_path):
            count = self.registry.import_json(json_path)
            self.get_logger().info(f"Loaded {count} reminders from {json_path}")

        # Start scheduler
        self.scheduler.start()
        self.scheduler.add_all()

        self.get_logger().info(
            f"Reminder node started | "
            f"{len(self.registry)} reminders | "
            f"{self.scheduler.job_count()} jobs scheduled"
        )

    def _ros_notify(self, reminder, message: str):
        """Override notifier to publish via ROS2 topics."""
        # Terminal print always
        from datetime import datetime
        now = datetime.now().strftime("%H:%M:%S")
        print(f"\n⏰ [{now}] REMINDER: {message}\n")

        # Choose topic based on priority
        msg      = String()
        msg.data = message

        if reminder.priority.value >= Priority.HIGH.value:
            self.pub_tts_urgent.publish(msg)
            self.get_logger().warn(f"🚨 Urgent reminder: {reminder.name}")
        else:
            self.pub_tts.publish(msg)
            self.get_logger().info(f"💬 Reminder: {reminder.name}")

        # Publish structured alert for decision engine
        alert_data = json.dumps({
            "id":       reminder.id,
            "type":     reminder.reminder_type,
            "name":     reminder.name,
            "priority": reminder.priority.name,
            "message":  message,
        })
        alert_msg      = String()
        alert_msg.data = alert_data
        self.pub_alert.publish(alert_msg)

    def _on_ack(self, msg: String):
        """User/system acknowledged a reminder by ID."""
        reminder_id = msg.data.strip()
        reminder    = self.registry.get(reminder_id)
        if reminder:
            self.get_logger().info(f"Reminder acknowledged: {reminder_id}")
        else:
            self.get_logger().warning(f"Ack for unknown ID: {reminder_id}")

    def destroy_node(self):
        self.scheduler.stop()
        self.registry.export_json()
        super().destroy_node()


def main_ros():
    if not ROS_AVAILABLE:
        print("❌ ROS2 not available. Run main.py for standalone mode.")
        sys.exit(1)
    rclpy.init()
    node = ReminderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down reminder node")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main_ros()
