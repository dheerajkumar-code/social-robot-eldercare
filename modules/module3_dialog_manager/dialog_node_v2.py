#!/usr/bin/env python3
"""
Module 3 — Dialog Node (v2)
-----------------------------
ROS2 node that connects all perception inputs to the dialog manager
and routes responses to TTS output.

This is the module that was EMPTY in v1. This is now a full ROS2 node.

Subscribes:
  /asr_text          String  — transcribed speech (from M2)
  /emotion_label     String  — detected emotion   (from M12)
  /activity_label    String  — detected activity  (from M14)
  /speaker_id        String  — recognised speaker (from M13)
  /doa_active        Bool    — speech VAD active  (from M1)

Publishes:
  /tts_speak         String  — normal response text    (to M4)
  /tts_speak_urgent  String  — emergency response text (to M4, interrupts)
  /dialog_intent     String  — detected intent tag
  /dialog_action     String  — triggered action (play_music, call_emergency, etc.)

Standalone (no ROS2):
  python3 dialog_node_v2.py --interactive

ROS2:
  python3 dialog_node_v2.py --ros --model path/to/fer_model.h5
"""

import os
import sys
import argparse
import threading

# Optional ROS2
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String, Bool
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

# Import v2 dialog manager
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dialog_manager_v2 import DialogManager

# ─────────────────────────────────────────────────────────────
# ROS2 Node
# ─────────────────────────────────────────────────────────────

class DialogNodeV2(Node):
    """
    Central ROS2 dialog node.

    Receives ASR text + emotion + activity + speaker → generates response →
    publishes to TTS. Handles emergency routing automatically.

    Emergency intents go to /tts_speak_urgent (interrupts current speech).
    All other responses go to /tts_speak (queued).
    """

    EMERGENCY_INTENTS = {"emergency", "user_health_status"}

    def __init__(self):
        super().__init__("dialog_node_v2")

        # ROS2 parameters
        self.declare_parameter("intents_file", "intents.json")
        self.declare_parameter("gemini_api_key", "")

        # Dialog manager
        api_key = self.get_parameter("gemini_api_key").value or None
        self.dm = DialogManager(
            intents_file=self.get_parameter("intents_file").value,
            api_key=api_key,
        )

        # State from perception modules
        self._emotion    = "neutral"
        self._activity   = "unknown"
        self._speaker_id = "unknown"
        self._doa_active = False
        self._state_lock = threading.Lock()

        # Publishers
        self.pub_tts        = self.create_publisher(String, "/tts_speak",        10)
        self.pub_tts_urgent = self.create_publisher(String, "/tts_speak_urgent", 10)
        self.pub_intent     = self.create_publisher(String, "/dialog_intent",    10)
        self.pub_action     = self.create_publisher(String, "/dialog_action",    10)

        # Subscribers
        self.create_subscription(String, "/asr_text",      self._cb_asr,      10)
        self.create_subscription(String, "/emotion_label", self._cb_emotion,   10)
        self.create_subscription(String, "/activity_label",self._cb_activity,  10)
        self.create_subscription(String, "/speaker_id",    self._cb_speaker,   10)
        self.create_subscription(Bool,   "/doa_active",    self._cb_doa,       10)

        self.get_logger().info(
            "Dialog node v2 ready | "
            "subscribes: /asr_text /emotion_label /activity_label /speaker_id | "
            "publishes: /tts_speak /tts_speak_urgent /dialog_intent /dialog_action"
        )

    # ── Perception callbacks (just store state — never do heavy work here) ──

    def _cb_emotion(self, msg: String):
        with self._state_lock:
            self._emotion = msg.data

    def _cb_activity(self, msg: String):
        with self._state_lock:
            self._activity = msg.data

    def _cb_speaker(self, msg: String):
        with self._state_lock:
            self._speaker_id = msg.data

    def _cb_doa(self, msg: Bool):
        with self._state_lock:
            self._doa_active = msg.data

    def _cb_asr(self, msg: String):
        """
        Main callback — triggered when ASR produces transcribed text.
        Runs dialog processing in a thread to avoid blocking ROS2 executor.
        """
        text = msg.data.strip()
        if not text:
            return

        # Capture current state snapshot (thread-safe)
        with self._state_lock:
            emotion    = self._emotion
            activity   = self._activity
            speaker_id = self._speaker_id

        self.get_logger().info(
            f"ASR: '{text}' | emotion={emotion} activity={activity} "
            f"speaker={speaker_id}"
        )

        # Process in background thread — Gemini calls can take 1-3 seconds
        t = threading.Thread(
            target=self._process_and_publish,
            args=(text, emotion, activity, speaker_id),
            daemon=True,
        )
        t.start()

    def _process_and_publish(self, text: str, emotion: str,
                              activity: str, speaker_id: str):
        """Background: run dialog manager and publish result."""
        result = self.dm.process_input(
            text       = text,
            emotion    = emotion,
            activity   = activity,
            speaker_id = speaker_id,
        )

        response = result["response"]
        intent   = result["intent"]
        action   = result["action"]

        # Route to urgent or normal TTS topic
        if intent in self.EMERGENCY_INTENTS:
            msg = String(); msg.data = response
            self.pub_tts_urgent.publish(msg)
            self.get_logger().warn(f"🚨 URGENT [{intent}]: {response}")
        else:
            msg = String(); msg.data = response
            self.pub_tts.publish(msg)
            self.get_logger().info(f"💬 [{intent}] → '{response}'")

        # Publish intent
        msg_intent      = String()
        msg_intent.data = intent
        self.pub_intent.publish(msg_intent)

        # Publish action if present
        if action:
            msg_action      = String()
            msg_action.data = action
            self.pub_action.publish(msg_action)
            self.get_logger().info(f"⚡ Action: {action}")

    def destroy_node(self):
        super().destroy_node()


# ─────────────────────────────────────────────────────────────
# Standalone interactive test
# ─────────────────────────────────────────────────────────────

CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def run_interactive():
    print(f"{BOLD}{CYAN}{'='*60}")
    print("  Module 3 — Dialog Manager v2 — Interactive Test")
    print(f"{'='*60}{RESET}")
    print("Commands:")
    print(f"  {YELLOW}/emotion [name]{RESET}   — set emotion (happy/sad/angry/neutral/fearful)")
    print(f"  {YELLOW}/activity [name]{RESET}  — set activity (sitting/walking/standing)")
    print(f"  {YELLOW}/speaker [name]{RESET}   — set speaker identity")
    print(f"  {YELLOW}/history{RESET}          — show last 5 conversation turns")
    print(f"  {YELLOW}/prefs{RESET}            — show disliked responses count")
    print(f"  {YELLOW}/reset_prefs{RESET}      — clear all preferences")
    print(f"  {YELLOW}quit{RESET}              — exit")
    print()
    print(f"  {GREEN}💡 Try: 'i don't like that' after any response")
    print(f"  💡 Try: 'I tumbled' or 'help me' to test emergency detection")
    print(f"  💡 Try: 'tell me a joke' then 'I don't like that' multiple times{RESET}")
    print()

    dm = DialogManager()

    emotion    = "neutral"
    activity   = "sitting"
    speaker_id = "Dheeraj"

    while True:
        print(f"\n{BOLD}[{speaker_id} | {emotion} | {activity}]{RESET}")
        try:
            user_text = input(f"{YELLOW}You: {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_text:
            continue

        if user_text.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        # State commands
        if user_text.startswith("/emotion"):
            parts = user_text.split()
            if len(parts) > 1:
                emotion = parts[1]
                print(f"{CYAN}Emotion set: {emotion}{RESET}")
            continue

        if user_text.startswith("/activity"):
            parts = user_text.split()
            if len(parts) > 1:
                activity = parts[1]
                print(f"{CYAN}Activity set: {activity}{RESET}")
            continue

        if user_text.startswith("/speaker"):
            parts = user_text.split()
            if len(parts) > 1:
                speaker_id = parts[1]
                print(f"{CYAN}Speaker set: {speaker_id}{RESET}")
            continue

        if user_text.lower() == "/history":
            history = dm.get_history(5)
            print(f"\n{BOLD}Last {len(history)} turns:{RESET}")
            for i, turn in enumerate(history, 1):
                print(f"  {i}. [{turn['intent']}] "
                      f"{turn['user_input']!r} → {turn['response']!r}")
            continue

        if user_text.lower() == "/prefs":
            count = dm.preferences.count_disliked(speaker_id)
            print(f"{CYAN}Disliked responses for {speaker_id}: {count}{RESET}")
            continue

        if user_text.lower() == "/reset_prefs":
            dm.reset_preferences(speaker_id)
            continue

        # Process
        result = dm.process_input(
            text=user_text, emotion=emotion,
            activity=activity, speaker_id=speaker_id
        )

        intent_color = RED if result["intent"] == "emergency" else GREEN
        print(f"{BOLD}Robot{RESET} [{intent_color}{result['intent']}{RESET}]: "
              f"{result['response']}")

        if result.get("action"):
            print(f"  {CYAN}⚡ Action: {result['action']}{RESET}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Dialog node v2")
    p.add_argument("--ros",         action="store_true", help="Run as ROS2 node")
    p.add_argument("--interactive", action="store_true", help="Interactive standalone test")
    return p.parse_args()


def main():
    args = parse_args()

    if args.ros:
        if not ROS_AVAILABLE:
            print("❌ rclpy not available. Install ROS2 Humble.")
            sys.exit(1)
        rclpy.init()
        node = DialogNodeV2()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            node.get_logger().info("Shutting down dialog node v2")
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        run_interactive()


if __name__ == "__main__":
    main()
