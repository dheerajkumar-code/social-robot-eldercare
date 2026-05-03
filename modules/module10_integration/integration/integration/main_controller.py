#!/usr/bin/env python3
"""
Module 10 — Integration Controller
====================================
Central coordinator for the elderly care robot.
Connects M2, M3, M4, M6, M11, M12, M14, M16 into one unified system.

Workflow:
  1. M11 (face) or M16 (voice) identifies the user
  2. M12 (emotion), M14 (activity), M3 (dialog) activate in parallel
  3. Controller routes data, handles priorities, manages emergencies
  4. M4 (TTS) speaks all responses
  5. M6 (reminders) runs in background continuously

Priority Order:
  P1 — EMERGENCY     : Fall detected (M14)
  P2 — SAFETY        : Unacknowledged reminder x 3
  P3 — REMINDER      : Scheduled alert (M6)
  P4 — EMOTION       : Emotional state (M12)
  P5 — CONVERSATION  : User speech (M2 -> M3)
  P6 — GREETING      : Idle / user just arrived
"""

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String, Bool, Float32
import json
import time
from datetime import datetime

from integration.session_manager import UserSession
from integration.event_logger    import EventLogger


IDLE      = "IDLE"
ACTIVE    = "ACTIVE"
EMERGENCY = "EMERGENCY"

EMOTION_RESPONSES = {
    "angry":     "I can see you are angry. Would you like to talk about it?",
    "sad":       "You seem sad. Would you like to talk?",
    "happy":     "You look happy today! That is wonderful.",
    "fearful":   "You seem a little scared. Everything is okay, I am here.",
    "disgusted": "Something seems to be bothering you. How can I help?",
    "surprised": "You look surprised! Did something happen?",
}


class IntegrationController(Node):

    def __init__(self):
        super().__init__('integration_controller')

        self.session  = UserSession()
        self.logger   = EventLogger()
        self.state    = IDLE

        # --- FIX: greeting guard so we never greet more than once per session ---
        self._greeted          = False
        self._person_absent_since = 0.0   # timestamp when face last disappeared
        self.PERSON_LEAVE_TIMEOUT = 10.0  # seconds absent before resetting session

        # Reminder ACK tracking
        self.pending_reminder_id  = ""
        self.pending_reminder_msg = ""
        self.reminder_retries     = 0
        self.MAX_RETRIES          = 3
        self.ack_timer            = None

        # Emergency tracking
        self.emergency_timer      = None

        # Idle greeting cooldown (5 min)
        self.last_greeting_time   = 0.0
        self.GREETING_COOLDOWN    = 300.0

        # Speaker confidence cache
        self._speaker_confidence  = 0.0

        # ── Publishers ────────────────────────────────────────────────────────
        self.pub_tts        = self.create_publisher(String, '/tts_speak',        10)
        self.pub_tts_urgent = self.create_publisher(String, '/tts_speak_urgent', 10)
        self.pub_ack        = self.create_publisher(String, '/reminder_ack',     10)
        self.pub_state      = self.create_publisher(String, '/system_state',     10)
        self.pub_identity   = self.create_publisher(String, '/user_identity',    10)
        self.pub_emergency  = self.create_publisher(Bool,   '/emergency_flag',   10)

        # ── Subscribers ───────────────────────────────────────────────────────

        # Identification (M11, M16)
        self.create_subscription(Bool,    '/person_detected',    self._on_person_detected,    10)
        self.create_subscription(String,  '/person_name',        self._on_person_name,        10)
        self.create_subscription(String,  '/speaker_id',         self._on_speaker_id,         10)
        self.create_subscription(Float32, '/speaker_confidence', self._on_speaker_confidence, 10)

        # Perception (M12, M14)
        self.create_subscription(String,  '/emotion_label',      self._on_emotion,            10)
        self.create_subscription(String,  '/activity_label',     self._on_activity,           10)
        self.create_subscription(String,  '/activity_alert',     self._on_activity_alert,     10)

        # Communication (M2, M3, M4)
        self.create_subscription(String,  '/asr_text',           self._on_speech,             10)
        self.create_subscription(String,  '/dialog_intent',      self._on_dialog_intent,      10)
        self.create_subscription(Bool,    '/tts_speaking',       self._on_tts_busy,           10)

        # Reminders (M6)
        self.create_subscription(String,  '/reminder_alert',     self._on_reminder,           10)
        self.create_subscription(String,  '/reminder_ack',       self._on_reminder_ack,       10)

        # ── Timers ────────────────────────────────────────────────────────────
        self.create_timer(5.0,  self._session_watchdog)   # check person absence
        self.create_timer(10.0, self._heartbeat)           # health check

        # ── Boot ──────────────────────────────────────────────────────────────
        self._set_state(IDLE)
        self.logger.log("SYSTEM_START", {"time": datetime.now().isoformat()})
        self.get_logger().info("=" * 55)
        self.get_logger().info("  Integration Controller READY")
        self.get_logger().info("  Waiting for user (M11 face / M16 voice)...")
        self.get_logger().info("=" * 55)

    # ══════════════════════════════════════════════════════════════════════════
    # IDENTIFICATION LAYER
    # ══════════════════════════════════════════════════════════════════════════

    def _on_person_detected(self, msg: Bool):
        """M11: person visible/not visible in camera frame."""
        if not msg.data:
            # Face left frame — note the time, session_watchdog will reset later
            if self._person_absent_since == 0.0:
                self._person_absent_since = time.time()
        else:
            # Face back in frame — cancel absence timer
            self._person_absent_since = 0.0

    def _session_watchdog(self):
        """
        Reset session only after person has been absent for PERSON_LEAVE_TIMEOUT seconds.
        This prevents rapid IDLE/ACTIVE toggling when face briefly leaves frame.
        """
        if (self._person_absent_since > 0.0 and
                time.time() - self._person_absent_since > self.PERSON_LEAVE_TIMEOUT):
            if self.session.identified:
                self.get_logger().info(
                    f"{self.session.name} left. Resetting session."
                )
                self.logger.log("PERSON_LEFT", {"name": self.session.name})
            self.session.reset()
            self._greeted = False
            self._person_absent_since = 0.0
            self._set_state(IDLE)

    def _on_person_name(self, msg: String):
        """M11: face recognised — we know WHO it is."""
        if self.session.identify(msg.data, "face", 1.0):
            self._on_user_identified()

    def _on_speaker_id(self, msg: String):
        """M16: voice recognised — we know WHO is speaking."""
        if self.session.identify(msg.data, "voice", self._speaker_confidence):
            self._on_user_identified()

    def _on_speaker_confidence(self, msg: Float32):
        self._speaker_confidence = float(msg.data)

    def _on_user_identified(self):
        """
        Called when a user is positively identified.
        Greets the user ONCE per session — never again until they leave and return.
        """
        # FIX: guard against repeated calls on every camera frame
        if self._greeted:
            return

        self._greeted = True
        self._set_state(ACTIVE)
        self.pub_identity.publish(self._str(self.session.to_json()))

        self.logger.log("USER_IDENTIFIED", {
            "name":   self.session.name,
            "source": self.session.source,
        })
        self.get_logger().info(
            f"User identified: {self.session.name} via {self.session.source}"
        )

        # Greet once
        self._greet(self.session.name)

    # ══════════════════════════════════════════════════════════════════════════
    # EMERGENCY RESPONSE  (M14 fall detection — Priority 1)
    # ══════════════════════════════════════════════════════════════════════════

    def _on_activity_alert(self, msg: String):
        if msg.data != "FALL_DETECTED":
            return

        self.get_logger().error("EMERGENCY: Fall detected!")
        self._set_state(EMERGENCY)
        self._speak_urgent("Are you okay? Please respond.")
        self.logger.log("FALL_DETECTED", {"time": datetime.now().isoformat()})

        if self.emergency_timer:
            self.emergency_timer.cancel()
        self.emergency_timer = self.create_timer(10.0, self._escalate_emergency)

    def _escalate_emergency(self):
        self.emergency_timer.cancel()
        self.emergency_timer = None
        self.get_logger().error("No response — calling emergency contact!")
        self._speak_urgent(
            "No response received. Calling the emergency contact now. "
            "Please stay calm, help is on the way."
        )
        flag = Bool(); flag.data = True
        self.pub_emergency.publish(flag)
        self.logger.log("EMERGENCY_ESCALATED", {"time": datetime.now().isoformat()})
        self._set_state(ACTIVE)

    # ══════════════════════════════════════════════════════════════════════════
    # EMOTION-AWARE CONVERSATION  (M12 — Priority 4)
    # ══════════════════════════════════════════════════════════════════════════

    def _on_emotion(self, msg: String):
        label = msg.data.lower().strip()
        if not label or label == "neutral":
            self.session.update_emotion(label)
            return

        if self.state != ACTIVE:
            return
        if label == self.session.last_emotion:
            return   # already responded

        self.session.update_emotion(label)
        self.session.last_emotion = label

        response = EMOTION_RESPONSES.get(label)
        if response:
            self.get_logger().info(f"Emotion: {label}")
            self._speak(response)
            self.logger.log("EMOTION_RESPONSE", {"emotion": label})

    def _on_activity(self, msg: String):
        self.session.update_activity(msg.data)

    # ══════════════════════════════════════════════════════════════════════════
    # CONVERSATION PIPELINE  (M2 STT -> M3 Dialog -> M4 TTS — Priority 5)
    # ══════════════════════════════════════════════════════════════════════════

    def _on_speech(self, msg: String):
        """
        M2 transcribes speech and publishes to /asr_text.
        M3 dialog manager subscribes to /asr_text directly and replies via /tts_speak.
        Controller only:
          - tracks conversation state
          - handles emergency cancel if user responds after fall
          - logs interactions
        """
        text = msg.data.strip()
        if not text:
            return

        self.session.in_conversation = True
        self.get_logger().info(f"User said: '{text}'")
        self.logger.log("USER_SPEECH", {"text": text})

        # If in emergency, any speech cancels escalation
        if self.state == EMERGENCY:
            self.get_logger().info("User responded — cancelling emergency.")
            if self.emergency_timer:
                self.emergency_timer.cancel()
                self.emergency_timer = None
            self._speak("I am glad you are okay. Please be careful.")
            self._set_state(ACTIVE)
            self.logger.log("EMERGENCY_CANCELLED", {"response": text})

    def _on_dialog_intent(self, msg: String):
        if msg.data:
            self.session.in_conversation = True

    def _on_tts_busy(self, msg: Bool):
        self.session.tts_busy = msg.data

    # ══════════════════════════════════════════════════════════════════════════
    # REMINDER SYSTEM  (M6 — Priority 3)
    # ══════════════════════════════════════════════════════════════════════════

    def _on_reminder(self, msg: String):
        if self.pending_reminder_id:
            return  # don't stack reminders

        try:
            data  = json.loads(msg.data)
            r_id  = data.get("id",      "unknown")
            r_msg = data.get("message", msg.data)
        except Exception:
            r_id  = "unknown"
            r_msg = msg.data

        self.pending_reminder_id  = r_id
        self.pending_reminder_msg = r_msg
        self.reminder_retries     = 0

        self.get_logger().info(f"Reminder: {r_msg}")
        self._speak(r_msg)
        self.logger.log("REMINDER_SENT", {"id": r_id, "message": r_msg})

        if self.ack_timer:
            self.ack_timer.cancel()
        self.ack_timer = self.create_timer(30.0, self._reminder_timeout)

    def _on_reminder_ack(self, msg: String):
        if msg.data != self.pending_reminder_id:
            return
        self.get_logger().info(f"Reminder acknowledged: {self.pending_reminder_id}")
        self.logger.log("REMINDER_ACK", {"id": self.pending_reminder_id})
        self._clear_reminder()

    def _reminder_timeout(self):
        if not self.pending_reminder_id:
            return
        self.reminder_retries += 1
        if self.reminder_retries <= self.MAX_RETRIES:
            self.get_logger().warn(
                f"Reminder retry {self.reminder_retries}/{self.MAX_RETRIES}"
            )
            self._speak(f"Reminder: {self.pending_reminder_msg}")
            self.logger.log("REMINDER_RETRY", {
                "id": self.pending_reminder_id, "retry": self.reminder_retries
            })
            if self.ack_timer:
                self.ack_timer.cancel()
            self.ack_timer = self.create_timer(30.0, self._reminder_timeout)
        else:
            self.get_logger().error("Reminder unacknowledged — alerting caregiver.")
            self._speak_urgent(
                "The patient has not responded to the reminder. "
                "Caregiver please check on the patient."
            )
            self.logger.log("REMINDER_ESCALATED", {"id": self.pending_reminder_id})
            self._clear_reminder()

    def _clear_reminder(self):
        self.pending_reminder_id  = ""
        self.pending_reminder_msg = ""
        self.reminder_retries     = 0
        if self.ack_timer:
            self.ack_timer.cancel()
            self.ack_timer = None

    # ══════════════════════════════════════════════════════════════════════════
    # GREETING  (once on arrival + 5-min idle)
    # ══════════════════════════════════════════════════════════════════════════

    def _greet(self, name=""):
        hour = datetime.now().hour
        if   hour < 12: msg = f"Good morning{', ' + name if name else ''}! How are you feeling today?"
        elif hour < 17: msg = f"Good afternoon{', ' + name if name else ''}! Are you comfortable?"
        elif hour < 21: msg = f"Good evening{', ' + name if name else ''}! How was your day?"
        else:           msg = f"Hello{', ' + name if name else ''}! Is everything alright?"

        self._speak(msg)
        self.last_greeting_time = time.time()
        self.logger.log("GREETING", {"message": msg})

    # ══════════════════════════════════════════════════════════════════════════
    # HEARTBEAT  (every 10 s)
    # ══════════════════════════════════════════════════════════════════════════

    def _heartbeat(self):
        self.get_logger().info(
            f"[HEARTBEAT] state={self.state} | user={self.session.name} | "
            f"emotion={self.session.emotion} | activity={self.session.activity}"
        )

        # Idle greeting every 5 min when user is present but quiet
        if (self.state == ACTIVE
                and not self.session.tts_busy
                and not self.session.in_conversation
                and (time.time() - self.last_greeting_time) > self.GREETING_COOLDOWN):
            self._greet(self.session.name)

        self.session.in_conversation = False

    # ══════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    def _speak(self, text: str):
        self.pub_tts.publish(self._str(text))
        self.get_logger().info(f"[TTS] {text}")

    def _speak_urgent(self, text: str):
        self.pub_tts_urgent.publish(self._str(text))
        self.get_logger().warn(f"[TTS URGENT] {text}")

    def _set_state(self, new_state: str):
        if new_state != self.state:
            self.get_logger().info(f"State: {self.state} -> {new_state}")
            self.state = new_state
            self.logger.log("STATE_CHANGE", {"state": new_state})
        self.pub_state.publish(self._str(new_state))

    @staticmethod
    def _str(text: str) -> String:
        m = String(); m.data = str(text); return m


def main(args=None):
    rclpy.init(args=args)
    node = IntegrationController()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
