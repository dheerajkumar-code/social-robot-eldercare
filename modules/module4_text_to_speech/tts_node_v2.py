#!/usr/bin/env python3
"""
Module 4 — TTS Node (v2)
--------------------------
Production text-to-speech for elderly healthcare robot.

What this fixes vs tts_node.py (v1):
  - v1: runAndWait() blocks ROS2 executor thread for entire speech duration
  - v2: Speech runs in background thread — ROS2 callbacks never blocked

  - v1: No priority — fall alert waits behind "Good morning"
  - v2: PriorityQueue — URGENT (fall/emergency) interrupts current speech instantly

  - v1: No way to interrupt speech in progress
  - v2: Urgent messages kill current speech process and speak immediately

  - v1: No speaking status — ASR transcribes robot's own voice
  - v2: Publishes /tts_speaking (Bool) — ASR mutes while True

  - v1: Rate/volume hardcoded at 150
  - v2: Emergency = faster + louder, reminders = normal pace

  - v1: No deduplication — same message spoken twice if duplicate msg received
  - v2: Dedup within 5-second window

Speech Backend (Pi5):
  Uses espeak subprocess — truly interruptible via process.terminate()
  Falls back to pyttsx3 if espeak not available

Topics subscribed:
  /tts_speak        std_msgs/String  — normal speech (reminders, dialogue)
  /tts_speak_urgent std_msgs/String  — urgent speech (fall alert, emergency)
                                       interrupts whatever is currently playing

Topics published:
  /tts_speaking     std_msgs/Bool    — True while speaking (mutes ASR)
  /tts_done         std_msgs/String  — echoes text when speech finishes

Standalone:
  python3 tts_node_v2.py --text "Hello, how are you?"
  python3 tts_node_v2.py --interactive

ROS2:
  python3 tts_node_v2.py --ros

Test urgent interrupt:
  ros2 topic pub /tts_speak std_msgs/String "data: 'This is a long message...'"
  ros2 topic pub /tts_speak_urgent std_msgs/String "data: 'EMERGENCY! Are you okay?'"
"""

import os
import sys
import time
import queue
import logging
import threading
import argparse
import hashlib
import subprocess
import shutil
from collections import deque

# Optional ROS2
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String, Bool
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

# Optional pyttsx3 (fallback if espeak not available)
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
RATE_NORMAL    = 135    # wpm — comfortable for elderly
RATE_URGENT    = 155    # wpm — emergency alerts slightly faster
VOLUME_NORMAL  = 80     # espeak amplitude 0-200
VOLUME_URGENT  = 150    # louder for emergencies

DEDUP_WINDOW_SEC = 5.0  # ignore identical messages within this window
QUEUE_MAXSIZE    = 20   # max queued messages before dropping old ones

# Priority levels — lower number = higher priority
PRIORITY_URGENT = 0
PRIORITY_NORMAL = 1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("tts_v2")


# ─────────────────────────────────────────────────────────────
# Speech Backend
# ─────────────────────────────────────────────────────────────

class SpeechBackend:
    """
    Wraps either espeak (preferred, interruptible) or pyttsx3 (fallback).

    espeak is preferred because:
    - subprocess.terminate() instantly kills it mid-sentence
    - pyttsx3 engine.stop() is unreliable on Linux/Pi5
    - espeak is always available on Pi5 (required by pyttsx3 anyway)
    """

    def __init__(self):
        self._proc      = None          # current espeak subprocess
        self._proc_lock = threading.Lock()
        self._pyttsx3_engine = None

        # Detect backend
        if shutil.which("espeak") or shutil.which("espeak-ng"):
            self._binary  = shutil.which("espeak-ng") or shutil.which("espeak")
            self._backend = "espeak"
            logger.info(f"✅ TTS backend: espeak ({self._binary})")
        elif PYTTSX3_AVAILABLE:
            self._backend = "pyttsx3"
            self._pyttsx3_engine = pyttsx3.init()
            logger.info("✅ TTS backend: pyttsx3 (espeak not found)")
        else:
            raise RuntimeError(
                "No TTS backend found.\n"
                "Install espeak: sudo apt-get install espeak\n"
                "Or pyttsx3:    pip install pyttsx3"
            )

    def speak(self, text: str, rate: int = RATE_NORMAL, volume: int = VOLUME_NORMAL):
        """
        Speak text. Blocks until speech finishes.
        Call from background thread only.
        """
        if not text:
            return

        if self._backend == "espeak":
            self._speak_espeak(text, rate, volume)
        else:
            self._speak_pyttsx3(text, rate, volume)

    def _speak_espeak(self, text: str, rate: int, volume: int):
        """Run espeak as subprocess. Interruptible via stop()."""
        cmd = [
            self._binary,
            "-r", str(rate),
            "-a", str(volume),
            "-v", "en",
            text
        ]
        with self._proc_lock:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        try:
            self._proc.wait()
        except Exception:
            pass
        finally:
            with self._proc_lock:
                self._proc = None

    def _speak_pyttsx3(self, text: str, rate: int, volume: int):
        """pyttsx3 fallback. Less interruptible."""
        e = self._pyttsx3_engine
        e.setProperty("rate",   rate)
        e.setProperty("volume", min(1.0, volume / 100.0))
        e.say(text)
        e.runAndWait()

    def stop(self):
        """
        Interrupt current speech immediately.
        Returns instantly — does not wait for process to die.
        """
        if self._backend == "espeak":
            with self._proc_lock:
                if self._proc and self._proc.poll() is None:
                    self._proc.terminate()
                    logger.debug("Speech interrupted (espeak terminated)")
        else:
            try:
                self._pyttsx3_engine.stop()
            except Exception:
                pass

    @property
    def backend_name(self) -> str:
        return self._backend


# ─────────────────────────────────────────────────────────────
# TTS Engine (priority queue + background thread)
# ─────────────────────────────────────────────────────────────

class TTSEngine:
    """
    Thread-safe TTS engine with priority queue and speech interruption.

    Usage:
        engine = TTSEngine(on_speaking=callback)
        engine.speak("Hello")                        # normal
        engine.speak_urgent("EMERGENCY! Fall!")      # interrupts immediately
    """

    def __init__(self, on_speaking=None, on_done=None):
        """
        Args:
            on_speaking : callback(is_speaking: bool) — called when state changes
            on_done     : callback(text: str) — called when utterance finishes
        """
        self.on_speaking = on_speaking
        self.on_done     = on_done

        self._backend    = SpeechBackend()
        self._queue      = queue.PriorityQueue(maxsize=QUEUE_MAXSIZE)
        self._stop_flag  = threading.Event()
        self._is_speaking = False
        self._seq        = 0              # tie-breaker for equal priorities

        # Deduplication: hash → timestamp
        self._recent: deque = deque(maxlen=50)

        # Background worker thread
        self._thread = threading.Thread(target=self._worker, daemon=True, name="tts_worker")
        self._thread.start()

        logger.info(f"TTS engine ready (backend={self._backend.backend_name})")

    # ── Public API ──

    def speak(self, text: str, rate: int = RATE_NORMAL, volume: int = VOLUME_NORMAL):
        """Enqueue text at normal priority. Non-blocking."""
        self._enqueue(text, PRIORITY_NORMAL, rate, volume)

    def speak_urgent(self, text: str, rate: int = RATE_URGENT, volume: int = VOLUME_URGENT):
        """
        Enqueue text at urgent priority.
        Interrupts any currently playing speech instantly.
        """
        logger.info(f"🚨 URGENT: {text}")
        self._backend.stop()               # kill current speech NOW
        self._enqueue(text, PRIORITY_URGENT, rate, volume)

    def stop(self):
        """Stop current speech and clear queue."""
        self._backend.stop()
        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def shutdown(self):
        """Clean shutdown."""
        self._stop_flag.set()
        self.stop()

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    # ── Internal ──

    def _enqueue(self, text: str, priority: int, rate: int, volume: int):
        """Add to priority queue with dedup check."""
        text = text.strip()
        if not text:
            return

        # Dedup: skip if same text spoken within DEDUP_WINDOW_SEC
        text_hash = hashlib.md5(text.encode()).hexdigest()
        now = time.time()
        for h, t in list(self._recent):
            if h == text_hash and (now - t) < DEDUP_WINDOW_SEC:
                logger.debug(f"Dedup skip: {text!r}")
                return
        self._recent.append((text_hash, now))

        self._seq += 1
        try:
            self._queue.put_nowait((priority, self._seq, text, rate, volume))
        except queue.Full:
            logger.warning("TTS queue full — dropping oldest message")
            # Drop lowest priority item (can't easily do with PriorityQueue)
            # Instead just skip this normal-priority item
            if priority == PRIORITY_NORMAL:
                return
            # For urgent: force-add by draining one item first
            try:
                self._queue.get_nowait()
                self._queue.put_nowait((priority, self._seq, text, rate, volume))
            except Exception:
                pass

    def _worker(self):
        """Background thread: dequeues and speaks. Never stops until shutdown."""
        while not self._stop_flag.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            _, _, text, rate, volume = item

            self._set_speaking(True)
            try:
                logger.info(f"🗣️ Speaking: {text!r}")
                self._backend.speak(text, rate=rate, volume=volume)
                if self.on_done:
                    self.on_done(text)
            except Exception as e:
                logger.error(f"Speech error: {e}")
            finally:
                self._set_speaking(False)

    def _set_speaking(self, state: bool):
        if self._is_speaking != state:
            self._is_speaking = state
            if self.on_speaking:
                self.on_speaking(state)


# ─────────────────────────────────────────────────────────────
# ROS2 Node
# ─────────────────────────────────────────────────────────────

class TTSNodeV2(Node):
    """
    ROS2 TTS node with priority queue and speech interruption.

    Subscribes:
      /tts_speak        String — normal speech (dialogue, reminders)
      /tts_speak_urgent String — urgent speech (fall alert, emergency)
                                 INTERRUPTS current speech immediately

    Publishes:
      /tts_speaking     Bool   — True while speaking (use to mute ASR)
      /tts_done         String — echoes text after speech completes
    """

    def __init__(self):
        super().__init__("tts_node_v2")

        # ROS2 parameters
        self.declare_parameter("rate_normal", RATE_NORMAL)
        self.declare_parameter("rate_urgent", RATE_URGENT)
        self.declare_parameter("volume_normal", VOLUME_NORMAL)
        self.declare_parameter("volume_urgent", VOLUME_URGENT)

        # Publishers
        self.pub_speaking = self.create_publisher(Bool,   "/tts_speaking", 10)
        self.pub_done     = self.create_publisher(String, "/tts_done",     10)

        # TTS engine with callbacks
        self.engine = TTSEngine(
            on_speaking=self._on_speaking_changed,
            on_done=self._on_done,
        )

        # Subscribers
        self.create_subscription(String, "/tts_speak",        self._cb_normal, 10)
        self.create_subscription(String, "/tts_speak_urgent", self._cb_urgent, 10)

        self.get_logger().info(
            f"TTS node v2 ready | backend={self.engine._backend.backend_name} | "
            f"rate={RATE_NORMAL}wpm | topics: /tts_speak, /tts_speak_urgent"
        )

    def _cb_normal(self, msg: String):
        """Callback for /tts_speak — normal priority."""
        text = msg.data.strip()
        if text:
            rate   = self.get_parameter("rate_normal").value
            volume = self.get_parameter("volume_normal").value
            self.engine.speak(text, rate=rate, volume=volume)
            self.get_logger().info(f"Queued (normal): {text!r}")

    def _cb_urgent(self, msg: String):
        """Callback for /tts_speak_urgent — interrupts immediately."""
        text = msg.data.strip()
        if text:
            rate   = self.get_parameter("rate_urgent").value
            volume = self.get_parameter("volume_urgent").value
            self.engine.speak_urgent(text, rate=rate, volume=volume)
            self.get_logger().warn(f"🚨 Urgent queued: {text!r}")

    def _on_speaking_changed(self, is_speaking: bool):
        """Called from TTS background thread when speaking state changes."""
        msg = Bool()
        msg.data = is_speaking
        self.pub_speaking.publish(msg)

    def _on_done(self, text: str):
        """Called from TTS background thread when utterance finishes."""
        msg = String()
        msg.data = text
        self.pub_done.publish(msg)

    def destroy_node(self):
        self.engine.shutdown()
        super().destroy_node()


# ─────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────

def run_standalone_text(text: str):
    """Speak a single message and exit."""
    engine = TTSEngine()
    engine.speak(text)
    time.sleep(0.2)
    while engine.is_speaking:
        time.sleep(0.1)
    engine.shutdown()


def run_interactive():
    """Interactive mode to test TTS and interrupt behavior."""
    print("=" * 55)
    print("  Module 4 — TTS v2 (interactive test)")
    print("=" * 55)
    print("  Commands:")
    print("    <text>          → speak normally")
    print("    !<text>         → speak URGENT (interrupts)")
    print("    stop            → stop current speech")
    print("    quit / exit     → exit\n")

    spoken = []

    def on_done(text):
        spoken.append(text)

    engine = TTSEngine(on_done=on_done)

    try:
        while True:
            try:
                line = input(">>> ").strip()
            except EOFError:
                break

            if not line:
                continue
            if line.lower() in ("quit", "exit"):
                break
            if line.lower() == "stop":
                engine.stop()
                print("    [stopped]")
            elif line.startswith("!"):
                engine.speak_urgent(line[1:].strip())
                print("    [urgent queued — interrupting]")
            else:
                engine.speak(line)
                print("    [queued]")

    except KeyboardInterrupt:
        pass
    finally:
        engine.shutdown()
        print(f"\nSpoke {len(spoken)} utterances.")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="TTS node v2")
    p.add_argument("--ros",         action="store_true", help="Run as ROS2 node")
    p.add_argument("--interactive", action="store_true", help="Interactive standalone test")
    p.add_argument("--text",        type=str, default=None, help="Speak one message and exit")
    return p.parse_args()


def main():
    args = parse_args()

    if args.text:
        run_standalone_text(args.text)
    elif args.interactive:
        run_interactive()
    elif args.ros:
        if not ROS_AVAILABLE:
            logger.error("rclpy not available. Install ROS2 Humble.")
            sys.exit(1)
        rclpy.init()
        node = TTSNodeV2()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            node.get_logger().info("Shutting down TTS node v2")
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        run_interactive()


if __name__ == "__main__":
    main()
