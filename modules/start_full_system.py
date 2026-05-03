#!/usr/bin/env python3
"""
Full System Launch Script — DIAT Social Robot
----------------------------------------------
Starts all required modules as background subprocesses.

Modules launched:
  M1  — DOA Node          (doa_node_v2.py)
  M2  — ASR Node          (asr_node_v2.py)
  M3  — Dialog Manager    (dialog_node_v2.py)
  M4  — TTS Node          (tts_node_v2.py)
  M6  — Reminder System   (reminder_node.py)
  M11 — Person Detection  (person_node.py) [+ integrated M12 Emotion]
  M14 — Activity Node     (activity_node_v2.py)
  M16 — Speaker Recognition (speaker_recognition_node.py)
  M17 — System Architecture Orchestrator (module17_node.py)

Usage:
  cd modules/
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python3 start_full_system.py

Press Ctrl+C to shut down all modules gracefully.
"""

import os
import sys
import time
import signal
import warnings
import subprocess

# Suppress protobuf deprecation warnings globally
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

BASE = os.path.dirname(os.path.abspath(__file__))

# ─── Module definitions ─────────────────────────────────────
# Each entry: (name, script_path, extra_args)
MODULES = [
    ("M1-DOA",       os.path.join(BASE, "module1_doa/doa_node_v2.py"),                  ["--ros", "--channels", "2"]),
    ("M2-ASR",       os.path.join(BASE, "module2_Speech_to_text/asr_node_v2.py"),        ["--ros"]),
    ("M3-Dialog",    os.path.join(BASE, "module3_dialog_manager/dialog_node_v2.py"),     ["--ros"]),
    ("M4-TTS",       os.path.join(BASE, "module4_text_to_speech/tts_node_v2.py"),        ["--ros"]),
    ("M6-Reminder",  os.path.join(BASE, "module6_reminder_system/reminder_node.py"),     ["--ros"]),
    ("M11-Person",   os.path.join(BASE, "module11_person_detection/person_node.py"),     ["--ros"]),
    # M12-Emotion is integrated directly into M11 (person_node.py) to avoid camera conflict
    ("M14-Activity", os.path.join(BASE, "module14_human_activity/activity_node_v2.py"),
                     ["--ros", "--model", os.path.join(BASE, "module14_human_activity/models/activity_model_v2.pkl")]),
    ("M16-Speaker",  os.path.join(BASE, "module16_speaker_recognition/speaker_recognition_node.py"), ["--ros"]),
]

M10_DIR = os.path.join(BASE, "module10_integration")

processes = []

def start_module(name, script, args):
    """Start a Python module as a background subprocess."""
    if not os.path.exists(script):
        print(f"  ⚠️  [{name}] Script not found: {script} — skipping")
        return None

    cmd = [sys.executable, script] + args
    env = os.environ.copy()
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    env["PYTHONWARNINGS"] = "ignore"   # suppress protobuf deprecation spam
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(script),
            env=env
            # stdout/stderr NOT piped — visible directly in terminal
        )
        print(f"  ✅ [{name}] Started (PID {proc.pid})")
        return proc
    except Exception as e:
        print(f"  ❌ [{name}] Failed to start: {e}")
        return None

def shutdown_all():
    """Gracefully shut down all started processes."""
    print("\n🛑 Shutting down all modules...")
    for name, proc in processes:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
                print(f"  ✅ [{name}] Stopped")
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"  ⚠️  [{name}] Force-killed")

def sigint_handler(sig, frame):
    shutdown_all()
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)
    
    print("=" * 60)
    print("  DIAT Social Robot — Full System Startup")
    print("=" * 60)

    # Step 1: Start all perception/interaction modules
    print("\n📡 Starting perception & interaction modules...\n")
    for name, script, args in MODULES:
        proc = start_module(name, script, args)
        processes.append((name, proc))
        time.sleep(1.0)  # Stagger startup to avoid race conditions

    print(f"\n⏳ Waiting 3 seconds for modules to initialize...\n")
    time.sleep(3)

    # Step 2: Check which modules actually started
    alive = [name for name, proc in processes if proc and proc.poll() is None]
    dead  = [name for name, proc in processes if not proc or proc.poll() is not None]
    
    if alive:
        print(f"  🟢 Running: {', '.join(alive)}")
    if dead:
        print(f"  🔴 Not running: {', '.join(dead)}")

    # Step 3: Launch Module 17 System Architecture Orchestrator
    print("\n🧠 Launching Module 17 System Architecture...\n")
    m17_cmd = [sys.executable, os.path.join(BASE, "module17_currect_file/module17_node.py"), "--ros"]
    
    try:
        m17_proc = subprocess.Popen(
            m17_cmd,
            cwd=BASE,
            text=True
        )
        processes.append(("M17-Orchestrator", m17_proc))
        print(f"  ✅ [M17-Orchestrator] Launched (PID {m17_proc.pid})")
    except Exception as e:
        print(f"  ❌ [M17-Orchestrator] Failed: {e}")
    
    print("\n" + "=" * 60)
    print("  All modules running. Press Ctrl+C to stop.")
    print("=" * 60 + "\n")

    # Step 4: Keep alive — monitor processes (report each death only once)
    reported_dead = set()
    try:
        while True:
            for name, proc in processes:
                if proc and proc.poll() is not None and name not in reported_dead:
                    print(f"  ⚠️  [{name}] Died (exit code {proc.returncode})")
                    reported_dead.add(name)
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_all()

if __name__ == "__main__":
    main()
