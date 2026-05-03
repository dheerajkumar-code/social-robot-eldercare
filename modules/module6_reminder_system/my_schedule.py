#!/usr/bin/env python3
"""
Module 6 — My Custom Schedule
================================
Create your own reminders interactively.
When a reminder fires → text shows on screen + voice speaks through speakers.

Usage:
  python3 my_schedule.py
"""

import os
import sys
import json
import shutil
import subprocess
import threading
import time
from datetime import datetime

# ── Audio output fix (route to Speaker+Headphones) ──────────────
os.environ.setdefault("PULSE_SINK",
    "alsa_output.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__hw_sofhdadsp__sink")

# ── APScheduler ──────────────────────────────────────────────────
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron         import CronTrigger
    from apscheduler.triggers.interval     import IntervalTrigger
    import logging
    logging.getLogger("apscheduler").setLevel(logging.ERROR)
except ImportError:
    print("❌ APScheduler not installed. Run: pip install apscheduler")
    sys.exit(1)

# ── Colours ──────────────────────────────────────────────────────
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
MAGENTA= "\033[95m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

SCHEDULE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_schedule.json")

# ── TTS engine ───────────────────────────────────────────────────
ESPEAK       = shutil.which("espeak-ng") or shutil.which("espeak")
SPEAKER_SINK = (
    "alsa_output.pci-0000_00_1f.3-platform-skl_hda_dsp_generic"
    ".HiFi__hw_sofhdadsp__sink"
)

def _speak_thread(text: str, urgent: bool):
    """Run espeak-ng in a background thread with correct audio sink."""
    speed  = 145 if urgent else 135
    volume = 180 if urgent else 160

    # Force PulseAudio to use Speaker+Headphones sink
    env = os.environ.copy()
    env["PULSE_SINK"] = SPEAKER_SINK

    # Set default sink via pactl
    subprocess.run(
        ["pactl", "set-default-sink", SPEAKER_SINK],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    subprocess.run(
        ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "100%"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    # espeak-ng uses -s for speed (words/min) and -a for amplitude
    cmd = [ESPEAK, "-s", str(speed), "-a", str(volume), "-v", "en", text]
    try:
        subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"  [TTS error: {e}]")

def speak(text: str, urgent: bool = False):
    """Speak text via espeak-ng (non-blocking — runs in background thread)."""
    if not ESPEAK:
        return
    t = threading.Thread(target=_speak_thread, args=(text, urgent), daemon=True)
    t.start()


# ── Reminder storage ─────────────────────────────────────────────
reminders: list[dict] = []

def load_schedule():
    global reminders
    if os.path.exists(SCHEDULE_FILE):
        try:
            with open(SCHEDULE_FILE) as f:
                reminders = json.load(f)
            print(f"  📂 Loaded {len(reminders)} saved reminder(s) from {SCHEDULE_FILE}")
        except Exception:
            reminders = []

def save_schedule():
    try:
        with open(SCHEDULE_FILE, "w") as f:
            json.dump(reminders, f, indent=2)
    except Exception as e:
        print(f"  ⚠️  Could not save schedule: {e}")


# ── Reminder firing ───────────────────────────────────────────────
def fire_reminder(name: str, message: str, rtype: str, urgent: bool):
    """Called by scheduler when a reminder fires."""
    now = datetime.now().strftime("%H:%M:%S")

    # Pick emoji
    emoji = {
        "medicine": "💊",
        "food":     "🍽️",
        "water":    "💧",
        "walking":  "🚶",
        "custom":   "⏰",
    }.get(rtype, "⏰")

    border = "🚨" * 30 if urgent else "─" * 50

    # ── Print to screen ──────────────────────────────────────────
    print(f"\n{BOLD}{RED if urgent else YELLOW}{border}{RESET}")
    print(f"{BOLD}{emoji}  [{now}]  REMINDER: {name}{RESET}")
    print(f"{CYAN}   {message}{RESET}")
    print(f"{BOLD}{RED if urgent else YELLOW}{border}{RESET}\n>>> ", end="", flush=True)

    # ── Speak out loud ───────────────────────────────────────────
    tts_text = f"Reminder. {name}. {message}"
    speak(tts_text, urgent=urgent)


# ── Schedule builder ──────────────────────────────────────────────
scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
scheduler.start()

def schedule_reminder(r: dict):
    """Add a single reminder dict to the APScheduler."""
    job_id = r["id"]
    name   = r["name"]
    msg    = r["message"]
    rtype  = r.get("type", "custom")
    urgent = r.get("urgent", False)

    if r.get("mode") == "interval":
        minutes = r.get("interval_minutes", 60)
        trigger = IntervalTrigger(minutes=minutes)
    else:
        hour, minute = map(int, r["time"].split(":"))
        trigger = CronTrigger(hour=hour, minute=minute)

    # Remove existing job if updating
    try:
        scheduler.remove_job(job_id)
    except Exception:
        pass

    scheduler.add_job(
        func    = fire_reminder,
        trigger = trigger,
        id      = job_id,
        kwargs  = {"name": name, "message": msg, "rtype": rtype, "urgent": urgent},
        replace_existing=True,
    )


def schedule_all():
    """Schedule all reminders in the list."""
    for r in reminders:
        try:
            schedule_reminder(r)
        except Exception as e:
            print(f"  ⚠️  Could not schedule [{r['name']}]: {e}")


# ── Input helpers ─────────────────────────────────────────────────
def get_input(prompt, default=None):
    try:
        val = input(f"  {YELLOW}{prompt}{RESET}" +
                    (f" [{default}]" if default else "") + ": ").strip()
        return val if val else default
    except (KeyboardInterrupt, EOFError):
        print()
        return default


def print_header():
    print(f"\n{BOLD}{CYAN}{'═'*55}")
    print("   🤖  My Custom Reminder Schedule")
    print(f"{'═'*55}{RESET}")


def print_menu():
    print(f"\n{BOLD}{'─'*45}{RESET}")
    print(f"  {YELLOW}1{RESET}. ➕  Add medicine reminder")
    print(f"  {YELLOW}2{RESET}. 🍽️   Add food/meal reminder")
    print(f"  {YELLOW}3{RESET}. 💧  Add water/drink reminder")
    print(f"  {YELLOW}4{RESET}. 🚶  Add walking/exercise reminder")
    print(f"  {YELLOW}5{RESET}. ⏰  Add custom reminder")
    print(f"  {YELLOW}6{RESET}. 📋  View all reminders")
    print(f"  {YELLOW}7{RESET}. 🗑️   Delete a reminder")
    print(f"  {YELLOW}8{RESET}. 🔔  Test voice (speak a message now)")
    print(f"  {YELLOW}9{RESET}. ▶️   START running the schedule")
    print(f"  {YELLOW}0{RESET}. 💾  Save & exit")
    print(f"{BOLD}{'─'*45}{RESET}")


def make_id():
    import random, string
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=6))


def print_reminders():
    if not reminders:
        print(f"\n  {RED}No reminders yet. Add some first!{RESET}")
        return
    print(f"\n{BOLD}  {'#':3}  {'Type':8}  {'Name':22}  {'Time/Interval':18}  {'Priority'}{RESET}")
    print(f"  {'─'*70}")
    for i, r in enumerate(reminders, 1):
        if r.get("mode") == "interval":
            timing = f"every {r.get('interval_minutes', 60)} min"
        else:
            timing = r.get("time", "?")
        priority = "🚨 URGENT" if r.get("urgent") else "normal"
        print(f"  {i:3}.  {r.get('type','custom'):8}  {r['name']:22}  {timing:18}  {priority}")


# ── Add reminder helpers ──────────────────────────────────────────
def add_medicine():
    print(f"\n{GREEN}{BOLD}─── Add Medicine Reminder ───{RESET}")
    name    = get_input("Medicine name (e.g. Morning Aspirin)", "Morning Medicine")
    med     = get_input("Medicine / tablet name",               "Aspirin 75mg")
    dosage  = get_input("Dosage (e.g. 1 tablet, 5ml)",          "1 tablet")
    instr   = get_input("Instructions (e.g. after food)",       "Take with water")
    ttime   = get_input("Time to remind (HH:MM, 24h)",          "08:00")
    urgent  = get_input("Mark as URGENT? (yes/no)",             "yes").lower() == "yes"
    msg     = f"Please take {dosage} of {med}. {instr}."
    r = {"id": make_id(), "name": name, "type": "medicine",
         "message": msg, "time": ttime, "mode": "fixed", "urgent": urgent}
    reminders.append(r)
    save_schedule()
    print(f"\n  {GREEN}✅ Added: {name} at {ttime}{RESET}")
    return r


def add_food():
    print(f"\n{GREEN}{BOLD}─── Add Food / Meal Reminder ───{RESET}")
    meal  = get_input("Meal type (breakfast/lunch/dinner/snack)", "lunch")
    name  = get_input("Reminder name", f"{meal.capitalize()} Time")
    note  = get_input("Any dietary note (press Enter to skip)",   "")
    ttime = get_input("Time to remind (HH:MM, 24h)",              "13:00")
    emoji = {"breakfast": "🌅", "lunch": "☀️", "dinner": "🌙", "snack": "🍎"}.get(meal, "🍽️")
    msg   = f"{emoji} It's {meal} time! Please have your meal."
    if note:
        msg += f" Note: {note}."
    r = {"id": make_id(), "name": name, "type": "food",
         "message": msg, "time": ttime, "mode": "fixed", "urgent": False}
    reminders.append(r)
    save_schedule()
    print(f"\n  {GREEN}✅ Added: {name} at {ttime}{RESET}")
    return r


def add_water():
    print(f"\n{GREEN}{BOLD}─── Add Water / Drink Reminder ───{RESET}")
    name     = get_input("Reminder name",                      "Hydration Check")
    amount   = get_input("How much to drink (e.g. 1 glass)",   "1 glass of water")
    interval = get_input("Remind every how many minutes?",     "60")
    try:
        interval = int(interval)
    except ValueError:
        interval = 60
    msg = f"💧 Time to drink {amount}! Staying hydrated is important for your health."
    r = {"id": make_id(), "name": name, "type": "water",
         "message": msg, "interval_minutes": interval, "mode": "interval", "urgent": False}
    reminders.append(r)
    save_schedule()
    print(f"\n  {GREEN}✅ Added: {name} every {interval} minutes{RESET}")
    return r


def add_walking():
    print(f"\n{GREEN}{BOLD}─── Add Walking / Exercise Reminder ───{RESET}")
    name     = get_input("Reminder name",                         "Morning Walk")
    activity = get_input("Activity type (walking/stretching/yoga/exercise)", "walking")
    duration = get_input("Duration in minutes",                   "20")
    ttime    = get_input("Time to remind (HH:MM, 24h)",           "09:00")
    emoji    = {"walking": "🚶", "stretching": "🧘", "yoga": "🧘", "exercise": "💪"}.get(activity, "🏃")
    msg      = f"{emoji} Time for your {duration}-minute {activity}! Gentle movement is great for your health."
    r = {"id": make_id(), "name": name, "type": "walking",
         "message": msg, "time": ttime, "mode": "fixed", "urgent": False}
    reminders.append(r)
    save_schedule()
    print(f"\n  {GREEN}✅ Added: {name} at {ttime}{RESET}")
    return r


def add_custom():
    print(f"\n{GREEN}{BOLD}─── Add Custom Reminder ───{RESET}")
    name   = get_input("Reminder name",                         "My Reminder")
    msg    = get_input("Message to show and speak",             "Time for your reminder!")
    mode   = get_input("Mode: (1) Fixed time  (2) Every N mins", "1")
    urgent = get_input("Mark as URGENT? (yes/no)",              "no").lower() == "yes"
    if mode == "2":
        interval = int(get_input("Remind every how many minutes?", "30"))
        r = {"id": make_id(), "name": name, "type": "custom",
             "message": msg, "interval_minutes": interval, "mode": "interval", "urgent": urgent}
        reminders.append(r)
        save_schedule()
        print(f"\n  {GREEN}✅ Added: {name} every {interval} minutes{RESET}")
    else:
        ttime = get_input("Time to remind (HH:MM, 24h)", "10:00")
        r = {"id": make_id(), "name": name, "type": "custom",
             "message": msg, "time": ttime, "mode": "fixed", "urgent": urgent}
        reminders.append(r)
        save_schedule()
        print(f"\n  {GREEN}✅ Added: {name} at {ttime}{RESET}")
    return r


def delete_reminder():
    print_reminders()
    if not reminders:
        return
    choice = get_input("Enter number to delete (or Enter to cancel)", "")
    if not choice:
        return
    try:
        idx = int(choice) - 1
        r = reminders[idx]
        try:
            scheduler.remove_job(r["id"])
        except Exception:
            pass
        reminders.pop(idx)
        save_schedule()
        print(f"\n  {RED}🗑️  Deleted: {r['name']}{RESET}")
    except (ValueError, IndexError):
        print(f"  {RED}Invalid choice{RESET}")


def run_schedule():
    """Start the live schedule and wait for reminders to fire."""
    if not reminders:
        print(f"\n  {RED}No reminders added! Add at least one first.{RESET}")
        return

    schedule_all()

    print(f"\n{BOLD}{GREEN}{'═'*55}")
    print("   ▶️  Schedule is now RUNNING!")
    print(f"{'═'*55}{RESET}")
    print(f"  {len(reminders)} reminder(s) scheduled.")
    print(f"  Text will appear here and voice will speak when reminders fire.")
    print(f"\n  {YELLOW}Upcoming reminders:{RESET}")
    for job in scheduler.get_jobs():
        nxt = str(job.next_run_time)[:16] if job.next_run_time else "interval"
        print(f"   • {job.kwargs.get('name', job.id):25s}  next: {nxt}")
    print(f"\n  {BOLD}Press Ctrl+C to stop and return to menu.{RESET}\n")

    # Speak confirmation
    speak("Schedule is now running. I will remind you at the right times.", urgent=False)

    try:
        while True:
            now = datetime.now().strftime("%H:%M:%S")
            print(f"  ⏱  Running... {now}  ({len(reminders)} reminders active)  ", end="\r")
            time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n\n  {YELLOW}⏸  Schedule paused. Back to menu.{RESET}")


# ── Main loop ─────────────────────────────────────────────────────
def main():
    # Set audio output to speakers
    subprocess.run(
        ["pactl", "set-default-sink",
         "alsa_output.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__hw_sofhdadsp__sink"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    print_header()
    load_schedule()

    print(f"\n  {CYAN}Welcome! Build your personal reminder schedule below.{RESET}")
    if ESPEAK:
        print(f"  {GREEN}✅ Voice: espeak-ng ready (will speak through speakers){RESET}")
        speak("Welcome! Your reminder system is ready.")
    else:
        print(f"  {RED}⚠️  espeak-ng not found — voice disabled{RESET}")

    while True:
        print_menu()
        choice = get_input("Choose an option", "6")

        if   choice == "1": add_medicine()
        elif choice == "2": add_food()
        elif choice == "3": add_water()
        elif choice == "4": add_walking()
        elif choice == "5": add_custom()
        elif choice == "6": print_reminders()
        elif choice == "7": delete_reminder()
        elif choice == "8":
            msg = get_input("Type a message to test voice", "Hello! Your reminder system is working perfectly.")
            print(f"  {CYAN}Speaking: {msg}{RESET}")
            speak(msg)
        elif choice == "9":
            run_schedule()
        elif choice == "0":
            save_schedule()
            scheduler.shutdown(wait=False)
            print(f"\n  {GREEN}💾 Schedule saved. Goodbye! 👋{RESET}\n")
            break
        else:
            print(f"  {RED}Invalid choice. Try again.{RESET}")


if __name__ == "__main__":
    main()
