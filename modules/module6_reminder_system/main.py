#!/usr/bin/env python3
"""
Module 6 — main.py
---------------------
Entry point for the DIAT Robot Reminder System.

Two modes:
  1. Demo mode  (default) — runs a 60-second demo with test reminders
  2. CLI mode   (--cli)   — interactive menu to manage reminders

Usage:
  python3 main.py           # demo mode
  python3 main.py --cli     # interactive CLI
  python3 main.py --test    # unit tests only
  python3 main.py --ros     # launch as ROS2 node

The demo creates realistic reminders for an elderly patient and
shows the scheduler triggering them in real time.
"""

import os
import sys
import time
import json
import logging
import argparse
import threading
from datetime import datetime, timedelta

# Set up logging before any imports
logging.basicConfig(
    level=logging.WARNING,   # suppress APScheduler internal noise
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# Keep our loggers visible
for name in ("ReminderRegistry", "ReminderScheduler", "Notifier"):
    logging.getLogger(name).setLevel(logging.INFO)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reminder_base  import Priority, RepeatMode
from reminder_types import (MedicineReminder, FoodReminder,
                             DrinkingReminder, WalkingReminder,
                             TYPE_REGISTRY, from_dict_any)
from registry       import ReminderRegistry
from notifier       import Notifier
from scheduler      import ReminderScheduler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────
# Demo helpers — schedule reminders to fire a few seconds ahead
# ─────────────────────────────────────────────────────────────

def _time_ahead(seconds: int) -> str:
    """Return HH:MM for N seconds from now (for demo scheduling)."""
    t = datetime.now() + timedelta(seconds=seconds)
    return t.strftime("%H:%M")


def build_demo_reminders(registry: ReminderRegistry):
    """
    Create a realistic set of reminders for an elderly patient.
    Demo reminders fire 10, 25, 40s from now for immediate demo feedback.
    """
    print("\n📋 Creating demo reminders...")

    # ── Medicine reminders ──
    med1 = MedicineReminder(
        name          = "Morning Aspirin",
        medicine_name = "Aspirin 75mg",
        dosage        = "1 tablet",
        trigger_time  = _time_ahead(10),     # fires in 10s (demo)
        instructions  = "Take with a glass of water after breakfast",
        priority      = Priority.HIGH,
        repeat_mode   = RepeatMode.ONCE,     # one-time for demo
    )

    med2 = MedicineReminder(
        name          = "Evening Metformin",
        medicine_name = "Metformin 500mg",
        dosage        = "2 tablets",
        trigger_time  = "20:00",
        instructions  = "Take with dinner. Do not skip",
        priority      = Priority.HIGH,
        repeat_mode   = RepeatMode.DAILY,
    )

    # ── Food reminders ──
    food1 = FoodReminder(
        name         = "Breakfast",
        meal_type    = "breakfast",
        trigger_time = _time_ahead(25),     # fires in 25s (demo)
        food_note    = "Low sodium, high fibre diet",
        priority     = Priority.MEDIUM,
        repeat_mode  = RepeatMode.ONCE,
    )

    food2 = FoodReminder(
        name         = "Lunch",
        meal_type    = "lunch",
        trigger_time = "13:00",
        food_note    = "Diabetic-friendly meal",
        priority     = Priority.MEDIUM,
        repeat_mode  = RepeatMode.DAILY,
    )

    food3 = FoodReminder(
        name         = "Dinner",
        meal_type    = "dinner",
        trigger_time = "19:30",
        priority     = Priority.MEDIUM,
        repeat_mode  = RepeatMode.DAILY,
    )

    # ── Drinking reminder ──
    water = DrinkingReminder(
        name             = "Hydration Check",
        interval_minutes = 1,               # every 1 min for demo (real = 120)
        amount_ml        = 200,
        drink_type       = "water",
        priority         = Priority.MEDIUM,
    )

    # ── Walking reminder ──
    walk = WalkingReminder(
        name             = "Morning Walk",
        trigger_time     = _time_ahead(40),  # fires in 40s (demo)
        duration_minutes = 20,
        steps_goal       = 2000,
        activity_type    = "walking",
        priority         = Priority.MEDIUM,
        repeat_mode      = RepeatMode.ONCE,
    )

    walk2 = WalkingReminder(
        name             = "Evening Stretch",
        trigger_time     = "18:00",
        duration_minutes = 10,
        steps_goal       = 500,
        activity_type    = "stretching",
        priority         = Priority.LOW,
        repeat_mode      = RepeatMode.DAILY,
    )

    # Add all to registry
    reminders = [med1, med2, food1, food2, food3, water, walk, walk2]
    for r in reminders:
        registry.add(r)
        print(f"  ✅ Added: {r}")

    return reminders


# ─────────────────────────────────────────────────────────────
# Demo mode
# ─────────────────────────────────────────────────────────────

def run_demo():
    """
    60-second demo showing the reminder system in action.
    Expected output:
      t=10s  Medicine reminder fires
      t=25s  Breakfast reminder fires
      t=40s  Walking reminder fires
      t=60s  Water reminder fires (1-min interval)
    """
    print("=" * 60)
    print("  DIAT Robot — Reminder System Demo")
    print("=" * 60)
    print("  Duration: 70 seconds")
    print("  Expected triggers:")
    print("    t+10s  → Medicine (Aspirin)")
    print("    t+25s  → Food (Breakfast)")
    print("    t+40s  → Walking reminder")
    print("    t+60s  → Water reminder (1-min interval)")
    print("=" * 60)

    # Clear any old data for clean demo
    registry  = ReminderRegistry()
    registry.clear_all()

    notifier  = Notifier(use_tts=False)   # terminal only (no hardware needed)
    scheduler = ReminderScheduler(registry, notifier, timezone="Asia/Kolkata")

    # Add demo reminders
    build_demo_reminders(registry)

    # Print registry summary
    print(f"\n📊 Registry summary: {registry.summary()}")
    print("\n📋 All registered reminders:")
    registry.print_all()

    # Start scheduler
    scheduler.start()
    scheduler.add_all()

    print(f"\n⏳ Scheduler running with {scheduler.job_count()} jobs")
    print("   Waiting for reminders to fire... (Ctrl+C to stop)\n")

    scheduler.list_jobs()

    # Export to JSON
    json_path = registry.export_json()
    print(f"💾 Reminders exported to: {json_path}\n")

    # Run for 70 seconds
    try:
        for remaining in range(70, 0, -1):
            print(f"  ⏱  {remaining:2d}s remaining...  "
                  f"(triggers: {sum(r.trigger_count for r in registry.get_all())})",
                  end="\r")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Stopped by user")

    print("\n\n📊 Final trigger counts:")
    for r in registry.get_all():
        print(f"  [{r.id}] {r.name:30s} fired {r.trigger_count}x  "
              f"last={r.last_triggered or 'never'}")

    scheduler.stop()
    print("\n✅ Demo complete")


# ─────────────────────────────────────────────────────────────
# Interactive CLI
# ─────────────────────────────────────────────────────────────

def run_cli():
    """Interactive command-line interface for managing reminders."""

    registry  = ReminderRegistry()
    notifier  = Notifier(use_tts=False)
    scheduler = ReminderScheduler(registry, notifier, timezone="Asia/Kolkata")
    scheduler.start()
    scheduler.add_all()

    CYAN  = "\033[96m"
    GREEN = "\033[92m"
    YELLOW= "\033[93m"
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    def print_menu():
        print(f"\n{BOLD}{CYAN}{'='*50}")
        print("  DIAT Reminder System — Main Menu")
        print(f"{'='*50}{RESET}")
        print(f"  {YELLOW}1{RESET}. List all reminders")
        print(f"  {YELLOW}2{RESET}. Add medicine reminder")
        print(f"  {YELLOW}3{RESET}. Add food reminder")
        print(f"  {YELLOW}4{RESET}. Add drinking reminder")
        print(f"  {YELLOW}5{RESET}. Add walking reminder")
        print(f"  {YELLOW}6{RESET}. Pause a reminder")
        print(f"  {YELLOW}7{RESET}. Resume a reminder")
        print(f"  {YELLOW}8{RESET}. Delete a reminder")
        print(f"  {YELLOW}9{RESET}. Show scheduled jobs")
        print(f"  {YELLOW}10{RESET}. Export reminders to JSON")
        print(f"  {YELLOW}11{RESET}. Import reminders from JSON")
        print(f"  {YELLOW}12{RESET}. Test notification now")
        print(f"  {YELLOW}0{RESET}.  Exit")
        print(f"{CYAN}{'='*50}{RESET}")

    def get_input(prompt, default=None):
        val = input(f"  {prompt}: ").strip()
        return val if val else default

    def add_medicine():
        print(f"\n{GREEN}--- Add Medicine Reminder ---{RESET}")
        name    = get_input("Reminder name",      "Morning Medicine")
        med     = get_input("Medicine name",       "Aspirin 75mg")
        dosage  = get_input("Dosage",              "1 tablet")
        ttime   = get_input("Time (HH:MM)",        "08:00")
        instr   = get_input("Instructions (Enter to skip)", "")
        prio    = int(get_input("Priority 1=Low 2=Med 3=High 4=Urgent", "3"))

        r = MedicineReminder(
            name=name, medicine_name=med, dosage=dosage,
            trigger_time=ttime, instructions=instr,
            priority=Priority(prio),
        )
        registry.add(r)
        scheduler.add_job(r)
        print(f"\n  {GREEN}✅ Added: [{r.id}] {r.name}{RESET}")

    def add_food():
        print(f"\n{GREEN}--- Add Food Reminder ---{RESET}")
        name      = get_input("Reminder name",        "Breakfast")
        meal_type = get_input("Meal type (breakfast/lunch/dinner/snack)", "lunch")
        ttime     = get_input("Time (HH:MM)",         "13:00")
        note      = get_input("Dietary note (Enter to skip)", "")

        r = FoodReminder(
            name=name, meal_type=meal_type,
            trigger_time=ttime, food_note=note,
        )
        registry.add(r)
        scheduler.add_job(r)
        print(f"\n  {GREEN}✅ Added: [{r.id}] {r.name}{RESET}")

    def add_drinking():
        print(f"\n{GREEN}--- Add Drinking Reminder ---{RESET}")
        name     = get_input("Reminder name",          "Hydration")
        interval = int(get_input("Interval (minutes)", "120"))
        amount   = int(get_input("Amount (ml)",        "200"))
        dtype    = get_input("Drink type",             "water")

        r = DrinkingReminder(
            name=name, interval_minutes=interval,
            amount_ml=amount, drink_type=dtype,
        )
        registry.add(r)
        scheduler.add_job(r)
        print(f"\n  {GREEN}✅ Added: [{r.id}] every {interval} min{RESET}")

    def add_walking():
        print(f"\n{GREEN}--- Add Walking Reminder ---{RESET}")
        name     = get_input("Reminder name",          "Morning Walk")
        ttime    = get_input("Time HH:MM (or Enter for interval)", None)
        interval = None
        if not ttime:
            interval = int(get_input("Interval (minutes)", "60"))
        duration = int(get_input("Duration (minutes)", "15"))
        steps    = int(get_input("Steps goal",         "1000"))
        atype    = get_input("Activity type (walking/stretching/exercise)", "walking")

        r = WalkingReminder(
            name=name, trigger_time=ttime,
            interval_minutes=interval, duration_minutes=duration,
            steps_goal=steps, activity_type=atype,
        )
        registry.add(r)
        scheduler.add_job(r)
        print(f"\n  {GREEN}✅ Added: [{r.id}] {r.name}{RESET}")

    # ── Main CLI loop ──
    print(f"\n{BOLD}Welcome to DIAT Reminder System{RESET}")
    print(f"  {len(registry)} reminders loaded, {scheduler.job_count()} scheduled\n")

    while True:
        print_menu()
        choice = get_input("Choose", "0")

        if choice == "0":
            print("\n  Saving and exiting...")
            registry.export_json()
            scheduler.stop()
            print("  Goodbye! 👋")
            break

        elif choice == "1":
            print(f"\n{BOLD}All Reminders ({len(registry)} total):{RESET}")
            registry.print_all()
            print(f"\nSummary: {registry.summary()}")

        elif choice == "2":  add_medicine()
        elif choice == "3":  add_food()
        elif choice == "4":  add_drinking()
        elif choice == "5":  add_walking()

        elif choice == "6":
            rid = get_input("Reminder ID to pause")
            if registry.pause(rid):
                scheduler.pause_job(rid)
                print(f"  ⏸️  Paused: {rid}")
            else:
                print(f"  ❌ ID not found: {rid}")

        elif choice == "7":
            rid = get_input("Reminder ID to resume")
            if registry.resume(rid):
                r = registry.get(rid)
                if r: scheduler.add_job(r)
                print(f"  ▶️  Resumed: {rid}")
            else:
                print(f"  ❌ ID not found: {rid}")

        elif choice == "8":
            rid = get_input("Reminder ID to delete")
            if registry.delete(rid):
                scheduler.remove_job(rid)
                print(f"  🗑️  Deleted: {rid}")
            else:
                print(f"  ❌ ID not found: {rid}")

        elif choice == "9":
            print(f"\n{BOLD}Scheduled Jobs:{RESET}")
            scheduler.list_jobs()

        elif choice == "10":
            path = registry.export_json()
            print(f"  💾 Exported to: {path}")

        elif choice == "11":
            path = get_input("JSON file path", "reminders.json")
            count = registry.import_json(path)
            if count:
                scheduler.add_all()
            print(f"  📥 Imported {count} reminders")

        elif choice == "12":
            print("  Sending test notification...")
            notifier.test_notification()

        else:
            print(f"  ❌ Invalid choice: {choice}")


# ─────────────────────────────────────────────────────────────
# Unit Tests
# ─────────────────────────────────────────────────────────────

def run_tests():
    """Fast unit tests — no scheduler, no timing needed."""
    import tempfile

    print("=" * 55)
    print("  Module 6 — Unit Tests")
    print("=" * 55)

    # T1: All reminder types create and build messages
    print("\nT1 Reminder type creation + message building")
    med   = MedicineReminder("Morning Med", "Aspirin", "1 tablet", "08:00")
    food  = FoodReminder("Lunch", "lunch", "13:00", "Low sodium")
    water = DrinkingReminder("Water", interval_minutes=120, amount_ml=250)
    walk  = WalkingReminder("Walk", trigger_time="09:00", duration_minutes=20)

    for r in [med, food, water, walk]:
        msg = r.build_message()
        assert len(msg) > 10, f"Empty message for {r.reminder_type}"
        assert r.reminder_type in msg.lower() or any(
            word in msg for word in ["medicine","meal","hydration","activity",
                                     "breakfast","lunch","water","walk",
                                     "aspirin","stretch"]
        )
        print(f"  ✅ {r.reminder_type:10s}: {msg[:60]}...")

    # T2: Registry CRUD
    print("\nT2 Registry CRUD operations")
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_db = f.name
    reg = ReminderRegistry(db_path=tmp_db)
    reg.clear_all()

    rid = reg.add(med)
    assert reg.get(rid) is not None
    assert len(reg.get_all()) == 1
    assert len(reg.get_by_type("medicine")) == 1
    assert len(reg.get_by_type("food")) == 0
    print("  ✅ add / get / get_all / get_by_type")

    reg.add(food); reg.add(water); reg.add(walk)
    assert len(reg) == 4
    print(f"  ✅ 4 reminders in registry")

    # Update
    med.name = "Updated Medicine"
    reg.update(med)
    assert reg.get(rid).name == "Updated Medicine"
    print("  ✅ update works")

    # Pause / Resume
    reg.pause(rid)
    assert not reg.get(rid).is_active()
    reg.resume(rid)
    assert reg.get(rid).is_active()
    print("  ✅ pause / resume")

    # Delete
    assert reg.delete(rid) is True
    assert reg.get(rid) is None
    assert reg.delete("nonexistent") is False
    print("  ✅ delete (existing + nonexistent)")

    # T3: Persistence (JSON export/import)
    print("\nT3 JSON export / import")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        tmp_json = f.name

    export_path = reg.export_json(tmp_json)
    assert os.path.exists(export_path)

    reg2 = ReminderRegistry(db_path=tmp_db + "2")
    reg2.clear_all()
    count = reg2.import_json(tmp_json)
    assert count == len(reg), f"Expected {len(reg)} imported, got {count}"
    print(f"  ✅ Exported and re-imported {count} reminders")

    # T4: to_dict / from_dict round-trip
    print("\nT4 Serialisation round-trip")
    for original in [med, food, water, walk]:
        d   = original.to_dict()
        r2  = from_dict_any(d)
        assert r2.reminder_type == original.reminder_type
        assert r2.name          == original.name
        assert r2.priority      == original.priority
        print(f"  ✅ {original.reminder_type:10s} round-trip OK")

    # T5: on_trigger updates state
    print("\nT5 Trigger state update")
    r = DrinkingReminder("Test Water", interval_minutes=60)
    assert r.trigger_count    == 0
    assert r.last_triggered  is None
    msg = r.on_trigger()
    assert r.trigger_count    == 1
    assert r.last_triggered  is not None
    assert "💧" in msg
    print(f"  ✅ trigger_count={r.trigger_count}  last_triggered={r.last_triggered[:10]}")

    # T6: TYPE_REGISTRY extensibility
    print("\nT6 TYPE_REGISTRY")
    assert "medicine" in TYPE_REGISTRY
    assert "food"     in TYPE_REGISTRY
    assert "drinking" in TYPE_REGISTRY
    assert "walking"  in TYPE_REGISTRY
    print(f"  ✅ Registry types: {list(TYPE_REGISTRY.keys())}")

    # T7: Registry summary
    print("\nT7 Registry summary")
    summary = reg.summary()
    assert "total"     in summary
    assert "by_type"   in summary
    assert "by_status" in summary
    print(f"  ✅ Summary: {summary}")

    # Cleanup
    import os as _os
    for p in [tmp_db, tmp_db+"2", tmp_json]:
        try: _os.unlink(p)
        except: pass

    print()
    print("=" * 55)
    print("  ✅ All 7 tests passed")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DIAT Reminder System")
    p.add_argument("--demo",  action="store_true", help="Run 70s demo (default)")
    p.add_argument("--cli",   action="store_true", help="Interactive CLI")
    p.add_argument("--test",  action="store_true", help="Run unit tests")
    p.add_argument("--ros",   action="store_true", help="Launch as ROS2 node")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.test:
        run_tests()
    elif args.cli:
        run_cli()
    elif args.ros:
        import subprocess
        subprocess.run(["python3", "reminder_node.py"])
    else:
        run_demo()   # default
