#!/usr/bin/env python3
"""
Module 6 — reminder_types.py
-------------------------------
Concrete reminder type implementations.

To add a NEW reminder type:
  1. Create a class that inherits from BaseReminder
  2. Implement: reminder_type, build_message(), to_dict(), from_dict()
  3. Register it in TYPE_REGISTRY at the bottom of this file
  4. That's it — no other file needs to change

Current types:
  - MedicineReminder   → fixed time, dosage tracking
  - FoodReminder       → meal times, dietary notes
  - DrinkingReminder   → interval-based hydration
  - WalkingReminder    → activity scheduling
"""

from datetime import datetime
from typing import Optional
from reminder_base import BaseReminder, Priority, RepeatMode, ReminderStatus


# ─────────────────────────────────────────────────────────────
# Medicine Reminder
# ─────────────────────────────────────────────────────────────

class MedicineReminder(BaseReminder):
    """
    Reminder for taking medication.
    Tracks medicine name, dosage, and any special instructions.
    """

    def __init__(self,
                 name:          str,
                 medicine_name: str,
                 dosage:        str,
                 trigger_time:  str,
                 instructions:  str = "",
                 priority:      Priority   = Priority.HIGH,
                 repeat_mode:   RepeatMode = RepeatMode.DAILY,
                 enabled:       bool = True,
                 reminder_id:   Optional[str] = None):
        """
        Args:
            medicine_name : e.g. "Aspirin", "Metformin 500mg"
            dosage        : e.g. "2 tablets", "5ml syrup"
            trigger_time  : "HH:MM"  e.g. "08:00"
            instructions  : e.g. "Take after food", "Avoid dairy"
        """
        super().__init__(
            name=name,
            message="",               # built dynamically
            priority=priority,
            repeat_mode=repeat_mode,
            trigger_time=trigger_time,
            enabled=enabled,
            reminder_id=reminder_id,
        )
        self.medicine_name = medicine_name
        self.dosage        = dosage
        self.instructions  = instructions

    @property
    def reminder_type(self) -> str:
        return "medicine"

    def build_message(self) -> str:
        msg = f"💊 Medicine Time! Please take {self.dosage} of {self.medicine_name}."
        if self.instructions:
            msg += f" {self.instructions}."
        msg += f" (Reminder: {self.name})"
        return msg

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "medicine_name": self.medicine_name,
            "dosage":        self.dosage,
            "instructions":  self.instructions,
        })
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "MedicineReminder":
        obj = cls(
            name          = data["name"],
            medicine_name = data.get("medicine_name", ""),
            dosage        = data.get("dosage", ""),
            trigger_time  = data.get("trigger_time", "08:00"),
            instructions  = data.get("instructions", ""),
            priority      = Priority(data.get("priority", 3)),
            repeat_mode   = RepeatMode(data.get("repeat_mode", "daily")),
            enabled       = data.get("enabled", True),
            reminder_id   = data.get("id"),
        )
        obj.status          = ReminderStatus(data.get("status", "active"))
        obj.created_at      = data.get("created_at", obj.created_at)
        obj.last_triggered  = data.get("last_triggered")
        obj.trigger_count   = data.get("trigger_count", 0)
        return obj


# ─────────────────────────────────────────────────────────────
# Food Reminder
# ─────────────────────────────────────────────────────────────

class FoodReminder(BaseReminder):
    """
    Reminder for meal times.
    Supports breakfast, lunch, dinner, and snack reminders.
    """

    MEAL_TYPES = ["breakfast", "lunch", "dinner", "snack", "custom"]

    def __init__(self,
                 name:         str,
                 meal_type:    str,
                 trigger_time: str,
                 food_note:    str = "",
                 priority:     Priority   = Priority.MEDIUM,
                 repeat_mode:  RepeatMode = RepeatMode.DAILY,
                 enabled:      bool = True,
                 reminder_id:  Optional[str] = None):
        """
        Args:
            meal_type   : "breakfast" | "lunch" | "dinner" | "snack" | "custom"
            trigger_time: "HH:MM"
            food_note   : e.g. "Low sodium diet", "Diabetic meal"
        """
        super().__init__(
            name=name,
            message="",
            priority=priority,
            repeat_mode=repeat_mode,
            trigger_time=trigger_time,
            enabled=enabled,
            reminder_id=reminder_id,
        )
        self.meal_type = meal_type.lower()
        self.food_note = food_note

    @property
    def reminder_type(self) -> str:
        return "food"

    def build_message(self) -> str:
        meal_emoji = {
            "breakfast": "🌅", "lunch": "☀️",
            "dinner": "🌙", "snack": "🍎", "custom": "🍽️"
        }.get(self.meal_type, "🍽️")

        msg = f"{meal_emoji} {self.meal_type.capitalize()} time! Time for your meal."
        if self.food_note:
            msg += f" Note: {self.food_note}."
        return msg

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "meal_type": self.meal_type,
            "food_note": self.food_note,
        })
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "FoodReminder":
        obj = cls(
            name         = data["name"],
            meal_type    = data.get("meal_type", "custom"),
            trigger_time = data.get("trigger_time", "12:00"),
            food_note    = data.get("food_note", ""),
            priority     = Priority(data.get("priority", 2)),
            repeat_mode  = RepeatMode(data.get("repeat_mode", "daily")),
            enabled      = data.get("enabled", True),
            reminder_id  = data.get("id"),
        )
        obj.status         = ReminderStatus(data.get("status", "active"))
        obj.created_at     = data.get("created_at", obj.created_at)
        obj.last_triggered = data.get("last_triggered")
        obj.trigger_count  = data.get("trigger_count", 0)
        return obj


# ─────────────────────────────────────────────────────────────
# Drinking Reminder
# ─────────────────────────────────────────────────────────────

class DrinkingReminder(BaseReminder):
    """
    Interval-based hydration reminder.
    Fires every N minutes to remind the user to drink water.
    Elderly people often forget to hydrate — this is safety-critical.
    """

    def __init__(self,
                 name:             str,
                 interval_minutes: int   = 120,
                 amount_ml:        int   = 200,
                 drink_type:       str   = "water",
                 priority:         Priority   = Priority.MEDIUM,
                 enabled:          bool = True,
                 reminder_id:      Optional[str] = None):
        """
        Args:
            interval_minutes: How often to remind (default 2 hours)
            amount_ml       : How much to drink (default 200ml = 1 glass)
            drink_type      : "water" | "juice" | "medicine" etc.
        """
        super().__init__(
            name=name,
            message="",
            priority=priority,
            repeat_mode=RepeatMode.INTERVAL,
            interval_minutes=interval_minutes,
            enabled=enabled,
            reminder_id=reminder_id,
        )
        self.amount_ml  = amount_ml
        self.drink_type = drink_type

    @property
    def reminder_type(self) -> str:
        return "drinking"

    def build_message(self) -> str:
        return (f"💧 Hydration Reminder! Please drink {self.amount_ml}ml of "
                f"{self.drink_type}. Staying hydrated is important for your health!")

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "amount_ml":  self.amount_ml,
            "drink_type": self.drink_type,
        })
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "DrinkingReminder":
        obj = cls(
            name             = data["name"],
            interval_minutes = data.get("interval_minutes", 120),
            amount_ml        = data.get("amount_ml", 200),
            drink_type       = data.get("drink_type", "water"),
            priority         = Priority(data.get("priority", 2)),
            enabled          = data.get("enabled", True),
            reminder_id      = data.get("id"),
        )
        obj.status         = ReminderStatus(data.get("status", "active"))
        obj.created_at     = data.get("created_at", obj.created_at)
        obj.last_triggered = data.get("last_triggered")
        obj.trigger_count  = data.get("trigger_count", 0)
        return obj


# ─────────────────────────────────────────────────────────────
# Walking Reminder
# ─────────────────────────────────────────────────────────────

class WalkingReminder(BaseReminder):
    """
    Physical activity and walking reminder.
    Can be fixed-time or interval-based.
    Includes step goal and duration suggestions.
    """

    def __init__(self,
                 name:             str,
                 trigger_time:     Optional[str] = None,
                 interval_minutes: Optional[int] = None,
                 duration_minutes: int  = 15,
                 steps_goal:       int  = 1000,
                 activity_type:    str  = "walking",
                 priority:         Priority   = Priority.MEDIUM,
                 repeat_mode:      RepeatMode = RepeatMode.DAILY,
                 enabled:          bool = True,
                 reminder_id:      Optional[str] = None):
        """
        Args:
            trigger_time     : "HH:MM" for fixed-time reminder
            interval_minutes : Minutes between reminders (for interval mode)
            duration_minutes : Suggested activity duration
            steps_goal       : Step target for this session
            activity_type    : "walking" | "stretching" | "exercise" | "yoga"
        """
        # Determine repeat mode based on what's provided
        if interval_minutes:
            repeat_mode = RepeatMode.INTERVAL

        super().__init__(
            name=name,
            message="",
            priority=priority,
            repeat_mode=repeat_mode,
            trigger_time=trigger_time,
            interval_minutes=interval_minutes,
            enabled=enabled,
            reminder_id=reminder_id,
        )
        self.duration_minutes = duration_minutes
        self.steps_goal       = steps_goal
        self.activity_type    = activity_type

    @property
    def reminder_type(self) -> str:
        return "walking"

    def build_message(self) -> str:
        activity_emoji = {
            "walking":    "🚶",
            "stretching": "🧘",
            "exercise":   "💪",
            "yoga":       "🧘‍♀️",
        }.get(self.activity_type, "🏃")

        return (f"{activity_emoji} Activity Time! "
                f"Time for your {self.duration_minutes}-minute {self.activity_type}. "
                f"Target: {self.steps_goal} steps. "
                f"Gentle movement is great for your health!")

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "duration_minutes": self.duration_minutes,
            "steps_goal":       self.steps_goal,
            "activity_type":    self.activity_type,
        })
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "WalkingReminder":
        obj = cls(
            name             = data["name"],
            trigger_time     = data.get("trigger_time"),
            interval_minutes = data.get("interval_minutes"),
            duration_minutes = data.get("duration_minutes", 15),
            steps_goal       = data.get("steps_goal", 1000),
            activity_type    = data.get("activity_type", "walking"),
            priority         = Priority(data.get("priority", 2)),
            repeat_mode      = RepeatMode(data.get("repeat_mode", "daily")),
            enabled          = data.get("enabled", True),
            reminder_id      = data.get("id"),
        )
        obj.status         = ReminderStatus(data.get("status", "active"))
        obj.created_at     = data.get("created_at", obj.created_at)
        obj.last_triggered = data.get("last_triggered")
        obj.trigger_count  = data.get("trigger_count", 0)
        return obj


# ─────────────────────────────────────────────────────────────
# TYPE REGISTRY — maps string → class
# Add new types here when you create them.
# ─────────────────────────────────────────────────────────────

TYPE_REGISTRY: dict = {
    "medicine": MedicineReminder,
    "food":     FoodReminder,
    "drinking": DrinkingReminder,
    "walking":  WalkingReminder,
}


def from_dict_any(data: dict) -> BaseReminder:
    """
    Reconstruct any reminder from a dict by looking up the type.
    Used by registry when loading from JSON or SQLite.
    """
    rtype = data.get("type", "")
    cls   = TYPE_REGISTRY.get(rtype)
    if cls is None:
        raise ValueError(
            f"Unknown reminder type: '{rtype}'. "
            f"Available types: {list(TYPE_REGISTRY.keys())}"
        )
    return cls.from_dict(data)
