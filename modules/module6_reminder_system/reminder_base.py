#!/usr/bin/env python3
"""
Module 6 — reminder_base.py
------------------------------
Abstract base class for all reminder types.

Every reminder type (Medicine, Food, Drinking, Walking) inherits from
BaseReminder. This ensures all reminders share a consistent interface
so the Scheduler and Registry can handle them uniformly.

Key design principle:
  - Adding a NEW reminder type = create a new class in reminder_types.py
  - Zero changes needed in registry.py, scheduler.py, or main.py
"""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Optional


class Priority(Enum):
    """Priority levels for reminders. Higher number = more urgent."""
    LOW    = 1
    MEDIUM = 2
    HIGH   = 3
    URGENT = 4


class RepeatMode(Enum):
    """How often a reminder repeats."""
    ONCE       = "once"        # one-time only
    DAILY      = "daily"       # every day at the same time
    WEEKLY     = "weekly"      # once a week
    INTERVAL   = "interval"    # every N minutes/hours


class ReminderStatus(Enum):
    """Current state of a reminder."""
    ACTIVE   = "active"
    PAUSED   = "paused"
    DONE     = "done"
    SNOOZED  = "snoozed"


class BaseReminder(ABC):
    """
    Abstract base class for all reminder types.

    Subclasses MUST implement:
      - reminder_type (property) → str identifier e.g. "medicine"
      - build_message() → str  the notification text
      - to_dict()       → dict for serialisation (call super().to_dict() first)

    Subclasses MAY override:
      - on_trigger()  → custom action when reminder fires
    """

    def __init__(self,
                 name:          str,
                 message:       str,
                 priority:      Priority    = Priority.MEDIUM,
                 repeat_mode:   RepeatMode  = RepeatMode.DAILY,
                 trigger_time:  Optional[str] = None,
                 interval_minutes: Optional[int] = None,
                 enabled:       bool = True,
                 reminder_id:   Optional[str] = None):
        """
        Args:
            name             : Human-readable name  e.g. "Morning Medicine"
            message          : Notification text    e.g. "Take 2 tablets of Aspirin"
            priority         : Priority enum value
            repeat_mode      : How often to repeat
            trigger_time     : "HH:MM" for DAILY/WEEKLY reminders  e.g. "08:00"
            interval_minutes : Minutes between triggers for INTERVAL mode
            enabled          : Whether the reminder is active
            reminder_id      : Unique ID (auto-generated if not provided)
        """
        self.id               = reminder_id or str(uuid.uuid4())[:8]
        self.name             = name
        self.message          = message
        self.priority         = priority
        self.repeat_mode      = repeat_mode
        self.trigger_time     = trigger_time         # "HH:MM"
        self.interval_minutes = interval_minutes
        self.enabled          = enabled
        self.status           = ReminderStatus.ACTIVE if enabled else ReminderStatus.PAUSED
        self.created_at       = datetime.now().isoformat()
        self.last_triggered   = None
        self.trigger_count    = 0

    # ── Abstract interface ──

    @property
    @abstractmethod
    def reminder_type(self) -> str:
        """Return the string type identifier e.g. 'medicine'."""
        pass

    @abstractmethod
    def build_message(self) -> str:
        """
        Build the full notification message.
        Called each time the reminder triggers.
        """
        pass

    # ── Concrete methods ──

    def on_trigger(self):
        """
        Called by the Scheduler when this reminder fires.
        Updates state, builds message, returns text for notification.
        Can be overridden by subclasses for custom behaviour.
        """
        self.last_triggered = datetime.now().isoformat()
        self.trigger_count += 1

        if self.repeat_mode == RepeatMode.ONCE:
            self.status = ReminderStatus.DONE

        return self.build_message()

    def pause(self):
        """Pause this reminder — it will not trigger until resumed."""
        self.status = ReminderStatus.PAUSED
        self.enabled = False

    def resume(self):
        """Resume a paused reminder."""
        self.status = ReminderStatus.ACTIVE
        self.enabled = True

    def snooze(self, minutes: int = 10):
        """Mark as snoozed for N minutes (scheduler handles re-trigger)."""
        self.status = ReminderStatus.SNOOZED
        return minutes

    def is_active(self) -> bool:
        return self.status == ReminderStatus.ACTIVE and self.enabled

    def to_dict(self) -> dict:
        """
        Serialise to dict for JSON/SQLite storage.
        Subclasses should call super().to_dict() and add their own fields.
        """
        return {
            "id":               self.id,
            "type":             self.reminder_type,
            "name":             self.name,
            "message":          self.message,
            "priority":         self.priority.value,
            "repeat_mode":      self.repeat_mode.value,
            "trigger_time":     self.trigger_time,
            "interval_minutes": self.interval_minutes,
            "enabled":          self.enabled,
            "status":           self.status.value,
            "created_at":       self.created_at,
            "last_triggered":   self.last_triggered,
            "trigger_count":    self.trigger_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BaseReminder":
        """
        Reconstruct from dict (used when loading from JSON/SQLite).
        Subclasses override this and call their own constructor.
        """
        raise NotImplementedError("Subclasses must implement from_dict()")

    def __repr__(self) -> str:
        return (f"<{self.__class__.__name__} id={self.id} "
                f"name='{self.name}' status={self.status.value}>")

    def __str__(self) -> str:
        time_str = self.trigger_time or f"every {self.interval_minutes}min"
        return (f"[{self.reminder_type.upper()}] {self.name} | "
                f"{time_str} | {self.priority.name} | {self.status.value}")
