#!/usr/bin/env python3
"""
Module 6 — registry.py
------------------------
Reminder Registry — stores, manages, and persists all reminders.

Storage:
  - Primary: SQLite (robot.db) — fast queries, survives restarts
  - Export:  JSON  (reminders.json) — human-readable, easy to edit

CRUD operations:
  - add(reminder)       → store new reminder
  - get(id)             → fetch one by ID
  - get_all()           → fetch all reminders
  - get_by_type(type)   → fetch by type string
  - update(reminder)    → update existing
  - delete(id)          → remove by ID
  - pause(id)           → pause one reminder
  - resume(id)          → resume one reminder
  - export_json(path)   → save all to JSON file
  - import_json(path)   → load from JSON file

Design:
  The registry knows nothing about reminder types — it only deals with dicts.
  Type reconstruction (from_dict_any) is handled by reminder_types.py.
  This makes the registry truly type-agnostic.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional

from reminder_types import BaseReminder, from_dict_any

logger = logging.getLogger("ReminderRegistry")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "robot.db")
JSON_PATH = os.path.join(BASE_DIR, "reminders.json")


class ReminderRegistry:
    """
    Central store for all reminders.

    In-memory dict for O(1) access + SQLite for persistence.
    The two are always kept in sync.

    Usage:
        registry = ReminderRegistry()
        registry.add(MedicineReminder(...))
        reminder = registry.get("abc12345")
        registry.delete("abc12345")
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path   = db_path
        self._store: Dict[str, BaseReminder] = {}
        self._init_db()
        self._load_from_db()
        logger.info(f"Registry loaded: {len(self._store)} reminders")

    # ─────────────────────────────────────────────────────────
    # SQLite setup
    # ─────────────────────────────────────────────────────────

    def _init_db(self):
        """Create reminders table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id          TEXT PRIMARY KEY,
                    type        TEXT NOT NULL,
                    name        TEXT NOT NULL,
                    data        TEXT NOT NULL,   -- full JSON dict
                    status      TEXT DEFAULT 'active',
                    created_at  TEXT,
                    updated_at  TEXT
                )
            """)
            conn.commit()

    def _load_from_db(self):
        """Load all reminders from SQLite into memory on startup."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT id, data FROM reminders"
                ).fetchall()
            for row_id, data_json in rows:
                try:
                    data     = json.loads(data_json)
                    reminder = from_dict_any(data)
                    self._store[reminder.id] = reminder
                except Exception as e:
                    logger.warning(f"Skipping corrupt reminder {row_id}: {e}")
        except Exception as e:
            logger.error(f"Failed to load from DB: {e}")

    def _save_to_db(self, reminder: BaseReminder):
        """Insert or update one reminder in SQLite."""
        data_json = json.dumps(reminder.to_dict())
        now       = datetime.now().isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO reminders (id, type, name, data, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        data       = excluded.data,
                        status     = excluded.status,
                        updated_at = excluded.updated_at
                """, (reminder.id, reminder.reminder_type, reminder.name,
                      data_json, reminder.status.value,
                      reminder.created_at, now))
                conn.commit()
        except Exception as e:
            logger.error(f"DB save failed for {reminder.id}: {e}")

    def _delete_from_db(self, reminder_id: str):
        """Remove a reminder from SQLite."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
                conn.commit()
        except Exception as e:
            logger.error(f"DB delete failed for {reminder_id}: {e}")

    # ─────────────────────────────────────────────────────────
    # CRUD operations
    # ─────────────────────────────────────────────────────────

    def add(self, reminder: BaseReminder) -> str:
        """
        Add a new reminder.
        Returns the reminder ID.
        Raises ValueError if ID already exists.
        """
        if reminder.id in self._store:
            raise ValueError(f"Reminder ID '{reminder.id}' already exists. "
                             f"Use update() to modify.")
        self._store[reminder.id] = reminder
        self._save_to_db(reminder)
        logger.info(f"Added: {reminder}")
        return reminder.id

    def get(self, reminder_id: str) -> Optional[BaseReminder]:
        """Fetch a reminder by ID. Returns None if not found."""
        return self._store.get(reminder_id)

    def get_all(self) -> List[BaseReminder]:
        """Return all reminders as a list."""
        return list(self._store.values())

    def get_active(self) -> List[BaseReminder]:
        """Return only active (enabled) reminders."""
        return [r for r in self._store.values() if r.is_active()]

    def get_by_type(self, reminder_type: str) -> List[BaseReminder]:
        """Return all reminders of a specific type."""
        return [r for r in self._store.values()
                if r.reminder_type == reminder_type]

    def get_by_priority(self, priority) -> List[BaseReminder]:
        """Return all reminders at or above a given priority."""
        return [r for r in self._store.values()
                if r.priority.value >= priority.value]

    def update(self, reminder: BaseReminder):
        """
        Update an existing reminder.
        Raises KeyError if reminder ID not found.
        """
        if reminder.id not in self._store:
            raise KeyError(f"Reminder '{reminder.id}' not found. "
                           f"Use add() to create new reminders.")
        self._store[reminder.id] = reminder
        self._save_to_db(reminder)
        logger.info(f"Updated: {reminder}")

    def delete(self, reminder_id: str) -> bool:
        """
        Delete a reminder by ID.
        Returns True if deleted, False if not found.
        """
        if reminder_id not in self._store:
            logger.warning(f"Delete: ID '{reminder_id}' not found")
            return False
        del self._store[reminder_id]
        self._delete_from_db(reminder_id)
        logger.info(f"Deleted reminder: {reminder_id}")
        return True

    def pause(self, reminder_id: str) -> bool:
        """Pause a reminder. Returns True on success."""
        reminder = self.get(reminder_id)
        if not reminder:
            return False
        reminder.pause()
        self._save_to_db(reminder)
        logger.info(f"Paused: {reminder_id}")
        return True

    def resume(self, reminder_id: str) -> bool:
        """Resume a paused reminder. Returns True on success."""
        reminder = self.get(reminder_id)
        if not reminder:
            return False
        reminder.resume()
        self._save_to_db(reminder)
        logger.info(f"Resumed: {reminder_id}")
        return True

    def clear_all(self):
        """Delete ALL reminders. Use with caution."""
        self._store.clear()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM reminders")
            conn.commit()
        logger.warning("All reminders cleared")

    # ─────────────────────────────────────────────────────────
    # Persistence: JSON export/import
    # ─────────────────────────────────────────────────────────

    def export_json(self, path: str = JSON_PATH):
        """Export all reminders to a JSON file."""
        data = {r.id: r.to_dict() for r in self._store.values()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported {len(data)} reminders → {path}")
        return path

    def import_json(self, path: str = JSON_PATH, overwrite: bool = False):
        """
        Import reminders from a JSON file.
        If overwrite=True, replaces existing reminder if ID conflicts.
        Returns count of imported reminders.
        """
        if not os.path.exists(path):
            logger.error(f"JSON file not found: {path}")
            return 0

        with open(path) as f:
            data = json.load(f)

        count = 0
        for rid, rdata in data.items():
            try:
                reminder = from_dict_any(rdata)
                if reminder.id in self._store and not overwrite:
                    logger.debug(f"Skipping existing ID: {rid}")
                    continue
                self._store[reminder.id] = reminder
                self._save_to_db(reminder)
                count += 1
            except Exception as e:
                logger.warning(f"Skipping import of {rid}: {e}")

        logger.info(f"Imported {count} reminders from {path}")
        return count

    # ─────────────────────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return a summary of all reminders grouped by type and status."""
        by_type   = {}
        by_status = {}
        for r in self._store.values():
            by_type[r.reminder_type]   = by_type.get(r.reminder_type, 0) + 1
            by_status[r.status.value]  = by_status.get(r.status.value, 0) + 1
        return {
            "total":     len(self._store),
            "by_type":   by_type,
            "by_status": by_status,
        }

    def print_all(self):
        """Pretty-print all reminders to terminal."""
        if not self._store:
            print("  (no reminders registered)")
            return
        for r in sorted(self._store.values(),
                        key=lambda x: (x.reminder_type, x.trigger_time or "")):
            status_icon = {"active":"🟢","paused":"⏸️","done":"✅","snoozed":"💤"}.get(r.status.value,"⚪")
            print(f"  {status_icon} [{r.id}] {r}")

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, reminder_id: str) -> bool:
        return reminder_id in self._store
