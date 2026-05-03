import sqlite3
import json
import os
from datetime import datetime


class EventLogger:
    """Logs all system events to SQLite for caregiver review."""

    def __init__(self, db_path=None):
        if db_path is None:
            db_dir  = os.path.expanduser("~/.ros/elderly_robot")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "events.db")
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp  TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    module     TEXT DEFAULT 'M10',
                    payload    TEXT
                )
            """)
            conn.commit()

    def log(self, event_type, payload=None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO events (timestamp, event_type, module, payload) VALUES (?,?,?,?)",
                (datetime.now().isoformat(), event_type, "M10",
                 json.dumps(payload) if payload else None)
            )
            conn.commit()
