#!/usr/bin/env python3
"""
Module 6 — scheduler.py
--------------------------
Central scheduler for all reminders.
Uses APScheduler BackgroundScheduler for non-blocking operation.

Supports:
  - Fixed time (CronTrigger):    "medicine at 08:00 daily"
  - Interval (IntervalTrigger):  "water every 2 hours"
  - Weekly (CronTrigger):        "physio every Monday at 09:00"
  - One-time (DateTrigger):      "appointment at 14:30 today"

Architecture:
  - One APScheduler job per reminder
  - Registry is the source of truth — scheduler reads from it
  - Adding/removing a reminder: call scheduler.add_job / remove_job
  - Notifier handles the actual output when job fires

Usage:
    scheduler = ReminderScheduler(registry, notifier)
    scheduler.start()
    scheduler.add_all()      # schedule everything in registry
    scheduler.stop()
"""

import logging
import threading
from datetime import datetime
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron     import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.date     import DateTrigger
from apscheduler.events            import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR

from reminder_base import BaseReminder, RepeatMode
from registry      import ReminderRegistry
from notifier      import Notifier

logger = logging.getLogger("ReminderScheduler")


class ReminderScheduler:
    """
    Manages APScheduler jobs for all reminders in the registry.

    Key methods:
      start()           — start the background scheduler
      stop()            — stop gracefully
      add_all()         — schedule all active reminders from registry
      add_job(reminder) — schedule one reminder
      remove_job(id)    — unschedule one reminder
      pause_job(id)     — pause one job
      resume_job(id)    — resume one job
      list_jobs()       — print all scheduled jobs
    """

    def __init__(self,
                 registry: ReminderRegistry,
                 notifier: Notifier,
                 timezone: str = "Asia/Kolkata"):
        """
        Args:
            registry : ReminderRegistry instance
            notifier : Notifier instance
            timezone : Timezone string (default: IST for India)
        """
        self.registry  = registry
        self.notifier  = notifier
        self._lock     = threading.Lock()

        self._scheduler = BackgroundScheduler(
            timezone=timezone,
            job_defaults={
                "coalesce":       True,   # merge missed runs into one
                "max_instances":  1,      # never run same job twice simultaneously
                "misfire_grace_time": 60, # allow up to 60s late execution
            }
        )

        # Listen for job events for logging
        self._scheduler.add_listener(
            self._on_job_event,
            EVENT_JOB_EXECUTED | EVENT_JOB_ERROR
        )

        self._running = False
        logger.info(f"Scheduler initialised (timezone={timezone})")

    # ── Lifecycle ──

    def start(self):
        """Start the background scheduler thread."""
        if not self._running:
            self._scheduler.start()
            self._running = True
            logger.info("✅ Scheduler started")

    def stop(self):
        """Stop the scheduler gracefully."""
        if self._running:
            self._scheduler.shutdown(wait=False)
            self._running = False
            logger.info("Scheduler stopped")

    # ── Job management ──

    def add_all(self):
        """Schedule all active reminders from registry."""
        active = self.registry.get_active()
        count  = 0
        for reminder in active:
            try:
                self.add_job(reminder)
                count += 1
            except Exception as e:
                logger.error(f"Failed to schedule {reminder.id}: {e}")
        logger.info(f"Scheduled {count}/{len(active)} active reminders")

    def add_job(self, reminder: BaseReminder):
        """
        Schedule a single reminder based on its repeat_mode.
        Removes existing job with same ID first (idempotent).
        """
        with self._lock:
            # Remove old job if exists (update scenario)
            self._remove_job_safe(reminder.id)

            if not reminder.is_active():
                logger.debug(f"Skipping inactive: {reminder.id}")
                return

            trigger = self._build_trigger(reminder)
            if trigger is None:
                logger.error(f"Cannot build trigger for {reminder.id} — check config")
                return

            self._scheduler.add_job(
                func       = self._fire_reminder,
                trigger    = trigger,
                id         = reminder.id,
                name       = f"{reminder.reminder_type}:{reminder.name}",
                kwargs     = {"reminder_id": reminder.id},
            )
            logger.info(f"Scheduled: [{reminder.id}] {reminder.name} "
                        f"({reminder.repeat_mode.value})")

    def remove_job(self, reminder_id: str):
        """Remove a scheduled job by reminder ID."""
        with self._lock:
            self._remove_job_safe(reminder_id)

    def pause_job(self, reminder_id: str):
        """Pause a running job without removing it."""
        try:
            self._scheduler.pause_job(reminder_id)
            self.registry.pause(reminder_id)
            logger.info(f"Paused job: {reminder_id}")
        except Exception as e:
            logger.warning(f"Could not pause {reminder_id}: {e}")

    def resume_job(self, reminder_id: str):
        """Resume a paused job."""
        try:
            self._scheduler.resume_job(reminder_id)
            self.registry.resume(reminder_id)
            logger.info(f"Resumed job: {reminder_id}")
        except Exception as e:
            logger.warning(f"Could not resume {reminder_id}: {e}")

    def reschedule_job(self, reminder: BaseReminder):
        """Update an existing scheduled job (e.g. after time change)."""
        self.add_job(reminder)   # add_job handles remove + re-add

    # ── Trigger builder ──

    def _build_trigger(self, reminder: BaseReminder):
        """
        Build the correct APScheduler trigger based on reminder config.

        RepeatMode.DAILY    → CronTrigger  (every day at HH:MM)
        RepeatMode.WEEKLY   → CronTrigger  (every Monday at HH:MM)
        RepeatMode.INTERVAL → IntervalTrigger (every N minutes)
        RepeatMode.ONCE     → DateTrigger   (single fire today at HH:MM)
        """
        mode = reminder.repeat_mode

        if mode in (RepeatMode.DAILY, RepeatMode.WEEKLY, RepeatMode.ONCE):
            if not reminder.trigger_time:
                logger.error(f"{reminder.id}: trigger_time required for {mode.value}")
                return None
            hour, minute = map(int, reminder.trigger_time.split(":"))

            if mode == RepeatMode.DAILY:
                return CronTrigger(hour=hour, minute=minute)

            elif mode == RepeatMode.WEEKLY:
                # Default to Monday if not specified
                day_of_week = getattr(reminder, "day_of_week", "mon")
                return CronTrigger(day_of_week=day_of_week,
                                   hour=hour, minute=minute)

            elif mode == RepeatMode.ONCE:
                now = datetime.now()
                run_date = now.replace(hour=hour, minute=minute, second=0)
                if run_date <= now:
                    logger.warning(f"{reminder.id}: one-time trigger is in the past")
                    return None
                return DateTrigger(run_date=run_date)

        elif mode == RepeatMode.INTERVAL:
            if not reminder.interval_minutes:
                logger.error(f"{reminder.id}: interval_minutes required for INTERVAL mode")
                return None
            return IntervalTrigger(minutes=reminder.interval_minutes)

        return None

    # ── Job executor ──

    def _fire_reminder(self, reminder_id: str):
        """
        Called by APScheduler when a job fires.
        Retrieves reminder from registry, gets message, notifies.
        """
        reminder = self.registry.get(reminder_id)
        if not reminder:
            logger.warning(f"Job fired but reminder not found: {reminder_id}")
            return

        if not reminder.is_active():
            logger.debug(f"Reminder inactive, skipping: {reminder_id}")
            return

        # Get message and update state
        message = reminder.on_trigger()

        # Update registry with new state (trigger_count, last_triggered)
        self.registry.update(reminder)

        # Send notification
        self.notifier.notify(reminder, message)

        # Auto-remove ONCE reminders after firing
        if reminder.repeat_mode == RepeatMode.ONCE:
            self.remove_job(reminder_id)

    # ── Event listener ──

    def _on_job_event(self, event):
        """Log job execution events."""
        if event.exception:
            logger.error(f"Job {event.job_id} failed: {event.exception}")
        else:
            logger.debug(f"Job {event.job_id} executed successfully")

    # ── Helpers ──

    def _remove_job_safe(self, job_id: str):
        """Remove job silently if it exists."""
        try:
            self._scheduler.remove_job(job_id)
        except Exception:
            pass

    def list_jobs(self):
        """Print all currently scheduled jobs."""
        jobs = self._scheduler.get_jobs()
        if not jobs:
            print("  (no jobs currently scheduled)")
            return
        print(f"\n{'─'*55}")
        print(f"  {'ID':10s}  {'Name':30s}  {'Next Run'}")
        print(f"{'─'*55}")
        for job in jobs:
            next_run = str(job.next_run_time)[:19] if job.next_run_time else "paused"
            print(f"  {job.id:10s}  {job.name:30s}  {next_run}")
        print(f"{'─'*55}\n")

    def job_count(self) -> int:
        return len(self._scheduler.get_jobs())

    @property
    def is_running(self) -> bool:
        return self._running
