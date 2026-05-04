"""
Module 10 — Integration Package
Exports the three core components so ROS2 can import them cleanly.
"""
from integration.session_manager import UserSession
from integration.event_logger    import EventLogger

__all__ = ["UserSession", "EventLogger"]
__version__ = "2.0.0"