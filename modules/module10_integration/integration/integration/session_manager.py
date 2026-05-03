from datetime import datetime
import json


class UserSession:
    """Tracks who the user is and their current state."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.name            = "Unknown"
        self.source          = ""          # "face" or "voice"
        self.confidence      = 0.0
        self.identified      = False
        self.session_start   = None
        self.emotion         = "neutral"
        self.activity        = "unknown"
        self.last_emotion    = ""          # for deduplication
        self.tts_busy        = False
        self.in_conversation = False

    def identify(self, name, source, confidence):
        if not name or name in ("None", "unknown", ""):
            return False
        self.name          = name
        self.source        = source
        self.confidence    = round(float(confidence), 2)
        self.identified    = True
        self.session_start = self.session_start or datetime.now()
        return True

    def update_emotion(self, label):
        self.emotion = label

    def update_activity(self, label):
        self.activity = label

    def to_json(self):
        return json.dumps({
            "name":          self.name,
            "source":        self.source,
            "confidence":    self.confidence,
            "identified":    self.identified,
            "session_start": self.session_start.isoformat() if self.session_start else None,
            "emotion":       self.emotion,
            "activity":      self.activity,
        })
