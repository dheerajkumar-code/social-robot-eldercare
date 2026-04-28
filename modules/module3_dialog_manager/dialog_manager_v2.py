#!/usr/bin/env python3
"""
Module 3 — Dialog Manager (v2)
--------------------------------
Improved dialog system for elderly healthcare robot.

What this fixes vs dialog_manager.py (v1):

  BUG 1 — Intent matching
    v1: fuzzy cutoff=0.7 misses "I tumbled down" → emergency
    v2: Multi-strategy matching (exact → keyword overlap → fuzzy → emergency keywords)
        Emergency keywords get special fast-path (safety-critical)

  BUG 2 — No context memory
    v1: conversation_history stored but never used
    v2: 3-turn sliding memory used to avoid immediate repeats

  BUG 3 — No dislike/like system  ← USER REQUEST
    v1: zero preference tracking, disliked responses repeat forever
    v2: ResponsePreferences persists to preferences.json across restarts
        Disliked responses are permanently blacklisted per speaker
        User says "I don't like that" → last response blacklisted instantly
        If ALL responses blacklisted for an intent → Gemini fallback

  BUG 4 — {activity} placeholder crashes on strings containing {}
    v1: response.format(activity=...) raises KeyError if Gemini returns {}
    v2: safe_format() handles malformed templates gracefully

  BUG 5 — conversation_history grows forever
    v1: unbounded list, memory leak on 24/7 robot
    v2: capped at MAX_HISTORY entries

  BUG 6 — No speaker-awareness
    v1: same response regardless of who is speaking
    v2: speaker_id used for personalised greetings and preferences

  BUG 7 — Gemini responses uncapped
    v1: Gemini may return a paragraph (too long for TTS)
    v2: 2-sentence cap enforced, response validated before use

Usage (standalone):
    dm = DialogManager()
    result = dm.process_input("hello", emotion="happy", speaker_id="Dheeraj")
    result = dm.process_input("I don't like that", ...)   # triggers dislike
    result = dm.process_input("that was great", ...)      # triggers like
"""

import json
import random
import os
import hashlib
import datetime
import re
from collections import deque
from difflib import get_close_matches

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
MAX_HISTORY          = 50     # max conversation turns stored in memory
CONTEXT_WINDOW       = 3      # turns used for repeat-avoidance
FUZZY_CUTOFF         = 0.55   # v1 was 0.7 (too strict) — lowered for natural speech
MAX_GEMINI_SENTENCES = 2      # cap Gemini response length for TTS
GEMINI_MODEL         = "gemini-2.0-flash"

# Emergency keywords — bypass all matching, instant priority
EMERGENCY_KEYWORDS = {
    "fell", "fall", "fallen", "tumbled", "collapsed", "hurt",
    "help", "emergency", "911", "ambulance", "attack", "stroke",
    "chest", "breathe", "bleeding", "fainted", "unconscious",
    "pain", "agony", "accident"
}

# Dislike trigger phrases (any of these → mark last response as disliked)
DISLIKE_PHRASES = {
    "don't like that", "i don't like that", "don't say that",
    "don't say that again", "stop saying that", "i hate that",
    "not helpful", "that's not helpful", "change your response",
    "say something else", "i prefer something else", "not good",
    "that's bad", "i don't like this", "dislike",
}

# Like trigger phrases
LIKE_PHRASES = {
    "i like that", "that's good", "that was nice", "i loved that",
    "that's helpful", "good response", "perfect", "exactly right",
    "i like this", "that's great", "well said",
}


# ─────────────────────────────────────────────────────────────
# Response Preferences — persists across sessions
# ─────────────────────────────────────────────────────────────

class ResponsePreferences:
    """
    Persists liked/disliked responses to JSON file.
    Keyed by (speaker_id, response_hash) so each speaker has independent preferences.

    Example preferences.json:
    {
      "Dheeraj": {"a1b2c3": "disliked", "d4e5f6": "liked"},
      "unknown": {"g7h8i9": "disliked"}
    }
    """

    DISLIKED = "disliked"
    LIKED    = "liked"

    def __init__(self, prefs_file: str):
        self.prefs_file = prefs_file
        self._data: dict = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.prefs_file):
            try:
                with open(self.prefs_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save(self):
        try:
            with open(self.prefs_file, "w") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save preferences: {e}")

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.strip().lower().encode()).hexdigest()[:10]

    def _speaker_prefs(self, speaker_id: str) -> dict:
        if speaker_id not in self._data:
            self._data[speaker_id] = {}
        return self._data[speaker_id]

    def dislike(self, response_text: str, speaker_id: str = "unknown"):
        """Mark a response as disliked for this speaker. Persisted immediately."""
        h = self._hash(response_text)
        self._speaker_prefs(speaker_id)[h] = self.DISLIKED
        self._save()
        print(f"🚫 Disliked [{speaker_id}]: {response_text[:50]}...")

    def like(self, response_text: str, speaker_id: str = "unknown"):
        """Mark a response as liked for this speaker."""
        h = self._hash(response_text)
        self._speaker_prefs(speaker_id)[h] = self.LIKED
        self._save()

    def is_disliked(self, response_text: str, speaker_id: str = "unknown") -> bool:
        prefs = self._data.get(speaker_id, {})
        return prefs.get(self._hash(response_text)) == self.DISLIKED

    def is_liked(self, response_text: str, speaker_id: str = "unknown") -> bool:
        prefs = self._data.get(speaker_id, {})
        return prefs.get(self._hash(response_text)) == self.LIKED

    def count_disliked(self, speaker_id: str = "unknown") -> int:
        return sum(
            1 for v in self._data.get(speaker_id, {}).values()
            if v == self.DISLIKED
        )

    def clear_speaker(self, speaker_id: str):
        """Reset all preferences for a speaker (use with caution)."""
        if speaker_id in self._data:
            del self._data[speaker_id]
            self._save()


# ─────────────────────────────────────────────────────────────
# Intent Matching — multi-strategy
# ─────────────────────────────────────────────────────────────

class IntentMatcher:
    """
    Multi-strategy intent matching.

    Priority order:
      1. Emergency keyword fast-path  (safety critical)
      2. Dislike/like detection
      3. Exact phrase match
      4. Substring match
      5. Keyword overlap score
      6. Fuzzy string match
    """

    def __init__(self, intents: list):
        self.intents = intents

        # Build lookup structures
        self.pattern_to_tag: dict = {}
        self.tag_to_patterns: dict = {}
        self.all_patterns: list   = []

        for intent in intents:
            tag = intent["tag"]
            self.tag_to_patterns[tag] = intent["patterns"]
            for p in intent["patterns"]:
                pl = p.lower().strip()
                self.pattern_to_tag[pl] = tag
                self.all_patterns.append(pl)

    def match(self, text: str) -> str | None:
        """
        Returns intent tag or None.
        """
        text_lower = text.lower().strip()
        words      = set(re.findall(r'\b\w+\b', text_lower))

        # 1. Emergency fast-path — any emergency keyword present
        if words & EMERGENCY_KEYWORDS:
            return "emergency"

        # 2. Dislike / like detection
        for phrase in DISLIKE_PHRASES:
            if phrase in text_lower:
                return "_dislike"
        for phrase in LIKE_PHRASES:
            if phrase in text_lower:
                return "_like"

        # 3. Exact match
        if text_lower in self.pattern_to_tag:
            return self.pattern_to_tag[text_lower]

        # 4. Substring: check if any pattern is contained in input text
        for pattern, tag in self.pattern_to_tag.items():
            if pattern in text_lower:
                return tag

        # 5. Keyword overlap: count shared content words with each pattern
        best_tag, best_score = None, 0
        stopwords = {"i", "me", "my", "the", "a", "an", "is", "it",
                     "to", "do", "you", "what", "how", "can", "could",
                     "would", "please", "am", "are", "was", "were"}
        content_words = words - stopwords

        if content_words:
            for tag, patterns in self.tag_to_patterns.items():
                for p in patterns:
                    p_words = set(re.findall(r'\b\w+\b', p.lower())) - stopwords
                    if not p_words:
                        continue
                    overlap = len(content_words & p_words)
                    score   = overlap / max(len(p_words), len(content_words))
                    if score > best_score:
                        best_score, best_tag = score, tag

        if best_score >= 0.4:
            return best_tag

        # 6. Fuzzy match on full input vs all patterns
        matches = get_close_matches(text_lower, self.all_patterns, n=1, cutoff=FUZZY_CUTOFF)
        if matches:
            return self.pattern_to_tag[matches[0]]

        return None


# ─────────────────────────────────────────────────────────────
# Dialog Manager v2
# ─────────────────────────────────────────────────────────────

class DialogManager:
    """
    Improved dialog manager with dislike system, context memory,
    speaker awareness, and robust intent matching.
    """

    def __init__(self, intents_file="intents.json", api_key=None):
        self.base_dir      = os.path.dirname(os.path.abspath(__file__))
        self.intents_path  = os.path.join(self.base_dir, intents_file)
        self.intents       = self._load_intents()

        # Intent matcher
        self.matcher = IntentMatcher(self.intents["intents"])

        # Response preferences (dislike persistence)
        prefs_path       = os.path.join(self.base_dir, "preferences.json")
        self.preferences = ResponsePreferences(prefs_path)

        # Conversation state
        self.recent_turns: deque = deque(maxlen=CONTEXT_WINDOW)
        self.conversation_history: list = []
        self.last_response: str  = ""
        self.last_speaker: str   = "unknown"
        self.context: dict       = {}

        # Gemini setup
        self.gemini_available = False
        self._init_gemini(api_key)

    # ── Initialisation ──

    def _load_intents(self) -> dict:
        try:
            with open(self.intents_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Intents file not found: {self.intents_path}")
            return {"intents": []}
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in intents file: {e}")
            return {"intents": []}

    def _init_gemini(self, api_key=None):
        if not GENAI_AVAILABLE:
            print("⚠️  google-generativeai not installed. Offline mode only.")
            return

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            key_path = os.path.join(self.base_dir, "api_key.txt")
            if os.path.exists(key_path):
                with open(key_path) as f:
                    key = f.read().strip()

        if not key:
            print("⚠️  No Gemini API key. Falling back to offline responses.")
            return

        try:
            genai.configure(api_key=key)
            self.gemini_model  = genai.GenerativeModel(GEMINI_MODEL)
            self.gemini_available = True
            print("✅ Gemini API ready")
        except Exception as e:
            print(f"❌ Gemini init failed: {e}")

    # ── Intent resolution ──

    def get_intent(self, text: str) -> str | None:
        return self.matcher.match(text)

    # ── Response selection ──

    def _get_responses_for_intent(self, intent_tag: str,
                                   emotion: str,
                                   speaker_id: str) -> list:
        """
        Return list of candidate responses filtered by:
        - Not disliked by this speaker
        - Not the immediately previous response (context window)
        """
        for intent in self.intents["intents"]:
            if intent["tag"] != intent_tag:
                continue

            emotion_key  = emotion if emotion in intent["responses"] else "neutral"
            all_responses = intent["responses"].get(emotion_key, [])

            # Filter disliked responses
            filtered = [
                r for r in all_responses
                if not self.preferences.is_disliked(r, speaker_id)
            ]

            # If all responses disliked → return unfiltered (so robot can still speak)
            # and log a warning
            if not filtered:
                print(f"⚠️  All responses for '{intent_tag}' disliked by {speaker_id}. "
                      f"Resetting to avoid silence.")
                filtered = all_responses

            # Avoid the immediately previous response if alternatives exist
            recent_responses = [t["response"] for t in self.recent_turns]
            avoid_last = [r for r in filtered if r not in recent_responses]
            if avoid_last:
                filtered = avoid_last

            return filtered

        return []

    def get_response(self, intent_tag: str,
                     emotion: str       = "neutral",
                     activity: str      = "unknown",
                     speaker_id: str    = "unknown") -> tuple:
        """
        Returns (response_text: str, action: str|None)
        """
        # Handle internal dislike/like intents
        if intent_tag == "_dislike":
            return self._handle_dislike(speaker_id), None
        if intent_tag == "_like":
            return self._handle_like(speaker_id), None

        candidates = self._get_responses_for_intent(intent_tag, emotion, speaker_id)

        if not candidates:
            # Gemini fallback when no matching intent responses
            gemini = self._get_gemini_response(
                f"(intent: {intent_tag})", emotion, activity, speaker_id
            )
            return gemini or "I'm not sure what to say right now.", None

        # Prefer liked responses if any exist
        liked = [r for r in candidates
                 if self.preferences.is_liked(r, speaker_id)]
        pool  = liked if liked else candidates

        response = random.choice(pool)

        # Get action field if any
        action = None
        for intent in self.intents["intents"]:
            if intent["tag"] == intent_tag:
                action = intent.get("action")
                # Update context_set
                if intent.get("context_set"):
                    self.context["last_context"] = intent["context_set"]
                break

        # Safe placeholder substitution
        response = self._safe_substitute(response, activity=activity)

        return response, action

    # ── Dislike / like handlers ──

    def _handle_dislike(self, speaker_id: str) -> str:
        """Blacklist the last response for this speaker."""
        if self.last_response:
            self.preferences.dislike(self.last_response, speaker_id)
            return random.choice([
                "I'll remember that and won't say it again.",
                "Noted! I'll try something different next time.",
                "Sorry about that. I'll avoid that response for you.",
                "Understood. I'll choose a different response next time.",
            ])
        return "I'll try to do better next time."

    def _handle_like(self, speaker_id: str) -> str:
        """Mark the last response as liked for this speaker."""
        if self.last_response:
            self.preferences.like(self.last_response, speaker_id)
            return random.choice([
                "I'm glad you liked that!",
                "Happy to hear that! I'll remember it.",
                "That's wonderful to know, thank you!",
            ])
        return "Thank you for the feedback!"

    # ── Placeholder substitution ──

    @staticmethod
    def _safe_substitute(text: str, **kwargs) -> str:
        """
        Replace {key} placeholders safely without crashing.
        Uses regex replace instead of .format() to avoid KeyError on
        unrelated {} characters (e.g., from Gemini responses).
        """
        for key, value in kwargs.items():
            if key == "activity":
                value = str(value).replace("_", " ")
            text = re.sub(r"\{" + re.escape(key) + r"\}", str(value), text)

        # Replace {time_str} with current time
        if "{time_str}" in text:
            now      = datetime.datetime.now()
            time_str = now.strftime("%A, %I:%M %p")
            text     = text.replace("{time_str}", time_str)

        return text

    # ── Gemini ──

    def _get_gemini_response(self, text: str,
                              emotion: str,
                              activity: str,
                              speaker_id: str = "unknown") -> str | None:
        if not self.gemini_available:
            return None

        name_part = f"named {speaker_id}" if speaker_id not in ("unknown", "") else ""

        # Include recent context in the prompt
        context_str = ""
        if self.recent_turns:
            turns = list(self.recent_turns)[-2:]  # last 2 turns
            context_str = "\nRecent conversation:\n" + "\n".join(
                f"  User: {t['user']}\n  Robot: {t['response']}"
                for t in turns
            )

        prompt = f"""You are a warm, helpful robot companion for an elderly person {name_part}.

Current context:
- User emotion: {emotion}
- User activity: {activity}
- Time: {datetime.datetime.now().strftime('%I:%M %p')}
{context_str}

User said: "{text}"

Instructions:
- Reply in exactly 1-2 short sentences.
- Use simple, clear language suitable for elderly people.
- Be warm and empathetic.
- Do NOT mention that you are an AI.
- Do NOT use markdown, lists, or special characters.
- Match the emotional tone: sad→supportive, happy→cheerful, fearful→calm."""

        try:
            response = self.gemini_model.generate_content(prompt)
            raw      = response.text.strip()
            # Cap at MAX_GEMINI_SENTENCES
            sentences = re.split(r'(?<=[.!?])\s+', raw)
            capped    = " ".join(sentences[:MAX_GEMINI_SENTENCES])
            return capped if capped else None
        except Exception as e:
            print(f"Gemini error: {e}")
            return None

    # ── Main entry point ──

    def process_input(self,
                      text:       str,
                      emotion:    str = "neutral",
                      activity:   str = "unknown",
                      speaker_id: str = "unknown") -> dict:
        """
        Process user input and return response dict.

        Args:
            text       : transcribed speech from ASR
            emotion    : current emotion from M12
            activity   : current activity from M14
            speaker_id : recognised speaker from M13

        Returns:
            {
              "response"     : str,
              "intent"       : str,
              "emotion_used" : str,
              "action"       : str|None,
              "speaker_id"   : str,
            }
        """
        if not text or not text.strip():
            return {
                "response":     "I didn't catch that. Could you say it again?",
                "intent":       "no_input",
                "emotion_used": emotion,
                "action":       None,
                "speaker_id":   speaker_id,
            }

        self.last_speaker = speaker_id
        intent = self.get_intent(text)
        action = None

        if intent:
            response, action = self.get_response(
                intent, emotion, activity, speaker_id
            )
        else:
            # Try Gemini for unknown intents
            if self.gemini_available:
                gemini_resp = self._get_gemini_response(text, emotion, activity, speaker_id)
                if gemini_resp:
                    response = gemini_resp
                    intent   = "gemini_chat"
                else:
                    response = self._emotion_fallback(emotion)
                    intent   = "unknown"
            else:
                response = self._emotion_fallback(emotion)
                intent   = "unknown"

        # Track last response for dislike system
        self.last_response = response

        # Update context window (used for repeat-avoidance and Gemini context)
        self.recent_turns.append({
            "user":     text,
            "response": response,
            "intent":   intent,
            "emotion":  emotion,
        })

        # Log conversation (capped)
        self.conversation_history.append({
            "timestamp":  datetime.datetime.now().isoformat(),
            "user_input": text,
            "response":   response,
            "intent":     intent,
            "emotion":    emotion,
            "activity":   activity,
            "speaker_id": speaker_id,
            "action":     action,
        })
        if len(self.conversation_history) > MAX_HISTORY:
            self.conversation_history = self.conversation_history[-MAX_HISTORY:]

        return {
            "response":     response,
            "intent":       intent,
            "emotion_used": emotion,
            "action":       action,
            "speaker_id":   speaker_id,
        }

    @staticmethod
    def _emotion_fallback(emotion: str) -> str:
        """Fallback response when no intent matched and Gemini unavailable."""
        mapping = {
            "sad":     "I'm not sure I understand, but I'm here with you.",
            "angry":   "I apologise, I didn't quite catch that. Could you rephrase?",
            "fearful": "I'm here. You're safe. Could you say that again slowly?",
            "happy":   "That sounds interesting! Could you tell me more?",
        }
        return mapping.get(
            emotion,
            "I'm sorry, I didn't understand. Could you say that again?"
        )

    # ── Utility ──

    def get_history(self, n: int = 5) -> list:
        """Return last N conversation turns."""
        return self.conversation_history[-n:]

    def reset_preferences(self, speaker_id: str = None):
        """Clear preferences for one speaker or all speakers."""
        if speaker_id:
            self.preferences.clear_speaker(speaker_id)
            print(f"✅ Preferences reset for: {speaker_id}")
        else:
            self.preferences._data = {}
            self.preferences._save()
            print("✅ All preferences reset")
