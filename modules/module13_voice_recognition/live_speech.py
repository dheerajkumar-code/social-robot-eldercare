#!/usr/bin/env python3
import speech_recognition as sr
import time

class LiveSpeech:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.adjust_for_noise()

    def adjust_for_noise(self):
        """
        Adjusts the recognizer sensitivity to ambient noise.
        """
        print("🎧 Adjusting for ambient noise... Please remain silent for 2 seconds.")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        print("✅ Ready to listen!")

    def listen(self):
        """
        Listens for a single phrase and returns the text.
        Returns:
            str: Recognized text, or None if error/silence.
        """
        try:
            with self.microphone as source:
                print("\n🎤 Listening...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
            print("⏳ Recognizing...")
            text = self.recognizer.recognize_google(audio)
            print(f"🗣️  You said: '{text}'")
            
            # Extract raw audio for speaker ID
            # Convert to 16kHz mono 16-bit PCM
            raw_data = audio.get_raw_data(convert_rate=16000, convert_width=2)
            import numpy as np
            audio_np = np.frombuffer(raw_data, dtype=np.int16)
            
            return text.lower(), audio_np
            
        except sr.WaitTimeoutError:
            print("❌ Timeout: No speech detected.")
            return None, None
        except sr.UnknownValueError:
            print("❌ Could not understand audio.")
            return None, None
        except sr.RequestError as e:
            print(f"❌ Could not request results; {e}")
            return None, None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, None

if __name__ == "__main__":
    speech = LiveSpeech()
    while True:
        try:
            text = speech.listen()
            if text and text in ["quit", "exit", "stop"]:
                break
        except KeyboardInterrupt:
            break
