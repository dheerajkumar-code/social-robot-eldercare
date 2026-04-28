#!/usr/bin/env python3
import sys
import os
import time

# Add Module 3 to path to import DialogManager
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../module3_dialog_manager')))

from dialog_manager import DialogManager
from media_player import MediaPlayer
from live_speech import LiveSpeech
from speaker_identity import SpeakerIdentity
import numpy as np
import sounddevice as sd

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

def print_header():
    print(f"{BOLD}{CYAN}="*60)
    print("Module 13 + 3: Voice-Enabled Dialog Demo (with Speaker ID)")
    print("="*60 + f"{RESET}")
    print("Speak to the robot! Say 'quit' or 'exit' to stop.\n")

def main():
    # Initialize Modules
    print(f"{YELLOW}Initializing modules...{RESET}")
    speech = LiveSpeech()
    dm = DialogManager()
    player = MediaPlayer()
    speaker_id = SpeakerIdentity()
    
    print_header()

    # Default state
    current_emotion = "neutral"
    current_activity = "sitting"
    current_speaker = "Guest"

    while True:
        try:
            # 1. Listen for Voice Input
            user_text, audio_data = speech.listen()
            
            if not user_text:
                continue
                
            if user_text in ['quit', 'exit', 'stop']:
                print("Goodbye!")
                player.stop_music()
                break
            
            if user_text in ['stop music', 'quiet']:
                player.stop_music()
                print(f"{CYAN}Music stopped.{RESET}")
                continue
                
            # Perform Speaker Identification on the captured audio
            if audio_data is not None:
                name = speaker_id.identify(audio_data)
                if name != "Unknown":
                    current_speaker = name
                    print(f"{GREEN}🎤 Speaker Identified: {current_speaker}{RESET}")
                else:
                    current_speaker = "Unknown"
                    print(f"{RED}🎤 Speaker: Unknown - Access Denied{RESET}")
                    print(f"{RED}Robot: I only respond to registered users.{RESET}")
                    continue # Skip processing for unknown users

            # 2. Process with Dialog Manager
            result = dm.process_input(user_text, current_emotion, current_activity)
            
            # 3. Display Output
            response = result['response']
            
            # Personalize if speaker is known
            # Only add name if it's a greeting or specifically asked, to avoid repetition
            if current_speaker != "Guest":
                if result['intent'] == "greeting":
                     response = f"Hello {current_speaker}! " + response.replace("Hello! ", "").replace("Hi there! ", "")
                elif result['intent'] == "name":
                     response = f"I am your robot, {current_speaker}."
            
            print(f"{BOLD}{GREEN}Robot ({result['intent']}):{RESET} {response}")
            
            # 4. Handle Actions (Music)
            if result.get('action') == 'play_music':
                import random
                print(f"{CYAN}▶️  Action Triggered: Playing Music...{RESET}")
                
                if current_emotion in ["sad", "fearful", "lonely"]:
                    genre = random.choice(["relaxing", "nature"])
                else:
                    genre = random.choice(["upbeat", "jazz"])
                    
                player.play_music(genre=genre)

        except KeyboardInterrupt:
            print("\nExiting...")
            player.stop_music()
            break

if __name__ == "__main__":
    main()
