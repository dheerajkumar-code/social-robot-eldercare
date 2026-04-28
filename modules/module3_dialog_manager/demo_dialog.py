#!/usr/bin/env python3
"""
Interactive Demo for Module 3: Dialog Manager
Allows testing the dialog system with simulated inputs, emotions, and activities.
"""

import os
import sys
from dialog_manager import DialogManager
from media_player import MediaPlayer

# ANSI Colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

def print_header():
    print(f"{BOLD}{CYAN}="*60)
    print("Module 3: Dialog Manager - Interactive Demo")
    print("="*60 + f"{RESET}")
    print("Test how the robot responds to different inputs, emotions, and activities.")
    print("Type 'quit' or 'exit' to stop.\n")

def get_user_input(prompt, default=None):
    try:
        text = input(f"{YELLOW}{prompt}{RESET} ")
        if not text and default:
            return default
        return text
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)

def main():
    dm = DialogManager()
    player = MediaPlayer()
    print_header()

    # Default state
    current_emotion = "neutral"
    current_activity = "sitting"

    while True:
        print(f"\n{BOLD}--- Current State ---{RESET}")
        print(f"Emotion: {GREEN}{current_emotion}{RESET} | Activity: {GREEN}{current_activity}{RESET}")
        print("-" * 20)
        
        user_text = get_user_input("You (Text):")
        
        if user_text.lower() in ['quit', 'exit']:
            print("Goodbye!")
            player.stop_music()
            break
            
        if user_text.lower() in ['stop music', 'stop', 'quiet']:
            player.stop_music()
            print(f"{CYAN}Music stopped.{RESET}")
            continue
            
        # Optional: Allow changing state commands
        if user_text.startswith("/emotion"):
            parts = user_text.split()
            if len(parts) > 1:
                current_emotion = parts[1]
                print(f"{CYAN}Set emotion to: {current_emotion}{RESET}")
            else:
                print(f"{RED}Usage: /emotion [happy|sad|angry|neutral|fearful|surprised|disgusted]{RESET}")
            continue
            
        if user_text.startswith("/activity"):
            parts = user_text.split()
            if len(parts) > 1:
                current_activity = parts[1]
                print(f"{CYAN}Set activity to: {current_activity}{RESET}")
            else:
                print(f"{RED}Usage: /activity [sitting|walking|drinking|reading]{RESET}")
            continue

        if user_text.startswith("/help"):
            print("Commands:")
            print("  /emotion [name]   - Change simulated user emotion")
            print("  /activity [name]  - Change simulated user activity")
            print("  quit              - Exit demo")
            continue

        # Process Input
        result = dm.process_input(user_text, current_emotion, current_activity)
        
        # Display Output
        print(f"{BOLD}Robot ({result['intent']}):{RESET} {result['response']}")
        
        # Handle Actions
        if result.get('action') == 'play_music':
            import random
            print(f"{CYAN}▶️  Action Triggered: Playing Music...{RESET}")
            
            if current_emotion in ["sad", "fearful", "lonely"]:
                genre = random.choice(["relaxing", "nature"])
            else:
                genre = random.choice(["upbeat", "jazz"])
                
            player.play_music(genre=genre)

if __name__ == "__main__":
    main()
