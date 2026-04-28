#!/usr/bin/env python3
import sys
from dialog_manager import DialogManager

def main():
    print("======================================================================")
    print("Module 3 - Enhanced Dialog Manager (with Gemini AI)")
    print("======================================================================")
    print("The robot can now handle unlimited topics!")
    print("Type 'quit' or 'exit' to stop.")
    print("----------------------------------------------------------------------")
    
    # Initialize Dialog Manager
    dm = DialogManager()
    
    if dm.gemini_available:
        print("✅ Gemini AI is ACTIVE. You can ask about anything.")
    else:
        print("⚠️ Gemini AI is NOT active. Limited to pre-defined responses.")
    print("")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                print("Robot: Goodbye! Have a wonderful day.")
                break
                
            # Process input (simulating neutral emotion for now)
            result = dm.process_input(user_input, emotion="neutral")
            
            response = result['response']
            intent = result['intent']
            
            # Add a visual indicator if it came from Gemini
            prefix = ""
            if intent == "gemini_chat":
                prefix = "✨ "
            
            print(f"Robot: {prefix}{response}")
            print("")
            
        except KeyboardInterrupt:
            print("\nRobot: Goodbye!")
            break

if __name__ == "__main__":
    main()
