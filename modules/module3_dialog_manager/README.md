# Module 3: Dialog Manager

## Overview
The Dialog Manager is the core conversational engine for the Elderly Robot Head. It is designed to be **elderly-friendly**, **context-aware**, and **emotionally adaptive**.

## Features
- **Intent Classification**: Uses fuzzy matching to understand user intent (Greetings, Health, Weather, etc.).
- **Emotion Adaptation**: Responses change based on the user's detected emotion (e.g., empathetic responses when sad).
- **Activity Awareness**: Can comment on the user's current activity (e.g., "I see you are reading").
- **Context Management**: Remembers conversation history and active contexts (e.g., health concerns).
- **Offline Capable**: Does not require external APIs for core logic.

## Files
- `dialog_manager.py`: Main class containing the logic.
- `intents.json`: Configuration file defining intents, patterns, and responses.
- `demo_dialog.py`: Interactive CLI tool for testing.
- `test_module3.py`: Unit tests.

## Usage

### Running the Demo
```bash
python3 demo_dialog.py
```

### Using in Code
```python
from dialog_manager import DialogManager

dm = DialogManager()
result = dm.process_input("I feel sick", emotion="sad", activity="sitting")
print(result['response'])
```

## Configuration
You can add new intents or modify responses by editing `intents.json`. No code changes are required for adding simple Q&A pairs.
