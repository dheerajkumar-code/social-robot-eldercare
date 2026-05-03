# Module 10: Integration Controller — Complete System Documentation

## Overview

The Integration Controller (Module 10 v2) is a central coordination node that connects all perception and communication modules into one unified intelligent robot assistant system for elderly care.

**Architecture:** Event-driven, priority-based, state machine with real-time synchronization.

---

## System Architecture

### State Machine
```
IDLE ← User leaves
  ↓
USER_PRESENT ← M11 (face) or M16 (voice) identifies user
  ↓
CONVERSATION_ACTIVE ← User speaks (M2) or TTS busy (M4)
  ↓
EMERGENCY ← Fall detected (M14) or critical alert
```

### Core Components

| Component | File | Purpose |
|---|---|---|
| **MainControllerNode** | `main_controller.py` | Central ROS2 node, orchestrates all modules |
| **UserSession** | `session_manager.py` | Tracks user identity, emotion, activity, session state |
| **PriorityEngine** | `priority_engine.py` | Thread-safe priority queue for event coordination |
| **EventLogger** | `event_logger.py` | SQLite database for audit/analytics |

---

## Module Connections

### Input Topics (What Controller Receives)
```
M11 (Face)        → /person_detected, /person_name
M16 (Voice)       → /speaker_id, /speaker_confidence
M2 (STT)          → /asr_text, /asr_status
M12 (Emotion)     → /emotion_label, /emotion_confidence, /emotion_active
M14 (Activity)    → /activity_label, /activity_alert
M6 (Reminders)    → /reminder_alert
M3 (Dialog)       → /dialog_intent
M4 (TTS)          → /tts_speaking
```

### Output Topics (What Controller Sends)
```
→ M4 (TTS)        : /tts_speak (normal), /tts_speak_urgent (emergency)
→ M6 (Reminders)  : /reminder_ack
→ All             : /system_state, /user_identity, /controller/heartbeat
```

---

## Priority System

| Priority | Level | Triggers | Action |
|---|---|---|---|
| **P1** | EMERGENCY | Fall detected (M14) | Immediate assistance, escalate if no response |
| **P2** | SAFETY | Safety alerts | Interrupt low-priority tasks |
| **P3** | CONVERSATION | Speech input or active dialog | Route to M3, set conversation state |
| **P4** | REMINDER | Medication reminders (M6) | Announce & wait for ACK (3 retries) |
| **P5** | BACKGROUND | Idle greetings | Time-aware greeting (5 min cooldown) |

---

## Running the Controller

### Build
```bash
source /opt/ros/humble/setup.bash
cd /home/harsh/Desktop/Dheeraj\ Project1/Dheeraj\ Project/elderly-robot-head
colcon build --packages-select integration
source install/setup.bash
```

### Run
```bash
# Direct run
ros2 run integration main_controller

# Or via launch file
ros2 launch integration integration.launch.py

# Run in background
ros2 run integration main_controller &
```

### Stop
```bash
pkill -f main_controller
```

---

## Testing Scenarios

### Scenario 1: User Identification (Face)
```bash
# Terminal 1: Start controller
ros2 run integration main_controller

# Terminal 2: Simulate person detected
ros2 topic pub /person_detected std_msgs/msg/Bool '{data: true}' -1

# Terminal 3: Simulate face recognition
ros2 topic pub /person_name std_msgs/msg/String '{data: "Dheeraj"}' -1
```

**Expected Output:**
```
[INFO] Main Controller initialized. System state: IDLE
[INFO] System state: IDLE → USER_PRESENT
[INFO] User identified via face: Dheeraj (confidence: 1.0)
[TTS] Good morning Dheeraj! How are you feeling today?
```

### Scenario 2: Emotion Detection → Dialog Response
```bash
# Simulate emotion (M12)
ros2 topic pub /emotion_label std_msgs/msg/String '{data: "sad"}' -1
```

**Expected:**
- Controller logs: "EMOTION_DETECTED: sad"
- M3 automatically sends contextual response
- M4 speaks the response

### Scenario 3: Fall Detection (Emergency)
```bash
# Simulate fall (M14)
ros2 topic pub /activity_alert std_msgs/msg/String '{data: "FALL_DETECTED"}' -1
```

**Expected Output:**
```
[ERROR] EMERGENCY: Fall detected!
[System state] USER_PRESENT → EMERGENCY
[TTS] Are you okay?
[10s timeout] → [TTS] Calling emergency contact now.
```

### Scenario 4: Medication Reminder (with ACK)
```bash
# Send reminder (M6)
ros2 topic pub /reminder_alert std_msgs/msg/String \
  '{data: "{\"id\":\"med_1\",\"message\":\"Time for medication\",\"priority\":3}"}' -1

# (Later) Send acknowledgment
ros2 topic pub /reminder_ack std_msgs/msg/String '{data: "med_1"}' -1
```

**Expected:**
```
[TTS] Time for medication
[System] Waiting for acknowledgment (30s)...
[ACK received] Reminder cleared
```

If no ACK after 30s:
- Retry (up to 3 times)
- After 3 retries: escalate to emergency TTS + alert caregiver

### Scenario 5: Monitor System Health
```bash
ros2 topic echo /controller/heartbeat
```

**Output:**
```json
{
  "timestamp": "2026-05-02T22:16:35.123456",
  "system_state": "USER_PRESENT",
  "user_session": {
    "name": "Dheeraj",
    "source": "face",
    "confidence": 0.95,
    "emotion": "neutral",
    "activity": "standing"
  },
  "tts_busy": false,
  "queue_size": 0
}
```

### Scenario 6: Monitor User Identity
```bash
ros2 topic echo /user_identity
```

**Output:**
```json
{
  "name": "Dheeraj",
  "source": "face",
  "confidence": 0.95,
  "session_start": "2026-05-02T22:16:23.090777",
  "identified": true,
  "emotion": "neutral",
  "activity": "unknown",
  "conversation_active": false
}
```

---

## Event Logging

All events are logged to SQLite database: `~/.ros/elderly_robot/events.db`

### View Events
```bash
# Last 20 events
sqlite3 ~/.ros/elderly_robot/events.db \
  "SELECT timestamp, event_type, payload FROM events ORDER BY id DESC LIMIT 20;"

# Events of specific type
sqlite3 ~/.ros/elderly_robot/events.db \
  "SELECT timestamp, payload FROM events WHERE event_type='EMOTION_DETECTED';"

# Fall detection history
sqlite3 ~/.ros/elderly_robot/events.db \
  "SELECT timestamp, payload FROM events WHERE event_type LIKE 'FALL%';"
```

### Event Types Logged
- `SYSTEM_START` — Controller started
- `PERSON_DETECTED` — Person detected/left
- `USER_IDENTIFIED_FACE` — Face recognition
- `USER_IDENTIFIED_VOICE` — Voice recognition
- `SPEECH_INPUT` — User spoke
- `EMOTION_DETECTED` — Emotion recognized
- `REMINDER_SENT` — Reminder announced
- `REMINDER_ACK` — User acknowledged
- `REMINDER_TIMEOUT` — No ACK, retrying
- `REMINDER_ESCALATED` — Unacknowledged after retries
- `FALL_EMERGENCY` — Fall detected
- `FALL_ESCALATED` — Emergency escalated
- `STATE_CHANGE` — System state transition
- `GREETING_SENT` — Greeting announced

---

## Code Structure

```
modules/module10_integration/integration/integration/
├── main_controller.py          (450+ lines)
│   ├── MainControllerNode (ROS2 node)
│   ├── Identification layer    (M11, M16)
│   ├── Perception layer        (M2, M12, M14)
│   ├── Reminder layer          (M6 with ACK/retry)
│   ├── Dialog & TTS layer      (M3, M4)
│   ├── Priority engine         (event dispatch)
│   └── Monitoring              (heartbeat, health)
│
├── session_manager.py          (~80 lines)
│   └── UserSession class       (identify, reset, track state)
│
├── priority_engine.py          (~60 lines)
│   ├── Priority enum           (5 levels)
│   ├── Event class
│   └── PriorityEngine          (thread-safe queue)
│
└── event_logger.py             (SQLite logging)
    └── EventLogger class
```

---

## Time-Aware Greetings

When user is identified, controller sends context-aware greeting:

```python
hour = datetime.now().hour
if hour < 12:
    greeting = f"Good morning {name}! How are you feeling today?"
elif hour < 17:
    greeting = f"Good afternoon {name}! Are you comfortable?"
elif hour < 21:
    greeting = f"Good evening {name}! How was your day?"
else:
    greeting = f"Hello {name}! Is everything alright?"
```

---

## Integration with M3 (Dialog Manager)

M3 subscribes directly to:
- `/asr_text` (from M2)
- `/emotion_label` (from M12)
- `/activity_label` (from M14)
- `/speaker_id` (from M16)

**Controller's role:** Monitor M3's `/dialog_intent` output and escalate emergencies.

---

## Thread Safety

- **PriorityEngine:** Thread-safe queue with lock
- **ROS2 callbacks:** Single-threaded via MultiThreadedExecutor
- **SQLite logging:** Context manager with commit
- **Timers:** ROS2 timer callbacks (safe in executor)

---

## Performance Notes

- **Callback latency:** <10ms (simple state updates)
- **Priority dispatch:** O(1) event pop (PriorityQueue)
- **Database writes:** <5ms (SQLite in-process)
- **Event rate:** ~10-50 events/minute during normal operation
- **Memory footprint:** ~30-50 MB (Python ROS2 node)

---

## Troubleshooting

### Controller doesn't respond to input
- Check topics: `ros2 topic list | grep -E "asr|emotion|activity|reminder|person|speaker"`
- Verify module connectivity: `ros2 topic info /person_name` should show 1 publisher

### Reminders not acknowledged
- Check `/reminder_ack` topic: `ros2 topic echo /reminder_ack`
- Verify M6 is publishing: `ros2 topic echo /reminder_alert`
- Check timeout logic in `_reminder_ack_timeout()`

### Fall detection not triggering
- Verify M14 publishes to `/activity_alert`
- Check value is exactly: `"FALL_DETECTED"`
- Monitor: `ros2 topic echo /activity_alert`

### Database permissions error
- Ensure `~/.ros/elderly_robot/` is writable
- Create manually: `mkdir -p ~/.ros/elderly_robot`

---

## Next Steps

1. **Run alongside other modules** (M2-M6, M11-M12, M14, M16)
2. **Test end-to-end scenarios** with real hardware (Raspberry Pi 5)
3. **Monitor event logs** for analytics and debugging
4. **Integrate with caregiver dashboard** (M7 backend)
5. **Deploy to production** on elderly care robot

---

## Architecture Diagram

```
User Physical Interaction
       ↓
M11 (Face) ──→ /person_name
M16 (Voice) ──→ /speaker_id
       ↓
[USER IDENTIFICATION]
       ↓
M12 (Emotion) ──→ /emotion_label
M14 (Activity) ──→ /activity_label (or /activity_alert)
       ↓
[STATE UPDATE + PRIORITY CHECK]
       ↓
M2 (STT) ──→ /asr_text ──→ [CONVERSATION ROUTING]
              ↓
       M3 (Dialog) ← (subscribes directly to /asr_text, /emotion_label)
              ↓
       M4 (TTS) ← /tts_speak (normal) or /tts_speak_urgent (emergency)
              ↓
         Speaker Output
         
[BACKGROUND PROCESSES]
M6 (Reminders) ──→ /reminder_alert ──→ [REMINDER HANDLER]
                                           ↓
                                    ← /reminder_ack

[MONITORING]
Every 10s: /controller/heartbeat (JSON: state, session, queue)
           /system_state (IDLE/USER_PRESENT/CONVERSATION/EMERGENCY)
           /user_identity (JSON: who is current user)

[LOGGING]
All events → SQLite: ~/.ros/elderly_robot/events.db
```

---

## References

- ROS2 Documentation: https://docs.ros.org/en/humble/
- Elderly Care Robot Design: See `/home/harsh/Desktop/Dheeraj Project1/Dheeraj Project/elderly-robot-head/CLAUDE.md`
- Module Specifications: See individual module READMEs
