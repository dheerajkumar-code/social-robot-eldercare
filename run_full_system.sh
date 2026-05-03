#!/bin/bash
# ============================================================
#  Elderly Care Robot — Full System Launcher
#  Starts all 9 modules + Integration Controller together
# ============================================================

BASE="/home/harsh/Desktop/Dheeraj Project1/Dheeraj Project/elderly-robot-head"
cd "$BASE"

# Colours
RED='\033[0;31m'; GREEN='\033[0;32m'
YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

echo -e "${CYAN}"
echo "  ╔══════════════════════════════════════════╗"
echo "  ║     Elderly Care Robot — Starting Up     ║"
echo "  ╚══════════════════════════════════════════╝"
echo -e "${NC}"

# ── Environment ──────────────────────────────────────────────
source /opt/ros/humble/setup.bash
source "$BASE/install/setup.bash"
[ -d "$BASE/venv" ] && source "$BASE/venv/bin/activate"
echo -e "${GREEN}✓ Environment ready${NC}"

mkdir -p "$BASE/logs"
PIDS="/tmp/robot_pids.txt"; > "$PIDS"

# ── Helper ────────────────────────────────────────────────────
start_module() {
    local NAME="$1"; local CMD="$2"
    echo -e "${YELLOW}  Starting ${NAME}...${NC}"
    eval "$CMD" > "$BASE/logs/${NAME}.log" 2>&1 &
    echo $! >> "$PIDS"
    sleep 1.5
    if kill -0 $! 2>/dev/null; then
        echo -e "${GREEN}  ✓ ${NAME} running (PID $!)${NC}"
    else
        echo -e "${RED}  ✗ ${NAME} failed — see logs/${NAME}.log${NC}"
    fi
}

# ── Launch order (dependencies first) ────────────────────────
echo -e "\n${CYAN}Starting modules...${NC}"

# M4 first — TTS must be ready before anything speaks
start_module "M4_TTS" \
    "python3 '$BASE/modules/module4_text_to_speech/tts_node_v2.py' --ros"

# M6 — Reminder system (background, no hardware)
start_module "M6_Reminder" \
    "python3 -c \"
import sys, os
sys.path.insert(0, '$BASE/modules/module6_reminder_system')
from reminder_node import main_ros
main_ros()
\""

# M2 — Speech to Text (microphone)
start_module "M2_STT" \
    "python3 '$BASE/modules/module2_Speech_to_text/asr_node_v2.py' --ros"

# M3 — Dialog Manager (needs M2 and M4 ready)
start_module "M3_Dialog" \
    "python3 '$BASE/modules/module3_dialog_manager/dialog_node_v2.py' --ros"

# M11 — Face Detection / Person Recognition (camera)
start_module "M11_Face" \
    "python3 '$BASE/modules/module11_person_detection/person_node.py' --ros --camera 0"

# M12 — Emotion Detection (camera)
start_module "M12_Emotion" \
    "python3 '$BASE/modules/module12_emotion_subtitle/emotion_node_v2.py' --ros --src 0"

# M14 — Activity / Fall Detection (camera)
start_module "M14_Activity" \
    "python3 '$BASE/modules/module14_human_activity/activity_node_v2.py' \
     --ros --cam 0 \
     --model '$BASE/modules/module14_human_activity/models/activity_model_v2.pkl'"

# M16 — Speaker Recognition (microphone)
start_module "M16_Speaker" \
    "python3 '$BASE/modules/module16_speaker_recognition/speaker_recognition_node.py' --ros"

# M10 — Integration Controller (starts last — all others must be ready)
echo -e "\n${CYAN}  Starting Integration Controller (brain)...${NC}"
sleep 2
start_module "M10_Controller" \
    "ros2 run integration main_controller"

# ── Status report ─────────────────────────────────────────────
echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Robot System is RUNNING!${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo -e "\n${YELLOW}Monitor the system:${NC}"
echo "  ros2 topic echo /system_state"
echo "  ros2 topic echo /user_identity"
echo "  ros2 topic echo /tts_speak"
echo "  ros2 topic echo /emotion_label"
echo "  ros2 topic echo /activity_label"

echo -e "\n${YELLOW}Watch controller log live:${NC}"
echo "  tail -f $BASE/logs/M10_Controller.log"

echo -e "\n${YELLOW}Test manually:${NC}"
echo "  # Simulate face recognition:"
echo "  ros2 topic pub /person_name std_msgs/msg/String '{data: \"Dheeraj\"}' -1"
echo ""
echo "  # Simulate emotion:"
echo "  ros2 topic pub /emotion_label std_msgs/msg/String '{data: \"sad\"}' -1"
echo ""
echo "  # Simulate fall:"
echo "  ros2 topic pub /activity_alert std_msgs/msg/String '{data: \"FALL_DETECTED\"}' -1"
echo ""
echo "  # Simulate reminder:"
echo '  ros2 topic pub /reminder_alert std_msgs/msg/String '"'"'{data: "{\"id\":\"med1\",\"message\":\"Time to take your medicine.\"}"}'"'"' -1'

echo -e "\n${YELLOW}Stop everything:${NC}"
echo "  bash $BASE/stop_system.sh"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

# ── Keep running until Ctrl+C ─────────────────────────────────
cleanup() {
    echo -e "\n${RED}Stopping all modules...${NC}"
    while read PID; do kill "$PID" 2>/dev/null; done < "$PIDS"
    pkill -f "person_node|emotion_node|activity_node|main_controller|asr_node|dialog_node|tts_node|reminder_node|speaker_recognition" 2>/dev/null
    echo -e "${GREEN}✓ All stopped.${NC}"
    exit 0
}
trap cleanup INT TERM
wait
