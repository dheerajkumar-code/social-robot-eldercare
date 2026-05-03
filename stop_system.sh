#!/bin/bash
echo "Stopping all robot modules..."
pkill -f "main_controller"          2>/dev/null
pkill -f "asr_node_v2"              2>/dev/null
pkill -f "dialog_node_v2"           2>/dev/null
pkill -f "tts_node_v2"              2>/dev/null
pkill -f "reminder_node"            2>/dev/null
pkill -f "person_node"              2>/dev/null
pkill -f "emotion_node_v2"          2>/dev/null
pkill -f "activity_node_v2"         2>/dev/null
pkill -f "speaker_recognition_node" 2>/dev/null
[ -f /tmp/robot_pids.txt ] && while read P; do kill "$P" 2>/dev/null; done < /tmp/robot_pids.txt
rm -f /tmp/robot_pids.txt
sleep 1
echo "✓ All modules stopped."
