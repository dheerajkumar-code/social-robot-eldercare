#!/bin/bash
# Quick demo script for Module 14 - Human Activity Recognition

echo "======================================================================"
echo "Module 14 - Human Activity Recognition Demo"
echo "======================================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "activity_node.py" ]; then
    echo "❌ Error: Please run this script from the module14_human_activity directory"
    exit 1
fi

# Function to run demo
run_demo() {
    local activity=$1
    echo ""
    echo "🎬 Testing activity: $activity"
    echo "----------------------------------------------------------------------"
    python3 demo_activity_recognition.py --activity "$activity"
    echo ""
    read -p "Press Enter to continue..."
}

# Main menu
while true; do
    clear
    echo "======================================================================"
    echo "Module 14 - Human Activity Recognition"
    echo "======================================================================"
    echo ""
    echo "Choose an option:"
    echo ""
    echo "  1) Demo: Waving activity"
    echo "  2) Demo: Walking activity"
    echo "  3) Demo: Standing activity"
    echo "  4) Demo: Sitting activity"
    echo "  5) Demo: Laying activity"
    echo "  6) Demo: Falling activity"
    echo "  7) Run all tests"
    echo "  8) Validate model"
    echo "  9) Live webcam detection (requires camera)"
    echo "  0) Exit"
    echo ""
    read -p "Enter choice [0-9]: " choice

    case $choice in
        1) run_demo "waving" ;;
        2) run_demo "walking" ;;
        3) run_demo "standing" ;;
        4) run_demo "sitting" ;;
        5) run_demo "laying" ;;
        6) run_demo "falling" ;;
        7)
            echo ""
            echo "🧪 Running comprehensive test suite..."
            echo "----------------------------------------------------------------------"
            python3 test_module14.py
            echo ""
            read -p "Press Enter to continue..."
            ;;
        8)
            echo ""
            echo "📊 Validating model on all pose data..."
            echo "----------------------------------------------------------------------"
            python3 validate_model.py
            echo ""
            read -p "Press Enter to continue..."
            ;;
        9)
            echo ""
            echo "🎥 Starting live webcam detection..."
            echo "Press 'q' to quit"
            echo "----------------------------------------------------------------------"
            python3 activity_node.py --cam 0 --model models/pose_activity_model_new.pkl
            ;;
        0)
            echo ""
            echo "👋 Goodbye!"
            exit 0
            ;;
        *)
            echo ""
            echo "❌ Invalid choice. Please try again."
            sleep 2
            ;;
    esac
done
