#!/usr/bin/bash
# Quick demo script for Module 12 - Emotion Subtitle System

echo "======================================================================"
echo "Module 12 - Emotion Subtitle System"
echo "======================================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "emotion_subtitle_huggingface.py" ]; then
    echo "❌ Error: Please run this script from the module12_emotion_subtitle directory"
    exit 1
fi

# Main menu
while true; do
    clear
    echo "======================================================================"
    echo "Module 12 - Emotion Subtitle System"
    echo "======================================================================"
    echo ""
    echo "Choose an option:"
    echo ""
    echo "  1) Run Emotion Detection (Auto-detect best model) - RECOMMENDED"
    echo "  2) Run TensorFlow Emotion Detection (FER2013 - Manual)"
    echo "  3) Run Demo Mode (All 7 Emotions Cycle - No Camera)"
    echo "  4) Collect Training Samples"
    echo "  5) Run Tests"
    echo "  0) Exit"
    echo ""
    read -p "Enter choice [0-5]: " choice

    case $choice in
        1)
            echo ""
            echo "🎥 Starting Emotion Detection (Auto-select best model)..."
            echo "Press 'q' to quit"
            echo "----------------------------------------------------------------------"
            python3 run_emotion_detection.py --src 0
            ;;
        2)
            echo ""
            echo "🎥 Starting TensorFlow Emotion Detection (FER2013)..."
            echo "Press 'q' to quit"
            echo "----------------------------------------------------------------------"
            # Ensure model exists
            if [ ! -f "fer_rebuilt_v2.h5" ]; then
                echo "⚠️  Model not found. Rebuilding..."
                python3 rebuild_improved.py
            fi
            python3 emotion_subtitle_enhanced.py --model fer_rebuilt_v2.h5 --src 0 --debug
            ;;
        3)
            echo ""
            echo "🎭 Starting Demo Mode..."
            echo "----------------------------------------------------------------------"
            python3 demo_all_emotions.py
            echo ""
            read -p "Press Enter to continue..."
            ;;
        4)
            echo ""
            echo "📸 Starting Sample Collection..."
            echo "----------------------------------------------------------------------"
            python3 collect_emotion_samples.py
            ;;
        5)
            echo ""
            echo "🧪 Running comprehensive test suite..."
            echo "----------------------------------------------------------------------"
            python3 test_module12.py
            echo ""
            read -p "Press Enter to continue..."
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
