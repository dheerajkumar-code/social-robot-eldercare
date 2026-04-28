def test_emotion_subtitle_files_exist():
    import os
    base = "modules/module12_emotion_subtitle"
    assert os.path.exists(f"{base}/emotion_subtitle_node.py")
    assert os.path.exists(f"{base}/requirements.txt")

