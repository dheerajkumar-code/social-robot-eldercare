#!/usr/bin/env python3
"""
Comprehensive test suite for Module 12 - Emotion Subtitle System

Tests:
1. Environment and dependencies
2. Model loading
3. Face detection
4. Emotion prediction
5. Subtitle rendering
6. Performance benchmarks
"""

import os
import sys
import unittest
import time
import numpy as np
from pathlib import Path

# Add module to path
MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR))

# Check dependencies
DEPENDENCIES_OK = True
MISSING_DEPS = []

try:
    import cv2
except ImportError:
    DEPENDENCIES_OK = False
    MISSING_DEPS.append("opencv-python")

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    DEPENDENCIES_OK = False
    MISSING_DEPS.append("tensorflow")

try:
    from emotion_subtitle_node import (
        EmotionSubtitle,
        preprocess_face,
        detect_faces,
        draw_boxes,
        CLASSES,
        EMOJI_MAP,
        COLOR_MAP
    )
    EMOTION_NODE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import from emotion_subtitle_node: {e}")
    EMOTION_NODE_AVAILABLE = False


class TestEnvironment(unittest.TestCase):
    """Test environment and dependencies"""
    
    def test_dependencies_installed(self):
        """Test that all required dependencies are installed"""
        if not DEPENDENCIES_OK:
            self.fail(f"Missing dependencies: {', '.join(MISSING_DEPS)}")
        
        print(f"✅ TensorFlow version: {tf.__version__}")
        print(f"✅ Keras version: {keras.__version__}")
        print(f"✅ OpenCV version: {cv2.__version__}")
    
    def test_model_files_exist(self):
        """Test that model files exist"""
        model_files = [
            MODULE_DIR / "fer_fixed_model.h5",
            MODULE_DIR / "model.h5"
        ]
        
        for model_file in model_files:
            if model_file.exists():
                size_mb = model_file.stat().st_size / (1024 * 1024)
                print(f"✅ Found {model_file.name} ({size_mb:.1f} MB)")
                self.assertTrue(model_file.exists())
        
        # At least one model should exist
        exists = any(mf.exists() for mf in model_files)
        self.assertTrue(exists, "At least one model file should exist")
    
    def test_emotion_classes_defined(self):
        """Test that emotion classes are properly defined"""
        if not EMOTION_NODE_AVAILABLE:
            self.skipTest("emotion_subtitle_node not available")
        
        self.assertEqual(len(CLASSES), 7, "Should have 7 emotion classes")
        self.assertEqual(len(EMOJI_MAP), 7, "Should have 7 emoji mappings")
        self.assertEqual(len(COLOR_MAP), 7, "Should have 7 color mappings")
        
        expected_emotions = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
        self.assertEqual(sorted(CLASSES), sorted(expected_emotions))
        
        print(f"✅ Emotion classes: {CLASSES}")


class TestModelLoading(unittest.TestCase):
    """Test model loading and initialization"""
    
    def setUp(self):
        """Load model for testing"""
        if not DEPENDENCIES_OK:
            self.skipTest("Dependencies not available")
        
        self.model_path = MODULE_DIR / "fer_fixed_model.h5"
        if not self.model_path.exists():
            self.model_path = MODULE_DIR / "model.h5"
        
        if not self.model_path.exists():
            self.skipTest("No model file found")
    
    def test_model_loads(self):
        """Test that the model loads without errors"""
        print(f"📥 Loading model: {self.model_path.name}")
        
        try:
            model = tf.keras.models.load_model(str(self.model_path))
            print(f"✅ Model loaded successfully")
            
            # Check model structure
            self.assertIsNotNone(model, "Model should not be None")
            print(f"✅ Model type: {type(model).__name__}")
            
            # Check input shape
            if hasattr(model, 'input_shape'):
                print(f"✅ Input shape: {model.input_shape}")
            
            # Check output shape
            if hasattr(model, 'output_shape'):
                print(f"✅ Output shape: {model.output_shape}")
            
            self.model = model
            
        except Exception as e:
            self.fail(f"Failed to load model: {e}")
    
    def test_model_prediction_shape(self):
        """Test that model produces correct output shape"""
        if not hasattr(self, 'model'):
            self.skipTest("Model not loaded")
        
        # Create dummy input (224x224x3)
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        
        try:
            predictions = self.model.predict(dummy_input, verbose=0)
            
            self.assertEqual(predictions.shape[0], 1, "Should predict for 1 sample")
            self.assertEqual(predictions.shape[1], 7, "Should have 7 emotion classes")
            
            print(f"✅ Prediction shape: {predictions.shape}")
            print(f"✅ Sample prediction: {predictions[0]}")
            
        except Exception as e:
            self.fail(f"Prediction failed: {e}")


class TestFaceDetection(unittest.TestCase):
    """Test face detection functionality"""
    
    def setUp(self):
        """Setup face detector"""
        if not DEPENDENCIES_OK:
            self.skipTest("Dependencies not available")
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
        if self.face_cascade.empty():
            self.skipTest("Haar cascade not available")
    
    def test_face_cascade_loaded(self):
        """Test that face cascade loads correctly"""
        self.assertFalse(self.face_cascade.empty(), "Face cascade should load")
        print("✅ Haar cascade loaded successfully")
    
    def test_detect_faces_function(self):
        """Test detect_faces function with dummy image"""
        if not EMOTION_NODE_AVAILABLE:
            self.skipTest("emotion_subtitle_node not available")
        
        # Create dummy grayscale image
        dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        try:
            faces = detect_faces(dummy_frame, self.face_cascade)
            self.assertIsInstance(faces, (tuple, np.ndarray), "Should return array of faces")
            print(f"✅ detect_faces works (found {len(faces)} faces in dummy image)")
        except Exception as e:
            self.fail(f"detect_faces failed: {e}")


class TestEmotionSubtitle(unittest.TestCase):
    """Test EmotionSubtitle class"""
    
    def setUp(self):
        """Create EmotionSubtitle instance"""
        if not EMOTION_NODE_AVAILABLE:
            self.skipTest("emotion_subtitle_node not available")
        
        self.subtitle = EmotionSubtitle()
    
    def test_subtitle_initialization(self):
        """Test subtitle initializes correctly"""
        self.assertEqual(self.subtitle.current_emotion, "neutral")
        self.assertEqual(self.subtitle.fade, 1.0)
        print("✅ EmotionSubtitle initialized correctly")
    
    def test_subtitle_update(self):
        """Test subtitle update functionality"""
        self.subtitle.update("happy")
        self.assertEqual(self.subtitle.current_emotion, "happy")
        
        # Update with same emotion
        time.sleep(0.1)
        self.subtitle.update("happy")
        self.assertLess(self.subtitle.fade, 1.0, "Fade should decrease over time")
        
        print("✅ Subtitle update works correctly")
    
    def test_subtitle_draw(self):
        """Test subtitle drawing on frame"""
        # Create dummy frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        try:
            result = self.subtitle.draw(frame)
            self.assertEqual(result.shape, frame.shape, "Output should have same shape")
            print("✅ Subtitle drawing works")
        except Exception as e:
            self.fail(f"Subtitle drawing failed: {e}")


class TestPreprocessing(unittest.TestCase):
    """Test preprocessing functions"""
    
    def test_preprocess_face(self):
        """Test face preprocessing"""
        if not EMOTION_NODE_AVAILABLE:
            self.skipTest("emotion_subtitle_node not available")
        
        # Create dummy face image
        face = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        try:
            processed = preprocess_face(face)
            
            self.assertEqual(processed.shape, (1, 224, 224, 3), "Should resize to 224x224")
            self.assertTrue(np.all(processed >= 0) and np.all(processed <= 1), 
                          "Should normalize to [0, 1]")
            
            print("✅ Face preprocessing works correctly")
        except Exception as e:
            self.fail(f"Preprocessing failed: {e}")


class TestPerformance(unittest.TestCase):
    """Test performance benchmarks"""
    
    def setUp(self):
        """Load model for performance testing"""
        if not DEPENDENCIES_OK:
            self.skipTest("Dependencies not available")
        
        model_path = MODULE_DIR / "fer_fixed_model.h5"
        if not model_path.exists():
            model_path = MODULE_DIR / "model.h5"
        
        if not model_path.exists():
            self.skipTest("No model file found")
        
        try:
            self.model = tf.keras.models.load_model(str(model_path))
        except Exception:
            self.skipTest("Could not load model")
    
    def test_inference_speed(self):
        """Benchmark inference speed"""
        # Create dummy input
        dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
        
        # Warm up
        for _ in range(3):
            self.model.predict(dummy_input, verbose=0)
        
        # Benchmark
        n_iterations = 20
        start_time = time.time()
        
        for _ in range(n_iterations):
            self.model.predict(dummy_input, verbose=0)
        
        elapsed = time.time() - start_time
        avg_time_ms = (elapsed / n_iterations) * 1000
        fps = 1000 / avg_time_ms
        
        print(f"✅ Inference speed: {avg_time_ms:.2f} ms/frame ({fps:.1f} FPS)")
        
        # Should be reasonably fast (< 500ms per frame)
        self.assertLess(avg_time_ms, 500, "Inference should be < 500ms")


def run_tests():
    """Run all tests and generate report"""
    print("=" * 70)
    print("Module 12 - Emotion Subtitle System Test Suite")
    print("=" * 70)
    print()
    
    if not DEPENDENCIES_OK:
        print("❌ Missing dependencies:")
        for dep in MISSING_DEPS:
            print(f"   - {dep}")
        print("\nInstall with: pip install -r requirements.txt")
        return 1
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEnvironment))
    suite.addTests(loader.loadTestsFromTestCase(TestModelLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestFaceDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestEmotionSubtitle))
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
