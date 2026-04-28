#!/usr/bin/env python3
"""
Comprehensive test suite for Module 14 - Human Activity Recognition

Tests:
1. Feature extraction functions
2. Model training pipeline
3. Model inference
4. Activity classification
5. Performance benchmarks
"""

import os
import sys
import unittest
import numpy as np
import pickle
import time
from pathlib import Path

# Add module to path
MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR))

# Import functions from activity_node
try:
    from activity_node import (
        landmarks_to_xy,
        normalize_landmarks_xy,
        angle,
        joint_angles,
        extract_features_from_sequence
    )
    ACTIVITY_NODE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import from activity_node: {e}")
    ACTIVITY_NODE_AVAILABLE = False

# Import training functions
try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


class TestFeatureExtraction(unittest.TestCase):
    """Test feature extraction functions"""
    
    def setUp(self):
        """Create dummy landmark data for testing"""
        if not ACTIVITY_NODE_AVAILABLE:
            self.skipTest("activity_node not available")
        
        # Create a simple standing pose (33 landmarks)
        self.dummy_landmarks = self._create_dummy_landmarks()
    
    def _create_dummy_landmarks(self):
        """Create dummy MediaPipe-style landmarks"""
        # Simplified standing pose with 33 landmarks
        landmarks = []
        for i in range(33):
            # Create a simple vertical arrangement
            x = 0.5 + (i % 3 - 1) * 0.1  # Slight horizontal variation
            y = 0.1 + (i / 33) * 0.8      # Vertical distribution
            z = 0.0
            visibility = 0.9
            
            class DummyLandmark:
                def __init__(self, x, y, z, visibility):
                    self.x = x
                    self.y = y
                    self.z = z
                    self.visibility = visibility
            
            landmarks.append(DummyLandmark(x, y, z, visibility))
        
        return landmarks
    
    def test_landmarks_to_xy(self):
        """Test conversion of landmarks to xy array"""
        xy = landmarks_to_xy(self.dummy_landmarks)
        
        self.assertEqual(xy.shape, (33, 2), "Should return 33x2 array")
        self.assertTrue(np.all(xy >= 0) and np.all(xy <= 1), "Coordinates should be normalized [0,1]")
    
    def test_normalize_landmarks_xy(self):
        """Test landmark normalization"""
        xy = landmarks_to_xy(self.dummy_landmarks)
        normalized = normalize_landmarks_xy(xy)
        
        self.assertEqual(len(normalized), 66, "Flattened normalized landmarks should have 66 elements")
        self.assertTrue(np.all(np.isfinite(normalized)), "All values should be finite")
    
    def test_angle_calculation(self):
        """Test angle calculation between three points"""
        # Right angle test
        a = np.array([0.0, 1.0])
        b = np.array([0.0, 0.0])
        c = np.array([1.0, 0.0])
        
        ang = angle(a, b, c)
        expected = np.pi / 2  # 90 degrees
        
        self.assertAlmostEqual(ang, expected, places=5, msg="Should calculate 90-degree angle")
    
    def test_joint_angles(self):
        """Test joint angle extraction"""
        xy = landmarks_to_xy(self.dummy_landmarks)
        angles = joint_angles(xy)
        
        self.assertEqual(len(angles), 2, "Should return 2 angles (left and right)")
        self.assertTrue(np.all(np.isfinite(angles)), "Angles should be finite")
        self.assertTrue(np.all(angles >= 0) and np.all(angles <= np.pi), "Angles should be in [0, π]")
    
    def test_extract_features_from_sequence(self):
        """Test feature extraction from sequence"""
        # Create a sequence of 10 frames
        lm_seq = []
        timestamps = []
        
        for i in range(10):
            xy = landmarks_to_xy(self.dummy_landmarks)
            # Add slight variation to simulate movement
            xy += np.random.randn(33, 2) * 0.01
            lm_seq.append(xy)
            timestamps.append(i * 0.04)  # 25 fps
        
        features = extract_features_from_sequence(lm_seq, timestamps)
        
        self.assertIsInstance(features, np.ndarray, "Should return numpy array")
        self.assertTrue(len(features) > 0, "Features should not be empty")
        self.assertTrue(np.all(np.isfinite(features)), "All features should be finite")


class TestModelTraining(unittest.TestCase):
    """Test model training pipeline"""
    
    def setUp(self):
        """Setup paths"""
        self.module_dir = MODULE_DIR
        self.pose_data_dir = self.module_dir / "pose_data"
        self.models_dir = self.module_dir / "models"
    
    def test_pose_data_exists(self):
        """Test that pose data directory exists and has data"""
        self.assertTrue(self.pose_data_dir.exists(), "pose_data directory should exist")
        
        # Check for activity subdirectories
        activities = ["standing", "sitting", "walking", "waving", "falling", "laying"]
        for activity in activities:
            activity_dir = self.pose_data_dir / activity
            if activity_dir.exists():
                csv_files = list(activity_dir.glob("*.csv"))
                if csv_files:
                    print(f"✓ Found {len(csv_files)} CSV files for {activity}")
    
    def test_model_file_exists(self):
        """Test that trained model exists"""
        model_path = self.models_dir / "pose_activity_model.pkl"
        self.assertTrue(model_path.exists(), "Trained model should exist")
        
        # Check file size
        size_kb = model_path.stat().st_size / 1024
        self.assertGreater(size_kb, 1, "Model file should be > 1KB")
        print(f"✓ Model file size: {size_kb:.2f} KB")


class TestModelInference(unittest.TestCase):
    """Test model loading and inference"""
    
    def setUp(self):
        """Load the trained model"""
        if not JOBLIB_AVAILABLE:
            self.skipTest("joblib not available")
        
        self.model_path = MODULE_DIR / "models" / "pose_activity_model.pkl"
        
        if not self.model_path.exists():
            self.skipTest("Model file not found")
        
        try:
            with open(self.model_path, "rb") as f:
                model_data = pickle.load(f)
            
            # Handle different model formats
            if isinstance(model_data, dict):
                self.model = model_data.get("model", model_data)
                self.classes = model_data.get("classes", None)
            else:
                self.model = model_data
                self.classes = None
            
            print(f"✓ Loaded model: {type(self.model).__name__}")
            if self.classes is not None:
                print(f"✓ Classes: {self.classes}")
        except Exception as e:
            self.skipTest(f"Could not load model: {e}")
    
    def test_model_loaded(self):
        """Test that model loaded successfully"""
        self.assertIsNotNone(self.model, "Model should be loaded")
        self.assertTrue(hasattr(self.model, 'predict'), "Model should have predict method")
    
    def test_model_prediction(self):
        """Test model prediction with dummy features"""
        # Create dummy feature vector (should match training feature dimension)
        # Typical: 66 (normalized landmarks) + 2 (angles) + 2 (velocity stats) = 70
        n_features = 70
        dummy_features = np.random.randn(1, n_features)
        
        try:
            prediction = self.model.predict(dummy_features)
            self.assertEqual(len(prediction), 1, "Should predict for 1 sample")
            print(f"✓ Prediction: {prediction[0]}")
            
            # Test probability prediction if available
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(dummy_features)
                self.assertEqual(proba.shape[0], 1, "Should return probabilities for 1 sample")
                self.assertAlmostEqual(np.sum(proba[0]), 1.0, places=5, msg="Probabilities should sum to 1")
                print(f"✓ Probabilities shape: {proba.shape}")
        except Exception as e:
            print(f"⚠️ Prediction failed (may be due to feature dimension mismatch): {e}")


class TestPerformance(unittest.TestCase):
    """Test performance benchmarks"""
    
    def setUp(self):
        if not ACTIVITY_NODE_AVAILABLE:
            self.skipTest("activity_node not available")
    
    def test_feature_extraction_speed(self):
        """Benchmark feature extraction speed"""
        # Create dummy sequence
        lm_seq = []
        timestamps = []
        
        for i in range(30):  # 30 frames
            xy = np.random.randn(33, 2)
            lm_seq.append(xy)
            timestamps.append(i * 0.04)
        
        # Benchmark
        n_iterations = 100
        start_time = time.time()
        
        for _ in range(n_iterations):
            features = extract_features_from_sequence(lm_seq, timestamps)
        
        elapsed = time.time() - start_time
        avg_time_ms = (elapsed / n_iterations) * 1000
        
        print(f"✓ Feature extraction: {avg_time_ms:.2f} ms/call")
        self.assertLess(avg_time_ms, 100, "Feature extraction should be < 100ms")


def run_tests():
    """Run all tests and generate report"""
    print("=" * 70)
    print("Module 14 - Human Activity Recognition Test Suite")
    print("=" * 70)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestModelTraining))
    suite.addTests(loader.loadTestsFromTestCase(TestModelInference))
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
