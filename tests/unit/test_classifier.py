"""
LendenClub Voice Assistant - Classifier Test
Unit tests for the free intent classifier
"""

import unittest
import logging
import json
from pathlib import Path
import sys
import os

# Add parent directory to path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.intent_classification.models.free_intent_classifier import FreeIntentClassifier
    from config.settings import INTENT_CATEGORIES
except ImportError as e:
    print(f"Import error: {e}")

class TestFreeIntentClassifier(unittest.TestCase):
    """Test cases for the Free Intent Classifier."""
    
    @classmethod
    def setUpClass(cls):
        """Setup for all test cases."""
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Load test data
        test_data_path = Path(__file__).parent.parent / "tests" / "data" / "sample_queries.json"
        
        try:
            with open(test_data_path, 'r', encoding='utf-8') as f:
                cls.test_data = json.load(f)
        except Exception as e:
            print(f"Failed to load test data: {e}")
            cls.test_data = []
    
    def setUp(self):
        """Setup before each test case."""
        self.classifier = FreeIntentClassifier(model_name="bart")
    
    def test_initialization(self):
        """Test classifier initialization."""
        self.assertIsNotNone(self.classifier)
        self.assertEqual(self.classifier.model_name, "bart")
        self.assertIsNotNone(self.classifier.intent_labels)
        self.assertGreater(len(self.classifier.intent_labels), 0)
    
    def test_predict_single(self):
        """Test single prediction functionality."""
        test_query = "What is the minimum credit score required for a loan?"
        result = self.classifier.predict_single(test_query)
        
        self.assertIsNotNone(result)
        self.assertIn("intent", result)
        self.assertIn("confidence", result)
        self.assertIn(result["intent"], self.classifier.intent_labels)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
    
    def test_empty_query(self):
        """Test handling of empty queries."""
        result = self.classifier.predict_single("")
        
        # Should default to fallback intent with low confidence
        self.assertIn("intent", result)
        self.assertIn("confidence", result)
        self.assertLess(result["confidence"], 0.5)
    
    def test_batch_prediction(self):
        """Test batch prediction functionality."""
        test_queries = ["How do I reset my password?", "What is the interest rate?"]
        results = self.classifier.predict_batch(test_queries)
        
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertIn("intent", result)
            self.assertIn("confidence", result)
            self.assertIn(result["intent"], self.classifier.intent_labels)
    
    def test_performance_on_samples(self):
        """Test performance on sample queries."""
        if not self.test_data:
            self.skipTest("No test data available")
        
        # Limit to first 10 samples for speed
        samples = self.test_data[:10]
        
        correct_count = 0
        total_diff = 0
        
        for sample in samples:
            query = sample["query"]
            expected_intent = sample["intent"]
            
            result = self.classifier.predict_single(query)
            predicted_intent = result["intent"]
            
            # Compare with expected intent
            if predicted_intent == expected_intent:
                correct_count += 1
            
            # Track confidence difference
            if "confidence" in sample:
                confidence_diff = abs(result["confidence"] - sample["confidence"])
                total_diff += confidence_diff
        
        accuracy = correct_count / len(samples)
        avg_confidence_diff = total_diff / len(samples)
        
        print(f"Test accuracy: {accuracy:.2f}")
        print(f"Average confidence difference: {avg_confidence_diff:.2f}")
        
        # Should maintain reasonable accuracy
        self.assertGreaterEqual(accuracy, 0.5)
    
    def test_model_info(self):
        """Test model info retrieval."""
        info = self.classifier.get_model_info()
        
        self.assertIn("model_name", info)
        self.assertIn("intent_labels", info)
        self.assertIn("num_intents", info)
    
    def test_error_handling(self):
        """Test error handling during prediction."""
        # Temporarily break the classifier to test error handling
        original_classifier = self.classifier.classifier
        self.classifier.classifier = None
        
        try:
            result = self.classifier.predict_single("Test query")
            self.assertIn("error", result)
        finally:
            # Restore the classifier
            self.classifier.classifier = original_classifier

if __name__ == '__main__':
    unittest.main()
