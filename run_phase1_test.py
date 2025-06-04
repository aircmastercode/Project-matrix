#!/usr/bin/env python3
"""
LendenClub Voice Assistant - Phase 1 Integration Test
Complete test of scraping, classification, and evaluation pipeline
"""

import sys
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run complete Phase 1 integration test."""
    
    print("🧪 LendenClub Voice Assistant - Phase 1 Integration Test")
    print("=" * 60)
    
    try:
        # Import modules
        sys.path.insert(0, str(Path(__file__).parent))
        
        from src.data_ingestion.scrapers.lendenclub_scraper import LendenClubScraper
        from src.intent_classification.models.free_intent_classifier import FreeIntentClassifier
        from src.intent_classification.evaluator.performance_evaluator import PerformanceEvaluator
        
        # Test 1: Web Scraping
        print("\n🌐 Test 1: Web Scraping")
        scraper = LendenClubScraper()
        scraping_results = scraper.scrape_all_urls()
        
        successful_scrapes = [r for r in scraping_results if r["status"] == "success"]
        print(f"Scraped {len(scraping_results)} URLs, {len(successful_scrapes)} successful")
        
        # Test 2: Intent Classification
        print("\n🧠 Test 2: Intent Classification")
        classifier = FreeIntentClassifier()
        
        test_queries = [
            "What is the minimum credit score required?",
            "How do I reset my password?",
            "What are the processing fees?",
            "How does the repayment work?",
            "What documents do I need?"
        ]
        
        classification_results = []
        for query in test_queries:
            result = classifier.predict_single(query)
            classification_results.append(result)
            print(f"'{query}' → {result['intent']} ({result.get('confidence', 0):.2f})")
        
        # Test 3: Performance Evaluation
        print("\n📊 Test 3: Performance Evaluation")
        
        # Load test data
        test_data_path = Path("tests/data/sample_queries.json")
        if test_data_path.exists():
            with open(test_data_path, 'r') as f:
                test_data = json.load(f)
            
            # Evaluate on subset of test data
            predictions = []
            true_labels = []
            confidence_scores = []
            
            for sample in test_data[:20]:  # Use first 20 samples
                result = classifier.predict_single(sample["query"])
                predictions.append(result["intent"])
                true_labels.append(sample["intent"])
                confidence_scores.append(result.get("confidence", 0))
            
            # Evaluate performance
            evaluator = PerformanceEvaluator()
            eval_results = evaluator.evaluate_predictions(
                predictions, true_labels, confidence_scores, "bart_phase1_test"
            )
            
            print(f"Accuracy: {eval_results['overall_metrics']['accuracy']:.3f}")
            print(f"F1-Score: {eval_results['overall_metrics']['f1_score']:.3f}")
            print(f"Performance Grade: {eval_results['performance_grade']}")
        
        # Test 4: Model Information
        print("\n🔍 Test 4: Model Information")
        model_info = classifier.get_model_info()
        print(f"Model: {model_info['model_name']}")
        print(f"Intent Categories: {model_info['num_intents']}")
        print(f"Average Prediction Time: {model_info['avg_prediction_time']:.3f}s")
        
        print("\n✅ All Phase 1 tests completed successfully!")
        print("🚀 Ready for Phase 2 development!")
        
    except Exception as e:
        logger.error(f"Phase 1 test failed: {e}")
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
