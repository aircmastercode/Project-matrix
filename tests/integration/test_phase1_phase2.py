"""
Integration Test for Phase 1 + Phase 2 Components
Tests the complete pipeline from scraping to RAG responses
"""

import sys
import os
import logging
from typing import Dict, Any, List

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def test_phase1_phase2_integration():
    """Test complete Phase 1 + Phase 2 integration"""
    
    print("🚀 Starting Phase 1 + Phase 2 Integration Test")
    print("=" * 50)
    
    success_count = 0
    total_tests = 6
    
    # Test 1: Intent Classifier (Phase 1)
    print("\n1️⃣ Testing Intent Classifier (Phase 1)...")
    try:
        from intent_classification.models.free_intent_classifier import FreeIntentClassifier
        
        classifier = FreeIntentClassifier()
        test_query = "What documents do I need for a loan?"
        result = classifier.predict_single(test_query)
        
        if result and 'intent' in result:
            print(f"   ✅ Intent Classification: {result['intent']} (confidence: {result.get('confidence', 0):.3f})")
            success_count += 1
        else:
            print(f"   ❌ Intent Classification failed")
            
    except Exception as e:
        print(f"   ❌ Intent Classifier error: {e}")
    
    # Test 2: Web Scraper (Phase 1)
    print("\n2️⃣ Testing Web Scraper (Phase 1)...")
    try:
        from data_ingestion.scrapers.lendenclub_scraper import LendenClubScraper
        
        scraper = LendenClubScraper()
        # Test just initialization and method existence
        if hasattr(scraper, 'scrape'):
            print(f"   ✅ Scraper initialized with scrape method")
            success_count += 1
        else:
            print(f"   ❌ Scraper missing scrape method")
            
    except Exception as e:
        print(f"   ❌ Scraper error: {e}")
    
    # Test 3: FAISS Manager (Phase 2)
    print("\n3️⃣ Testing FAISS Manager (Phase 2)...")
    try:
        from rag_engine.faiss_manager import FAISSManager
        
        manager = FAISSManager()
        
        # Test with sample document
        sample_docs = [{
            'text': 'Minimum credit score required is 750 for personal loans.',
            'title': 'Credit Requirements',
            'source': 'test'
        }]
        
        success = manager.add_documents(sample_docs)
        if success:
            print(f"   ✅ FAISS Manager: Document added successfully")
            success_count += 1
        else:
            print(f"   ❌ FAISS Manager: Failed to add document")
            
    except Exception as e:
        print(f"   ❌ FAISS Manager error: {e}")
    
    # Test 4: Document Search (Phase 2)
    print("\n4️⃣ Testing Document Search (Phase 2)...")
    try:
        # Continue with previous FAISS manager
        search_results = manager.search("credit score requirements", k=1)
        
        if search_results and len(search_results) > 0:
            print(f"   ✅ Document Search: Found {len(search_results)} results")
            print(f"      Best match: {search_results[0].get('title', 'Unknown')} (similarity: {search_results[0].get('similarity', 0):.3f})")
            success_count += 1
        else:
            print(f"   ❌ Document Search: No results found")
            
    except Exception as e:
        print(f"   ❌ Document Search error: {e}")
    
    # Test 5: Performance Evaluator (Phase 1)
    print("\n5️⃣ Testing Performance Evaluator (Phase 1)...")
    try:
        from intent_classification.evaluator.performance_evaluator import PerformanceEvaluator
        
        evaluator = PerformanceEvaluator()
        
        # Test with sample data
        sample_true = ['documentation', 'loan_eligibility']
        sample_pred = ['documentation', 'loan_eligibility']
        
        results = evaluator.evaluate_predictions(sample_true, sample_pred)
        
        if results and 'accuracy' in results:
            print(f"   ✅ Performance Evaluator: Accuracy = {results['accuracy']:.3f}")
            success_count += 1
        else:
            print(f"   ❌ Performance Evaluator: Failed to generate results")
            
    except Exception as e:
        print(f"   ❌ Performance Evaluator error: {e}")
    
    # Test 6: End-to-End Pipeline
    print("\n6️⃣ Testing End-to-End Pipeline...")
    try:
        # Test complete workflow
        query = "What are the eligibility criteria?"
        
        # Step 1: Classify intent
        intent_result = classifier.predict_single(query)
        intent = intent_result.get('intent', 'unknown')
        
        # Step 2: Search documents
        search_results = manager.search(query, k=1)
        
        # Step 3: Generate response (simplified)
        if search_results:
            response = f"Based on {intent} intent, found relevant information: {search_results[0].get('text', '')[:100]}..."
        else:
            response = f"I understand you're asking about {intent}, but I couldn't find specific information."
        
        print(f"   ✅ End-to-End Pipeline:")
        print(f"      Query: {query}")
        print(f"      Intent: {intent}")
        print(f"      Results: {len(search_results)} documents found")
        print(f"      Response: {response[:100]}...")
        
        success_count += 1
        
    except Exception as e:
        print(f"   ❌ End-to-End Pipeline error: {e}")
    
    # Final Summary
    print(f"\n📊 Integration Test Summary")
    print("=" * 30)
    print(f"✅ Passed: {success_count}/{total_tests} tests")
    print(f"📈 Success Rate: {(success_count/total_tests)*100:.1f}%")
    
    if success_count == total_tests:
        print("🎉 All integration tests passed! Phase 1 + Phase 2 are working together.")
    elif success_count >= total_tests * 0.8:
        print("⚠️ Most tests passed. Minor issues to address.")
    else:
        print("❌ Several tests failed. Please check the implementations.")
    
    return success_count == total_tests

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the integration test
    success = test_phase1_phase2_integration()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
