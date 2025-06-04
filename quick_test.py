#!/usr/bin/env python3
"""
Quick test runner for Phase 2 components
"""

import sys
import os
sys.path.append('src')

def quick_test():
    print("🧪 Running Quick Phase 2 Tests...")
    
    # Test FAISS Manager
    try:
        from rag_engine.faiss_manager import FAISSManager
        manager = FAISSManager()
        print("✅ FAISS Manager: OK")
    except Exception as e:
        print(f"❌ FAISS Manager: {e}")
    
    # Test Scraper
    try:
        from data_ingestion.scrapers.lendenclub_scraper import LendenClubScraper
        scraper = LendenClubScraper()
        if hasattr(scraper, 'scrape'):
            print("✅ Scraper: OK (has scrape method)")
        else:
            print("❌ Scraper: Missing scrape method")
    except Exception as e:
        print(f"❌ Scraper: {e}")
    
    # Test Performance Evaluator
    try:
        from intent_classification.evaluator.performance_evaluator import PerformanceEvaluator
        evaluator = PerformanceEvaluator()
        print("✅ Performance Evaluator: OK")
    except Exception as e:
        print(f"❌ Performance Evaluator: {e}")
    
    print("\n🎉 Quick test complete!")

if __name__ == "__main__":
    quick_test()
