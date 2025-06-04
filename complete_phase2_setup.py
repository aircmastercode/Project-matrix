#!/usr/bin/env python3
"""
Complete Phase 2 Setup and Fix Script
Addresses all the issues found in your testing and sets up missing components
"""

import os
import shutil
import logging
from pathlib import Path

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def create_directory_structure():
    """Create the complete directory structure for Phase 2"""
    
    directories = [
        # Phase 1 directories (ensure they exist)
        "src/data_ingestion/scrapers",
        "src/intent_classification/models", 
        "src/intent_classification/evaluator",
        
        # Phase 2 new directories
        "src/rag_engine",
        "src/response_generation",
        
        # Test directories
        "tests/unit",
        "tests/integration",
        "tests/data",
        
        # Data directories
        "data/raw",
        "data/processed", 
        "data/vector_db",
        
        # Reports and logs
        "reports",
        "logs",
        
        # Config
        "config"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
        # Create __init__.py files for Python packages
        if "src/" in directory:
            init_file = os.path.join(directory, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, "w") as f:
                    f.write(f'"""Package: {directory}"""\n')

def fix_scraper_implementation():
    """Fix the LendenClubScraper implementation"""
    
    scraper_content = '''"""
Enhanced LendenClub Scraper with scrape() method
"""

import requests
from bs4 import BeautifulSoup
import time
import random
import logging
import json
from typing import List, Dict, Any
from datetime import datetime
import os

class LendenClubScraper:
    """Enhanced web scraper for LendenClub with anti-detection capabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        
        # URLs to scrape
        self.urls = [
            "https://www.lendenclub.com/terms-of-services/",
            "https://support.lendenclub.com/support/solutions"
        ]
        
        # Create data directories
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        
    def _get_headers(self):
        """Generate realistic browser headers"""
        user_agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        
        return {
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = ' '.join(text.strip().split())
        
        # Basic cleaning
        import re
        text = re.sub(r'[^\\w\\s\\.,!?-]', '', text)
        
        return text
    
    def _scrape_single_url(self, url: str, max_retries: int = 3) -> Dict[str, Any]:
        """Scrape a single URL with retry logic"""
        result = {
            'url': url,
            'timestamp': datetime.now().isoformat(),
            'status': 'failed',
            'content': [],
            'error': None,
            'metadata': {}
        }
        
        for attempt in range(max_retries):
            try:
                # Random delay between requests
                time.sleep(random.uniform(1, 3))
                
                # Make request with rotating headers
                headers = self._get_headers()
                response = self.session.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                # Parse content
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Store metadata
                result['metadata'] = {
                    'title': soup.title.string if soup.title else '',
                    'content_type': response.headers.get('content-type', ''),
                    'status_code': response.status_code,
                    'response_size': len(response.content)
                }
                
                # Extract content using multiple selectors
                content_selectors = ['main', '.article-content', '.content', 'article', '.main-content', 'body']
                
                content_found = False
                for selector in content_selectors:
                    elements = soup.select(selector)
                    if elements:
                        for element in elements:
                            text = self._clean_text(element.get_text())
                            if text and len(text) > 50:  # Only include substantial content
                                result['content'].append({
                                    'text': text,
                                    'source': url,
                                    'selector': selector,
                                    'length': len(text)
                                })
                                content_found = True
                
                if content_found:
                    result['status'] = 'success'
                    result['error'] = None
                    self.logger.info(f"✅ Successfully scraped {url}")
                    break
                else:
                    self.logger.warning(f"⚠️ No content found for {url}")
                    
            except requests.exceptions.HTTPError as e:
                error_msg = f"Error scraping {url}: {e}"
                result['error'] = error_msg
                self.logger.error(f"❌ HTTP Error (attempt {attempt + 1}): {error_msg}")
                
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    self.logger.info(f"⏳ Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                error_msg = f"Error scraping {url}: {e}"
                result['error'] = error_msg
                self.logger.error(f"❌ Unexpected error: {error_msg}")
                break
        
        return result
    
    def scrape(self) -> List[Dict[str, Any]]:
        """Main scraping method - THIS IS THE MISSING METHOD"""
        self.logger.info(f"🚀 Starting scraping of {len(self.urls)} URLs...")
        
        results = []
        for url in self.urls:
            self.logger.info(f"🔍 Scraping: {url}")
            result = self._scrape_single_url(url)
            results.append(result)
            
            # Brief pause between URLs
            time.sleep(random.uniform(2, 4))
        
        # Save results to file
        output_file = f"data/raw/scraped_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        successful = sum(1 for r in results if r['status'] == 'success')
        total_content = sum(len(r['content']) for r in results)
        
        self.logger.info(f"📊 Scraping complete:")
        self.logger.info(f"   - Total URLs: {len(results)}")
        self.logger.info(f"   - Successful: {successful}")
        self.logger.info(f"   - Content pieces: {total_content}")
        self.logger.info(f"   - Saved to: {output_file}")
        
        return results
    
    def get_processed_documents(self) -> List[Dict[str, Any]]:
        """Get documents in format suitable for RAG system"""
        results = self.scrape()
        documents = []
        
        for result in results:
            if result['status'] == 'success':
                for content in result['content']:
                    doc = {
                        'text': content['text'],
                        'title': result['metadata'].get('title', 'Unknown'),
                        'source': result['url'],
                        'timestamp': result['timestamp'],
                        'length': content['length']
                    }
                    documents.append(doc)
        
        self.logger.info(f"📝 Processed {len(documents)} documents for RAG system")
        return documents
'''
    
    # Write the fixed scraper
    with open("src/data_ingestion/scrapers/lendenclub_scraper.py", "w") as f:
        f.write(scraper_content)

def fix_performance_evaluator():
    """Fix the performance evaluator with proper label handling"""
    
    evaluator_content = '''"""
Fixed Performance Evaluator for LendenClub Voice Assistant
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import logging
from typing import List, Dict, Any
import json
from datetime import datetime
import os

class PerformanceEvaluator:
    """Performance evaluation with proper label handling"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Updated intent categories
        self.intent_categories = [
            'loan_eligibility',
            'documentation', 
            'interest_rates',
            'account_management',
            'fees_charges',
            'general_inquiry'
        ]
        
        os.makedirs("reports", exist_ok=True)
        
    def evaluate_predictions(self, y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        """Evaluate predictions with proper label handling"""
        
        # Get unique labels from both true and predicted
        unique_labels = sorted(list(set(y_true + y_pred)))
        
        # Only use labels that exist in data
        valid_labels = [label for label in unique_labels if label in self.intent_categories]
        
        if not valid_labels:
            valid_labels = unique_labels
        
        self.logger.info(f"📊 Evaluating with labels: {valid_labels}")
        
        try:
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, labels=valid_labels, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, labels=valid_labels, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, labels=valid_labels, average='weighted', zero_division=0)
            
            # Fixed classification report
            per_class_report = classification_report(
                y_true, y_pred, 
                labels=valid_labels,
                target_names=valid_labels,  # Same as labels
                output_dict=True,
                zero_division=0
            )
            
            results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'per_class_metrics': per_class_report,
                'evaluation_timestamp': datetime.now().isoformat(),
                'total_samples': len(y_true),
                'unique_labels': valid_labels
            }
            
            self.logger.info(f"✅ Evaluation complete - Accuracy: {accuracy:.3f}")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error in evaluation: {e}")
            return {
                'error': str(e),
                'accuracy': 0.0,
                'evaluation_timestamp': datetime.now().isoformat()
            }

if __name__ == "__main__":
    evaluator = PerformanceEvaluator()
    
    # Test with sample data
    sample_true = ['documentation', 'loan_eligibility', 'interest_rates']
    sample_pred = ['documentation', 'loan_eligibility', 'general_inquiry']
    
    results = evaluator.evaluate_predictions(sample_true, sample_pred)
    print(f"Test Results: Accuracy = {results.get('accuracy', 0):.3f}")
'''
    
    with open("src/intent_classification/evaluator/performance_evaluator.py", "w") as f:
        f.write(evaluator_content)

def main():
    """Main setup function"""
    logger = setup_logging()
    
    logger.info("🚀 Starting Complete Phase 2 Setup and Fix...")
    
    # Step 1: Create directory structure
    logger.info("📁 Creating directory structure...")
    create_directory_structure()
    
    # Step 2: Fix scraper with missing scrape() method
    logger.info("🔧 Fixing LendenClub scraper...")
    fix_scraper_implementation()
    
    # Step 3: Fix performance evaluator 
    logger.info("📊 Fixing performance evaluator...")
    fix_performance_evaluator()
    
    # Step 4: Copy other files if they exist
    file_mappings = [
        ("faiss_manager.py", "src/rag_engine/faiss_manager.py"),
        ("test_phase1_phase2_integration.py", "tests/integration/test_phase1_phase2.py"),
        ("requirements_phase2.txt", "requirements.txt")
    ]
    
    for source, destination in file_mappings:
        if os.path.exists(source):
            logger.info(f"📄 Moving {source} to {destination}")
            shutil.copy2(source, destination)
    
    # Step 5: Create test runner script
    test_runner_content = '''#!/usr/bin/env python3
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
    
    print("\\n🎉 Quick test complete!")

if __name__ == "__main__":
    quick_test()
'''
    
    with open("quick_test.py", "w") as f:
        f.write(test_runner_content)
    
    os.chmod("quick_test.py", 0o755)
    
    logger.info("✅ Setup complete! Next steps:")
    logger.info("1. Install dependencies: pip install -r requirements.txt")
    logger.info("2. Run quick test: python quick_test.py")
    logger.info("3. Test scraper: python -c \"from src.data_ingestion.scrapers.lendenclub_scraper import LendenClubScraper; scraper = LendenClubScraper(); print('✅ Scraper ready')\"")
    logger.info("4. Test FAISS: python -c \"from src.rag_engine.faiss_manager import FAISSManager; manager = FAISSManager(); print('✅ FAISS ready')\"")

if __name__ == "__main__":
    main()
