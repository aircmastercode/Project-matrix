"""
Enhanced LendenClub Scraper with 403 Error Handling
Implements anti-detection techniques for robust data collection
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
from fake_useragent import UserAgent

class LendenClubScraper:
    """Enhanced web scraper for LendenClub with anti-detection capabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ua = UserAgent()
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
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = ' '.join(text.strip().split())
        
        # Remove special characters but keep basic punctuation
        import re
        text = re.sub(r'[^\w\s\.,!?-]', '', text)
        
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
                content_selectors = [
                    'main',
                    '.article-content',
                    '.content',
                    'article',
                    '.main-content',
                    'body'
                ]
                
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
        """Main scraping method that returns scraped data"""
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

# Test functionality when run directly
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    scraper = LendenClubScraper()
    
    # Test scraping
    results = scraper.scrape()
    
    # Show results summary
    for result in results:
        print(f"\n🌐 URL: {result['url']}")
        print(f"📊 Status: {result['status']}")
        print(f"📄 Content pieces: {len(result['content'])}")
        if result['error']:
            print(f"❌ Error: {result['error']}")
        else:
            print(f"✅ Success: {sum(len(c['text']) for c in result['content'])} characters")
