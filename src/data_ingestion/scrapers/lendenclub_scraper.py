"""
LendenClub Voice Assistant - Enhanced Web Scraper
Advanced web scraper with anti-detection capabilities to handle 403 errors
"""

import requests
import time
import random
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from urllib.parse import urljoin, urlparse
import hashlib

try:
    from bs4 import BeautifulSoup
    from fake_useragent import UserAgent
    BS4_AVAILABLE = True
except ImportError as e:
    print(f"Warning: BeautifulSoup or fake_useragent not available: {e}")
    BS4_AVAILABLE = False

from config.settings import (
    SCRAPING_URLS, SCRAPING_CONFIG, CONTENT_SELECTORS, 
    DATA_PATHS, FILE_PATTERNS
)

class LendenClubScraper:
    """
    Enhanced web scraper for LendenClub data with anti-detection capabilities.
    
    Features:
    - User agent rotation to avoid detection
    - Random delays between requests
    - Retry logic with exponential backoff
    - Multiple content extraction strategies
    - Error handling for 403 and other HTTP errors
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the enhanced scraper.
        
        Args:
            config: Optional custom scraping configuration
        """
        self.config = config or SCRAPING_CONFIG
        self.session = requests.Session()
        self.user_agent = UserAgent() if BS4_AVAILABLE else None
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Setup session with anti-detection headers
        self._setup_session()
        
        # Statistics tracking
        self.stats = {
            "requests_made": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_content_extracted": 0,
            "start_time": datetime.now()
        }
        
        self.logger.info("Enhanced LendenClub scraper initialized")
    
    def _setup_session(self):
        """Setup session with anti-detection configuration."""
        # Set default headers
        self.session.headers.update(self.config["headers"])
        
        # Configure SSL verification
        self.session.verify = self.config.get("verify_ssl", True)
        
        # Set timeout
        self.session.timeout = self.config.get("timeout", 30)
        
        # Configure proxies if provided
        if self.config.get("proxies"):
            self.session.proxies = self.config["proxies"]
    
    def _get_random_user_agent(self) -> str:
        """Get a random user agent string."""
        if self.user_agent and self.config.get("user_agent_rotation", True):
            try:
                return self.user_agent.random
            except:
                pass
        
        # Fallback user agents
        fallback_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0"
        ]
        return random.choice(fallback_agents)
    
    def _add_random_delay(self):
        """Add random delay between requests."""
        delay_range = self.config.get("delay_range", (1, 3))
        delay = random.uniform(delay_range[0], delay_range[1])
        time.sleep(delay)
        self.logger.debug(f"Added {delay:.2f}s delay between requests")
    
    def _make_request(self, url: str, retries: int = None) -> Optional[requests.Response]:
        """
        Make HTTP request with anti-detection measures.
        
        Args:
            url: URL to request
            retries: Number of retry attempts
            
        Returns:
            Response object or None if failed
        """
        retries = retries or self.config.get("max_retries", 3)
        
        for attempt in range(retries + 1):
            try:
                # Update user agent for each attempt
                self.session.headers.update({
                    "User-Agent": self._get_random_user_agent()
                })
                
                # Add random delay (except for first attempt)
                if attempt > 0:
                    # Exponential backoff with jitter
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
                    self.logger.info(f"Retry attempt {attempt} after {delay:.2f}s delay")
                
                # Make the request
                self.logger.debug(f"Making request to {url} (attempt {attempt + 1})")
                response = self.session.get(url)
                
                # Update statistics
                self.stats["requests_made"] += 1
                
                # Check response status
                if response.status_code == 200:
                    self.stats["successful_requests"] += 1
                    self.logger.info(f"Successfully retrieved {url}")
                    return response
                    
                elif response.status_code == 403:
                    self.logger.warning(f"403 Forbidden for {url} (attempt {attempt + 1})")
                    if attempt < retries:
                        # Add extra delay for 403 errors
                        self._add_random_delay()
                        continue
                
                elif response.status_code == 429:
                    self.logger.warning(f"429 Rate Limited for {url} (attempt {attempt + 1})")
                    if attempt < retries:
                        # Longer delay for rate limiting
                        time.sleep(30 + random.uniform(0, 10))
                        continue
                
                else:
                    self.logger.warning(f"HTTP {response.status_code} for {url}")
                    if attempt < retries:
                        continue
                
                # If we get here, all retries failed
                response.raise_for_status()
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Request error for {url} (attempt {attempt + 1}): {e}")
                if attempt < retries:
                    # Add delay before retry
                    time.sleep(2 ** attempt + random.uniform(0, 1))
                    continue
                else:
                    self.stats["failed_requests"] += 1
                    return None
        
        self.stats["failed_requests"] += 1
        return None
    
    def _extract_content(self, html: str, url: str) -> List[Dict[str, Any]]:
        """
        Extract content from HTML using multiple strategies.
        
        Args:
            html: HTML content
            url: Source URL for context
            
        Returns:
            List of extracted content sections
        """
        if not BS4_AVAILABLE:
            return [{
                "text": html[:5000],  # Fallback: return raw HTML (truncated)
                "source": url,
                "selector": "raw_html",
                "length": len(html)
            }]
        
        soup = BeautifulSoup(html, 'lxml')
        extracted_content = []
        
        # Determine content selectors based on URL
        selectors = self._get_selectors_for_url(url)
        
        for selector in selectors:
            try:
                elements = soup.select(selector)
                
                for element in elements:
                    # Clean and extract text
                    text = self._clean_text(element.get_text())
                    
                    if text and len(text.strip()) > 50:  # Minimum content length
                        extracted_content.append({
                            "text": text,
                            "source": url,
                            "selector": selector,
                            "length": len(text)
                        })
                        
                        self.logger.debug(f"Extracted {len(text)} characters using selector '{selector}'")
                
                # If we found content with this selector, we're done
                if extracted_content:
                    break
                    
            except Exception as e:
                self.logger.warning(f"Failed to extract with selector '{selector}': {e}")
                continue
        
        # Fallback: extract from common elements
        if not extracted_content:
            extracted_content = self._fallback_extraction(soup, url)
        
        self.stats["total_content_extracted"] += len(extracted_content)
        return extracted_content
    
    def _get_selectors_for_url(self, url: str) -> List[str]:
        """Get appropriate content selectors based on URL."""
        if "terms-of-services" in url:
            return CONTENT_SELECTORS.get("lendenclub_terms", ["main", ".content"])
        elif "support" in url:
            return CONTENT_SELECTORS.get("support_solutions", ["main", ".article-content"])
        elif "faq" in url:
            return CONTENT_SELECTORS.get("general_faq", [".faq-content", ".qa-content"])
        else:
            # Default selectors
            return ["main", "article", ".content", ".article-content", "body"]
    
    def _fallback_extraction(self, soup: BeautifulSoup, url: str) -> List[Dict[str, Any]]:
        """Fallback content extraction using common elements."""
        fallback_selectors = [
            "main", "article", ".content", ".main-content", 
            ".article-content", "#main", "#content", "body"
        ]
        
        for selector in fallback_selectors:
            try:
                element = soup.select_one(selector)
                if element:
                    text = self._clean_text(element.get_text())
                    if text and len(text.strip()) > 100:
                        return [{
                            "text": text,
                            "source": url,
                            "selector": f"fallback:{selector}",
                            "length": len(text)
                        }]
            except Exception as e:
                continue
        
        # Last resort: return all visible text
        text = self._clean_text(soup.get_text())
        if text:
            return [{
                "text": text[:10000],  # Truncate to avoid too much noise
                "source": url,
                "selector": "fallback:body_text",
                "length": len(text)
            }]
        
        return []
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common navigation elements
        noise_patterns = [
            "Home", "Contact", "Login", "Register", "Menu", "Search",
            "Facebook", "Twitter", "LinkedIn", "Instagram",
            "Copyright", "All rights reserved", "Terms of Use", "Privacy Policy"
        ]
        
        # Simple noise removal (can be made more sophisticated)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) > 20:  # Keep substantial content
                # Check if line is mostly noise
                if not any(noise in line for noise in noise_patterns):
                    cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines) if cleaned_lines else text
    
    def scrape_url(self, url: str, max_retries: int = None) -> Dict[str, Any]:
        """
        Scrape a single URL.
        
        Args:
            url: URL to scrape
            max_retries: Override default retry count
            
        Returns:
            Scraping result with content and metadata
        """
        self.logger.info(f"Starting to scrape {url}")
        start_time = datetime.now()
        
        # Make request
        response = self._make_request(url, retries=max_retries)
        
        if not response:
            return {
                "url": url,
                "timestamp": start_time.isoformat(),
                "status": "failed",
                "content": [],
                "error": f"Failed to retrieve {url} after retries",
                "metadata": {}
            }
        
        try:
            # Extract content
            content = self._extract_content(response.text, url)
            
            # Prepare metadata
            metadata = {
                "title": self._extract_title(response.text),
                "content_type": response.headers.get("content-type", ""),
                "status_code": response.status_code,
                "response_size": len(response.text),
                "extraction_time": (datetime.now() - start_time).total_seconds()
            }
            
            self.logger.info(f"Successfully scraped {url} - extracted {len(content)} sections")
            
            return {
                "url": url,
                "timestamp": start_time.isoformat(),
                "status": "success",
                "content": content,
                "error": None,
                "metadata": metadata
            }
            
        except Exception as e:
            self.logger.error(f"Content extraction failed for {url}: {e}")
            return {
                "url": url,
                "timestamp": start_time.isoformat(),
                "status": "failed",
                "content": [],
                "error": f"Content extraction error: {str(e)}",
                "metadata": {"status_code": response.status_code if response else None}
            }
    
    def _extract_title(self, html: str) -> str:
        """Extract page title from HTML."""
        if not BS4_AVAILABLE:
            return "Unknown Title"
        
        try:
            soup = BeautifulSoup(html, 'lxml')
            title_tag = soup.find('title')
            return title_tag.get_text().strip() if title_tag else "Unknown Title"
        except:
            return "Unknown Title"
    
    def scrape_all_urls(self, urls: List[Dict] = None) -> List[Dict[str, Any]]:
        """
        Scrape all configured URLs.
        
        Args:
            urls: Optional custom URL list, defaults to SCRAPING_URLS
            
        Returns:
            List of scraping results
        """
        urls = urls or SCRAPING_URLS
        results = []
        
        self.logger.info(f"Starting to scrape {len(urls)} URLs")
        
        for url_config in urls:
            url = url_config["url"]
            max_retries = url_config.get("retry_count", self.config.get("max_retries", 3))
            
            # Add random delay between URLs
            if results:  # Don't delay before first URL
                self._add_random_delay()
            
            result = self.scrape_url(url, max_retries=max_retries)
            results.append(result)
            
            # Log progress
            success_count = sum(1 for r in results if r["status"] == "success")
            self.logger.info(f"Progress: {len(results)}/{len(urls)} URLs processed, {success_count} successful")
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], filename: str = None) -> str:
        """
        Save scraping results to file.
        
        Args:
            results: Scraping results to save
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lendenclub_scraping_{timestamp}.json"
        
        output_path = DATA_PATHS["raw_data"] / filename
        
        # Ensure directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Scraping results saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scraping statistics."""
        runtime = datetime.now() - self.stats["start_time"]
        
        return {
            **self.stats,
            "runtime_seconds": runtime.total_seconds(),
            "success_rate": (self.stats["successful_requests"] / max(self.stats["requests_made"], 1)) * 100,
            "avg_requests_per_minute": (self.stats["requests_made"] / max(runtime.total_seconds() / 60, 1))
        }

# Convenience function for quick scraping
def scrape_lendenclub_data(save_results: bool = True) -> List[Dict[str, Any]]:
    """
    Quick function to scrape all LendenClub data.
    
    Args:
        save_results: Whether to save results to file
        
    Returns:
        List of scraping results
    """
    scraper = LendenClubScraper()
    results = scraper.scrape_all_urls()
    
    if save_results:
        scraper.save_results(results)
    
    return results

# Example usage and testing
if __name__ == "__main__":
    print("🌐 Testing Enhanced LendenClub Scraper")
    print("=" * 50)
    
    # Initialize scraper
    scraper = LendenClubScraper()
    
    # Test scraping
    results = scraper.scrape_all_urls()
    
    # Print results summary
    print(f"\n📊 Scraping Results Summary:")
    print(f"Total URLs: {len(results)}")
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        total_content = sum(len(r["content"]) for r in successful)
        print(f"Total content sections extracted: {total_content}")
    
    # Print detailed results
    print(f"\n📋 Detailed Results:")
    for result in results:
        print(f"URL: {result['url']}")
        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"Content sections: {len(result['content'])}")
            if result['content']:
                avg_length = sum(c['length'] for c in result['content']) / len(result['content'])
                print(f"Average section length: {avg_length:.0f} characters")
        else:
            print(f"Error: {result['error']}")
        print("-" * 30)
    
    # Save results
    if results:
        filepath = scraper.save_results(results)
        print(f"\n💾 Results saved to: {filepath}")
    
    # Print statistics
    stats = scraper.get_statistics()
    print(f"\n📈 Statistics:")
    print(f"Requests made: {stats['requests_made']}")
    print(f"Success rate: {stats['success_rate']:.1f}%")
    print(f"Runtime: {stats['runtime_seconds']:.1f} seconds")
