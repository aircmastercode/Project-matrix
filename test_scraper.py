#!/usr/bin/env python3
"""Quick test of web scraper"""

from src.data_ingestion.scrapers.lendenclub_scraper import LendenClubScraper

print("🌐 Testing Enhanced Web Scraper")
print("=" * 40)

scraper = LendenClubScraper()
results = scraper.scrape_all_urls()

for result in results:
    print(f"URL: {result['url']}")
    print(f"Status: {result['status']}")
    if result['status'] == 'success':
        print(f"Content sections: {len(result['content'])}")
    else:
        print(f"Error: {result['error']}")
    print("-" * 30)

print("\n✅ Scraper test completed!")
