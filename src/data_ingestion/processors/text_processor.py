"""
LendenClub Voice Assistant - Text Processor
Basic text processing utilities for Phase 1
"""

import re
import string
from typing import List, Dict, Any

class TextProcessor:
    """Basic text processing for scraped content and user queries."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text.strip()
    
    @staticmethod
    def extract_keywords(text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction
        words = text.lower().split()
        
        # Filter out common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = [word.strip(string.punctuation) for word in words if word not in stop_words and len(word) > 2]
        
        return list(set(keywords))
    
    @staticmethod
    def normalize_query(query: str) -> str:
        """Normalize user query for classification."""
        query = TextProcessor.clean_text(query)
        return query.lower().strip()
