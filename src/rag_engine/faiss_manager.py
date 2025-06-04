"""
FAISS Manager Implementation for LendenClub Voice Assistant
Handles vector database operations for document storage and retrieval
"""

import faiss
import numpy as np
import pickle
import os
import logging
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import json
from datetime import datetime

class FAISSManager:
    """
    FAISS-based vector database manager for document storage and retrieval
    Supports offline operation with sentence transformers for embeddings
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 index_path: str = "data/vector_db/faiss_index",
                 metadata_path: str = "data/vector_db/metadata.json"):
        
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.index_path = index_path
        self.metadata_path = metadata_path
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        # Initialize sentence transformer
        try:
            self.encoder = SentenceTransformer(model_name)
            self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
            self.logger.info(f"✅ Sentence transformer loaded: {model_name}")
        except Exception as e:
            self.logger.error(f"❌ Failed to load sentence transformer: {e}")
            raise
        
        # Initialize FAISS index
        self.index = None
        self.metadata = []
        self.doc_count = 0
        
        # Load existing index if available
        self._load_index()
        
    def _load_index(self):
        """Load existing FAISS index and metadata"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                self.index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                self.doc_count = len(self.metadata)
                self.logger.info(f"✅ Loaded existing index with {self.doc_count} documents")
            else:
                self._create_new_index()
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load existing index: {e}")
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index"""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata = []
        self.doc_count = 0
        self.logger.info("✅ Created new FAISS index")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the vector database"""
        try:
            texts = [doc.get('text', '') for doc in documents]
            if not texts:
                self.logger.warning("⚠️ No texts to add")
                return False
            
            # Generate embeddings
            embeddings = self.encoder.encode(texts, convert_to_numpy=True)
            
            # Add to FAISS index
            self.index.add(embeddings.astype('float32'))
            
            # Store metadata
            for i, doc in enumerate(documents):
                metadata_entry = {
                    'id': self.doc_count + i,
                    'text': doc.get('text', ''),
                    'title': doc.get('title', ''),
                    'source': doc.get('source', ''),
                    'timestamp': datetime.now().isoformat(),
                    'length': len(doc.get('text', ''))
                }
                self.metadata.append(metadata_entry)
            
            self.doc_count += len(documents)
            self.logger.info(f"✅ Added {len(documents)} documents to index")
            
            # Save index and metadata
            self._save_index()
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error adding documents: {e}")
            return False
    
    def search(self, query: str, k: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            if self.doc_count == 0:
                self.logger.warning("⚠️ No documents in index")
                return []
            
            # Generate query embedding
            query_embedding = self.encoder.encode([query], convert_to_numpy=True)
            
            # Search FAISS index
            distances, indices = self.index.search(query_embedding.astype('float32'), k)
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.metadata):
                    # Convert distance to similarity score (0-1)
                    similarity = max(0, 1 - (distance / 10))  # Normalize distance
                    
                    if similarity >= threshold:
                        result = self.metadata[idx].copy()
                        result['similarity'] = similarity
                        result['rank'] = i + 1
                        results.append(result)
            
            self.logger.info(f"✅ Found {len(results)} relevant documents for query")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error searching documents: {e}")
            return []
    
    def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            self.logger.debug(f"💾 Saved index with {self.doc_count} documents")
        except Exception as e:
            self.logger.error(f"❌ Error saving index: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            'total_documents': self.doc_count,
            'embedding_dimension': self.embedding_dim,
            'model_name': self.model_name,
            'index_type': 'IndexFlatL2',
            'last_updated': datetime.now().isoformat()
        }
    
    def clear_index(self):
        """Clear all documents from the index"""
        self._create_new_index()
        self._save_index()
        self.logger.info("🗑️ Cleared all documents from index")

# Test functionality when run directly
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize FAISS manager
    manager = FAISSManager()
    
    # Test with sample documents
    sample_docs = [
        {
            'text': 'Minimum salary requirement for personal loans is Rs. 25,000 per month.',
            'title': 'Loan Eligibility Criteria',
            'source': 'lendenclub.com/eligibility'
        },
        {
            'text': 'Interest rates vary from 12% to 36% annually based on credit score.',
            'title': 'Interest Rate Information',
            'source': 'lendenclub.com/rates'
        }
    ]
    
    # Add documents
    success = manager.add_documents(sample_docs)
    print(f"✅ Documents added successfully: {success}")
    
    # Test search
    results = manager.search("What is minimum salary needed?", k=2)
    print(f"🔍 Search results: {len(results)} found")
    
    for result in results:
        print(f"  - {result['title']}: {result['similarity']:.3f}")
    
    # Print stats
    stats = manager.get_stats()
    print(f"📊 Database stats: {stats}")
