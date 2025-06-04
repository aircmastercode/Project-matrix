# FAISS Vector Database Manager for LendenClub RAG System
# Handles document embedding, storage, and retrieval

import numpy as np
import faiss
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSVectorManager:
    """
    Manages FAISS vector database for document storage and retrieval
    Completely free and offline solution for RAG system
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.embedding_model = SentenceTransformer(config['embedding_model'])
        self.vector_dimension = config['vector_dimension']
        self.index = None
        self.documents = []
        self.metadata = []
        
        # Initialize paths
        self.index_path = Path(config['index_path'])
        self.metadata_path = Path(config['metadata_path'])
        
        # Create directories
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized FAISS Vector Manager with {config['embedding_model']}")
    
    def create_index(self):
        """Create a new FAISS index"""
        if self.config['index_type'] == "IndexFlatIP":
            # Inner Product for cosine similarity (after L2 normalization)
            self.index = faiss.IndexFlatIP(self.vector_dimension)
        elif self.config['index_type'] == "IndexFlatL2":
            # L2 distance (Euclidean)
            self.index = faiss.IndexFlatL2(self.vector_dimension)
        else:
            # Default to Flat IP
            self.index = faiss.IndexFlatIP(self.vector_dimension)
            
        logger.info(f"Created new FAISS index: {self.config['index_type']}")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Convert texts to embeddings using sentence transformer"""
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            
            # Normalize for cosine similarity (if using IndexFlatIP)
            if self.config['index_type'] == "IndexFlatIP":
                faiss.normalize_L2(embeddings)
                
            return embeddings.astype('float32')
        except Exception as e:
            logger.error(f"Error embedding texts: {str(e)}")
            return np.array([])
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to the vector database"""
        if not documents:
            logger.warning("No documents provided for addition")
            return
            
        # Create index if it doesn't exist
        if self.index is None:
            self.create_index()
        
        # Generate embeddings
        embeddings = self.embed_texts(documents)
        if embeddings.size == 0:
            logger.error("Failed to generate embeddings")
            return
        
        # Add to index
        self.index.add(embeddings)
        
        # Store documents and metadata
        start_idx = len(self.documents)
        self.documents.extend(documents)
        
        if metadata is None:
            metadata = [{"id": start_idx + i, "source": "unknown"} for i in range(len(documents))]
        
        self.metadata.extend(metadata)
        
        logger.info(f"Added {len(documents)} documents to vector database")
        logger.info(f"Total documents: {len(self.documents)}")
    
    def search(self, query: str, k: int = 5, threshold: float = 0.7) -> List[Dict]:
        """Search for similar documents"""
        if self.index is None or len(self.documents) == 0:
            logger.warning("No documents in database for search")
            return []
        
        # Embed query
        query_embedding = self.embed_texts([query])
        if query_embedding.size == 0:
            logger.error("Failed to embed query")
            return []
        
        # Search
        scores, indices = self.index.search(query_embedding, min(k, len(self.documents)))
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold and idx != -1:  # -1 indicates no match found
                results.append({
                    "document": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "score": float(score),
                    "index": int(idx)
                })
        
        logger.info(f"Found {len(results)} relevant documents for query")
        return results
    
    def save_index(self):
        """Save index and metadata to disk"""
        try:
            if self.index is not None:
                faiss.write_index(self.index, str(self.index_path))
                
            # Save metadata and documents
            metadata_dict = {
                "documents": self.documents,
                "metadata": self.metadata,
                "config": self.config
            }
            
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved index to {self.index_path}")
            logger.info(f"Saved metadata to {self.metadata_path}")
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
    
    def load_index(self):
        """Load index and metadata from disk"""
        try:
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                logger.info(f"Loaded index from {self.index_path}")
            
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    metadata_dict = json.load(f)
                    
                self.documents = metadata_dict.get("documents", [])
                self.metadata = metadata_dict.get("metadata", [])
                
                logger.info(f"Loaded {len(self.documents)} documents from metadata")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "vector_dimension": self.vector_dimension,
            "index_type": self.config['index_type'],
            "model_name": self.config['embedding_model']
        }
    
    def clear_database(self):
        """Clear all documents and reset index"""
        self.documents = []
        self.metadata = []
        self.create_index()
        logger.info("Cleared vector database")

# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    test_config = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "vector_dimension": 384,
        "index_type": "IndexFlatIP",
        "index_path": "data/vectorstore/test_faiss_index.bin",
        "metadata_path": "data/vectorstore/test_metadata.json"
    }
    
    # Initialize manager
    vector_manager = FAISSVectorManager(test_config)
    
    # Test documents
    test_docs = [
        "LendenClub requires minimum salary of Rs. 25,000 for loan eligibility",
        "Interest rates range from 12% to 36% per annum based on risk assessment", 
        "Required documents include salary slip, bank statement, and PAN card",
        "EMI can be modified once per loan tenure with 7 days notice",
        "Processing fee is 2.5% of loan amount with minimum Rs. 1,000"
    ]
    
    test_metadata = [
        {"source": "eligibility_criteria", "category": "loan_eligibility"},
        {"source": "rate_structure", "category": "interest_rates"},
        {"source": "documentation", "category": "documentation"},
        {"source": "payment_terms", "category": "repayment_terms"},
        {"source": "fee_structure", "category": "fees_charges"}
    ]
    
    # Add documents
    vector_manager.add_documents(test_docs, test_metadata)
    
    # Test search
    results = vector_manager.search("What salary is needed for loan?", k=3)
    
    print("Search Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['score']:.3f}")
        print(f"   Document: {result['document']}")
        print(f"   Category: {result['metadata']['category']}")
        print()
    
    # Save for future use
    vector_manager.save_index()
    
    print("✅ FAISS Vector Manager test completed!")
