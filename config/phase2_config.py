# Fixed Phase 2 Configuration - Minimal Version
from pathlib import Path

# Intent Classification Settings (FIXED)
INTENT_CONFIG = {
    "model_type": "bart_large_mnli",  # Fixed: was "bart_zero_shot"
    "model_name": "facebook/bart-large-mnli",  # Correct identifier
    "confidence_threshold": 0.6
}

# RAG Settings
RAG_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "vector_dimension": 384,
    "similarity_threshold": 0.7,
    "max_retrieved_docs": 5
}

# FAISS Settings  
FAISS_CONFIG = {
    "index_type": "IndexFlatIP",
    "index_path": "data/vectorstore/faiss_index.bin",
    "metadata_path": "data/vectorstore/metadata.json",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "vector_dimension": 384
}

# Response Settings
RESPONSE_CONFIG = {
    "max_response_length": 500,
    "include_sources": True,
    "fallback_response": "Please contact LendenClub support for assistance."
}
