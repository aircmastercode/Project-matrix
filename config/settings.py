"""
LendenClub Voice Assistant - Configuration Settings
Central configuration for all phases with focus on free models and Phase 1 setup.
"""

import os
from pathlib import Path
from typing import Dict, List, Any

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# =============================================================================
# PHASE 1: INTENT CLASSIFICATION SETTINGS (FREE MODELS)
# =============================================================================

# BART-Large-MNLI Configuration (Recommended Primary Model)
BART_CONFIG = {
    "model_name": "facebook/bart-large-mnli",
    "task": "zero-shot-classification",
    "device": "auto",  # Automatically detect GPU/CPU
    "max_length": 512,
    "confidence_threshold": 0.7,
    "multi_label": False,
    "cache_dir": str(DATA_DIR / "models" / "bart"),
}

# Alternative Free Models Configuration
ALTERNATIVE_MODELS = {
    "distilbert": {
        "model_name": "Falconsai/intent_classification",
        "task": "text-classification", 
        "confidence_threshold": 0.8,
        "cache_dir": str(DATA_DIR / "models" / "distilbert"),
    },
    "setfit": {
        "model_name": "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "task": "few-shot-learning",
        "confidence_threshold": 0.75,
        "cache_dir": str(DATA_DIR / "models" / "setfit"),
        "few_shot_examples": 8,  # Examples per intent
    },
    "sklearn": {
        "vectorizer": "tfidf",
        "classifier": "svm",
        "confidence_threshold": 0.6,
        "ngram_range": (1, 2),
        "max_features": 10000,
    }
}

# Current active model (change this to switch models)
ACTIVE_MODEL = "bart"  # Options: "bart", "distilbert", "setfit", "sklearn"

# =============================================================================
# INTENT CATEGORIES DEFINITION
# =============================================================================

# Core financial intents based on LendenClub domain
INTENT_CATEGORIES = {
    # Primary Business Intents
    "loan_eligibility": {
        "description": "Questions about loan qualification criteria, credit scores, income requirements",
        "examples": [
            "What is the minimum credit score required?",
            "Do I qualify for a loan?",
            "What documents are needed for eligibility?"
        ],
        "keywords": ["eligibility", "qualify", "credit score", "minimum", "requirements"]
    },
    
    "repayment_terms": {
        "description": "Queries about payment schedules, EMI, interest rates, tenure",
        "examples": [
            "What are the repayment options?",
            "How do I change my payment date?",
            "What is the EMI amount?"
        ],
        "keywords": ["repayment", "EMI", "payment", "schedule", "tenure", "installment"]
    },
    
    "interest_rates": {
        "description": "Questions about APR, interest calculations, rate changes",
        "examples": [
            "What is the current interest rate?",
            "How is interest calculated?",
            "Will my rate change?"
        ],
        "keywords": ["interest", "rate", "APR", "percentage", "calculation"]
    },
    
    "documentation": {
        "description": "Requests for required paperwork, verification documents",
        "examples": [
            "What documents do I need?",
            "How to upload my PAN card?",
            "Document verification process"
        ],
        "keywords": ["documents", "paperwork", "upload", "verification", "PAN", "Aadhaar"]
    },
    
    "account_management": {
        "description": "Profile updates, password resets, account access issues",
        "examples": [
            "How to reset my password?",
            "Update my phone number",
            "Cannot access my account"
        ],
        "keywords": ["password", "profile", "account", "update", "access", "login"]
    },
    
    "fees_charges": {
        "description": "Questions about processing fees, charges, penalties",
        "examples": [
            "What is the processing fee?",
            "Are there any hidden charges?",
            "Late payment penalty"
        ],
        "keywords": ["fee", "charges", "penalty", "cost", "processing", "hidden"]
    },
    
    "investment_process": {
        "description": "Lumpsum lending, manual lending, portfolio management",
        "examples": [
            "How does lumpsum lending work?",
            "Manual lending process",
            "Portfolio management"
        ],
        "keywords": ["lumpsum", "manual", "lending", "investment", "portfolio"]
    },
    
    "general_inquiry": {
        "description": "Basic platform questions, company information, general help",
        "examples": [
            "How does LendenClub work?",
            "Company information",
            "General help"
        ],
        "keywords": ["help", "information", "how", "what", "general", "about"]
    }
}

# Default fallback intent for unclear queries
FALLBACK_INTENT = "general_inquiry"

# =============================================================================
# DATA INGESTION SETTINGS (WEB SCRAPING)
# =============================================================================

# Target URLs for data collection
SCRAPING_URLS = [
    {
        "url": "https://www.lendenclub.com/terms-of-services/",
        "priority": "high",
        "retry_count": 5,
        "content_type": "terms_and_conditions"
    },
    {
        "url": "https://support.lendenclub.com/support/solutions",
        "priority": "high", 
        "retry_count": 3,
        "content_type": "faq_support"
    },
    {
        "url": "https://www.lendenclub.com/faq/",
        "priority": "medium",
        "retry_count": 3,
        "content_type": "general_faq"
    }
]

# Anti-detection settings for web scraping
SCRAPING_CONFIG = {
    "delay_range": (1, 3),  # Random delay between requests (seconds)
    "max_retries": 5,
    "timeout": 30,
    "user_agent_rotation": True,
    "headers": {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Cache-Control": "max-age=0",
    },
    "proxies": None,  # Add proxy rotation if needed
    "verify_ssl": True
}

# Content selectors for different pages
CONTENT_SELECTORS = {
    "lendenclub_terms": [
        "main",
        ".content",
        ".article-content", 
        ".terms-content",
        "article"
    ],
    "support_solutions": [
        "main",
        ".article-content",
        ".kb-article",
        ".solution-content"
    ],
    "general_faq": [
        ".faq-content",
        ".question-answer",
        ".accordion-content"
    ]
}

# =============================================================================
# FILE PATHS AND STORAGE
# =============================================================================

# Data storage paths
DATA_PATHS = {
    "raw_data": DATA_DIR / "raw",
    "processed_data": DATA_DIR / "processed",
    "models": DATA_DIR / "models", 
    "embeddings": DATA_DIR / "embeddings",
    "feedback": DATA_DIR / "feedback",
    "test_data": PROJECT_ROOT / "tests" / "data"
}

# Ensure all data directories exist
for path in DATA_PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

# File naming patterns
FILE_PATTERNS = {
    "scraped_data": "scraped_{source}_{timestamp}.json",
    "processed_data": "processed_{source}_{timestamp}.json",
    "model_cache": "{model_name}_{version}.pkl",
    "embeddings": "embeddings_{source}_{timestamp}.npy",
    "logs": "voice_assistant_{date}.log"
}

# =============================================================================
# PERFORMANCE AND EVALUATION SETTINGS
# =============================================================================

# Evaluation metrics configuration
EVALUATION_CONFIG = {
    "test_split_ratio": 0.2,
    "cross_validation_folds": 5,
    "metrics": ["accuracy", "precision", "recall", "f1_score"],
    "confidence_buckets": [0.0, 0.5, 0.7, 0.8, 0.9, 1.0],
    "performance_thresholds": {
        "accuracy": 0.85,
        "precision": 0.80,
        "recall": 0.80,
        "f1_score": 0.80
    },
    "benchmark_queries_per_intent": 10
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s"
        }
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "level": "DEBUG",
            "formatter": "detailed",
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR / "voice_assistant.log"),
            "mode": "a"
        }
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "DEBUG",
            "propagate": False
        }
    }
}

# =============================================================================
# ENVIRONMENT VARIABLES (WITH DEFAULTS)
# =============================================================================

def get_env_var(key: str, default: Any = None) -> Any:
    """Get environment variable with default fallback."""
    return os.getenv(key, default)

# Model configuration from environment
BART_MODEL_NAME = get_env_var("BART_MODEL_NAME", BART_CONFIG["model_name"])
MODEL_CACHE_DIR = get_env_var("MODEL_CACHE_DIR", BART_CONFIG["cache_dir"])
CONFIDENCE_THRESHOLD = float(get_env_var("CONFIDENCE_THRESHOLD", BART_CONFIG["confidence_threshold"]))

# Data paths from environment
RAW_DATA_PATH = Path(get_env_var("RAW_DATA_PATH", DATA_PATHS["raw_data"]))
PROCESSED_DATA_PATH = Path(get_env_var("PROCESSED_DATA_PATH", DATA_PATHS["processed_data"]))

# Scraping configuration from environment
SCRAPING_DELAY = float(get_env_var("SCRAPING_DELAY", 1.0))
MAX_RETRIES = int(get_env_var("MAX_RETRIES", 3))
USER_AGENT_ROTATION = get_env_var("USER_AGENT_ROTATION", "true").lower() == "true"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_model_config(model_name: str = None) -> Dict[str, Any]:
    """Get configuration for specified model or active model."""
    model_name = model_name or ACTIVE_MODEL
    
    if model_name == "bart":
        return BART_CONFIG
    elif model_name in ALTERNATIVE_MODELS:
        return ALTERNATIVE_MODELS[model_name]
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_intent_labels() -> List[str]:
    """Get list of all intent category labels."""
    return list(INTENT_CATEGORIES.keys())

def get_intent_descriptions() -> Dict[str, str]:
    """Get mapping of intent labels to descriptions."""
    return {intent: config["description"] for intent, config in INTENT_CATEGORIES.items()}

def validate_config() -> bool:
    """Validate configuration settings."""
    try:
        # Check if required directories exist
        for path in DATA_PATHS.values():
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
        
        # Validate model configuration
        model_config = get_model_config()
        if not model_config:
            raise ValueError("Invalid model configuration")
        
        # Validate intent categories
        if not INTENT_CATEGORIES:
            raise ValueError("No intent categories defined")
        
        return True
    
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

# Validate configuration on import
if __name__ == "__main__":
    if validate_config():
        print("✅ Configuration validation successful")
        print(f"Active model: {ACTIVE_MODEL}")
        print(f"Intent categories: {len(INTENT_CATEGORIES)}")
        print(f"Data directory: {DATA_DIR}")
    else:
        print("❌ Configuration validation failed")
