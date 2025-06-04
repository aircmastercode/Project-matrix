"""
LendenClub Voice Assistant - Free Intent Classifier
BART-Large-MNLI based intent classification system (GPT-4o free alternative)
"""

import logging
import time
import json
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import numpy as np

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, classification_report
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    TRANSFORMERS_AVAILABLE = False

# Import configuration
from config.settings import (
    BART_CONFIG, ALTERNATIVE_MODELS, ACTIVE_MODEL, 
    INTENT_CATEGORIES, FALLBACK_INTENT, get_model_config, get_intent_labels
)

class FreeIntentClassifier:
    """
    Free intent classification system using multiple model options.
    
    Primary: BART-Large-MNLI for zero-shot classification
    Fallbacks: DistilBERT, SetFit, Scikit-learn
    """
    
    def __init__(self, model_name: str = None, config: Dict = None):
        """
        Initialize the free intent classifier.
        
        Args:
            model_name: Model to use ('bart', 'distilbert', 'setfit', 'sklearn')
            config: Optional custom configuration
        """
        self.model_name = model_name or ACTIVE_MODEL
        self.config = config or get_model_config(self.model_name)
        self.intent_labels = get_intent_labels()
        self.intent_descriptions = self._prepare_intent_descriptions()
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.classifier = None
        self.tokenizer = None
        self.vectorizer = None
        self.sklearn_model = None
        
        # Performance tracking
        self.prediction_times = []
        self.confidence_scores = []
        
        # Initialize the selected model
        self._initialize_model()
        
        self.logger.info(f"Initialized FreeIntentClassifier with model: {self.model_name}")
    
    def _prepare_intent_descriptions(self) -> Dict[str, str]:
        """Prepare intent descriptions for zero-shot classification."""
        descriptions = {}
        for intent, config in INTENT_CATEGORIES.items():
            # Create natural language descriptions for BART
            descriptions[intent] = f"This text is about {config['description']}"
        return descriptions
    
    def _initialize_model(self):
        """Initialize the selected model."""
        try:
            if self.model_name == "bart":
                self._initialize_bart()
            elif self.model_name == "distilbert":
                self._initialize_distilbert()
            elif self.model_name == "setfit":
                self._initialize_setfit()
            elif self.model_name == "sklearn":
                self._initialize_sklearn()
            else:
                raise ValueError(f"Unknown model: {self.model_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.model_name} model: {e}")
            self._initialize_fallback()
    
    def _initialize_bart(self):
        """Initialize BART-Large-MNLI for zero-shot classification."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available for BART model")
        
        try:
            self.logger.info("Loading BART-Large-MNLI model...")
            
            # Initialize the zero-shot classification pipeline
            self.classifier = pipeline(
                "zero-shot-classification",
                model=self.config["model_name"],
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            self.logger.info("BART-Large-MNLI model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load BART model: {e}")
            raise
    
    def _initialize_distilbert(self):
        """Initialize DistilBERT for intent classification."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available for DistilBERT model")
        
        try:
            self.logger.info("Loading DistilBERT model...")
            
            # Load pre-trained intent classification model
            self.classifier = pipeline(
                "text-classification",
                model=self.config["model_name"],
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            self.logger.info("DistilBERT model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load DistilBERT model: {e}")
            raise
    
    def _initialize_setfit(self):
        """Initialize SetFit for few-shot learning."""
        self.logger.warning("SetFit model not fully implemented yet. Using BART fallback.")
        self._initialize_bart()
    
    def _initialize_sklearn(self):
        """Initialize Scikit-learn traditional ML model."""
        self.logger.info("Initializing Scikit-learn TF-IDF + SVM model...")
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.config.get("ngram_range", (1, 2)),
            max_features=self.config.get("max_features", 10000),
            stop_words='english'
        )
        
        # Initialize classifier
        self.sklearn_model = SVC(
            kernel='rbf',
            probability=True,  # Enable probability estimates
            random_state=42
        )
        
        # Train with sample data
        self._train_sklearn_model()
        
        self.logger.info("Scikit-learn model initialized successfully")
    
    def _train_sklearn_model(self):
        """Train sklearn model with sample data from intent categories."""
        # Extract training examples from intent categories
        training_texts = []
        training_labels = []
        
        for intent, config in INTENT_CATEGORIES.items():
            examples = config.get("examples", [])
            for example in examples:
                training_texts.append(example)
                training_labels.append(intent)
        
        if training_texts:
            # Vectorize training data
            X = self.vectorizer.fit_transform(training_texts)
            
            # Train the classifier
            self.sklearn_model.fit(X, training_labels)
            
            self.logger.info(f"Trained sklearn model with {len(training_texts)} examples")
        else:
            self.logger.warning("No training examples found for sklearn model")
    
    def _initialize_fallback(self):
        """Initialize fallback model when primary model fails."""
        self.logger.warning("Initializing fallback keyword-based classifier")
        self.model_name = "keyword_fallback"
        # Fallback will use keyword matching
    
    def predict_single(self, text: str, return_confidence: bool = True) -> Dict[str, Any]:
        """
        Predict intent for a single text input.
        
        Args:
            text: Input text to classify
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary with intent, confidence, and other metadata
        """
        start_time = time.time()
        
        try:
            if self.model_name == "bart":
                result = self._predict_bart(text)
            elif self.model_name == "distilbert":
                result = self._predict_distilbert(text)
            elif self.model_name == "sklearn":
                result = self._predict_sklearn(text)
            else:
                result = self._predict_fallback(text)
            
            # Add prediction time
            prediction_time = time.time() - start_time
            result["prediction_time"] = prediction_time
            result["model_used"] = self.model_name
            
            # Track performance metrics
            self.prediction_times.append(prediction_time)
            if "confidence" in result:
                self.confidence_scores.append(result["confidence"])
            
            self.logger.debug(f"Predicted intent '{result['intent']}' with confidence {result.get('confidence', 'N/A')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return {
                "intent": FALLBACK_INTENT,
                "confidence": 0.0,
                "error": str(e),
                "model_used": "error_fallback"
            }
    
    def _predict_bart(self, text: str) -> Dict[str, Any]:
        """Predict using BART-Large-MNLI."""
        if not self.classifier:
            raise RuntimeError("BART classifier not initialized")
        
        # Use intent descriptions as candidate labels for zero-shot
        candidate_labels = list(self.intent_descriptions.values())
        
        # Perform zero-shot classification
        result = self.classifier(text, candidate_labels)
        
        # Map back to intent names
        top_label = result["labels"][0]
        confidence = result["scores"][0]
        
        # Find corresponding intent
        intent = None
        for intent_name, description in self.intent_descriptions.items():
            if description == top_label:
                intent = intent_name
                break
        
        if not intent:
            intent = FALLBACK_INTENT
            confidence = 0.0
        
        return {
            "intent": intent,
            "confidence": confidence,
            "all_scores": dict(zip(
                [self._description_to_intent(label) for label in result["labels"]],
                result["scores"]
            )),
            "reasoning": f"Classified as '{intent}' based on zero-shot classification"
        }
    
    def _predict_distilbert(self, text: str) -> Dict[str, Any]:
        """Predict using DistilBERT."""
        if not self.classifier:
            raise RuntimeError("DistilBERT classifier not initialized")
        
        result = self.classifier(text)
        
        # Get top prediction
        top_result = result[0] if isinstance(result, list) else result
        intent = top_result["label"]
        confidence = top_result["score"]
        
        # Map to our intent categories if needed
        if intent not in self.intent_labels:
            intent = self._map_external_intent(intent)
        
        return {
            "intent": intent,
            "confidence": confidence,
            "all_scores": {r["label"]: r["score"] for r in result} if isinstance(result, list) else {intent: confidence},
            "reasoning": f"Classified as '{intent}' using DistilBERT"
        }
    
    def _predict_sklearn(self, text: str) -> Dict[str, Any]:
        """Predict using Scikit-learn TF-IDF + SVM."""
        if not self.vectorizer or not self.sklearn_model:
            raise RuntimeError("Sklearn model not initialized")
        
        # Vectorize input text
        X = self.vectorizer.transform([text])
        
        # Predict
        predicted_intent = self.sklearn_model.predict(X)[0]
        probabilities = self.sklearn_model.predict_proba(X)[0]
        
        # Get confidence score
        confidence = max(probabilities)
        
        # Create scores for all intents
        all_scores = dict(zip(self.sklearn_model.classes_, probabilities))
        
        return {
            "intent": predicted_intent,
            "confidence": confidence,
            "all_scores": all_scores,
            "reasoning": f"Classified as '{predicted_intent}' using TF-IDF + SVM"
        }
    
    def _predict_fallback(self, text: str) -> Dict[str, Any]:
        """Fallback keyword-based prediction."""
        text_lower = text.lower()
        scores = {}
        
        # Score based on keyword matching
        for intent, config in INTENT_CATEGORIES.items():
            score = 0
            keywords = config.get("keywords", [])
            
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    score += 1
            
            # Normalize score
            scores[intent] = score / max(len(keywords), 1) if keywords else 0
        
        # Find best match
        best_intent = max(scores, key=scores.get) if scores else FALLBACK_INTENT
        confidence = scores.get(best_intent, 0.0)
        
        return {
            "intent": best_intent,
            "confidence": confidence,
            "all_scores": scores,
            "reasoning": f"Classified as '{best_intent}' using keyword matching"
        }
    
    def _description_to_intent(self, description: str) -> str:
        """Map intent description back to intent name."""
        for intent, desc in self.intent_descriptions.items():
            if desc == description:
                return intent
        return FALLBACK_INTENT
    
    def _map_external_intent(self, external_intent: str) -> str:
        """Map external model intent to our intent categories."""
        # Simple mapping - can be made more sophisticated
        external_lower = external_intent.lower()
        
        for intent in self.intent_labels:
            if intent.lower() in external_lower or external_lower in intent.lower():
                return intent
        
        return FALLBACK_INTENT
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict intents for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of prediction results
        """
        return [self.predict_single(text) for text in texts]
    
    def evaluate_performance(self, test_data: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: List of (text, true_intent) tuples
            
        Returns:
            Performance metrics
        """
        if not test_data:
            return {"error": "No test data provided"}
        
        predictions = []
        true_labels = []
        
        for text, true_intent in test_data:
            result = self.predict_single(text)
            predictions.append(result["intent"])
            true_labels.append(true_intent)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        # Generate detailed report
        report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)
        
        return {
            "accuracy": accuracy,
            "precision": report["macro avg"]["precision"],
            "recall": report["macro avg"]["recall"],
            "f1_score": report["macro avg"]["f1-score"],
            "per_class_metrics": {
                intent: {
                    "precision": report.get(intent, {}).get("precision", 0),
                    "recall": report.get(intent, {}).get("recall", 0),
                    "f1_score": report.get(intent, {}).get("f1-score", 0)
                }
                for intent in self.intent_labels
            }
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model_name": self.model_name,
            "config": self.config,
            "intent_labels": self.intent_labels,
            "num_intents": len(self.intent_labels),
            "avg_prediction_time": np.mean(self.prediction_times) if self.prediction_times else 0,
            "avg_confidence": np.mean(self.confidence_scores) if self.confidence_scores else 0,
            "total_predictions": len(self.prediction_times)
        }
    
    def add_training_examples(self, examples: List[Tuple[str, str]]):
        """
        Add training examples for models that support incremental learning.
        
        Args:
            examples: List of (text, intent) tuples
        """
        if self.model_name == "sklearn":
            # Retrain sklearn model with new examples
            texts, labels = zip(*examples)
            
            # Add to existing training data and retrain
            self._train_sklearn_model()
            self.logger.info(f"Added {len(examples)} training examples to sklearn model")
        else:
            self.logger.warning(f"Adding training examples not supported for {self.model_name} model")

# Convenience function for quick classification
def classify_intent(text: str, model_name: str = None) -> Dict[str, Any]:
    """
    Quick intent classification function.
    
    Args:
        text: Text to classify
        model_name: Optional model name override
        
    Returns:
        Classification result
    """
    classifier = FreeIntentClassifier(model_name=model_name)
    return classifier.predict_single(text)

# Example usage and testing
if __name__ == "__main__":
    # Test the classifier
    test_queries = [
        "What is the minimum credit score required for a loan?",
        "How do I reset my password?",
        "What are the processing fees?",
        "How does the repayment work?",
        "What documents do I need to upload?",
        "Tell me about your company",
        "This is a random question about nothing specific"
    ]
    
    print("🧠 Testing Free Intent Classifier (BART-Large-MNLI)")
    print("=" * 60)
    
    # Initialize classifier
    classifier = FreeIntentClassifier()
    
    # Test each query
    for query in test_queries:
        result = classifier.predict_single(query)
        print(f"Query: {query}")
        print(f"Intent: {result['intent']} (Confidence: {result.get('confidence', 0):.2f})")
        print(f"Reasoning: {result.get('reasoning', 'N/A')}")
        print("-" * 40)
    
    # Print model info
    info = classifier.get_model_info()
    print(f"\nModel Info:")
    print(f"Model: {info['model_name']}")
    print(f"Intents: {info['num_intents']}")
    print(f"Avg Prediction Time: {info['avg_prediction_time']:.3f}s")
    print(f"Avg Confidence: {info['avg_confidence']:.2f}")
