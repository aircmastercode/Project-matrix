# Fixed Free Intent Classifier - Minimal Version
import logging
from transformers import pipeline

logger = logging.getLogger(__name__)

class FreeIntentClassifier:
    def __init__(self, model_type="bart_large_mnli"):
        logger.info(f"Loading intent classifier: {model_type}")
        
        # Map model types to correct identifiers
        model_mapping = {
            "bart_zero_shot": "facebook/bart-large-mnli",
            "bart_large_mnli": "facebook/bart-large-mnli"
        }
        
        model_name = model_mapping.get(model_type, "facebook/bart-large-mnli")
        
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name
        )
        
        self.intent_labels = [
            "loan eligibility", "repayment terms", "interest rates",
            "documentation", "account management", "fees and charges",
            "investment process", "general inquiry"
        ]
        
        logger.info("✅ Intent classifier loaded successfully!")
    
    def predict_single(self, query):
        try:
            result = self.classifier(query, self.intent_labels)
            
            intent_mapping = {
                "loan eligibility": "loan_eligibility",
                "repayment terms": "repayment_terms", 
                "interest rates": "interest_rates",
                "documentation": "documentation",
                "account management": "account_management",
                "fees and charges": "fees_charges",
                "investment process": "investment_process",
                "general inquiry": "general_inquiry"
            }
            
            top_intent = result['labels'][0]
            mapped_intent = intent_mapping.get(top_intent, "general_inquiry")
            
            return {
                "intent": mapped_intent,
                "confidence": float(result['scores'][0])
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {"intent": "general_inquiry", "confidence": 0.0}
