# Fixed Main Pipeline - Minimal Version
import sys
import logging

sys.path.append("src")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Test the fixed pipeline"""
    try:
        logger.info("Testing fixed intent classifier...")
        
        # Test the fixed classifier
        from intent_classification.models.free_intent_classifier import FreeIntentClassifier
        classifier = FreeIntentClassifier("bart_large_mnli")  # Fixed identifier
        
        # Test prediction
        result = classifier.predict_single("What documents do I need?")
        
        print("\n✅ Fix successful!")
        print(f"Test query: 'What documents do I need?'")
        print(f"Intent: {result['intent']}")
        print(f"Confidence: {result['confidence']:.3f}")
        
        print("\n🤖 LendenClub Assistant is now working!")
        print("You can now run the full pipeline.")
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    main()
