#!/usr/bin/env python3
"""Quick test of Phase 1 classifier"""

from src.intent_classification.models.free_intent_classifier import FreeIntentClassifier

# Test queries
queries = [
    "What is the minimum credit score?",
    "how the lenden club works and what are its policies also show me the emi process", 
    "What are the fees?",
    "EMI calculation method",
    "Document upload process"
]

print("🧠 Testing BART-Large-MNLI Classifier")
print("=" * 40)

classifier = FreeIntentClassifier()

for query in queries:
    result = classifier.predict_single(query)
    print(f"'{query}' → {result['intent']} ({result.get('confidence', 0):.2f})")

print("\n✅ Quick test completed!")
