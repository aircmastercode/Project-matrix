# Text-based Response Generator for LendenClub RAG System
# Generates contextual responses using retrieved documents and intent classification

import json
import logging
from typing import List, Dict, Optional
from pathlib import Path
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextResponseGenerator:
    """
    Generates contextual text responses for LendenClub queries
    Integrates with FAISS vector search and BART intent classification
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.templates = {}
        self.load_response_templates()
        
        logger.info("Initialized Text Response Generator")
    
    def load_response_templates(self):
        """Load response templates for different intent categories"""
        # Default templates for each intent
        self.templates = {
            "loan_eligibility": """
Based on LendenClub's requirements: {context}

Key eligibility criteria:
• {criteria}

For specific requirements related to your situation, I recommend contacting LendenClub support.
            """,
            
            "repayment_terms": """
Regarding repayment: {context}

Important details:
• {details}

Please note that terms may vary based on your specific loan agreement.
            """,
            
            "interest_rates": """
About interest rates: {context}

Rate information:
• {rate_info}

Interest rates are determined based on individual risk assessment and market conditions.
            """,
            
            "documentation": """
Required documentation: {context}

Documents needed:
• {doc_list}

Please ensure all documents are valid and clearly legible when submitting.
            """,
            
            "account_management": """
For account-related queries: {context}

Steps to follow:
• {steps}

If you continue to face issues, please contact customer support.
            """,
            
            "fees_charges": """
Regarding fees and charges: {context}

Fee structure:
• {fee_details}

All fees are as per the terms and conditions agreed at the time of loan sanction.
            """,
            
            "investment_process": """
About investment with LendenClub: {context}

Investment information:
• {investment_info}

Please review the risk factors before making any investment decisions.
            """,
            
            "general_inquiry": """
Here's what I found: {context}

Additional information:
• {additional_info}

For more detailed information, please visit our support section or contact us directly.
            """
        }
    
    def extract_key_points(self, documents: List[Dict], intent: str) -> Dict[str, str]:
        """Extract key points from retrieved documents based on intent"""
        if not documents:
            return {"context": "No specific information found.", "details": "Please contact support."}
        
        # Combine all document text
        full_context = " ".join([doc['document'] for doc in documents])
        
        # Intent-specific extraction
        if intent == "loan_eligibility":
            criteria = self._extract_criteria(full_context)
            return {
                "context": self._truncate_text(full_context, 200),
                "criteria": criteria or "Contact support for specific criteria"
            }
            
        elif intent == "repayment_terms":
            details = self._extract_payment_details(full_context)
            return {
                "context": self._truncate_text(full_context, 200),
                "details": details or "Refer to your loan agreement"
            }
            
        elif intent == "interest_rates":
            rate_info = self._extract_rate_info(full_context)
            return {
                "context": self._truncate_text(full_context, 200),
                "rate_info": rate_info or "Rates vary based on assessment"
            }
            
        elif intent == "documentation":
            doc_list = self._extract_document_list(full_context)
            return {
                "context": self._truncate_text(full_context, 200),
                "doc_list": doc_list or "Standard KYC documents required"
            }
            
        elif intent == "account_management":
            steps = self._extract_steps(full_context)
            return {
                "context": self._truncate_text(full_context, 200),
                "steps": steps or "Follow standard account procedures"
            }
            
        elif intent == "fees_charges":
            fee_details = self._extract_fee_details(full_context)
            return {
                "context": self._truncate_text(full_context, 200),
                "fee_details": fee_details or "Fees as per terms and conditions"
            }
            
        elif intent == "investment_process":
            investment_info = self._extract_investment_info(full_context)
            return {
                "context": self._truncate_text(full_context, 200),
                "investment_info": investment_info or "Review investment guidelines"
            }
            
        else:  # general_inquiry
            additional_info = self._extract_general_info(full_context)
            return {
                "context": self._truncate_text(full_context, 200),
                "additional_info": additional_info or "Additional details available on request"
            }
    
    def _extract_criteria(self, text: str) -> str:
        """Extract eligibility criteria from text"""
        patterns = [
            r"minimum.*salary.*?(\d+[,\d]*)",
            r"eligib.*?requirement.*?([^.]+)",
            r"criteria.*?([^.]+)",
            r"qualify.*?([^.]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""
    
    def _extract_payment_details(self, text: str) -> str:
        """Extract payment-related details"""
        patterns = [
            r"EMI.*?([^.]+)",
            r"payment.*?([^.]+)",
            r"repay.*?([^.]+)",
            r"due.*?([^.]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""
    
    def _extract_rate_info(self, text: str) -> str:
        """Extract interest rate information"""
        patterns = [
            r"(\d+%?\s*(?:to|-)?\s*\d+%?).*?per\s*annum",
            r"interest.*?rate.*?(\d+[^.]+)",
            r"APR.*?(\d+[^.]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""
    
    def _extract_document_list(self, text: str) -> str:
        """Extract required documents"""
        documents = []
        doc_patterns = [
            r"salary\s*slip", r"bank\s*statement", r"PAN\s*card",
            r"Aadhar", r"address\s*proof", r"income\s*proof",
            r"ITR", r"form\s*16", r"employment\s*letter"
        ]
        
        for pattern in doc_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                documents.append(pattern.replace(r"\s*", " ").replace("\\", ""))
        
        return ", ".join(documents) if documents else ""
    
    def _extract_steps(self, text: str) -> str:
        """Extract procedural steps"""
        steps = re.findall(r"(?:step\s*\d+|first|second|then|next|finally).*?([^.]+)", text, re.IGNORECASE)
        return "; ".join(steps[:3]) if steps else ""
    
    def _extract_fee_details(self, text: str) -> str:
        """Extract fee information"""
        patterns = [
            r"(\d+\.?\d*%?).*?fee",
            r"charges.*?(\d+[^.]+)",
            r"processing.*?fee.*?(\d+[^.]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""
    
    def _extract_investment_info(self, text: str) -> str:
        """Extract investment-related information"""
        patterns = [
            r"return.*?(\d+[^.]+)",
            r"FMPP.*?([^.]+)",
            r"lend.*?([^.]+)",
            r"portfolio.*?([^.]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""
    
    def _extract_general_info(self, text: str) -> str:
        """Extract general information"""
        # Return first meaningful sentence
        sentences = text.split('.')
        for sentence in sentences:
            if len(sentence.strip()) > 20:
                return sentence.strip()
        return ""
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to specified length"""
        if len(text) <= max_length:
            return text
        
        # Find the last complete sentence within the limit
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        
        if last_period > max_length * 0.7:  # If we have a good sentence boundary
            return truncated[:last_period + 1]
        else:
            return truncated + "..."
    
    def generate_response(
        self, 
        query: str, 
        intent: str, 
        documents: List[Dict], 
        confidence: float
    ) -> Dict[str, any]:
        """Generate a complete response for the user query"""
        
        # Handle low confidence predictions
        if confidence < 0.6:
            return {
                "response": self.config.get("fallback_response", 
                    "I couldn't fully understand your query. Could you please rephrase or provide more details?"),
                "confidence": confidence,
                "intent": "unclear",
                "sources": [],
                "suggestions": [
                    "Try asking about loan eligibility requirements",
                    "Ask about repayment terms or EMI details", 
                    "Inquire about required documentation",
                    "Contact LendenClub support directly"
                ]
            }
        
        # Handle case with no retrieved documents
        if not documents:
            fallback_info = {
                "loan_eligibility": "For loan eligibility, you typically need: minimum salary, good credit score, required documents, and age criteria.",
                "repayment_terms": "Repayment terms include EMI amount, tenure, due dates, and prepayment options.",
                "interest_rates": "Interest rates depend on your profile, loan amount, tenure, and risk assessment.",
                "documentation": "Common documents include salary slips, bank statements, PAN card, and address proof.",
                "account_management": "For account issues, try logging in again, check your credentials, or contact support.",
                "fees_charges": "Fees typically include processing fee, GST, and any applicable charges as per terms.",
                "investment_process": "Investment process involves registration, KYC, choosing plans, and monitoring returns.",
                "general_inquiry": "For general queries, please check our FAQ section or contact customer support."
            }
            
            response_text = fallback_info.get(intent, self.config.get("fallback_response"))
            
            return {
                "response": response_text,
                "confidence": confidence,
                "intent": intent,
                "sources": [],
                "note": "This is general information. For specific details, please contact LendenClub support."
            }
        
        # Generate response with retrieved documents
        key_points = self.extract_key_points(documents, intent)
        template = self.templates.get(intent, self.templates["general_inquiry"])
        
        try:
            response_text = template.format(**key_points)
        except KeyError as e:
            logger.warning(f"Template formatting error: {e}")
            response_text = f"Based on the available information: {key_points.get('context', 'Please contact support.')}"
        
        # Clean up response
        response_text = re.sub(r'\n\s*\n', '\n\n', response_text)  # Clean multiple newlines
        response_text = response_text.strip()
        
        # Add sources information
        sources = [
            {
                "content": doc['document'][:100] + "..." if len(doc['document']) > 100 else doc['document'],
                "relevance_score": doc['score'],
                "category": doc['metadata'].get('category', 'unknown')
            }
            for doc in documents[:3]  # Top 3 sources
        ]
        
        return {
            "response": response_text,
            "confidence": confidence,
            "intent": intent,
            "sources": sources,
            "query": query
        }

# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    test_config = {
        "max_response_length": 500,
        "include_sources": True,
        "fallback_response": "Please contact LendenClub support for assistance."
    }
    
    # Initialize generator
    response_gen = TextResponseGenerator(test_config)
    
    # Test documents
    test_docs = [
        {
            "document": "LendenClub requires minimum salary of Rs. 25,000 for loan eligibility",
            "score": 0.95,
            "metadata": {"category": "loan_eligibility"}
        },
        {
            "document": "Processing fee is 2.5% of loan amount with minimum Rs. 1,000",
            "score": 0.87,
            "metadata": {"category": "fees_charges"}
        }
    ]
    
    # Test response generation
    result = response_gen.generate_response(
        query="What salary is needed for loan?",
        intent="loan_eligibility",
        documents=test_docs,
        confidence=0.92
    )
    
    print("Generated Response:")
    print("=" * 50)
    print(result["response"])
    print("\n" + "=" * 50)
    print(f"Intent: {result['intent']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Sources: {len(result['sources'])}")
    print("\n✅ Text Response Generator test completed!")
