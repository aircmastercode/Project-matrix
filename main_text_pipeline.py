# LendenClub Voice Assistant - Fixed FastAPI Backend
import sys
import logging
from typing import Optional
import uvicorn
from datetime import datetime

# Add src to path for imports
sys.path.append("src")

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response Models
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    intent: str
    confidence: float
    response: str
    timestamp: str
    session_id: str

class HealthResponse(BaseModel):
    status: str
    classifier_loaded: bool
    timestamp: str
    version: str

class ClassifyRequest(BaseModel):
    query: str

class ClassifyResponse(BaseModel):
    intent: str
    confidence: float
    timestamp: str

class FeedbackRequest(BaseModel):
    message_id: Optional[str] = None
    rating: int  # 1-5 scale
    comment: Optional[str] = None
    session_id: Optional[str] = "default"

# Create FastAPI app
app = FastAPI(
    title="LendenClub Voice Assistant API",
    description="Backend API for LendenClub Voice Assistant with BART-Large-MNLI intent classification",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
classifier = None
classifier_loaded = False

# Response templates based on intent
RESPONSE_TEMPLATES = {
    "loan_eligibility": "For personal loans, you typically need a minimum monthly income of ₹25,000 and a credit score above 650. We also consider your employment history and debt-to-income ratio.",
    "documentation": "For loan applications, you'll need: 1) Valid ID proof (Aadhaar/PAN), 2) Income proof (salary slips/ITR), 3) Bank statements (3-6 months), 4) Employment verification, and 5) Address proof.",
    "interest_rates": "Our current personal loan interest rates range from 10.99% to 24% per annum, depending on your credit profile and loan amount. Rates are personalized based on your creditworthiness.",
    "account_management": "You can manage your account through our web portal or mobile app. This includes updating personal information, viewing loan status, downloading statements, and making payments.",
    "fees_charges": "We charge a processing fee of 1-3% of loan amount (minimum ₹999). There's no prepayment penalty after 6 months, and late payment charges are ₹500 per occurrence.",
    "repayment_terms": "You can choose EMI tenure from 12 to 60 months. We offer flexible payment dates, and you can change your EMI date once per year through the portal.",
    "investment_process": "LendenClub offers peer-to-peer lending where you can invest in loan portfolios. Minimum investment is ₹5,000 with expected returns of 8-12% annually.",
    "general_inquiry": "I'm here to help with information about loans, documentation, interest rates, and account management. Please feel free to ask specific questions about our services."
}

@app.on_event("startup")
async def startup_event():
    """Initialize the classifier on startup"""
    global classifier, classifier_loaded
    try:
        logger.info("Initializing BART-Large-MNLI intent classifier...")
        from intent_classification.models.free_intent_classifier import FreeIntentClassifier
        classifier = FreeIntentClassifier("bart_large_mnli")
        classifier_loaded = True
        logger.info("✅ Intent classifier loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load classifier: {str(e)}")
        classifier_loaded = False

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "LendenClub Voice Assistant API",
        "version": "1.0.0",
        "status": "running",
        "classifier_status": "loaded" if classifier_loaded else "not loaded",
        "docs": "/docs"
    }

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if classifier_loaded else "degraded",
        classifier_loaded=classifier_loaded,
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/api/chat/message", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    """Main chat endpoint for processing user queries"""
    try:
        if not classifier_loaded:
            raise HTTPException(status_code=503, detail="Classifier not available")
        
        logger.info(f"Processing chat message: {request.query}")
        
        # Classify intent
        result = classifier.predict_single(request.query)
        intent = result['intent']
        confidence = result['confidence']
        
        # Generate response based on intent
        base_response = RESPONSE_TEMPLATES.get(intent, RESPONSE_TEMPLATES["general_inquiry"])
        
        # Add confidence-based messaging
        if confidence < 0.6:
            response = f"{base_response}\n\nI'm not entirely certain about your question. Could you please rephrase it for better assistance?"
        else:
            response = base_response
        
        # Log interaction
        logger.info(f"Intent: {intent}, Confidence: {confidence:.3f}")
        
        return ChatResponse(
            intent=intent,
            confidence=confidence,
            response=response,
            timestamp=datetime.now().isoformat(),
            session_id=request.session_id
        )
        
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/classify", response_model=ClassifyResponse)
async def classify_intent(request: ClassifyRequest):
    """Standalone intent classification endpoint"""
    try:
        if not classifier_loaded:
            raise HTTPException(status_code=503, detail="Classifier not available")
        
        result = classifier.predict_single(request.query)
        
        return ClassifyResponse(
            intent=result['intent'],
            confidence=result['confidence'],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error classifying intent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/intents/categories")
async def get_intent_categories():
    """Get available intent categories"""
    if not classifier_loaded:
        raise HTTPException(status_code=503, detail="Classifier not available")
    
    categories = {
        "loan_eligibility": "Questions about loan qualification criteria",
        "documentation": "Required documents and paperwork",
        "interest_rates": "Interest rates and pricing information",
        "account_management": "Account operations and management",
        "fees_charges": "Fees, charges, and costs",
        "repayment_terms": "EMI and repayment options",
        "investment_process": "Investment and lending processes",
        "general_inquiry": "General questions and information"
    }
    
    return {
        "categories": categories,
        "total_categories": len(categories),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
    """Submit user feedback"""
    try:
        # Process feedback in background
        background_tasks.add_task(process_feedback, request)
        
        return {
            "status": "received",
            "message": "Thank you for your feedback!",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")

async def process_feedback(feedback: FeedbackRequest):
    """Process user feedback in background"""
    logger.info(f"Processing feedback: Rating={feedback.rating}, Session={feedback.session_id}")
    # Add your feedback processing logic here
    # This could include storing to database, analytics, etc.

@app.get("/api/stats")
async def get_system_stats():
    """Get system statistics"""
    return {
        "classifier_loaded": classifier_loaded,
        "model_type": "BART-Large-MNLI",
        "available_intents": 8,
        "api_version": "1.0.0",
        "uptime": datetime.now().isoformat(),
        "status": "operational"
    }

def main():
    """Main function to run the server directly"""
    logger.info("Starting LendenClub Voice Assistant API server...")
    uvicorn.run(
        "main_text_pipeline:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
