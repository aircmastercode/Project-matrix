import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json; charset=utf-8',
    'Accept': 'application/json'
  },
  timeout: 15000 // Increased timeout for BART-Large-MNLI processing
});

// Request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
    console.log('📤 Request data:', config.data);
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => {
    console.log(`✅ API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('❌ API Error:', error.response?.data || error.message);
    
    if (error.response?.status === 422) {
      console.error('🔍 422 Error Details:', error.response.data);
      return Promise.reject(new Error('Invalid request format. Please check the data being sent.'));
    }
    
    if (error.response?.status === 503) {
      return Promise.reject(new Error('AI service is currently unavailable. Please try again later.'));
    }
    
    if (error.code === 'ECONNABORTED') {
      return Promise.reject(new Error('Request timeout. Please check your connection.'));
    }
    
    return Promise.reject(new Error(error.response?.data?.detail || 'An unexpected error occurred.'));
  }
);

export const chatApi = {
  // Send message to BART-Large-MNLI backend
  async sendMessage(message) {
    try {
      const startTime = Date.now();
      
      // Generate or retrieve session ID
      let sessionId = sessionStorage.getItem('session_id');
      if (!sessionId) {
        sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        sessionStorage.setItem('session_id', sessionId);
      }
      
      // Fixed payload format to match backend expectations
      const requestPayload = {
        query: message.trim(), // Changed from 'message' to 'query'
        session_id: sessionId  // Backend expects snake_case
      };
      
      console.log('📨 Sending request payload:', requestPayload);
      
      const response = await apiClient.post('/api/chat/message', requestPayload);
      
      const endTime = Date.now();
      const responseTime = endTime - startTime;
      
      console.log('📥 Backend response:', response.data);
      
      return {
        response: response.data.response || 'I understand your query. Let me help you with that.',
        intent: response.data.intent || 'general_inquiry',
        confidence: response.data.confidence || 0.5,
        secondaryIntents: response.data.secondary_intents || [],
        responseTime: responseTime,
        retrievedDocs: response.data.retrieved_docs || [],
        reasoning: response.data.reasoning || '',
        sessionId: sessionId
      };
    } catch (error) {
      // Enhanced fallback response for development/testing
      console.warn('⚠️ API call failed, using fallback response:', error.message);
      
      return {
        response: `I received your message: "${message}". Currently in development mode - please ensure your backend is running on port 8000. Error: ${error.message}`,
        intent: 'general_inquiry',
        confidence: 0.1, // Low confidence for fallback
        secondaryIntents: [],
        responseTime: 100,
        retrievedDocs: [],
        reasoning: 'Fallback response due to API unavailability',
        sessionId: sessionStorage.getItem('session_id') || 'fallback'
      };
    }
  },

  // Submit user feedback for model improvement
  async submitFeedback(messageId, rating, comment = '') {
    try {
      const feedbackPayload = {
        message_id: messageId,
        rating: rating, // 1-5 or thumbs up/down
        comment: comment,
        session_id: sessionStorage.getItem('session_id') || 'default',
        timestamp: new Date().toISOString()
      };
      
      const response = await apiClient.post('/api/feedback', feedbackPayload);
      console.log('✅ Feedback submitted successfully');
      return response.data;
    } catch (error) {
      console.error('❌ Failed to submit feedback:', error.message);
      return { success: false, error: error.message };
    }
  },

  // Get intent categories from backend
  async getIntentCategories() {
    try {
      const response = await apiClient.get('/api/intents/categories');
      return response.data.categories || [
        'loan_eligibility',
        'documentation', 
        'interest_rates',
        'account_management',
        'fees_charges',
        'repayment_terms',
        'investment_process',
        'general_inquiry'
      ];
    } catch (error) {
      console.warn('⚠️ Failed to fetch intent categories:', error.message);
      return [
        'loan_eligibility',
        'documentation', 
        'interest_rates',
        'account_management',
        'fees_charges',
        'repayment_terms',
        'investment_process',
        'general_inquiry'
      ];
    }
  },

  // Health check with detailed system status
  async checkHealth() {
    try {
      const response = await apiClient.get('/api/health');
      console.log('🟢 Backend health check successful:', response.data);
      return {
        status: 'healthy',
        classifierLoaded: response.data.classifier_loaded || false,
        uptime: response.data.uptime || 0,
        version: response.data.version || '1.0.0',
        ...response.data
      };
    } catch (error) {
      console.error('🔴 Backend health check failed:', error.message);
      return { 
        status: 'unavailable', 
        message: error.message,
        classifierLoaded: false,
        uptime: 0
      };
    }
  },

  // Test intent classification without full chat processing
  async classifyIntent(query) {
    try {
      const response = await apiClient.post('/api/classify', {
        query: query.trim()
      });
      
      return {
        intent: response.data.intent,
        confidence: response.data.confidence,
        secondaryIntents: response.data.secondary_intents || []
      };
    } catch (error) {
      console.error('❌ Intent classification failed:', error.message);
      return {
        intent: 'general_inquiry',
        confidence: 0.1,
        secondaryIntents: []
      };
    }
  }
};

// Export additional utilities
export const sessionUtils = {
  clearSession: () => {
    sessionStorage.removeItem('session_id');
    console.log('🗑️ Session cleared');
  },
  
  getSessionId: () => {
    return sessionStorage.getItem('session_id');
  },
  
  generateNewSession: () => {
    const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    sessionStorage.setItem('session_id', newSessionId);
    console.log('🆕 New session generated:', newSessionId);
    return newSessionId;
  }
};

export default chatApi;
