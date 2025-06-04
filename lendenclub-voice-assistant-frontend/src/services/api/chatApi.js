import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    
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
      
      const response = await apiClient.post('/api/chat/message', {
        message: message.trim(),
        timestamp: new Date().toISOString(),
        session_id: sessionStorage.getItem('session_id') || 'default'
      });
      
      const endTime = Date.now();
      const responseTime = endTime - startTime;
      
      return {
        response: response.data.response || 'I understand your query. Let me help you with that.',
        intent: response.data.intent || 'general_inquiry',
        confidence: response.data.confidence || 0.5,
        secondaryIntents: response.data.secondary_intents || [],
        responseTime: responseTime,
        retrievedDocs: response.data.retrieved_docs || [],
        reasoning: response.data.reasoning || ''
      };
    } catch (error) {
      // Fallback response for development/testing
      console.warn('API call failed, using fallback response:', error.message);
      
      return {
        response: `I received your message: "${message}". Currently in development mode - please ensure your backend is running on port 8000.`,
        intent: 'general_inquiry',
        confidence: 0.5,
        secondaryIntents: [],
        responseTime: 100,
        retrievedDocs: [],
        reasoning: 'Fallback response due to API unavailability'
      };
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
      console.warn('Failed to fetch intent categories:', error.message);
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

  // Health check
  async checkHealth() {
    try {
      const response = await apiClient.get('/api/health');
      return response.data;
    } catch (error) {
      return { status: 'unavailable', message: error.message };
    }
  }
};
