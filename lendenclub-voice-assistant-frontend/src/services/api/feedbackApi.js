import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const feedbackApi = {
  // Submit user feedback
  async submitRating(feedbackData) {
    try {
      const response = await apiClient.post('/api/feedback', {
        ...feedbackData,
        user_agent: navigator.userAgent,
        session_id: sessionStorage.getItem('session_id') || 'default'
      });
      
      return response.data;
    } catch (error) {
      // Store feedback locally if API fails
      const localFeedback = JSON.parse(localStorage.getItem('localFeedback') || '[]');
      localFeedback.push({
        ...feedbackData,
        stored_locally: true,
        error: error.message
      });
      localStorage.setItem('localFeedback', JSON.stringify(localFeedback));
      
      console.warn('Feedback stored locally due to API error:', error.message);
      return { success: true, stored_locally: true };
    }
  },

  // Get feedback statistics
  async getFeedbackStats() {
    try {
      const response = await apiClient.get('/api/feedback/stats');
      return response.data;
    } catch (error) {
      console.warn('Failed to fetch feedback stats:', error.message);
      return {
        total_ratings: 0,
        average_rating: 0,
        positive_percentage: 0
      };
    }
  }
};