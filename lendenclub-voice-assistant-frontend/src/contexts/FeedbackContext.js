import React, { createContext, useContext, useReducer, useCallback } from 'react';
import { feedbackApi } from '../services/api/feedbackApi';

const FeedbackContext = createContext(null);

const initialState = {
  ratings: {},
  isSubmitting: false,
  error: null,
  feedbackStats: {
    totalRatings: 0,
    averageRating: 0,
    positivePercentage: 0
  }
};

function feedbackReducer(state, action) {
  switch (action.type) {
    case 'SET_RATING':
      return {
        ...state,
        ratings: {
          ...state.ratings,
          [action.payload.messageId]: action.payload.rating
        }
      };
    
    case 'SET_SUBMITTING':
      return {
        ...state,
        isSubmitting: action.payload
      };
    
    case 'SET_ERROR':
      return {
        ...state,
        error: action.payload,
        isSubmitting: false
      };
    
    case 'UPDATE_FEEDBACK_STATS':
      return {
        ...state,
        feedbackStats: {
          ...state.feedbackStats,
          ...action.payload
        }
      };
    
    default:
      return state;
  }
}

export const FeedbackProvider = ({ children }) => {
  const [state, dispatch] = useReducer(feedbackReducer, initialState);

  const submitRating = useCallback(async (messageId, rating, comment = '') => {
    dispatch({ type: 'SET_SUBMITTING', payload: true });

    try {
      await feedbackApi.submitRating({
        messageId,
        rating,
        comment,
        timestamp: new Date().toISOString()
      });

      dispatch({ type: 'SET_RATING', payload: { messageId, rating } });
      
      // Update feedback stats
      const newTotal = state.feedbackStats.totalRatings + 1;
      const newAverage = ((state.feedbackStats.averageRating * state.feedbackStats.totalRatings) + rating) / newTotal;
      const positiveCount = Object.values(state.ratings).filter(r => r > 0).length + (rating > 0 ? 1 : 0);
      
      dispatch({
        type: 'UPDATE_FEEDBACK_STATS',
        payload: {
          totalRatings: newTotal,
          averageRating: newAverage,
          positivePercentage: (positiveCount / newTotal) * 100
        }
      });

    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: error.message });
    } finally {
      dispatch({ type: 'SET_SUBMITTING', payload: false });
    }
  }, [state.feedbackStats, state.ratings]);

  const getRating = useCallback((messageId) => {
    return state.ratings[messageId] || null;
  }, [state.ratings]);

  const value = {
    ...state,
    submitRating,
    getRating
  };

  return (
    <FeedbackContext.Provider value={value}>
      {children}
    </FeedbackContext.Provider>
  );
};

export const useFeedback = () => {
  const context = useContext(FeedbackContext);
  if (!context) {
    throw new Error('useFeedback must be used within a FeedbackProvider');
  }
  return context;
};