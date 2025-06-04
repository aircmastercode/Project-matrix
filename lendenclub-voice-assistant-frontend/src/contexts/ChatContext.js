// src/contexts/ChatContext.js

import React, { createContext, useContext, useReducer, useCallback } from 'react';
import { chatApi } from '../services/api/chatApi';

const ChatContext = createContext(null);

const initialState = {
  messages: [],
  isLoading: false,
  error: null,
  stats: {
    totalQueries: 0,
    averageConfidence: 0,
    intentDistribution: {}
  }
};

function chatReducer(state, action) {
  switch (action.type) {
    case 'ADD_MESSAGE':
      return {
        ...state,
        messages: [...state.messages, action.payload],
        isLoading: false,
        error: null
      };

    case 'SET_LOADING':
      return {
        ...state,
        isLoading: action.payload
      };

    case 'SET_ERROR':
      return {
        ...state,
        error: action.payload,
        isLoading: false
      };

    case 'UPDATE_STATS':
      return {
        ...state,
        stats: {
          ...state.stats,
          ...action.payload
        }
      };

    case 'CLEAR_MESSAGES':
      return {
        ...state,
        messages: []
      };

    default:
      return state;
  }
}

export const ChatProvider = ({ children }) => {
  const [state, dispatch] = useReducer(chatReducer, initialState);

  const sendMessage = useCallback(async (message) => {
    if (!message.trim()) return;

    const userMessage = {
      id: Date.now(),
      text: message,
      type: 'user',
      timestamp: new Date().toISOString()
    };

    dispatch({ type: 'ADD_MESSAGE', payload: userMessage });
    dispatch({ type: 'SET_LOADING', payload: true });

    try {
      const response = await chatApi.sendMessage(message);

      const aiMessage = {
        id: Date.now() + 1,
        text: response.response,
        type: 'ai',
        timestamp: new Date().toISOString(),
        intent: response.intent,
        confidence: response.confidence,
        secondaryIntents: response.secondaryIntents || [],
        responseTime: response.responseTime
      };

      dispatch({ type: 'ADD_MESSAGE', payload: aiMessage });

      dispatch({
        type: 'UPDATE_STATS',
        payload: {
          totalQueries: state.stats.totalQueries + 1,
          averageConfidence: response.confidence,
          intentDistribution: {
            ...state.stats.intentDistribution,
            [response.intent]: (state.stats.intentDistribution[response.intent] || 0) + 1
          }
        }
      });

    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: error.message });

      const errorMessage = {
        id: Date.now() + 1,
        text: 'Sorry, I encountered an error processing your request. Please try again.',
        type: 'ai',
        timestamp: new Date().toISOString(),
        isError: true
      };

      dispatch({ type: 'ADD_MESSAGE', payload: errorMessage });
    }
  }, [state.stats.totalQueries, state.stats.intentDistribution]);

  const clearMessages = useCallback(() => {
    dispatch({ type: 'CLEAR_MESSAGES' });
  }, []);

  const value = {
    ...state,
    sendMessage,
    clearMessages
  };

  return (
    <ChatContext.Provider value={value}>
      {children}
    </ChatContext.Provider>
  );
};

export const useChat = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
};