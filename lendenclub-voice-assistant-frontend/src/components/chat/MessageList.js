import React, { useEffect, useRef } from 'react';
import styled from 'styled-components';
import { useChat } from '../../contexts/ChatContext';
import ChatMessage from './ChatMessage';
import TypingIndicator from './TypingIndicator';

const MessageContainer = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: ${props => props.theme.spacing.md};
  display: flex;
  flex-direction: column;
  gap: ${props => props.theme.spacing.md};
  background: linear-gradient(to bottom, #f8f9fa 0%, #ffffff 100%);

  /* Custom scrollbar */
  &::-webkit-scrollbar {
    width: 6px;
  }

  &::-webkit-scrollbar-track {
    background: ${props => props.theme.colors.border};
    border-radius: 3px;
  }

  &::-webkit-scrollbar-thumb {
    background: ${props => props.theme.colors.textSecondary};
    border-radius: 3px;
  }

  &::-webkit-scrollbar-thumb:hover {
    background: ${props => props.theme.colors.text};
  }
`;

const EmptyState = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  text-align: center;
  color: ${props => props.theme.colors.textSecondary};
  
  h4 {
    margin: 0 0 ${props => props.theme.spacing.sm};
    font-size: 1.1rem;
    font-weight: 600;
  }
  
  p {
    margin: 0;
    font-size: 0.9rem;
    line-height: 1.5;
  }
`;

const SuggestionButtons = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: ${props => props.theme.spacing.sm};
  margin-top: ${props => props.theme.spacing.md};
`;

const SuggestionButton = styled.button`
  background: white;
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.borderRadius};
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  font-size: 0.85rem;
  color: ${props => props.theme.colors.text};
  cursor: pointer;
  transition: all 0.2s ease;
  
  &:hover {
    background: ${props => props.theme.colors.primary};
    color: white;
    border-color: ${props => props.theme.colors.primary};
  }
`;

const WelcomeMessage = styled.div`
  background: linear-gradient(135deg, ${props => props.theme.colors.primary}, ${props => props.theme.colors.secondary});
  color: white;
  padding: ${props => props.theme.spacing.lg};
  border-radius: ${props => props.theme.borderRadius};
  margin-bottom: ${props => props.theme.spacing.md};
  
  h4 {
    margin: 0 0 ${props => props.theme.spacing.sm};
    font-size: 1.1rem;
  }
  
  p {
    margin: 0;
    font-size: 0.9rem;
    opacity: 0.9;
    line-height: 1.5;
  }
`;

function MessageList() {
  const { messages, isLoading, sendMessage } = useChat();
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleSuggestionClick = (suggestion) => {
    sendMessage(suggestion);
  };

  const suggestions = [
    "What documents do I need for a loan?",
    "What are the current interest rates?",
    "How can I check my account status?",
    "What are the processing fees?",
    "How does the investment process work?"
  ];

  if (messages.length === 0) {
    return (
      <MessageContainer>
        <WelcomeMessage>
          <h4>Welcome to LendenClub Voice Assistant!</h4>
          <p>
            I'm powered by BART-Large-MNLI for accurate intent classification 
            and can help you with loans, investments, documentation, and more.
          </p>
        </WelcomeMessage>
        
        <EmptyState>
          <h4>How can I help you today?</h4>
          <p>
            Ask me anything about LendenClub services, loan eligibility, 
            documentation requirements, or investment processes.
          </p>
          
          <SuggestionButtons>
            {suggestions.map((suggestion, index) => (
              <SuggestionButton
                key={index}
                onClick={() => handleSuggestionClick(suggestion)}
              >
                {suggestion}
              </SuggestionButton>
            ))}
          </SuggestionButtons>
        </EmptyState>
      </MessageContainer>
    );
  }

  return (
    <MessageContainer>
      {messages.map((message) => (
        <ChatMessage key={message.id} message={message} />
      ))}
      
      {isLoading && <TypingIndicator />}
      
      <div ref={messagesEndRef} />
    </MessageContainer>
  );
}

export default MessageList;