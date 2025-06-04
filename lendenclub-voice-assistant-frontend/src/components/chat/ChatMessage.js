import React from 'react';
import styled from 'styled-components';
import { format } from 'date-fns';
import { FiUser, FiMessageSquare, FiClock, FiTarget } from 'react-icons/fi';
import IntentBadge from '../ui/IntentBadge';
import ConfidenceIndicator from '../feedback/ConfidenceIndicator';
import RatingButtons from '../feedback/RatingButtons';

const MessageWrapper = styled.div`
  display: flex;
  justify-content: ${props => props.isUser ? 'flex-end' : 'flex-start'};
  margin-bottom: ${props => props.theme.spacing.md};
`;

const MessageContainer = styled.div`
  max-width: 70%;
  background: ${props => props.isUser 
    ? `linear-gradient(135deg, ${props.theme.colors.primary}, ${props.theme.colors.secondary})`
    : props.isError 
      ? props.theme.colors.error 
      : 'white'
  };
  color: ${props => props.isUser || props.isError ? 'white' : props.theme.colors.text};
  padding: ${props => props.theme.spacing.md};
  border-radius: ${props => props.theme.borderRadius};
  box-shadow: ${props => props.theme.boxShadow};
  position: relative;
`;

const MessageHeader = styled.div`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  margin-bottom: ${props => props.theme.spacing.sm};
  font-size: 0.85rem;
  opacity: 0.8;
`;

const MessageText = styled.div`
  line-height: 1.5;
  margin-bottom: ${props => props.hasMetadata ? props.theme.spacing.sm : '0'};
  
  p {
    margin: 0;
  }
`;

const MessageMetadata = styled.div`
  margin-top: ${props => props.theme.spacing.sm};
  padding-top: ${props => props.theme.spacing.sm};
  border-top: 1px solid ${props => props.isUser ? 'rgba(255,255,255,0.2)' : props.theme.colors.border};
  font-size: 0.85rem;
`;

const IntentSection = styled.div`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  margin-bottom: ${props => props.theme.spacing.xs};
  flex-wrap: wrap;
`;

const ResponseTimeSection = styled.div`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.xs};
  color: ${props => props.theme.colors.textSecondary};
  font-size: 0.8rem;
  margin-top: ${props => props.theme.spacing.xs};
`;

const SecondaryIntents = styled.div`
  margin-top: ${props => props.theme.spacing.xs};
  
  span {
    font-size: 0.75rem;
    opacity: 0.7;
    margin-right: ${props => props.theme.spacing.xs};
  }
`;

function ChatMessage({ message }) {
  const isUser = message.type === 'user';
  const isError = message.isError;
  const hasMetadata = !isUser && (message.intent || message.confidence !== undefined);

  return (
    <MessageWrapper isUser={isUser}>
      <MessageContainer isUser={isUser} isError={isError}>
        <MessageHeader>
          {isUser ? <FiUser /> : <FiMessageSquare />}
          <span>{isUser ? 'You' : 'Assistant'}</span>
          <span>•</span>
          <span>{format(new Date(message.timestamp), 'HH:mm')}</span>
        </MessageHeader>

        <MessageText hasMetadata={hasMetadata}>
          <p>{message.text}</p>
        </MessageText>

        {hasMetadata && (
          <MessageMetadata isUser={isUser}>
            <IntentSection>
              <IntentBadge intent={message.intent} />
              {message.confidence !== undefined && (
                <ConfidenceIndicator confidence={message.confidence} />
              )}
            </IntentSection>

            {message.secondaryIntents && message.secondaryIntents.length > 0 && (
              <SecondaryIntents>
                <span>Secondary intents:</span>
                {message.secondaryIntents.map((intent, index) => (
                  <IntentBadge 
                    key={index} 
                    intent={intent} 
                    isSecondary={true}
                  />
                ))}
              </SecondaryIntents>
            )}

            {message.responseTime && (
              <ResponseTimeSection>
                <FiClock />
                <span>{message.responseTime}ms</span>
              </ResponseTimeSection>
            )}

            <RatingButtons messageId={message.id} />
          </MessageMetadata>
        )}
      </MessageContainer>
    </MessageWrapper>
  );
}

export default ChatMessage;