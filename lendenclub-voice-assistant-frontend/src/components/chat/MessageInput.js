import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';
import { FiSend, FiMessageCircle } from 'react-icons/fi';
import { useChat } from '../../contexts/ChatContext';

const InputContainer = styled.div`
  padding: ${props => props.theme.spacing.md};
  background: white;
  border-top: 1px solid ${props => props.theme.colors.border};
`;

const InputWrapper = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.sm};
  align-items: flex-end;
`;

const TextAreaWrapper = styled.div`
  flex: 1;
  position: relative;
`;

const StyledTextArea = styled.textarea`
  width: 100%;
  min-height: 44px;
  max-height: 120px;
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  border: 2px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.borderRadius};
  font-family: inherit;
  font-size: 0.9rem;
  line-height: 1.4;
  resize: none;
  outline: none;
  transition: border-color 0.2s ease;

  &:focus {
    border-color: ${props => props.theme.colors.primary};
  }

  &::placeholder {
    color: ${props => props.theme.colors.textSecondary};
  }
`;

const SendButton = styled.button`
  background: ${props => props.disabled 
    ? props.theme.colors.border 
    : `linear-gradient(135deg, ${props.theme.colors.primary}, ${props.theme.colors.secondary})`
  };
  color: white;
  border: none;
  border-radius: ${props => props.theme.borderRadius};
  padding: ${props => props.theme.spacing.sm};
  width: 44px;
  height: 44px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: ${props => props.disabled ? 'not-allowed' : 'pointer'};
  transition: all 0.2s ease;

  &:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
  }

  &:active:not(:disabled) {
    transform: translateY(0);
  }

  svg {
    width: 18px;
    height: 18px;
  }
`;

const CharacterCount = styled.div`
  position: absolute;
  bottom: -20px;
  right: 0;
  font-size: 0.75rem;
  color: ${props => props.theme.colors.textSecondary};
`;

const QuickActions = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.xs};
  margin-bottom: ${props => props.theme.spacing.sm};
  flex-wrap: wrap;
`;

const QuickActionButton = styled.button`
  background: transparent;
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: 20px;
  padding: ${props => props.theme.spacing.xs} ${props => props.theme.spacing.sm};
  font-size: 0.8rem;
  color: ${props => props.theme.colors.text};
  cursor: pointer;
  transition: all 0.2s ease;
  white-space: nowrap;

  &:hover {
    background: ${props => props.theme.colors.primary};
    color: white;
    border-color: ${props => props.theme.colors.primary};
  }
`;

const MAX_CHARACTERS = 500;

function MessageInput() {
  const [message, setMessage] = useState('');
  const [showQuickActions, setShowQuickActions] = useState(true);
  const textAreaRef = useRef(null);
  const { sendMessage, isLoading } = useChat();

  useEffect(() => {
    if (textAreaRef.current) {
      textAreaRef.current.style.height = 'auto';
      textAreaRef.current.style.height = `${textAreaRef.current.scrollHeight}px`;
    }
  }, [message]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim() && !isLoading) {
      sendMessage(message);
      setMessage('');
      setShowQuickActions(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleQuickAction = (action) => {
    setMessage(action);
    setShowQuickActions(false);
    textAreaRef.current?.focus();
  };

  const quickActions = [
    "What documents do I need?",
    "Check loan eligibility",
    "Current interest rates",
    "Account management help",
    "Processing fees information"
  ];

  const isDisabled = !message.trim() || isLoading || message.length > MAX_CHARACTERS;

  return (
    <InputContainer>
      {showQuickActions && (
        <QuickActions>
          {quickActions.map((action, index) => (
            <QuickActionButton
              key={index}
              onClick={() => handleQuickAction(action)}
            >
              <FiMessageCircle style={{ marginRight: '4px', width: '12px', height: '12px' }} />
              {action}
            </QuickActionButton>
          ))}
        </QuickActions>
      )}

      <form onSubmit={handleSubmit}>
        <InputWrapper>
          <TextAreaWrapper>
            <StyledTextArea
              ref={textAreaRef}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              onFocus={() => setShowQuickActions(false)}
              placeholder={isLoading ? "Processing your message..." : "Ask me about loans, investments, documentation..."}
              disabled={isLoading}
              maxLength={MAX_CHARACTERS}
            />
            <CharacterCount>
              {message.length}/{MAX_CHARACTERS}
            </CharacterCount>
          </TextAreaWrapper>

          <SendButton type="submit" disabled={isDisabled}>
            <FiSend />
          </SendButton>
        </InputWrapper>
      </form>
    </InputContainer>
  );
}

export default MessageInput;