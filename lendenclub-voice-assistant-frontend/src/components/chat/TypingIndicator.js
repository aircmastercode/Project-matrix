import React from 'react';
import styled, { keyframes } from 'styled-components';
import { FiMessageSquare } from 'react-icons/fi';

const bounce = keyframes`
  0%, 60%, 100% {
    transform: translateY(0);
  }
  30% {
    transform: translateY(-10px);
  }
`;

const TypingContainer = styled.div`
  display: flex;
  justify-content: flex-start;
  margin-bottom: ${props => props.theme.spacing.md};
`;

const TypingBubble = styled.div`
  background: white;
  padding: ${props => props.theme.spacing.md};
  border-radius: ${props => props.theme.borderRadius};
  box-shadow: ${props => props.theme.boxShadow};
  max-width: 200px;
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  color: ${props => props.theme.colors.textSecondary};
`;

const DotsContainer = styled.div`
  display: flex;
  gap: 4px;
`;

const Dot = styled.div`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: ${props => props.theme.colors.primary};
  animation: ${bounce} 1.4s ease-in-out infinite;
  animation-delay: ${props => props.delay}s;
`;

function TypingIndicator() {
  return (
    <TypingContainer>
      <TypingBubble>
        <FiMessageSquare />
        <span>Assistant is thinking</span>
        <DotsContainer>
          <Dot delay={0} />
          <Dot delay={0.2} />
          <Dot delay={0.4} />
        </DotsContainer>
      </TypingBubble>
    </TypingContainer>
  );
}

export default TypingIndicator;