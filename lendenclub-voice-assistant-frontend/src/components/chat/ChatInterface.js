import React, { useEffect, useState } from 'react';
import styled from 'styled-components';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import { useChat } from '../../contexts/ChatContext';
import { chatApi } from '../../services/api/chatApi';
import { FiActivity, FiMessageCircle, FiTarget } from 'react-icons/fi';

const ChatContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  background: ${props => props.theme.colors.surface};
  border-radius: ${props => props.theme.borderRadius};
  box-shadow: ${props => props.theme.boxShadow};
  overflow: hidden;
`;

const ChatHeader = styled.div`
  background: linear-gradient(135deg, ${props => props.theme.colors.primary} 0%, ${props => props.theme.colors.secondary} 100%);
  padding: ${props => props.theme.spacing.md};
  color: white;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const HeaderInfo = styled.div`
  h3 {
    margin: 0;
    font-size: 1.2rem;
    font-weight: 600;
  }
  
  p {
    margin: 0;
    font-size: 0.9rem;
    opacity: 0.9;
  }
`;

const StatsContainer = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.md};
  font-size: 0.85rem;
`;

const StatItem = styled.div`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.xs};
  
  svg {
    width: 16px;
    height: 16px;
  }
`;

const StatusIndicator = styled.div`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: ${props => props.online ? '#27ae60' : '#e74c3c'};
  margin-left: ${props => props.theme.spacing.xs};
`;

function ChatInterface() {
  const { messages, stats, isLoading } = useChat();
  const [isOnline, setIsOnline] = useState(false);

  useEffect(() => {
    // Check backend health on component mount
    const checkHealth = async () => {
      try {
        const health = await chatApi.checkHealth();
        setIsOnline(health.status === 'healthy');
      } catch (error) {
        setIsOnline(false);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Check every 30 seconds

    return () => clearInterval(interval);
  }, []);

  return (
    <ChatContainer>
      <ChatHeader>
        <HeaderInfo>
          <h3>
            LendenClub Assistant
            <StatusIndicator online={isOnline} />
          </h3>
          <p>BART-Large-MNLI • Multi-Intent Classification</p>
        </HeaderInfo>
        
        <StatsContainer>
          <StatItem>
            <FiMessageCircle />
            {stats.totalQueries}
          </StatItem>
          <StatItem>
            <FiTarget />
            {(stats.averageConfidence * 100).toFixed(0)}%
          </StatItem>
          <StatItem>
            <FiActivity />
            {isLoading ? 'Processing...' : 'Ready'}
          </StatItem>
        </StatsContainer>
      </ChatHeader>
      
      <MessageList />
      <MessageInput />
    </ChatContainer>
  );
}

export default ChatInterface;