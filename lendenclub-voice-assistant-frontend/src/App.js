import React from 'react';
import styled, { ThemeProvider } from 'styled-components';
import ChatInterface from './components/chat/ChatInterface';
import { ChatProvider } from './contexts/ChatContext';
import { FeedbackProvider } from './contexts/FeedbackContext';

const theme = {
  colors: {
    primary: '#667eea',
    secondary: '#764ba2',
    success: '#27ae60',
    warning: '#f39c12',
    error: '#e74c3c',
    background: '#f8f9fa',
    surface: '#ffffff',
    text: '#2c3e50',
    textSecondary: '#7f8c8d',
    border: '#ecf0f1'
  },
  spacing: {
    xs: '0.25rem',
    sm: '0.5rem',
    md: '1rem',
    lg: '1.5rem',
    xl: '2rem'
  },
  borderRadius: '0.5rem',
  boxShadow: '0 2px 10px rgba(0, 0, 0, 0.1)'
};

const AppContainer = styled.div`
  height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: ${props => props.theme.spacing.md};
  background: linear-gradient(135deg, ${props => props.theme.colors.primary} 0%, ${props => props.theme.colors.secondary} 100%);
`;

const AppHeader = styled.div`
  text-align: center;
  margin-bottom: ${props => props.theme.spacing.lg};
  color: white;
  
  h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 700;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
  }
  
  p {
    margin: ${props => props.theme.spacing.sm} 0 0;
    font-size: 1.1rem;
    opacity: 0.9;
  }
`;

const MainContent = styled.div`
  width: 100%;
  max-width: 800px;
  height: 80vh;
  display: flex;
  flex-direction: column;
`;

function App() {
  return (
    <ThemeProvider theme={theme}>
      <ChatProvider>
        <FeedbackProvider>
          <AppContainer>
            <MainContent>
              <AppHeader>
                <h1>LendenClub Voice Assistant</h1>
                <p>AI-powered financial assistant with BART-Large-MNLI</p>
              </AppHeader>
              <ChatInterface />
            </MainContent>
          </AppContainer>
        </FeedbackProvider>
      </ChatProvider>
    </ThemeProvider>
  );
}

export default App;