import React from 'react';
import styled from 'styled-components';
import { FiTarget, FiTrendingUp, FiTrendingDown } from 'react-icons/fi';

const ConfidenceContainer = styled.div`
  display: inline-flex;
  align-items: center;
  gap: ${props => props.theme.spacing.xs};
  background: ${props => props.bgColor}15;
  color: ${props => props.color};
  border: 1px solid ${props => props.color}30;
  border-radius: 20px;
  padding: ${props => props.theme.spacing.xs} ${props => props.theme.spacing.sm};
  font-size: 0.75rem;
  font-weight: 500;
  
  svg {
    width: 12px;
    height: 12px;
  }
`;

const ConfidenceBar = styled.div`
  width: 40px;
  height: 6px;
  background: ${props => props.theme.colors.border};
  border-radius: 3px;
  overflow: hidden;
  position: relative;
`;

const ConfidenceFill = styled.div`
  height: 100%;
  width: ${props => props.percentage}%;
  background: ${props => props.color};
  border-radius: 3px;
  transition: width 0.3s ease;
`;

function ConfidenceIndicator({ confidence }) {
  const percentage = Math.round(confidence * 100);
  
  let color, bgColor, icon;
  
  if (confidence >= 0.8) {
    color = '#27ae60';
    bgColor = '#27ae60';
    icon = FiTrendingUp;
  } else if (confidence >= 0.6) {
    color = '#f39c12';
    bgColor = '#f39c12';
    icon = FiTarget;
  } else {
    color = '#e74c3c';
    bgColor = '#e74c3c';
    icon = FiTrendingDown;
  }
  
  const IconComponent = icon;

  return (
    <ConfidenceContainer color={color} bgColor={bgColor}>
      <IconComponent />
      <span>{percentage}%</span>
      <ConfidenceBar>
        <ConfidenceFill percentage={percentage} color={color} />
      </ConfidenceBar>
    </ConfidenceContainer>
  );
}

export default ConfidenceIndicator;