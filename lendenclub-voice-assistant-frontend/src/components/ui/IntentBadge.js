import React from 'react';
import styled from 'styled-components';
import { FiTarget, FiDollarSign, FiFileText, FiUser, FiCreditCard, FiCalendar, FiTrendingUp, FiHelpCircle } from 'react-icons/fi';

const intentConfig = {
  loan_eligibility: {
    label: 'Loan Eligibility',
    color: '#3498db',
    icon: FiDollarSign
  },
  documentation: {
    label: 'Documentation',
    color: '#e67e22',
    icon: FiFileText
  },
  interest_rates: {
    label: 'Interest Rates',
    color: '#27ae60',
    icon: FiTrendingUp
  },
  account_management: {
    label: 'Account Management',
    color: '#9b59b6',
    icon: FiUser
  },
  fees_charges: {
    label: 'Fees & Charges',
    color: '#e74c3c',
    icon: FiCreditCard
  },
  repayment_terms: {
    label: 'Repayment Terms',
    color: '#f39c12',
    icon: FiCalendar
  },
  investment_process: {
    label: 'Investment Process',
    color: '#1abc9c',
    icon: FiTrendingUp
  },
  general_inquiry: {
    label: 'General Inquiry',
    color: '#95a5a6',
    icon: FiHelpCircle
  }
};

const BadgeContainer = styled.div`
  display: inline-flex;
  align-items: center;
  gap: ${props => props.theme.spacing.xs};
  background: ${props => props.color}15;
  color: ${props => props.color};
  border: 1px solid ${props => props.color}30;
  border-radius: 20px;
  padding: ${props => props.theme.spacing.xs} ${props => props.theme.spacing.sm};
  font-size: ${props => props.isSecondary ? '0.7rem' : '0.75rem'};
  font-weight: 500;
  opacity: ${props => props.isSecondary ? 0.8 : 1};
  
  svg {
    width: 12px;
    height: 12px;
  }
`;

function IntentBadge({ intent, isSecondary = false }) {
  const config = intentConfig[intent] || intentConfig.general_inquiry;
  const IconComponent = config.icon;

  return (
    <BadgeContainer color={config.color} isSecondary={isSecondary}>
      <IconComponent />
      <span>{config.label}</span>
    </BadgeContainer>
  );
}

export default IntentBadge;