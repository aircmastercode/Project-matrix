import React, { useState } from 'react';
import styled from 'styled-components';
import { FiThumbsUp, FiThumbsDown, FiMessageSquare } from 'react-icons/fi';
import { useFeedback } from '../../contexts/FeedbackContext';

const RatingContainer = styled.div`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  margin-top: ${props => props.theme.spacing.sm};
`;

const RatingButton = styled.button`
  background: ${props => props.active 
    ? props.activeColor 
    : 'transparent'
  };
  color: ${props => props.active 
    ? 'white' 
    : props.theme.colors.textSecondary
  };
  border: 1px solid ${props => props.active 
    ? props.activeColor 
    : props.theme.colors.border
  };
  border-radius: 20px;
  padding: ${props => props.theme.spacing.xs} ${props => props.theme.spacing.sm};
  font-size: 0.75rem;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  gap: 4px;

  &:hover {
    background: ${props => props.active 
      ? props.activeColor 
      : props.hoverColor + '15'
    };
    color: ${props => props.active 
      ? 'white' 
      : props.hoverColor
    };
    border-color: ${props => props.hoverColor};
  }

  svg {
    width: 12px;
    height: 12px;
  }
`;

const CommentButton = styled(RatingButton)`
  margin-left: ${props => props.theme.spacing.xs};
`;

const CommentModal = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
`;

const CommentForm = styled.div`
  background: white;
  padding: ${props => props.theme.spacing.lg};
  border-radius: ${props => props.theme.borderRadius};
  width: 90%;
  max-width: 400px;
  box-shadow: ${props => props.theme.boxShadow};
`;

const CommentTextArea = styled.textarea`
  width: 100%;
  height: 100px;
  padding: ${props => props.theme.spacing.sm};
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.borderRadius};
  font-family: inherit;
  font-size: 0.9rem;
  resize: vertical;
  outline: none;

  &:focus {
    border-color: ${props => props.theme.colors.primary};
  }
`;

const CommentActions = styled.div`
  display: flex;
  gap: ${props => props.theme.spacing.sm};
  margin-top: ${props => props.theme.spacing.sm};
  justify-content: flex-end;
`;

const ActionButton = styled.button`
  background: ${props => props.primary 
    ? `linear-gradient(135deg, ${props.theme.colors.primary}, ${props.theme.colors.secondary})`
    : 'transparent'
  };
  color: ${props => props.primary ? 'white' : props.theme.colors.text};
  border: 1px solid ${props => props.primary 
    ? 'transparent' 
    : props.theme.colors.border
  };
  border-radius: ${props => props.theme.borderRadius};
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  font-size: 0.85rem;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    opacity: 0.8;
    transform: translateY(-1px);
  }
`;

function RatingButtons({ messageId }) {
  const { submitRating, getRating, isSubmitting } = useFeedback();
  const [showCommentModal, setShowCommentModal] = useState(false);
  const [comment, setComment] = useState('');
  const [pendingRating, setPendingRating] = useState(null);
  
  const currentRating = getRating(messageId);

  const handleRating = async (rating) => {
    if (isSubmitting) return;
    
    if (rating === 1) {
      // Positive rating - submit immediately
      await submitRating(messageId, rating);
    } else {
      // Negative rating - ask for comment
      setPendingRating(rating);
      setShowCommentModal(true);
    }
  };

  const handleCommentSubmit = async () => {
    if (pendingRating !== null) {
      await submitRating(messageId, pendingRating, comment);
      setShowCommentModal(false);
      setComment('');
      setPendingRating(null);
    }
  };

  const handleCommentCancel = () => {
    setShowCommentModal(false);
    setComment('');
    setPendingRating(null);
  };

  return (
    <>
      <RatingContainer>
        <RatingButton
          active={currentRating === 1}
          activeColor="#27ae60"
          hoverColor="#27ae60"
          onClick={() => handleRating(1)}
          disabled={isSubmitting}
        >
          <FiThumbsUp />
          Helpful
        </RatingButton>

        <RatingButton
          active={currentRating === -1}
          activeColor="#e74c3c"
          hoverColor="#e74c3c"
          onClick={() => handleRating(-1)}
          disabled={isSubmitting}
        >
          <FiThumbsDown />
          Not helpful
        </RatingButton>

        {currentRating === -1 && (
          <CommentButton
            hoverColor="#3498db"
            onClick={() => setShowCommentModal(true)}
          >
            <FiMessageSquare />
            Add comment
          </CommentButton>
        )}
      </RatingContainer>

      {showCommentModal && (
        <CommentModal onClick={handleCommentCancel}>
          <CommentForm onClick={(e) => e.stopPropagation()}>
            <h4 style={{ margin: '0 0 1rem', color: '#2c3e50' }}>
              Help us improve
            </h4>
            <p style={{ margin: '0 0 1rem', fontSize: '0.9rem', color: '#7f8c8d' }}>
              What could we do better? Your feedback helps improve our AI.
            </p>
            <CommentTextArea
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              placeholder="Tell us what went wrong or how we can improve..."
              maxLength={500}
            />
            <CommentActions>
              <ActionButton onClick={handleCommentCancel}>
                Cancel
              </ActionButton>
              <ActionButton 
                primary 
                onClick={handleCommentSubmit}
                disabled={isSubmitting}
              >
                {isSubmitting ? 'Submitting...' : 'Submit Feedback'}
              </ActionButton>
            </CommentActions>
          </CommentForm>
        </CommentModal>
      )}
    </>
  );
}

export default RatingButtons;