import React from 'react';
import { motion } from 'framer-motion';
import './ModelCard.css';

const ModelCard = ({
  modelName,
  modelId,
  status = 'generating',
  onClick,
  position,
  index,
  errorReason = null,
  onReplace = null
}) => {
  const normalizeModelName = (name) => {
    return String(name).split('/').pop().replace(':free', '');
  };

  const getStatusIcon = () => {
    switch (status) {
      case 'generating':
        return '⏳';
      case 'success':
        return '✓';
      case 'error':
        return '✕';
      case 'skipped':
        return '⊘';
      default:
        return '⏳';
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'generating':
        return 'Generating…';
      case 'success':
        return 'Ready';
      case 'error':
        return errorReason || 'Failed';
      case 'skipped':
        return 'Skipped';
      default:
        return 'Pending';
    }
  };

  const cardVariants = {
    hidden: {
      opacity: 0,
      scale: 0.8
    },
    generating: {
      opacity: 1,
      scale: 1,
      transition: {
        duration: 0.4,
        ease: [0.2, 0.8, 0.2, 1]
      }
    },
    success: {
      opacity: 1,
      scale: [1, 1.06, 1],
      transition: {
        duration: 0.5,
        ease: [0.2, 0.8, 0.2, 1]
      }
    }
  };

  const shimmerVariants = {
    generating: {
      x: ['-100%', '100%'],
      transition: {
        duration: 1.5,
        repeat: Infinity,
        ease: 'linear'
      }
    }
  };

  const isClickable = status === 'success';

  return (
    <motion.div
      layoutId={modelId}
      className={`model-card model-card-${status}`}
      data-model-id={modelId}
      variants={cardVariants}
      initial="hidden"
      animate={status === 'success' ? 'success' : 'generating'}
      style={position}
      onClick={isClickable ? onClick : undefined}
      role={isClickable ? 'button' : 'status'}
      tabIndex={isClickable ? 0 : -1}
      onKeyDown={isClickable ? (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick();
        }
      } : undefined}
      aria-label={`${normalizeModelName(modelName)}: ${getStatusText()}`}
      whileHover={isClickable ? { scale: 1.02 } : undefined}
      whileTap={isClickable ? { scale: 0.98 } : undefined}
    >
      <div className="model-card-header">
        <span className="model-card-name">{normalizeModelName(modelName)}</span>
        <motion.span
          className={`model-card-status-icon status-${status}`}
          animate={status === 'generating' ? { rotate: 360 } : {}}
          transition={status === 'generating' ? { duration: 2, repeat: Infinity, ease: 'linear' } : {}}
        >
          {getStatusIcon()}
        </motion.span>
      </div>

      <div className="model-card-body">
        {status === 'generating' && (
          <div className="shimmer-container">
            <motion.div
              className="shimmer-bar"
              variants={shimmerVariants}
              animate="generating"
            />
          </div>
        )}
        <span className={`model-card-status-text status-${status}`}>
          {getStatusText()}
        </span>
      </div>

      {status === 'success' && (
        <div className="model-card-footer">
          <span className="click-hint">Click to preview</span>
        </div>
      )}

      {status === 'error' && onReplace && (
        <div className="model-card-footer">
          <button
            className="replace-btn"
            onClick={(e) => {
              e.stopPropagation();
              onReplace(modelName);
            }}
          >
            Replace Agent
          </button>
        </div>
      )}
    </motion.div>
  );
};

export default ModelCard;
