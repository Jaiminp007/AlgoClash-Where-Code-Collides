import React, { useState } from 'react';
import './AlgorithmCard.css';

const AlgorithmCard = ({ modelName, code, index, status = 'completed', onReplace = null }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(code);
    // Simple visual feedback
    const button = document.querySelector(`#copy-btn-${index}`);
    if (button) {
      const originalText = button.textContent;
      button.textContent = '✓ Copied!';
      setTimeout(() => {
        button.textContent = originalText;
      }, 2000);
    }
  };

  const downloadAlgorithm = () => {
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${modelName.replace(/[^a-z0-9]/gi, '_')}_algorithm.py`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // If status is not completed, show loading/error state
  if (status === 'generating') {
    return (
      <div className="algorithm-card loading" data-index={index}>
        <div className="card-header">
          <div className="card-title">
            <span className="card-number">#{index + 1}</span>
            <span className="card-model">{modelName}</span>
          </div>
          <span className="loading-spinner">⏳</span>
        </div>
        <div className="card-status">Generating algorithm...</div>
      </div>
    );
  }

  if (status === 'failed') {
    return (
      <div className="algorithm-card failed" data-index={index}>
        <div className="card-header">
          <div className="card-title">
            <span className="card-number">#{index + 1}</span>
            <span className="card-model">{modelName}</span>
          </div>
          <span className="error-badge">❌ FAILED</span>
        </div>
        <div className="card-status">Algorithm generation failed</div>
        {onReplace && (
          <div className="card-actions-failed">
            <button
              className="replace-agent-btn"
              onClick={() => onReplace(modelName)}
            >
              🔄 Replace Agent
            </button>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className={`algorithm-card ${isExpanded ? 'expanded' : ''}`} data-index={index}>
      <div className="card-header" onClick={() => setIsExpanded(!isExpanded)}>
        <div className="card-title">
          <span className="card-number">#{index + 1}</span>
          <span className="card-model">{modelName}</span>
        </div>
        <div className="card-actions" onClick={(e) => e.stopPropagation()}>
          <button
            id={`copy-btn-${index}`}
            className="action-btn"
            onClick={copyToClipboard}
            title="Copy to clipboard"
            aria-label="Copy algorithm to clipboard"
          >
            📋 Copy
          </button>
          <button
            className="action-btn"
            onClick={downloadAlgorithm}
            title="Download algorithm"
            aria-label="Download algorithm file"
          >
            💾
          </button>
          <button
            className="expand-btn"
            onClick={() => setIsExpanded(!isExpanded)}
            title={isExpanded ? 'Collapse' : 'Expand'}
            aria-label={isExpanded ? 'Collapse code' : 'Expand code'}
          >
            {isExpanded ? '▲' : '▼'}
          </button>
        </div>
      </div>

      {isExpanded && (
        <div className="card-body">
          <pre className="code-block">
            <code>{code}</code>
          </pre>
        </div>
      )}

      {!isExpanded && (
        <div className="card-preview">
          <code>{code.slice(0, 150)}...</code>
        </div>
      )}
    </div>
  );
};

export default AlgorithmCard;
