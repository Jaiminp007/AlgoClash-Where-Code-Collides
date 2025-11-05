import React, { useState } from 'react';
import './ReplaceAgentModal.css';
import CustomDropdown from './CustomDropdown';

const ReplaceAgentModal = ({ isOpen, onClose, failedModel, agents, usedAgents, onReplace }) => {
  const [selectedAgent, setSelectedAgent] = useState(null);

  if (!isOpen) return null;

  const handleReplace = () => {
    if (selectedAgent) {
      onReplace(selectedAgent);
      setSelectedAgent(null);
    }
  };

  const handleClose = () => {
    setSelectedAgent(null);
    onClose();
  };

  // Create a set of disabled agents (already used agents except the failed one)
  const disabledAgents = new Set(usedAgents.filter(a => a !== failedModel));

  return (
    <div className="modal-overlay" onClick={handleClose}>
      <div className="replace-modal" onClick={(e) => e.stopPropagation()}>
        <div className="replace-modal-header">
          <h2>Replace Failed Agent</h2>
          <button className="close-btn" onClick={handleClose}>×</button>
        </div>

        <div className="replace-modal-body">
          <p className="failed-model-info">
            <span className="label">Failed Model:</span>
            <span className="model-name">{failedModel}</span>
          </p>

          <p className="instruction">Select a new AI agent to replace the failed one:</p>

          <div className="agent-selector">
            <CustomDropdown
              agents={agents}
              selected={selectedAgent}
              onSelect={setSelectedAgent}
              disabledAgents={disabledAgents}
              placeholder="Select replacement agent..."
            />
          </div>
        </div>

        <div className="replace-modal-footer">
          <button className="cancel-btn" onClick={handleClose}>
            Cancel
          </button>
          <button
            className="replace-confirm-btn"
            onClick={handleReplace}
            disabled={!selectedAgent}
          >
            Replace & Regenerate
          </button>
        </div>
      </div>
    </div>
  );
};

export default ReplaceAgentModal;
