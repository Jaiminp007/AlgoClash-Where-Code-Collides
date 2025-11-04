import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './ReviewScreen.css';

const ReviewScreen = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { generationId, selectedAgents, selectedStock } = location.state || {};

  const [generationData, setGenerationData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState(null);
  const [simulationStarted, setSimulationStarted] = useState(false);
  const [simulationId, setSimulationId] = useState(null);

  useEffect(() => {
    if (!generationId) {
      navigate('/');
      return;
    }

    // Poll for generation status
    const pollGeneration = async () => {
      try {
        const apiBase = process.env.REACT_APP_API_BASE_URL || '';
        const response = await fetch(`${apiBase}/api/generation/${generationId}`);
        const data = await response.json();

        // Always update local copy so we can render partial results while generating
        setGenerationData(data);

        if (data.status === 'completed') {
          setLoading(false);
        } else if (data.status === 'error') {
          setError(data.error || 'Generation failed');
          setLoading(false);
        } else {
          // Still running, poll again
          setTimeout(pollGeneration, 2000);
        }
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    pollGeneration();
  }, [generationId, navigate]);

  const handleStartSimulation = async () => {
    try {
      setSimulationStarted(true);
      const apiBase = process.env.REACT_APP_API_BASE_URL || '';
      const response = await fetch(`${apiBase}/api/simulate/${generationId}`, {
        method: 'POST'
      });

      const data = await response.json();

      if (response.ok) {
        setSimulationId(data.simulation_id);
        // Navigate to results page or update UI to show simulation progress
        navigate('/simulation-results', {
          state: {
            simulationId: data.simulation_id,
            selectedAgents,
            selectedStock
          }
        });
      } else {
        throw new Error(data.error || 'Failed to start simulation');
      }
    } catch (err) {
      setError(err.message);
      setSimulationStarted(false);
    }
  };

  const handleBack = () => {
    navigate('/');
  };

  const downloadAlgorithm = (modelName, code) => {
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${modelName.replace(/[^a-z0-9]/gi, '_')}_algorithm.py`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const copyToClipboard = (code) => {
    navigator.clipboard.writeText(code);
    alert('Algorithm copied to clipboard!');
  };

  if (loading) {
    return (
      <div className="review-screen">
        <div className="review-header">
          <h1>Generating Algorithms...</h1>
        </div>
        <div className="loading-container">
          <div className="spinner"></div>
          <p>{generationData?.message || 'Please wait while we generate trading algorithms...'}</p>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${generationData?.progress || 0}%` }}
            />
          </div>
        </div>

        {/* Show any algorithms that have arrived so far */}
        {generationData?.algorithms && Object.keys(generationData.algorithms).length > 0 && (
          <div className="algorithms-grid" style={{ marginTop: '1.5rem' }}>
            {Object.entries(generationData.algorithms).map(([modelName, code], index) => (
              <div key={modelName} className="algorithm-card">
                <div className="algorithm-header">
                  <h3>
                    {index + 1}. {modelName}
                  </h3>
                </div>
                <pre className="algorithm-preview">
                  <code>{String(code).slice(0, 300)}...</code>
                </pre>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  if (error) {
    return (
      <div className="review-screen">
        <div className="review-header">
          <h1>Error</h1>
        </div>
        <div className="error-container">
          <p>{error}</p>
          <button onClick={handleBack} className="back-btn">Back to Dashboard</button>
        </div>
      </div>
    );
  }

  const algorithms = generationData?.algorithms || {};
  const algorithmList = Object.entries(algorithms);

  return (
    <div className="review-screen">
      <div className="review-header">
        <h1>Review Generated Algorithms</h1>
        <p className="stock-info">Stock: {selectedStock?.replace('_data.csv', '').toUpperCase()}</p>
      </div>

      <div className="algorithms-grid">
        {algorithmList.map(([modelName, code], index) => (
          <div key={modelName} className="algorithm-card">
            <div className="algorithm-header">
              <h3>
                {index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : `${index + 1}.`}
                {' '}{modelName}
              </h3>
              <div className="algorithm-actions">
                <button
                  onClick={() => copyToClipboard(code)}
                  className="action-btn"
                  title="Copy to clipboard"
                >
                  📋
                </button>
                <button
                  onClick={() => downloadAlgorithm(modelName, code)}
                  className="action-btn"
                  title="Download"
                >
                  💾
                </button>
                <button
                  onClick={() => setSelectedAlgorithm({ modelName, code })}
                  className="action-btn"
                  title="View full code"
                >
                  🔍
                </button>
              </div>
            </div>
            <pre className="algorithm-preview">
              <code>{code.slice(0, 300)}...</code>
            </pre>
          </div>
        ))}
      </div>

      <div className="review-actions">
        <button onClick={handleBack} className="back-btn" disabled={simulationStarted}>
          Back
        </button>
        <button
          onClick={handleStartSimulation}
          className="start-simulation-btn"
          disabled={simulationStarted}
        >
          {simulationStarted ? 'Starting Simulation...' : 'Start Market Simulation'}
        </button>
      </div>

      {/* Full Algorithm Modal */}
      {selectedAlgorithm && (
        <div className="modal-overlay" onClick={() => setSelectedAlgorithm(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>{selectedAlgorithm.modelName}</h2>
              <button onClick={() => setSelectedAlgorithm(null)} className="close-btn">
                ✕
              </button>
            </div>
            <div className="modal-body">
              <pre><code>{selectedAlgorithm.code}</code></pre>
            </div>
            <div className="modal-footer">
              <button
                onClick={() => copyToClipboard(selectedAlgorithm.code)}
                className="modal-action-btn"
              >
                Copy to Clipboard
              </button>
              <button
                onClick={() => downloadAlgorithm(selectedAlgorithm.modelName, selectedAlgorithm.code)}
                className="modal-action-btn"
              >
                Download
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ReviewScreen;
