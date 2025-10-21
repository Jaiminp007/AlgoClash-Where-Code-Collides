import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './SimulationResults.css';

const SimulationResults = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { simulationId, selectedAgents, selectedStock } = location.state || {};

  const [simulationData, setSimulationData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [progress, setProgress] = useState(0);
  const [currentTask, setCurrentTask] = useState('');

  useEffect(() => {
    if (!simulationId) {
      navigate('/');
      return;
    }

    // Poll for simulation status
    const pollSimulation = async () => {
      try {
        const apiBase = process.env.REACT_APP_API_BASE_URL || '';
        const response = await fetch(`${apiBase}/api/simulation/${simulationId}`);
        const data = await response.json();

        setProgress(data.progress || 0);
        setCurrentTask(data.message || 'Running simulation...');

        if (data.status === 'completed') {
          setSimulationData(data);
          setLoading(false);
        } else if (data.status === 'error') {
          setError(data.error || 'Simulation failed');
          setLoading(false);
        } else {
          // Still running, poll again
          setTimeout(pollSimulation, 2000);
        }
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    pollSimulation();
  }, [simulationId, navigate]);

  const handleBack = () => {
    navigate('/');
  };

  if (loading) {
    return (
      <div className="simulation-results">
        <div className="results-header">
          <h1>Market Simulation Running...</h1>
        </div>
        <div className="loading-container">
          <div className="spinner"></div>
          <p className="task-message">{currentTask}</p>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="progress-text">{progress}%</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="simulation-results">
        <div className="results-header">
          <h1>Error</h1>
        </div>
        <div className="error-container">
          <p>{error}</p>
          <button onClick={handleBack} className="back-btn">Back to Dashboard</button>
        </div>
      </div>
    );
  }

  const results = simulationData?.results || {};
  const leaderboard = results.leaderboard || [];
  const winner = results.winner;

  return (
    <div className="simulation-results">
      <div className="results-header">
        <h1>Battle Results</h1>
        <p className="stock-info">Stock: {selectedStock?.replace('_data.csv', '').toUpperCase()}</p>
      </div>

      {winner && (
        <div className="winner-container">
          <div className="winner-badge">
            <div className="trophy">🏆</div>
            <h2 className="winner-title">Winner</h2>
            <h3 className="winner-name">{winner.name}</h3>
            <div className="winner-stats">
              <div className="stat">
                <span className="stat-label">ROI</span>
                <span className="stat-value">{winner.roi?.toFixed(2)}%</span>
              </div>
              <div className="stat">
                <span className="stat-label">Final Value</span>
                <span className="stat-value">${winner.current_value?.toFixed(2)}</span>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="leaderboard-container">
        <h2>Leaderboard</h2>
        <div className="leaderboard">
          {leaderboard.map((agent, index) => (
            <div key={agent.name} className={`leaderboard-item rank-${index + 1}`}>
              <div className="rank-badge">
                {index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : `${index + 1}`}
              </div>
              <div className="agent-info">
                <div className="agent-name">{agent.name}</div>
                <div className="agent-stats">
                  <span className="stat-item">
                    <span className="stat-label">ROI:</span>
                    <span className={`stat-value ${agent.roi >= 0 ? 'positive' : 'negative'}`}>
                      {agent.roi >= 0 ? '+' : ''}{agent.roi?.toFixed(2)}%
                    </span>
                  </span>
                  <span className="stat-item">
                    <span className="stat-label">Value:</span>
                    <span className="stat-value">${agent.current_value?.toFixed(2)}</span>
                  </span>
                  <span className="stat-item">
                    <span className="stat-label">Cash:</span>
                    <span className="stat-value">${agent.cash?.toFixed(2)}</span>
                  </span>
                  <span className="stat-item">
                    <span className="stat-label">Stock:</span>
                    <span className="stat-value">{agent.stock}</span>
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="results-actions">
        <button onClick={handleBack} className="back-btn">
          Back to Dashboard
        </button>
      </div>
    </div>
  );
};

export default SimulationResults;
