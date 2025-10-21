import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import './Dashboard.css';
import CustomDropdown from './CustomDropdown'; // Import the new component
import AlgorithmPreviewModal from './AlgorithmPreviewModal';

const Dashboard = () => {
  const navigate = useNavigate();
  const [navOpen, setNavOpen] = useState(false);
  const [active, setActive] = useState('home');
  const [agents, setAgents] = useState({});
  const [stocks, setStocks] = useState([]);
  const [selectedStock, setSelectedStock] = useState('');
  const [selectedAgents, setSelectedAgents] = useState({
    'left-1': null,
    'left-2': null,
    'left-3': null,
    'right-1': null,
    'right-2': null,
    'right-3': null,
  });
  const [simulationStatus, setSimulationStatus] = useState(null);
  const [simulationResults, setSimulationResults] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [showInstructions, setShowInstructions] = useState(true);
  const [progress, setProgress] = useState(0);
  const [currentTask, setCurrentTask] = useState('');
  const [genStates, setGenStates] = useState({}); // { modelName: 'pending'|'generating'|'done' }
  const lastGeneratingRef = useRef(null);
  const [codePreview, setCodePreview] = useState('');
  const [codePreviewModel, setCodePreviewModel] = useState('');
  const lastPreviewModelRef = useRef('');
  const [showAlgoPreview, setShowAlgoPreview] = useState(false);
  const [generationPhase, setGenerationPhase] = useState('idle'); // 'idle', 'generating', 'review', 'simulating', 'completed'
  const [generatedAlgos, setGeneratedAlgos] = useState([]);
  const [currentSimId, setCurrentSimId] = useState(null);

  useEffect(() => {
    const apiBase = process.env.REACT_APP_API_BASE_URL || '';
    fetch(`${apiBase}/api/ai_agents`)
      .then(response => response.json())
      .then(data => setAgents(data))
      .catch(error => console.error('Error fetching agents:', error));
  }, []);

  useEffect(() => {
    const apiBase = process.env.REACT_APP_API_BASE_URL || '';
    fetch(`${apiBase}/api/data_files`)
      .then(res => res.json())
      .then((data) => {
        // Accept either { stocks: [...] } or [...] directly
        const list = Array.isArray(data) ? data : data?.stocks;
        const safe = Array.isArray(list) ? list : [];
        setStocks(safe);
        if (safe.length > 0) {
          setSelectedStock(safe[0].filename || safe[0]);
        }
      })
      .catch(err => {
        console.error('Error fetching data files:', err);
        setStocks([]);
      });
  }, []);

  // Navigation handler for all nav links
  const handleNav = (e, to) => {
    e.preventDefault();
    if (to === 'home') {
      window.location.href = '/';
    } else if (to === 'models') {
      window.location.href = '/models';
    } else if (to === 'about') {
      window.location.href = '/about';
    } else if (to === 'contact') {
      window.location.href = '/contact';
    }
  };

  const handleAgentSelect = (id, agent) => {
    setSelectedAgents(prev => ({ ...prev, [id]: agent }));
  };

  const handleStartSimulation = async () => {
    // Validate all 6 agents are selected
    const agents = Object.values(selectedAgents).filter(Boolean);
    if (agents.length !== 6) {
      alert('Please select all 6 AI agents before starting the simulation.');
      return;
    }

    if (!selectedStock) {
      alert('Please select a stock dataset.');
      return;
    }

    setIsRunning(true);
    setSimulationResults(null);
    setSimulationStatus('Generating algorithms...');
    setShowInstructions(false);
    setProgress(0);
    // Initialize per-agent generation states
    const uniqueAgents = Array.from(new Set(agents));
    const initStates = {};
    uniqueAgents.forEach(name => { initStates[name] = 'pending'; });
    setGenStates(initStates);
    lastGeneratingRef.current = null;

    try {
      const apiBase = process.env.REACT_APP_API_BASE_URL || '';
      const response = await fetch(`${apiBase}/api/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          agents: agents,
          stock: selectedStock
        })
      });

      const data = await response.json();

      if (response.ok) {
        const genId = data.generation_id;
        setSimulationStatus('Algorithms generating...');

        // Navigate to review screen
        navigate('/review', {
          state: {
            generationId: genId,
            selectedAgents: agents,
            selectedStock: selectedStock
          }
        });
      } else {
        throw new Error(data.error || 'Failed to start generation');
      }
    } catch (error) {
      console.error('Generation error:', error);
      setSimulationStatus(`Error: ${error.message}`);
      setIsRunning(false);
      setGenerationPhase('idle');
    }
  };

  const handleStartMarketSimulation = async () => {
    if (!currentSimId) return;

    setGenerationPhase('simulating');
    setSimulationStatus('Starting market simulation...');
    setProgress(50);

    // Continue polling - backend continues with market simulation
    // Note: Backend doesn't have separate pause/resume, so we just resume polling
    // The backend actually runs the full simulation automatically.
    // This checkpoint is purely for frontend UX to let users review algorithms.
    // We'll resume polling and the backend simulation should already be running
    setTimeout(() => pollSimulationStatus(currentSimId), 1000);
  };

  const pollSimulationStatus = async (simId) => {
    try {
      const apiBase = process.env.REACT_APP_API_BASE_URL || '';
      const response = await fetch(`${apiBase}/api/simulation/${simId}`);
      const data = await response.json();

      if (data.status === 'completed') {
        setSimulationResults(data.results);
        setSimulationStatus('Simulation completed!');
        setIsRunning(false);
        setProgress(100);
        setGenerationPhase('completed');
        // keep last preview visible until user navigates back
      } else if (data.status === 'error') {
        setSimulationStatus(`Error: ${data.error}`);
        setIsRunning(false);
        setGenerationPhase('idle');
      } else {
        const pct = typeof data.progress === 'number' ? data.progress : 0;
        const message = data.message || `Running... ${pct}%`;
        setSimulationStatus(message);
        setProgress(pct);

        if (data.code_preview) {
          const incomingModel = String(data.preview_model || '');
          const incomingCode = String(data.code_preview);
          // Only update when a new model preview arrives or content changes
          if (incomingModel !== lastPreviewModelRef.current || incomingCode !== codePreview) {
            setCodePreview(incomingCode);
            setCodePreviewModel(incomingModel);
            lastPreviewModelRef.current = incomingModel;
          }
        }

        // Parse and reflect generation progress per model if present
        parseProgressMessage(message);

        // Check if all algorithms are generated (transition to review phase)
        if (message.includes('All algorithms generated successfully') && generationPhase === 'generating') {
          setGenerationPhase('review');
          setProgress(50);
          // Fetch generated algorithms list
          fetchGeneratedAlgorithms();
          // Don't continue polling - wait for user to click "Start Market Simulation"
          return;
        }

        // Only continue polling if not in review phase
        if (generationPhase !== 'review') {
          setTimeout(() => pollSimulationStatus(simId), 2000);
        }
      }
    } catch (error) {
      console.error('Polling error:', error);
      setSimulationStatus(`Polling error: ${error.message}`);
      setIsRunning(false);
      setGenerationPhase('idle');
    }
  };

  const fetchGeneratedAlgorithms = async () => {
    try {
      const apiBase = process.env.REACT_APP_API_BASE_URL || '';
      const response = await fetch(`${apiBase}/api/algos`);
      if (response.ok) {
        const algos = await response.json();
        setGeneratedAlgos(algos);
      }
    } catch (error) {
      console.error('Error fetching algorithms:', error);
    }
  };

  // Interpret backend status messages to maintain per-agent generation states
  const parseProgressMessage = (message = '') => {
    setCurrentTask(message);
    // Pattern: "Generating algorithm X/6 using <model>..."
    const genMatch = message.match(/Generating algorithm\s+(\d+)\/(\d+)\s+using\s+(.+?)\.\.\./i);
    if (genMatch) {
      const model = genMatch[3];
      const current = parseInt(genMatch[1]);
      const total = parseInt(genMatch[2]);

      // Mark this model as generating
      setGenStates(prevState => {
        const newState = { ...prevState };
        // Mark previous models as done
        Object.keys(newState).forEach(key => {
          if (key !== model && newState[key] === 'generating') {
            newState[key] = 'done';
          }
        });
        newState[model] = 'generating';
        return newState;
      });
      lastGeneratingRef.current = model;
      return;
    }
    // Mark all done after generation complete message
    if (/All algorithms generated successfully/i.test(message)) {
      setGenStates(prev => {
        const next = { ...prev };
        Object.keys(next).forEach(k => { next[k] = 'done'; });
        return next;
      });
      lastGeneratingRef.current = null;
      return;
    }
  };

  const handleBack = () => {
    // Return to instruction screen; keep agent selections
    setSimulationResults(null);
    setSimulationStatus(null);
    setIsRunning(false);
    setProgress(0);
    setCurrentTask('');
    setCodePreview('');
    setCodePreviewModel('');
    lastPreviewModelRef.current = '';
    setGenStates({});
    lastGeneratingRef.current = null;
    setShowInstructions(true);
    setGenerationPhase('idle');
    setGeneratedAlgos([]);
    setCurrentSimId(null);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="dashboard">
      {/* Header Title */}
      <header className="dashboard-header">
        <h1 className="dashboard-title">
          AlgoClash: <span className="subtitle">Where Code Collides</span>
        </h1>
        <nav className="dashboard-nav">
          <a
            href="#home"
            className={active === 'home' ? 'active' : ''}
            onClick={e => {
              handleNav(e, 'home');
              setActive('home');
            }}
          >
            Home
          </a>
          <a
            href="/models"
            className={active === 'models' ? 'active' : ''}
            onClick={e => {
              handleNav(e, 'models');
              setActive('models');
            }}
          >
            Models
          </a>
          <a
            href="/about"
            className={active === 'about' ? 'active' : ''}
            onClick={e => {
              handleNav(e, 'about');
              setActive('about');
            }}
          >
            About
          </a>
          <a
            href="/contact"
            className={active === 'contact' ? 'active' : ''}
            onClick={e => {
              handleNav(e, 'contact');
              setActive('contact');
            }}
          >
            Contact
          </a>
        </nav>
      </header>

      {/* Top controls below navbar: Preview button and Stock selector */}
      <div className="top-controls">
        <button
          className="preview-algos-button"
          onClick={() => setShowAlgoPreview(true)}
          title="View and manage generated algorithms"
        >
          📄 Preview Algorithms
        </button>

        <div className="stock-selector-group">
          <label htmlFor="stock-select">Stock data:</label>
          {stocks.length > 0 ? (
            <select
              id="stock-select"
              className="stock-select"
              value={selectedStock}
              onChange={(e) => setSelectedStock(e.target.value)}
            >
              {stocks.map((item) => {
                const ticker = item.ticker || String(item).replace(/_data\.csv$/i, '').toUpperCase();
                const filename = item.filename || String(item);
                return (
                  <option key={filename} value={filename}>{ticker}</option>
                );
              })}
            </select>
          ) : (
            <span className="stock-empty">No data files found</span>
          )}
        </div>
      </div>

  {/* Main Content Area */}
      <div className="dashboard-content">
        {[1, 2, 3].map(i => {
          const leftKey = `left-${i}`;
          const rightKey = `right-${i}`;
          const values = Object.values(selectedAgents);
          const disabledLeft = new Set(values.filter(a => a && a !== selectedAgents[leftKey]));
          const disabledRight = new Set(values.filter(a => a && a !== selectedAgents[rightKey]));
          return (
            <React.Fragment key={i}>
              <div className={`side-element left-element left-element-${i}`}>
                <div className="side-circle left-circle">
                  <CustomDropdown 
                    agents={agents} 
                    selected={selectedAgents[leftKey]} 
                    onSelect={(agent) => handleAgentSelect(leftKey, agent)}
                    disabledAgents={disabledLeft}
                  />
                </div>
              </div>
              <div className={`side-element right-element right-element-${i}`}>
                <div className="side-circle right-circle">
                  <CustomDropdown 
                    agents={agents} 
                    selected={selectedAgents[rightKey]} 
                    onSelect={(agent) => handleAgentSelect(rightKey, agent)}
                    disabledAgents={disabledRight}
                  />
                </div>
              </div>
            </React.Fragment>
          );
        })}
      
        {/* Center Blue Box with conditional content */}
        <div className="center-box">
          {/* Idle/Instructions State */}
          {generationPhase === 'idle' && !isRunning ? (
            <div className="instructions">
              <h2>How the Battle Works</h2>
              <p>
                AI Trader Battlefield simulates a fast market session where six AI agents trade the same stock.
                Strategies are generated, orders are matched in a live order book, and the top ROI wins.
              </p>

              <h3>What you do</h3>
              <ol>
                <li>
                  Choose a stock dataset from the dropdown at the top right (e.g., AAPL, MSFT).
                </li>
                <li>
                  Pick exactly six unique AI models using the left and right selectors (3 vs 3).
                </li>
                <li>
                  Click <strong>GENERATE ALGORITHMS</strong> to create unique trading strategies for each AI.
                </li>
                <li>
                  Review all generated algorithms, then click <strong>START SIMULATION</strong> to run the market battle.
                </li>
                <li>
                  Watch the winner and leaderboard after the simulation completes.
                </li>
              </ol>

              <div className="tips">
                <span className="tip-title">Tips</span>
                <ul>
                  <li>All agents must be selected before starting.</li>
                  <li>Mix models from different providers for diverse tactics.</li>
                  <li>You can preview and review algorithms before the battle starts.</li>
                </ul>
              </div>
            </div>
          ) : generationPhase === 'generating' ? (
            /* Algorithm Generation Phase */
            <div className="progress-panel">
              <h2>🔧 Generating Trading Algorithms</h2>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${Math.min(100, Math.max(0, progress))}%` }} />
              </div>
              <div className="current-task">{currentTask || 'Generating algorithms...'}</div>

              {/* Active generation status */}
              <div className="gen-list">
                {Object.keys(genStates).length > 0 ? (
                  Object.entries(genStates).map(([name, state]) => (
                    <div key={name} className={`gen-item ${state}`}>
                      <span className={`status-icon ${state}`}>
                        {state === 'done' ? '✅' : state === 'generating' ? '⏳' : '⏸️'}
                      </span>
                      <span className="model-name">{name}</span>
                      {state === 'generating' && <span className="generating-label">Generating...</span>}
                    </div>
                  ))
                ) : (
                  <div className="gen-empty">Preparing algorithm generation…</div>
                )}
              </div>

              {/* Live code preview during generation */}
              {codePreview && generationPhase === 'generating' && (
                <div className="code-preview">
                  <div className="code-preview-header">
                    Live Preview: {codePreviewModel || 'Algorithm'}
                  </div>
                  <pre><code>{codePreview}</code></pre>
                </div>
              )}
            </div>
          ) : generationPhase === 'review' ? (
            /* Review & Checkpoint Phase */
            <div className="review-panel">
              <h2>✅ Algorithms Generated Successfully!</h2>
              <p className="review-subtitle">
                All {generatedAlgos.length} trading algorithms have been generated. Review them below and start the market simulation when ready.
              </p>

              <div className="generated-algos-list">
                {generatedAlgos.map((algo, idx) => (
                  <div key={algo.filename} className="algo-card">
                    <div className="algo-info">
                      <span className="algo-number">#{idx + 1}</span>
                      <span className="algo-model">{algo.modelName}</span>
                      <span className="algo-size">{(algo.sizeBytes / 1024).toFixed(1)} KB</span>
                    </div>
                    <button
                      className="preview-btn"
                      onClick={() => {
                        setShowAlgoPreview(true);
                      }}
                    >
                      👁️ View
                    </button>
                  </div>
                ))}
              </div>

              <button
                className="start-market-sim-button"
                onClick={handleStartMarketSimulation}
              >
                🚀 Start Market Simulation
              </button>

              <button className="back-button-review" onClick={handleBack}>
                ← Back
              </button>
            </div>
          ) : generationPhase === 'simulating' || generationPhase === 'completed' ? (
            /* Market Simulation Phase */
            <div className="progress-panel">
              <h2>📈 Market Simulation Running</h2>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${Math.min(100, Math.max(0, progress))}%` }} />
              </div>
              <div className="current-task">{currentTask || simulationStatus || 'Running market simulation...'}</div>
            </div>
          ) : null}
        </div>


        {/* Start Button */}
        <button
          className={`start-button ${isRunning ? 'running' : ''}`}
          onClick={handleStartSimulation}
          disabled={isRunning}
        >
          {isRunning ? 'GENERATING...' : 'GENERATE ALGORITHMS'}
        </button>

        {/* Simulation Status */}
        {simulationStatus && (
          <div className="simulation-status">
            {simulationStatus}
          </div>
        )}

        {/* Results Display */}
        {simulationResults && (
          <div className="results-container">
            <h2>🏁 Battle Results</h2>
            <div className="winner-display">
              {simulationResults.winner && (
                <div className="winner">
                  🏆 Winner: {simulationResults.winner.name} 
                  <span className="roi">ROI: {simulationResults.winner.roi?.toFixed(2)}%</span>
                </div>
              )}
            </div>
            <div className="leaderboard">
              <h3>📊 Leaderboard</h3>
              {simulationResults.leaderboard?.map((agent, index) => (
                <div key={agent.name} className={`leaderboard-item rank-${index + 1}`}>
                  <span className="rank">
                    {index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : `${index + 1}.`}
                  </span>
                  <span className="name">{agent.name}</span>
                  <span className="roi">{agent.roi?.toFixed(2)}%</span>
                  <span className="value">${agent.current_value?.toFixed(2)}</span>
                </div>
              ))}
              <div className="results-actions">
                <button className="back-button" onClick={handleBack}>Back</button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Algorithm Preview Modal */}
      <AlgorithmPreviewModal
        isOpen={showAlgoPreview}
        onClose={() => setShowAlgoPreview(false)}
      />
    </div>
  );
};

export default Dashboard;
