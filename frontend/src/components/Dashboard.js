import React, { useState, useEffect, useRef } from 'react';
import './Dashboard.css';
import CustomDropdown from './CustomDropdown'; // Import the new component

const Dashboard = () => {
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
  // Persistent algorithm previews and modal control
  const [previews, setPreviews] = useState({}); // { modelName: code }
  const [isPreviewOpen, setIsPreviewOpen] = useState(false);
  const [activePreviewModel, setActivePreviewModel] = useState('');
  // Trade aggressiveness control
  const [volumeMultiplier, setVolumeMultiplier] = useState(1);
  const [aggressiveMode, setAggressiveMode] = useState(false);

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
    setSimulationStatus('Starting simulation...');
  setShowInstructions(false);
  setProgress(0);
  // Initialize per-agent generation states
  const uniqueAgents = Array.from(new Set(agents));
  const initStates = {};
  uniqueAgents.forEach(name => { initStates[name] = 'pending'; });
  setGenStates(initStates);
  lastGeneratingRef.current = null;
  // Reset previews for a clean run
  setPreviews({});
  setActivePreviewModel('');

    try {
      const apiBase = process.env.REACT_APP_API_BASE_URL || '';
      const response = await fetch(`${apiBase}/api/run`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          agents: agents,
          stock: selectedStock,
          volume_multiplier: volumeMultiplier,
          aggressive_mode: aggressiveMode,
        })
      });

      const data = await response.json();
      
      if (response.ok) {
        const simId = data.simulation_id;
        setSimulationStatus('Simulation running...');
        
        // Poll for results
        pollSimulationStatus(simId);
      } else {
        throw new Error(data.error || 'Failed to start simulation');
      }
    } catch (error) {
      console.error('Simulation error:', error);
      setSimulationStatus(`Error: ${error.message}`);
      setIsRunning(false);
    }
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
        // keep last preview visible until user navigates back
      } else if (data.status === 'error') {
        setSimulationStatus(`Error: ${data.error}`);
        setIsRunning(false);
      } else {
        const pct = typeof data.progress === 'number' ? data.progress : 0;
        const message = data.message || `Running... ${pct}%`;
        setSimulationStatus(message);
        setProgress(pct);
        if (data.code_preview) {
          const incomingModel = String(data.preview_model || 'Unknown');
          const incomingCode = String(data.code_preview);
          // Only update when a new model preview arrives or content changes
          if (incomingModel !== lastPreviewModelRef.current || incomingCode !== codePreview) {
            setCodePreview(incomingCode);
            setCodePreviewModel(incomingModel);
            lastPreviewModelRef.current = incomingModel;
            // Persist into previews map for modal access
            setPreviews(prev => ({ ...prev, [incomingModel]: incomingCode }));
            if (!activePreviewModel) {
              setActivePreviewModel(incomingModel);
            }
          }
        }
        // Parse and reflect generation progress per model if present
        parseProgressMessage(message);
        // Continue polling
        setTimeout(() => pollSimulationStatus(simId), 2000);
      }
    } catch (error) {
      console.error('Polling error:', error);
      setSimulationStatus(`Polling error: ${error.message}`);
      setIsRunning(false);
    }
  };

  // Interpret backend status messages to maintain per-agent generation states
  const parseProgressMessage = (message = '') => {
    setCurrentTask(message);
    // Pattern: "Generating algorithm X/6 using <model>..."
    const genMatch = message.match(/Generating algorithm\s+(\d+)\/(\d+)\s+using\s+(.+?)\.\.\./i);
    if (genMatch) {
      const model = genMatch[3];
      const prev = lastGeneratingRef.current;
      if (prev && prev !== model) {
        setGenStates(prevState => ({ ...prevState, [prev]: 'done' }));
      }
      lastGeneratingRef.current = model;
      setGenStates(prevState => ({ ...prevState, [model]: 'generating' }));
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
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Preview modal derived values and actions
  const previewModels = Object.keys(previews);
  const activeCode = activePreviewModel ? previews[activePreviewModel] : '';

  const handleCopyActive = async () => {
    try {
      if (activeCode) await navigator.clipboard.writeText(activeCode);
    } catch (e) {
      console.error('Copy failed:', e);
    }
  };

  const handleDownloadActive = () => {
    try {
      if (!activeCode) return;
      const blob = new Blob([activeCode], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      const safe = (activePreviewModel || 'algorithm').replace(/[^a-z0-9_-]+/gi, '_');
      a.href = url;
      a.download = `${safe}.py`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error('Download failed:', e);
    }
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

      {/* Top controls below navbar: Stock selector */}
      <div className="top-controls">
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
          {showInstructions && !isRunning ? (
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
                  Click <strong>START</strong> to generate strategies and run a ~60‑tick market battle.
                </li>
                <li>
                  Watch the status at the bottom; when finished, view the winner and leaderboard.
                </li>
              </ol>

              <div className="tips">
                <span className="tip-title">Tips</span>
                <ul>
                  <li>All agents must be selected before starting.</li>
                  <li>Mix models from different providers for diverse tactics.</li>
                  <li>Internet is required for market data and model generation.</li>
                </ul>
              </div>
              {/* Volume control */}
              <div className="volume-control">
                <label htmlFor="volume-mult">Trade size</label>
                <input
                  id="volume-mult"
                  type="range"
                  min="0.5"
                  max="5"
                  step="0.5"
                  value={volumeMultiplier}
                  onChange={(e) => setVolumeMultiplier(parseFloat(e.target.value))}
                />
                <span className="volume-value">{volumeMultiplier.toFixed(1)}x</span>
              </div>

              {/* Aggressive mode toggle */}
              <div className="aggressive-toggle">
                <label className="switch">
                  <input
                    type="checkbox"
                    checked={aggressiveMode}
                    onChange={(e) => setAggressiveMode(e.target.checked)}
                  />
                  <span className="slider round"></span>
                </label>
                <span className={`toggle-label ${aggressiveMode ? 'on' : ''}`}>Aggressive mode</span>
              </div>
            </div>
          ) : (
            <div className="progress-panel">
              <h2>Simulation Status</h2>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${Math.min(100, Math.max(0, progress))}%` }} />
              </div>
              <div className="current-task">{currentTask || simulationStatus || 'Preparing...'}</div>

              {/* Per-agent generation states */}
              <div className="gen-list">
                {Object.keys(genStates).length > 0 ? (
                  Object.entries(genStates).map(([name, state]) => (
                    <div key={name} className={`gen-item ${state}`}>
                      <span className={`status ${state}`}>
                        {state === 'done' ? '✔' : state === 'generating' ? '⏳' : '•'}
                      </span>
                      <span className="model-name">{name}</span>
                    </div>
                  ))
                ) : (
                  <div className="gen-empty">Waiting for algorithm generation…</div>
                )}
              </div>

              {/* Live code preview during generation */}
              {codePreview && (
                <div className="code-preview">
                  <div className="code-preview-header">
                    Preview {codePreviewModel ? `from ${codePreviewModel}` : ''}
                  </div>
                  <pre><code>{codePreview}</code></pre>
                </div>
              )}
            </div>
          )}
        </div>


        {/* Controls Row */}
        <button 
          className={`start-button ${isRunning ? 'running' : ''}`}
          onClick={handleStartSimulation}
          disabled={isRunning}
        >
          {isRunning ? 'RUNNING...' : 'START'}
        </button>

        <button
          className={`preview-button ${previewModels.length ? '' : 'disabled'}`}
          onClick={() => setIsPreviewOpen(true)}
          disabled={!previewModels.length}
          title={previewModels.length ? 'Preview generated algorithms' : 'No previews yet'}
        >
          Preview Algorithms
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
        {/* Preview Modal */}
        {isPreviewOpen && (
          <div className="preview-modal-overlay" onClick={() => setIsPreviewOpen(false)}>
            <div className="preview-modal" onClick={(e) => e.stopPropagation()}>
              <div className="preview-modal-header">
                <span>Algorithm Preview</span>
                <button className="preview-close" onClick={() => setIsPreviewOpen(false)}>✕</button>
              </div>
              <div className="preview-modal-body">
                {previewModels.length === 0 ? (
                  <div className="preview-empty">No previews yet. Start a simulation to generate code.</div>
                ) : (
                  <>
                    <div className="preview-tabs">
                      {previewModels.map(m => (
                        <button
                          key={m}
                          className={`preview-tab ${m === activePreviewModel ? 'active' : ''}`}
                          onClick={() => setActivePreviewModel(m)}
                        >
                          {m}
                        </button>
                      ))}
                    </div>
                    <div className="preview-actions">
                      <button onClick={handleCopyActive} disabled={!activeCode}>Copy</button>
                      <button onClick={handleDownloadActive} disabled={!activeCode}>Download</button>
                    </div>
                    <div className="preview-code">
                      {activeCode ? (
                        <pre><code>{activeCode}</code></pre>
                      ) : (
                        <div className="preview-empty">Select a model to view its algorithm.</div>
                      )}
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        )}
    </div>
  );
};

export default Dashboard;
