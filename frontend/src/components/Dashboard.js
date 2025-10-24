import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './Dashboard.css';
import CustomDropdown from './CustomDropdown';
import AlgorithmCard from './AlgorithmCard';
import ResultsDashboard from './ResultsDashboard';
import AllAlgorithmsModal from './AllAlgorithmsModal';
import ModelCard from './ModelCard';
import AlgorithmPreviewModal from './AlgorithmPreviewModal';

const Dashboard = () => {
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
  const [progress, setProgress] = useState(0);
  const [currentTask, setCurrentTask] = useState('');
  const [genStates, setGenStates] = useState({}); // { modelName: 'pending'|'generating'|'done' }
  const lastGeneratingRef = useRef(null);
  const [codePreview, setCodePreview] = useState('');
  const [codePreviewModel, setCodePreviewModel] = useState('');
  const lastPreviewModelRef = useRef('');
  const [generationPhase, setGenerationPhase] = useState('idle'); // 'idle', 'generating', 'review', 'simulating', 'completed'
  const [generatedAlgos, setGeneratedAlgos] = useState([]);
  const [currentGenId, setCurrentGenId] = useState(null);
  const pollingRef = useRef(null);
  const [showAllAlgos, setShowAllAlgos] = useState(false);

  // New state for preview modal
  const [previewModalOpen, setPreviewModalOpen] = useState(false);
  const [selectedModelPreview, setSelectedModelPreview] = useState(null);

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

  // Handle model card click to open preview modal
  const handleCardClick = (modelName, code) => {
    if (!code) return;
    setSelectedModelPreview({ modelName, code });
    setPreviewModalOpen(true);
  };

  // Close preview modal
  const closePreviewModal = () => {
    setPreviewModalOpen(false);
    setTimeout(() => setSelectedModelPreview(null), 300);
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
    setProgress(0);
    setGenerationPhase('generating');

    // Initialize per-agent generation states
    const uniqueAgents = Array.from(new Set(agents));
    const initStates = {};
    uniqueAgents.forEach(name => { initStates[name] = 'pending'; });
    setGenStates(initStates);
    lastGeneratingRef.current = null;

    // Initialize algorithm cards with pending state
    const initialAlgos = uniqueAgents.map((name, idx) => ({
      modelName: name,
      code: '',
      status: 'generating',
      index: idx
    }));
    setGeneratedAlgos(initialAlgos);

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
        setCurrentGenId(genId);
        setSimulationStatus('Algorithms generating...');

        // Start polling for generation status (stay on same page)
        pollGenerationStatus(genId);
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

  const normalizeModelId = (s = '') => String(s).toLowerCase().trim().replace(/:free$/,'');

  const pollGenerationStatus = async (genId) => {
    try {
      const apiBase = process.env.REACT_APP_API_BASE_URL || '';
      const response = await fetch(`${apiBase}/api/generation/${genId}`);
      const data = await response.json();

  if (data.status === 'completed') {
        // All algorithms generated successfully
        setProgress(50);
        setSimulationStatus('All algorithms generated!');

        // Update algorithm cards with actual code
        const algorithms = data.algorithms || {};
        // Build a normalized lookup for algorithm code by model id (strip :free, lowercase)
        const algoLookup = Object.entries(algorithms).reduce((acc, [k, v]) => {
          acc[normalizeModelId(k)] = v;
          return acc;
        }, {});
        setGeneratedAlgos(prev => {
          // If prev list is empty (e.g., previous state not initialized), seed from algorithms keys
          let base = prev && prev.length ? [...prev] : Object.keys(algorithms).map((k, idx) => ({
            modelName: k,
            code: algorithms[k] || '',
            status: algorithms[k] ? 'completed' : 'failed',
            index: idx
          }));

          // Update any existing entries by normalized id
          const seen = new Set(base.map(a => normalizeModelId(a.modelName)));
          base = base.map(algo => {
            const code = algoLookup[normalizeModelId(algo.modelName)] || '';
            return { ...algo, code, status: code ? 'completed' : 'failed' };
          });

          // Append any new algorithms that were not in base
          Object.entries(algorithms).forEach(([k, v]) => {
            const nk = normalizeModelId(k);
            if (!seen.has(nk)) {
              base.push({ modelName: k, code: v || '', status: v ? 'completed' : 'failed', index: base.length });
            }
          });
          return base;
        });

        // Mark all as done in genStates
        setGenStates(prev => {
          const next = { ...prev };
          Object.keys(next).forEach(k => { next[k] = 'done'; });
          return next;
        });

        // Transition to review phase (stay inline on Dashboard)
        setGenerationPhase('review');
        setIsRunning(false);
      } else if (data.status === 'error') {
        setSimulationStatus(`Error: ${data.error}`);
        setIsRunning(false);
        setGenerationPhase('idle');
  } else {
        // Still generating
        const pct = typeof data.progress === 'number' ? data.progress : 0;
        const message = data.message || 'Generating algorithms...';
        setSimulationStatus(message);
        setProgress(Math.min(45, pct)); // Cap at 45% during generation

        // Update algorithm codes as they arrive
        if (data.algorithms) {
          const algoLookup = Object.entries(data.algorithms).reduce((acc, [k, v]) => {
            acc[normalizeModelId(k)] = v;
            return acc;
          }, {});
          setGeneratedAlgos(prev => {
            const list = prev && prev.length ? [...prev] : [];
            const existing = new Set(list.map(a => normalizeModelId(a.modelName)));
            // Update existing
            const updated = list.map(algo => {
              const code = algoLookup[normalizeModelId(algo.modelName)];
              return { ...algo, code: code || algo.code, status: code ? 'completed' : algo.status };
            });
            // Append new ones that arrived but weren't seeded yet
            Object.entries(data.algorithms).forEach(([k, v]) => {
              const nk = normalizeModelId(k);
              if (!existing.has(nk)) {
                updated.push({ modelName: k, code: v || '', status: v ? 'completed' : 'generating', index: updated.length });
              }
            });
            return updated;
          });
        }

        // Update per-model generation states from backend if available (supports concurrency)
        if (data.model_states && typeof data.model_states === 'object') {
          setGenStates(prev => {
            const next = { ...prev };
            Object.entries(data.model_states).forEach(([model, state]) => {
              next[model] = state === 'generating' ? 'generating' : (state === 'done' ? 'done' : state);
            });
            return next;
          });
        } else {
          // Fallback to parsing human-readable message
          parseProgressMessage(message);
        }


        // Continue polling
        pollingRef.current = setTimeout(() => pollGenerationStatus(genId), 2000);
      }
    } catch (error) {
      console.error('Generation polling error:', error);
      setSimulationStatus(`Polling error: ${error.message}`);
      setIsRunning(false);
      setGenerationPhase('idle');
    }
  };

  const handleStartMarketSimulation = async () => {
    if (!currentGenId) return;

    setGenerationPhase('simulating');
    setSimulationStatus('Starting market simulation...');
    setProgress(50);
    setIsRunning(true);

    try {
      const apiBase = process.env.REACT_APP_API_BASE_URL || '';
      const response = await fetch(`${apiBase}/api/simulate/${currentGenId}`, {
        method: 'POST'
      });

      const data = await response.json();

      if (response.ok) {
        const simId = data.simulation_id;
        // Start polling for simulation results
        pollSimulationStatus(simId);
      } else {
        throw new Error(data.error || 'Failed to start simulation');
      }
    } catch (error) {
      console.error('Simulation start error:', error);
      setSimulationStatus(`Error: ${error.message}`);
      setIsRunning(false);
      setGenerationPhase('review');
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
        setGenerationPhase('completed');
      } else if (data.status === 'error') {
        setSimulationStatus(`Error: ${data.error}`);
        setIsRunning(false);
        setGenerationPhase('simulating');
      } else {
        // Scale progress from 50-100% during simulation
        const pct = typeof data.progress === 'number' ? data.progress : 50;
        const scaledProgress = 50 + (pct / 2); // 0-100% becomes 50-100%
        const message = data.message || 'Running simulation...';
        setSimulationStatus(message);
        setProgress(Math.min(100, scaledProgress));
        setCurrentTask(message);

        // Continue polling
        pollingRef.current = setTimeout(() => pollSimulationStatus(simId), 2000);
      }
    } catch (error) {
      console.error('Simulation polling error:', error);
      setSimulationStatus(`Polling error: ${error.message}`);
      setIsRunning(false);
      setGenerationPhase('review');
    }
  };

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollingRef.current) {
        clearTimeout(pollingRef.current);
      }
    };
  }, []);


  // Interpret backend status messages to maintain per-agent generation states
  const parseProgressMessage = (message = '') => {
    setCurrentTask(message);
    // Pattern: "Generating algorithm X/6 using <model>..."
    const genMatch = message.match(/Generating algorithm\s+(\d+)\/(\d+)\s+using\s+(.+?)\.\.\./i);
    if (genMatch) {
      const model = genMatch[3];

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
    // Clear any ongoing polling
    if (pollingRef.current) {
      clearTimeout(pollingRef.current);
      pollingRef.current = null;
    }

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
    setGenerationPhase('idle');
    setGeneratedAlgos([]);
    setCurrentGenId(null);

    // Reset modal state
    setPreviewModalOpen(false);
    setSelectedModelPreview(null);

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

      {/* Top controls below navbar: Stock selector */}
      <div className="top-controls">
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
        <AnimatePresence>
          {generationPhase === 'idle' && [1, 2, 3].map(i => {
            const leftKey = `left-${i}`;
            const rightKey = `right-${i}`;
            const values = Object.values(selectedAgents);
            const disabledLeft = new Set(values.filter(a => a && a !== selectedAgents[leftKey]));
            const disabledRight = new Set(values.filter(a => a && a !== selectedAgents[rightKey]));
            return (
              <React.Fragment key={i}>
                <motion.div
                  className={`side-element left-element left-element-${i}`}
                  layoutId={selectedAgents[leftKey] ? normalizeModelId(selectedAgents[leftKey]) : undefined}
                  initial={{ opacity: 1 }}
                  exit={{ opacity: 0, scale: 0.95, transition: { duration: 0.2 } }}
                >
                  <div className="side-circle left-circle">
                    <CustomDropdown
                      agents={agents}
                      selected={selectedAgents[leftKey]}
                      onSelect={(agent) => handleAgentSelect(leftKey, agent)}
                      disabledAgents={disabledLeft}
                    />
                  </div>
                </motion.div>
                <motion.div
                  className={`side-element right-element right-element-${i}`}
                  layoutId={selectedAgents[rightKey] ? normalizeModelId(selectedAgents[rightKey]) : undefined}
                  initial={{ opacity: 1 }}
                  exit={{ opacity: 0, scale: 0.95, transition: { duration: 0.2 } }}
                >
                  <div className="side-circle right-circle">
                    <CustomDropdown
                      agents={agents}
                      selected={selectedAgents[rightKey]}
                      onSelect={(agent) => handleAgentSelect(rightKey, agent)}
                      disabledAgents={disabledRight}
                    />
                  </div>
                </motion.div>
              </React.Fragment>
            );
          })}
        </AnimatePresence>
      
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

              {/* Model Cards Grid inside progress panel */}
              <div className="model-cards-in-progress">
                {Array.from(new Set(Object.values(selectedAgents).filter(Boolean))).map((agent, idx) => {
                  const algo = generatedAlgos.find(a => normalizeModelId(a.modelName) === normalizeModelId(agent));
                  const status = algo?.status === 'completed' ? 'success' : 'generating';
                  const code = algo?.code || '';

                  return (
                    <ModelCard
                      key={agent}
                      modelId={normalizeModelId(agent)}
                      modelName={agent}
                      status={status}
                      index={idx}
                      onClick={() => handleCardClick(agent, code)}
                    />
                  );
                })}
              </div>
            </div>
          ) : generationPhase === 'review' ? (
            /* Review & Checkpoint Phase */
            <div className="review-panel">
              <h2>✅ Algorithms Generated Successfully!</h2>
              <p className="review-subtitle">
                These are the algos generated by the agents. You can review them if you want. To continue, press the below button "MARKET SIMULATION".
              </p>

              <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '12px' }}>
                <button
                  className="preview-btn"
                  onClick={() => setShowAllAlgos(true)}
                >
                  View All Full Algorithms
                </button>
              </div>

              <div className="algorithms-container">
                {generatedAlgos.length === 0 ? (
                  <div className="gen-empty">No algorithms received yet. If this persists, check API key access and try again.</div>
                ) : (
                  generatedAlgos.map((algo, idx) => (
                    <AlgorithmCard
                      key={algo.modelName}
                      modelName={algo.modelName}
                      code={algo.code}
                      index={idx}
                      status={algo.status}
                    />
                  ))
                )}
              </div>

              <button
                className="start-market-sim-button"
                onClick={handleStartMarketSimulation}
                disabled={generatedAlgos.some(a => a.status !== 'completed')}
                title={generatedAlgos.some(a => a.status !== 'completed') ? 'Waiting for all algorithms to complete...' : 'Start market simulation'}
                aria-disabled={generatedAlgos.some(a => a.status !== 'completed')}
              >
                🚀 MARKET SIMULATION
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


        {/* Start Button - only show when idle */}
        {generationPhase === 'idle' && (
          <button
            className={`start-button ${isRunning ? 'running' : ''}`}
            onClick={handleStartSimulation}
            disabled={isRunning}
          >
            {isRunning ? 'GENERATING...' : 'GENERATE ALGORITHMS'}
          </button>
        )}

        {/* Simulation Status */}
        {simulationStatus && (
          <div className="simulation-status">
            {simulationStatus}
          </div>
        )}

        {/* Results Display */}
        {simulationResults && generationPhase === 'completed' && (
          <div className="results-overlay">
            <div className="results-modal">
              <ResultsDashboard results={simulationResults} onBack={handleBack} />
            </div>
          </div>
        )}

        {/* All Algorithms Modal */}
        <AllAlgorithmsModal
          isOpen={showAllAlgos}
          onClose={() => setShowAllAlgos(false)}
          algorithms={generatedAlgos}
        />

        {/* Single Algorithm Preview Modal */}
        <AlgorithmPreviewModal
          isOpen={previewModalOpen}
          onClose={closePreviewModal}
          modelName={selectedModelPreview?.modelName || ''}
          code={selectedModelPreview?.code || ''}
        />
      </div>

      {/* Inline review only; external preview modal removed */}
    </div>
  );
};

export default Dashboard;
