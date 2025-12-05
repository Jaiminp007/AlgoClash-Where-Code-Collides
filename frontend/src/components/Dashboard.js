import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './Dashboard.css';
import CustomDropdown from './CustomDropdown';
import AlgorithmCard from './AlgorithmCard';
import AllAlgorithmsModal from './AllAlgorithmsModal';
import ModelCard from './ModelCard';
import AlgorithmPreviewModal from './AlgorithmPreviewModal';
import ReplaceAgentModal from './ReplaceAgentModal';
import MarketSimulationChart from './MarketSimulationChart';

// Import provider icons
import googlePng from '../assets/google.png';
import anthropicPng from '../assets/anthropic.png';
import openaiPng from '../assets/openai.png';
import metaPng from '../assets/meta.png';
import qwenPng from '../assets/qwen.png';
import mistralPng from '../assets/mistral.png';
import deepseekPng from '../assets/deepseek.png';
import nousresearchPng from '../assets/nousresearch.png';
import agenticaPng from '../assets/agentica.png';
import moonshotaiPng from '../assets/moonshotai.png';
import openrouterPng from '../assets/openrouter.png';
import grokPng from '../assets/grok.png';
import alibabaPng from '../assets/alibaba.png';
import arliaiPng from '../assets/arliai.png';
import cognitivecomputationsPng from '../assets/cognitivecomputations.png';
import meituanPng from '../assets/meituan.png';
import microsoftPng from '../assets/microsoft.png';
import nvidiaPng from '../assets/nvidia.png';
import shisaaiPng from '../assets/shisaai.png';
import tencentPng from '../assets/tencent.png';
import tngtechPng from '../assets/tngtech.png';
import zaiPng from '../assets/zai.png';

const providerIcons = {
  google: googlePng,
  anthropic: anthropicPng,
  openai: openaiPng,
  meta: metaPng,
  qwen: qwenPng,
  mistral: mistralPng,
  mistralai: mistralPng,
  deepseek: deepseekPng,
  nousresearch: nousresearchPng,
  agentica: agenticaPng,
  moonshotai: moonshotaiPng,
  openrouter: openrouterPng,
  grok: grokPng,
  'x-ai': grokPng,
  alibaba: alibabaPng,
  arliai: arliaiPng,
  cognitivecomputations: cognitivecomputationsPng,
  meituan: meituanPng,
  microsoft: microsoftPng,
  nvidia: nvidiaPng,
  shisaai: shisaaiPng,
  'shisa-ai': shisaaiPng,
  tencent: tencentPng,
  tngtech: tngtechPng,
  zai: zaiPng,
  'z-ai': zaiPng,
};

// Parse agent name to extract provider and model
const parseAgentName = (fullName) => {
  const withoutPrefix = fullName.replace(/^generated_algo_/, '');
  const parts = withoutPrefix.split('_');

  if (parts.length === 0) return { provider: '', model: fullName, displayName: fullName };

  const provider = parts[0].toLowerCase();
  const modelParts = parts.slice(1);
  const formattedModel = modelParts
    .map((part, idx) => {
      const prevPart = idx > 0 ? modelParts[idx - 1] : '';
      if (/^\d+$/.test(part) && /^\d+$/.test(prevPart)) {
        return `.${part}`;
      }
      return part.charAt(0).toUpperCase() + part.slice(1);
    })
    .join(' ')
    .replace(/\s+\./g, '.')
    .replace(/_free$/i, '')
    .replace(/\s+Free$/i, '');

  return { provider, model: formattedModel, displayName: formattedModel };
};

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
  const [currentGenId, setCurrentGenId] = useState(() => {
    // Initialize from sessionStorage if available
    return sessionStorage.getItem('currentGenId') || null;
  });
  const pollingRef = useRef(null);
  const [showAllAlgos, setShowAllAlgos] = useState(false);
  const [chartData, setChartData] = useState([]);

  // New state for preview modal
  const [previewModalOpen, setPreviewModalOpen] = useState(false);
  const [selectedModelPreview, setSelectedModelPreview] = useState(null);

  // State for replace agent modal
  const [replaceModalOpen, setReplaceModalOpen] = useState(false);
  const [modelToReplace, setModelToReplace] = useState(null);


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

  // Handle replace agent
  const handleReplaceAgent = (failedModel) => {
    console.log('handleReplaceAgent called:', { failedModel, currentGenId, generationPhase });
    setModelToReplace(failedModel);
    setReplaceModalOpen(true);
  };

  // Close replace modal
  const closeReplaceModal = () => {
    setReplaceModalOpen(false);
    setTimeout(() => setModelToReplace(null), 300);
  };

  // Handle agent replacement
  const handleAgentReplacement = async (newAgent) => {
    console.log('handleAgentReplacement called with:', { newAgent, modelToReplace, currentGenId });

    if (!currentGenId) {
      alert('Error: No generation ID found. Please restart the generation process.');
      return;
    }

    if (!modelToReplace) {
      alert('Error: No model to replace specified.');
      return;
    }

    try {
      const apiBase = process.env.REACT_APP_API_BASE_URL || '';
      const url = `${apiBase}/api/generation/${currentGenId}/regenerate`;

      console.log('Calling regenerate endpoint:', url);
      console.log('Payload:', { old_model: modelToReplace, new_model: newAgent });

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          old_model: modelToReplace,
          new_model: newAgent
        })
      });

      // Check if response is JSON
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        throw new Error('Backend returned non-JSON response. Please ensure the Flask backend is running and restart it if you just added the regenerate endpoint.');
      }

      const data = await response.json();

      if (response.ok) {
        // Update the currentSessionAgents list
        const currentSessionAgents = JSON.parse(sessionStorage.getItem('currentSessionAgents') || '[]');
        const updatedAgents = currentSessionAgents.map(a =>
          normalizeModelId(a) === normalizeModelId(modelToReplace) ? newAgent : a
        );
        sessionStorage.setItem('currentSessionAgents', JSON.stringify(updatedAgents));
        console.log('Updated session agents:', updatedAgents);

        // Update the generatedAlgos state
        setGeneratedAlgos(prev => prev.map(algo =>
          normalizeModelId(algo.modelName) === normalizeModelId(modelToReplace)
            ? { ...algo, modelName: newAgent, status: 'generating', code: '' }
            : algo
        ));

        // Update genStates
        setGenStates(prev => {
          const next = { ...prev };
          delete next[modelToReplace];
          next[newAgent] = 'generating';
          return next;
        });

        closeReplaceModal();

        // Always restart polling to track the regeneration progress
        console.log('Restarting polling for replaced agent regeneration...');
        // Clear any existing polling first
        if (pollingRef.current) {
          clearTimeout(pollingRef.current);
          pollingRef.current = null;
        }
        // Start new polling
        pollGenerationStatus(currentGenId);
      } else {
        alert(`Failed to replace agent: ${data.error}`);
      }
    } catch (error) {
      console.error('Replace agent error:', error);
      alert(`Error replacing agent: ${error.message}\n\nIf you just added this feature, please restart the Flask backend server.`);
    }
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

    // Initialize algorithm cards with pending state - ONLY for current session
    const initialAlgos = uniqueAgents.map((name, idx) => ({
      modelName: name,
      code: '',
      status: 'generating',
      index: idx
    }));
    // Store the selected agents in sessionStorage to track current session
    sessionStorage.setItem('currentSessionAgents', JSON.stringify(uniqueAgents));
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
        console.log('Generation started with ID:', genId);
        setCurrentGenId(genId);
        // Persist to sessionStorage so it survives re-renders
        sessionStorage.setItem('currentGenId', genId);
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
        // Get the current session's agents
        const currentSessionAgents = JSON.parse(sessionStorage.getItem('currentSessionAgents') || '[]');
        const currentSessionSet = new Set(currentSessionAgents.map(a => normalizeModelId(a)));

        // Update algorithm cards with actual code - ONLY for current session agents
        const algorithms = data.algorithms || {};
        const model_states = data.model_states || {};

        // Build a normalized lookup for algorithm code by model id (strip :free, lowercase)
        // Filter to only include algorithms for current session agents
        const algoLookup = Object.entries(algorithms).reduce((acc, [k, v]) => {
          if (currentSessionSet.has(normalizeModelId(k))) {
            acc[normalizeModelId(k)] = v;
          }
          return acc;
        }, {});

        // Build a normalized lookup for model states
        const statesLookup = Object.entries(model_states).reduce((acc, [k, v]) => {
          acc[normalizeModelId(k)] = v;
          return acc;
        }, {});

        // Check if any current session agent is still generating
        const stillGenerating = currentSessionAgents.some(agent =>
          statesLookup[normalizeModelId(agent)] === 'generating'
        );

        console.log('Generation status check:', {
          currentSessionAgents,
          model_states,
          statesLookup,
          stillGenerating
        });

        if (stillGenerating) {
          // Some algorithms are still generating, continue polling
          console.log('⏳ Some algorithms still generating, continuing to poll...');
          setSimulationStatus('Generating algorithms...');
          pollingRef.current = setTimeout(() => pollGenerationStatus(genId), 2000);

          // Update what we have so far
          setGeneratedAlgos(prev => prev.map(algo => {
            const normId = normalizeModelId(algo.modelName);
            const code = algoLookup[normId] || algo.code;
            const state = statesLookup[normId];
            const status = state === 'done' ? 'completed' : (state === 'error' ? 'failed' : 'generating');
            return { ...algo, code, status };
          }));

          return; // Don't transition to review yet
        }

        // All algorithms generated successfully
        setProgress(50);
        setSimulationStatus('All algorithms generated!');

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
            const normId = normalizeModelId(algo.modelName);
            const code = algoLookup[normId] || '';
            const state = statesLookup[normId];
            // Use model_states to determine status, fallback to code-based check
            const status = state === 'done' ? 'completed' : (state === 'error' ? 'failed' : (code ? 'completed' : 'failed'));
            return { ...algo, code, status };
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

        // Stay in generating phase (don't transition to review)
        // setGenerationPhase('review'); // REMOVED - stay in generating phase
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
        if (data.algorithms || data.failures) {
          // Get the current session's agents
          const currentSessionAgents = JSON.parse(sessionStorage.getItem('currentSessionAgents') || '[]');
          const currentSessionSet = new Set(currentSessionAgents.map(a => normalizeModelId(a)));

          // Filter to only include algorithms for current session agents
          const algoLookup = Object.entries(data.algorithms || {}).reduce((acc, [k, v]) => {
            if (currentSessionSet.has(normalizeModelId(k))) {
              acc[normalizeModelId(k)] = v;
            }
            return acc;
          }, {});
          
          // Lookup failures
          const failureLookup = Object.entries(data.failures || {}).reduce((acc, [k, v]) => {
             acc[normalizeModelId(k)] = v;
             return acc;
          }, {});

          setGeneratedAlgos(prev => {
            const list = prev && prev.length ? [...prev] : [];
            const existing = new Set(list.map(a => normalizeModelId(a.modelName)));
            // Update existing
            const updated = list.map(algo => {
              const normId = normalizeModelId(algo.modelName);
              const code = algoLookup[normId];
              const failure = failureLookup[normId];
              
              let status = algo.status;
              if (code) status = 'completed';
              else if (failure) status = 'failed';
              
              return { ...algo, code: code || algo.code, status: status };
            });
            // Append new ones that arrived but weren't seeded yet
            Object.entries(data.algorithms || {}).forEach(([k, v]) => {
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
    setChartData([]); // Reset chart data for new simulation

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
      setGenerationPhase('generating'); // Go back to generating phase (not review)
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
        // Update chart data one last time
        if (data.chart_data) {
          setChartData(data.chart_data);
        }
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

        // Update chart data if available
        if (data.chart_data) {
          setChartData(data.chart_data);
        }

        // Continue polling
        pollingRef.current = setTimeout(() => pollSimulationStatus(simId), 2000);
      }
    } catch (error) {
      console.error('Simulation polling error:', error);
      setSimulationStatus(`Polling error: ${error.message}`);
      setIsRunning(false);
      setGenerationPhase('generating'); // Go back to generating phase (not review)
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

    // Call cleanup API to remove generated algorithm files
    // (algorithms are already saved in MongoDB)
    if (currentGenId) {
      fetch(`http://localhost:5000/api/cleanup/${currentGenId}`, {
        method: 'POST',
      })
        .then(response => response.json())
        .then(data => {
          console.log('Cleanup result:', data);
        })
        .catch(error => {
          console.error('Cleanup error:', error);
        });
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
    setChartData([]); // Reset chart data
    // Clear persisted generation ID
    sessionStorage.removeItem('currentGenId');

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
        <div className={`center-box ${generationPhase === 'generating' ? 'expanded' : ''} ${generationPhase === 'review' ? 'review-expanded' : ''} ${generationPhase === 'simulating' || generationPhase === 'completed' ? 'simulation-expanded' : ''}`}>
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
              <div className="current-task">{currentTask || simulationStatus || 'Starting algorithm generation...'}</div>

              {/* Model Cards Grid inside progress panel */}
              <div className="model-cards-in-progress">
                {generatedAlgos.map((algo, idx) => {
                  const status = algo?.status === 'completed' ? 'success' : (algo?.status === 'failed' ? 'error' : 'generating');
                  const code = algo?.code || '';

                  return (
                    <ModelCard
                      key={algo.modelName}
                      modelId={normalizeModelId(algo.modelName)}
                      modelName={algo.modelName}
                      status={status}
                      index={idx}
                      onClick={() => handleCardClick(algo.modelName, code)}
                      onReplace={status === 'error' ? handleReplaceAgent : null}
                    />
                  );
                })}
              </div>

              {/* Show Market Simulation button when all algorithms are complete */}
              {!isRunning && generatedAlgos.length > 0 && generatedAlgos.every(a => a.status === 'completed' || a.status === 'failed') && (
                <div style={{ marginTop: '24px', display: 'flex', justifyContent: 'center' }}>
                  <button
                    className="start-market-sim-button"
                    onClick={handleStartMarketSimulation}
                    disabled={generatedAlgos.some(a => a.status !== 'completed')}
                    title={generatedAlgos.some(a => a.status !== 'completed') ? 'Some algorithms failed. Replace them before continuing.' : 'Start market simulation'}
                  >
                    🚀 START MARKET SIMULATION
                  </button>
                </div>
              )}
            </div>
          ) : generationPhase === 'review' ? (
            /* Review & Checkpoint Phase */
            <div className="review-panel">
              <h2>✅ Algorithms Generated Successfully!</h2>
              <p className="review-subtitle">
                These are the algos generated by the agents. You can review them if you want. To continue, press the below button "MARKET SIMULATION".
              </p>
              {/* Debug info - remove after testing */}
              <div style={{ fontSize: '0.7rem', color: '#666', marginBottom: '6px', textAlign: 'center' }}>
                Gen ID: {currentGenId || 'NOT SET'}
              </div>

              <div className="review-actions-header" style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <button
                  className="preview-btn"
                  onClick={() => setShowAllAlgos(true)}
                >
                  View All Algorithms
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
                      onReplace={algo.status === 'failed' ? handleReplaceAgent : null}
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

              {/* Real-time Market Chart */}
              <MarketSimulationChart chartData={chartData} />
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

        {/* Results Display - Side by Side Layout */}
        {simulationResults && generationPhase === 'completed' && (
          <div className="results-overlay">
            <div className="results-modal-wide">
              {/* Winner Banner at Top */}
              {simulationResults.winner && (
                <div className="winner-banner">
                  <div className="trophy-large">🏆</div>
                  <div className="winner-info-banner">
                    <div className="winner-label">WINNER</div>
                    <div className="winner-name-large">{simulationResults.winner.name}</div>
                    <div className="winner-roi-large">
                      ROI: <span className={simulationResults.winner.roi >= 0 ? 'positive' : 'negative'}>
                        {simulationResults.winner.roi >= 0 ? '+' : ''}{((simulationResults.winner.roi || 0) * 100).toFixed(2)}%
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Side by Side Content */}
              <div className="results-content-split">
                {/* Left: Chart */}
                <div className="results-chart-container">
                  {chartData && chartData.length > 0 && (
                    <MarketSimulationChart chartData={chartData} hideAgentCards={true} />
                  )}
                </div>

                {/* Right: Leaderboard styled as agent cards */}
                <div className="results-leaderboard-container">
                  <h3 className="leaderboard-title">📊 Final Rankings</h3>
                  <div className="leaderboard-cards">
                    {simulationResults.leaderboard && simulationResults.leaderboard.map((agent, index) => {
                      const { provider, displayName } = parseAgentName(agent.name);
                      const icon = providerIcons[provider];
                      const initialCash = agent.initial_value || 10000;
                      const finalCash = agent.cash || 0;
                      const initialStock = agent.initial_stock || 0;
                      const finalStock = agent.stock || 0;
                      const roi = ((agent.roi || 0) * 100).toFixed(2);

                      return (
                        <div key={agent.name} className={`leaderboard-card rank-${index + 1}`}>
                          {/* Header with Rank and Agent Info */}
                          <div className="leaderboard-rank-header">
                            <div className="agent-card-header">
                              {icon && <img src={icon} alt={provider} className="agent-provider-icon-small" />}
                              <div className="agent-card-name">{displayName}</div>
                            </div>
                            <div className="rank-badge">
                              {index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : `#${index + 1}`}
                            </div>
                          </div>

                          {/* Body with Stats */}
                          <div className="leaderboard-card-body">
                            <div className="leaderboard-stat-row">
                              <span className="stat-label-lb">ROI</span>
                              <span className={`stat-value-lb-large ${agent.roi >= 0 ? 'positive' : 'negative'}`}>
                                {agent.roi >= 0 ? '+' : ''}{roi}%
                              </span>
                            </div>
                            <div className="leaderboard-stat-row">
                              <span className="stat-label-lb">Initial Cash</span>
                              <span className="stat-value-lb">${initialCash.toLocaleString()}</span>
                            </div>
                            <div className="leaderboard-stat-row">
                              <span className="stat-label-lb">Final Cash</span>
                              <span className="stat-value-lb">${finalCash.toLocaleString()}</span>
                            </div>
                            <div className="leaderboard-stat-row">
                              <span className="stat-label-lb">Initial Stock</span>
                              <span className="stat-value-lb">{initialStock}</span>
                            </div>
                            <div className="leaderboard-stat-row">
                              <span className="stat-label-lb">Final Stock</span>
                              <span className="stat-value-lb">{finalStock}</span>
                            </div>
                            <div className="leaderboard-stat-row">
                              <span className="stat-label-lb">Trades</span>
                              <span className="stat-value-lb">{agent.trades || 0}</span>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  {/* Back Button */}
                  <button onClick={handleBack} className="back-button-results">
                    ← Back to Dashboard
                  </button>
                </div>
              </div>
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

        {/* Replace Agent Modal */}
        <ReplaceAgentModal
          isOpen={replaceModalOpen}
          onClose={closeReplaceModal}
          failedModel={modelToReplace}
          agents={agents}
          usedAgents={generatedAlgos.map(a => a.modelName)}
          onReplace={handleAgentReplacement}
        />
      </div>

      {/* Inline review only; external preview modal removed */}
    </div>
  );
};

export default Dashboard;
