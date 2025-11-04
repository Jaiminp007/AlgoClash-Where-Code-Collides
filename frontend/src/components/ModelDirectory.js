import React, { useEffect, useState, useMemo } from 'react';
import './Models.css';

// Import provider icons (PNG only)
import openaiPng from '../assets/openai.png';
import googlePng from '../assets/google.png';
import anthropicPng from '../assets/anthropic.png';
import metaPng from '../assets/meta.png';
import qwenPng from '../assets/qwen.png';
import mistralPng from '../assets/mistral.png';
import deepseekPng from '../assets/deepseek.png';
import openrouterPng from '../assets/openrouter.png';
import nousresearchPng from '../assets/nousresearch.png';
import moonshotaiPng from '../assets/moonshotai.png';
import agenticaPng from '../assets/agentica.png';
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

const logoMap = {
  OpenAI: openaiPng,
  Google: googlePng,
  Anthropic: anthropicPng,
  Meta: metaPng,
  Qwen: qwenPng,
  Mistral: mistralPng,
  DeepSeek: deepseekPng,
  Grok: grokPng,
  OpenRouter: openrouterPng,
  Nousresearch: nousresearchPng,
  Moonshotai: moonshotaiPng,
  Agentica: agenticaPng,
  Alibaba: alibabaPng,
  Arliai: arliaiPng,
  Cognitivecomputations: cognitivecomputationsPng,
  Meituan: meituanPng,
  Microsoft: microsoftPng,
  NVIDIA: nvidiaPng,
  ShisaAI: shisaaiPng,
  Tencent: tencentPng,
  TNGTech: tngtechPng,
  ZAI: zaiPng,
};

function ProviderSection({ name, models, compact = false, limit = 6 }) {
  const shown = compact ? models.slice(0, limit) : models;
  const remaining = compact ? Math.max(0, models.length - shown.length) : 0;

  return (
    <div className="provider-group">
      <div className="provider-header">
        <img src={logoMap[name]} alt={`${name} logo`} className="provider-logo" />
        <div className="provider-meta">
          <div className="provider-name">{name}</div>
          <div className="provider-count">{models.length} models</div>
        </div>
      </div>
      <div className="models-chip-grid">
        {shown.map((m) => (
          <div key={m} className="model-chip" title={m}>{m}</div>
        ))}
        {remaining > 0 && (
          <a href="/models" className="model-chip more">+{remaining} more</a>
        )}
      </div>
    </div>
  );
}

export default function ModelDirectory({ compact = false, limitPerProvider = 6 }) {
  const [agents, setAgents] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const apiBase = process.env.REACT_APP_API_BASE_URL || '';
    fetch(`${apiBase}/api/ai_agents`)
      .then((r) => r.json())
      .then((data) => {
        setAgents(data || {});
        setLoading(false);
      })
      .catch((e) => {
        setError(e);
        setLoading(false);
      });
  }, []);

  const providerEntries = useMemo(() => {
    return Object.entries(agents)
      .sort(([a], [b]) => a.localeCompare(b));
  }, [agents]);

  if (loading) return <div className="models-loading">Loading models…</div>;
  if (error) return <div className="models-error">Failed to load models</div>;

  return (
    <div className={compact ? 'models-directory compact' : 'models-directory'}>
      {providerEntries.map(([provider, list]) => (
        <ProviderSection
          key={provider}
          name={provider}
          models={Array.isArray(list) ? list : []}
          compact={compact}
          limit={limitPerProvider}
        />)
      )}
    </div>
  );
}
