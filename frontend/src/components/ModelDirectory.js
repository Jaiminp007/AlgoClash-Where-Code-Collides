import React, { useEffect, useState, useMemo } from 'react';
import './Models.css';

// Prefer SVGs where available, fall back to PNGs that already exist
import openaiSvg from '../assets/openai.svg';
import openaiPng from '../assets/openai.png';
import googleSvg from '../assets/google.svg';
import googlePng from '../assets/google.png';
import anthropicSvg from '../assets/anthropic.svg';
import anthropicPng from '../assets/anthropic.png';
import metaSvg from '../assets/meta.svg';
import metaPng from '../assets/meta.png';
import qwenSvg from '../assets/qwen.svg';
import qwenPng from '../assets/qwen.png';
import mistralSvg from '../assets/mistral.svg';
import mistralPng from '../assets/mistral.png';
import deepseekSvg from '../assets/deepseek.svg';
import deepseekPng from '../assets/deepseek.png';
import openrouterSvg from '../assets/openrouter.svg';
import openrouterPng from '../assets/openrouter.png';
import nousresearchSvg from '../assets/nousresearch.svg';
import nousresearchPng from '../assets/nousresearch.png';
import moonshotaiSvg from '../assets/moonshotai.svg';
import moonshotaiPng from '../assets/moonshotai.png';
import agenticaSvg from '../assets/agentica.svg';
import grokPng from '../assets/grok.png';
// New providers added
import alibabaSvg from '../assets/alibaba.svg';
import meituanSvg from '../assets/meituan.svg';
import nvidiaSvg from '../assets/nvidia.svg';
import tencentSvg from '../assets/tencent.svg';
import microsoftSvg from '../assets/microsoft.svg';
import zaiSvg from '../assets/zai.svg';
import tngtechSvg from '../assets/tngtech.svg';
import arliaiSvg from '../assets/arliai.svg';
import shisaaiSvg from '../assets/shisaai.svg';
import cognitivecomputationsSvg from '../assets/cognitivecomputations.svg';

const logoMap = {
  OpenAI: openaiSvg || openaiPng,
  Google: googleSvg || googlePng,
  Anthropic: anthropicSvg || anthropicPng,
  Meta: metaSvg || metaPng,
  Qwen: qwenSvg || qwenPng,
  Mistral: mistralSvg || mistralPng,
  DeepSeek: deepseekSvg || deepseekPng,
  Grok: grokPng,
  OpenRouter: openrouterSvg || openrouterPng,
  Nousresearch: nousresearchSvg || nousresearchPng,
  Moonshotai: moonshotaiSvg || moonshotaiPng,
  Agentica: agenticaSvg,
  Alibaba: alibabaSvg,
  Meituan: meituanSvg,
  NVIDIA: nvidiaSvg,
  Tencent: tencentSvg,
  Microsoft: microsoftSvg,
  ZAI: zaiSvg,
  TNGTech: tngtechSvg,
  Arliai: arliaiSvg,
  ShisaAI: shisaaiSvg,
  Cognitivecomputations: cognitivecomputationsSvg,
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
