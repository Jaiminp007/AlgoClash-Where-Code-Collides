import React, { useState, useEffect, useRef } from 'react';
import './CustomDropdown.css';

// Import all images
import anthropicIcon from '../assets/anthropic.png';
import deepseekIcon from '../assets/deepseek.png';
import googleIcon from '../assets/google.png';
import grokIcon from '../assets/grok.png';
import metaIcon from '../assets/meta.png';
import mistralIcon from '../assets/mistral.png';
import openaiIcon from '../assets/openai.png';
import qwenIcon from '../assets/qwen.png';
import openrouterIcon from '../assets/openrouter.png';
import agenticaIcon from '../assets/agentica.png';
import moonshotaiIcon from '../assets/moonshotai.png';
import nousresearchIcon from '../assets/nousresearch.png';
import alibabaIcon from '../assets/alibaba.png';
import arliaiIcon from '../assets/arliai.png';
import cognitivecomputationsIcon from '../assets/cognitivecomputations.png';
import meituanIcon from '../assets/meituan.png';
import microsoftIcon from '../assets/microsoft.png';
import nvidiaIcon from '../assets/nvidia.png';
import shisaaiIcon from '../assets/shisaai.png';
import tencentIcon from '../assets/tencent.png';
import tngtechIcon from '../assets/tngtech.png';
import zaiIcon from '../assets/zai.png';

const imageMap = {
  'Anthropic': anthropicIcon,
  'DeepSeek': deepseekIcon,
  'Google': googleIcon,
  'Grok': grokIcon,
  'Meta': metaIcon,
  'Mistral': mistralIcon,
  'OpenAI': openaiIcon,
  'Qwen': qwenIcon,
  'OpenRouter': openrouterIcon,
  'Agentica': agenticaIcon,
  'Moonshotai': moonshotaiIcon,
  'Nousresearch': nousresearchIcon,
  'Alibaba': alibabaIcon,
  'Arliai': arliaiIcon,
  'Cognitivecomputations': cognitivecomputationsIcon,
  'Meituan': meituanIcon,
  'Microsoft': microsoftIcon,
  'NVIDIA': nvidiaIcon,
  'ShisaAI': shisaaiIcon,
  'Tencent': tencentIcon,
  'TNGTech': tngtechIcon,
  'ZAI': zaiIcon,
  'Z-AI': zaiIcon
};

const CustomDropdown = ({ agents, selected, onSelect, disabledAgents = new Set() }) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  const handleSelect = (agent) => {
    onSelect(agent);
    setIsOpen(false);
  };

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const getCompanyFromAgent = (agent) => {
    if (!agent) return null;
    return Object.keys(agents).find(company => agents[company].includes(agent));
  };

  const selectedCompany = getCompanyFromAgent(selected);

  return (
    <div className="custom-dropdown" ref={dropdownRef}>
      <div className="dropdown-selected" onClick={() => setIsOpen(!isOpen)}>
        {selectedCompany ? (
          <img 
            src={imageMap[selectedCompany]} 
            alt={selectedCompany} 
            className="agent-icon-selected"
          />
        ) : (
          <span className="placeholder">+</span>
        )}
      </div>
      {isOpen && (
        <div className="dropdown-options">
          {Object.entries(agents).map(([company, agentList]) => (
            <div key={company}>
              <div className="dropdown-company-header">{company}</div>
              {agentList.map((agent) => (
                <div
                  key={agent}
                  className={`dropdown-option ${selected === agent ? 'selected' : ''} ${disabledAgents.has(agent) && selected !== agent ? 'disabled' : ''}`}
                  aria-disabled={disabledAgents.has(agent) && selected !== agent}
                  onClick={() => {
                    if (disabledAgents.has(agent) && selected !== agent) return; // prevent selecting duplicates
                    handleSelect(agent);
                  }}
                >
                  <img 
                    src={imageMap[company]} 
                    alt={company} 
                    className="agent-icon-option"
                  />
                  <span>{agent.split('/')[1]}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default CustomDropdown;
