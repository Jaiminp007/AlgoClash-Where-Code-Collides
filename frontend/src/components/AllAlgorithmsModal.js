import React, { useMemo } from 'react';
import './AllAlgorithmsModal.css';

const AllAlgorithmsModal = ({ isOpen, onClose, algorithms = [] }) => {
  const hasAlgos = Array.isArray(algorithms) && algorithms.length > 0;

  const allText = useMemo(() => {
    if (!hasAlgos) return '';
    return algorithms
      .map(({ modelName, code }, idx) => `# ${idx + 1}. ${modelName}\n\n${code || ''}\n\n`)
      .join('\n');
  }, [algorithms, hasAlgos]);

  const copyAll = async () => {
    try {
      await navigator.clipboard.writeText(allText);
      // basic feedback by swapping button label briefly
      const el = document.getElementById('copy-all-btn');
      if (el) {
        const prev = el.textContent;
        el.textContent = '✓ Copied';
        setTimeout(() => (el.textContent = prev), 1500);
      }
    } catch (e) {}
  };

  if (!isOpen) return null;

  return (
    <div className="all-modal-overlay" onClick={onClose}>
      <div className="all-modal" onClick={(e) => e.stopPropagation()}>
        <div className="all-modal-header">
          <h3>All Algorithms (Full)</h3>
          <div className="all-modal-actions">
            <button id="copy-all-btn" className="all-action" onClick={copyAll} disabled={!hasAlgos}>
              Copy All
            </button>
            <button className="all-action" onClick={onClose}>Close</button>
          </div>
        </div>
        <div className="all-modal-body">
          {!hasAlgos ? (
            <div className="all-empty">No algorithms available yet.</div>
          ) : (
            algorithms.map(({ modelName, code }, idx) => (
              <div className="algo-section" key={modelName}>
                <div className="algo-title">{idx + 1}. {modelName}</div>
                <pre className="algo-code"><code>{code}</code></pre>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default AllAlgorithmsModal;
