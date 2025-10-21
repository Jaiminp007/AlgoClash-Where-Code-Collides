import React, { useState, useEffect, useCallback } from 'react';
import './AlgorithmPreviewModal.css';

const AlgorithmPreviewModal = ({ isOpen, onClose }) => {
  const [algorithms, setAlgorithms] = useState([]);
  const [selectedAlgo, setSelectedAlgo] = useState(null);
  const [algoContent, setAlgoContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [autoDelete, setAutoDelete] = useState(false);
  const [polling, setPolling] = useState(false);

  const apiBase = process.env.REACT_APP_API_BASE_URL || '';

  // Fetch algorithm list
  const fetchAlgorithms = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`${apiBase}/api/algos`);
      if (!response.ok) {
        throw new Error('Failed to fetch algorithms');
      }
      const data = await response.json();
      setAlgorithms(Array.isArray(data) ? data : []);
    } catch (err) {
      console.error('Error fetching algorithms:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  // Fetch specific algorithm content
  const fetchAlgorithmContent = async (filename) => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`${apiBase}/api/algos/${encodeURIComponent(filename)}`);
      if (!response.ok) {
        throw new Error('Failed to fetch algorithm content');
      }
      const content = await response.text();
      setAlgoContent(content);
      setSelectedAlgo(filename);
    } catch (err) {
      console.error('Error fetching algorithm content:', err);
      setError(err.message);
      setAlgoContent('');
    } finally {
      setLoading(false);
    }
  };

  // Delete single algorithm
  const deleteAlgorithm = async (filename) => {
    if (!window.confirm(`Are you sure you want to delete ${filename}?`)) {
      return;
    }

    try {
      const response = await fetch(`${apiBase}/api/algos/${encodeURIComponent(filename)}`, {
        method: 'DELETE',
      });
      if (!response.ok) {
        throw new Error('Failed to delete algorithm');
      }

      // Refresh list and clear preview if it was the selected one
      if (selectedAlgo === filename) {
        setSelectedAlgo(null);
        setAlgoContent('');
      }
      await fetchAlgorithms();

    } catch (err) {
      console.error('Error deleting algorithm:', err);
      setError(err.message);
    }
  };

  // Delete all algorithms
  const deleteAllAlgorithms = async () => {
    if (!window.confirm('Are you sure you want to delete ALL algorithm files?')) {
      return;
    }

    try {
      const response = await fetch(`${apiBase}/api/algos`, {
        method: 'DELETE',
      });
      if (!response.ok) {
        throw new Error('Failed to delete algorithms');
      }

      const result = await response.json();
      alert(result.message || `Deleted ${result.deleted} file(s)`);

      setSelectedAlgo(null);
      setAlgoContent('');
      await fetchAlgorithms();

    } catch (err) {
      console.error('Error deleting all algorithms:', err);
      setError(err.message);
    }
  };

  // Download algorithm
  const downloadAlgorithm = (filename) => {
    const url = `${apiBase}/api/algos/${encodeURIComponent(filename)}/download`;
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  // Copy to clipboard
  const copyToClipboard = () => {
    if (!algoContent) return;

    navigator.clipboard.writeText(algoContent)
      .then(() => {
        alert('Code copied to clipboard!');
      })
      .catch(err => {
        console.error('Failed to copy:', err);
        alert('Failed to copy to clipboard');
      });
  };

  // Close preview and auto-delete if enabled
  const closePreview = async () => {
    if (autoDelete && selectedAlgo) {
      if (window.confirm(`Auto-delete is ON. Delete ${selectedAlgo}?`)) {
        await deleteAlgorithm(selectedAlgo);
      }
    }
    setSelectedAlgo(null);
    setAlgoContent('');
  };

  // Initial load when modal opens
  useEffect(() => {
    if (isOpen) {
      fetchAlgorithms();
      setPolling(true);
    } else {
      setPolling(false);
      setSelectedAlgo(null);
      setAlgoContent('');
    }
  }, [isOpen, fetchAlgorithms]);

  // Polling to detect new files (every 5 seconds)
  useEffect(() => {
    if (!polling || !isOpen) return;

    const interval = setInterval(() => {
      fetchAlgorithms();
    }, 5000);

    return () => clearInterval(interval);
  }, [polling, isOpen, fetchAlgorithms]);

  // Format file size
  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  // Format date
  const formatDate = (isoString) => {
    const date = new Date(isoString);
    return date.toLocaleString();
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Algorithm Preview</h2>
          <button className="close-button" onClick={onClose}>×</button>
        </div>

        <div className="modal-body">
          {/* Left: File list */}
          <div className="file-list-panel">
            <div className="panel-header">
              <h3>Generated Algorithms ({algorithms.length})</h3>
              <div className="panel-controls">
                <label className="auto-delete-toggle">
                  <input
                    type="checkbox"
                    checked={autoDelete}
                    onChange={(e) => setAutoDelete(e.target.checked)}
                  />
                  Auto-delete after preview
                </label>
                {algorithms.length > 0 && (
                  <button
                    className="delete-all-button"
                    onClick={deleteAllAlgorithms}
                  >
                    Delete All
                  </button>
                )}
              </div>
            </div>

            {loading && !selectedAlgo && (
              <div className="loading-skeleton">
                <div className="skeleton-item"></div>
                <div className="skeleton-item"></div>
                <div className="skeleton-item"></div>
              </div>
            )}

            {error && !selectedAlgo && (
              <div className="error-message">
                Error: {error}
              </div>
            )}

            {!loading && algorithms.length === 0 && (
              <div className="empty-state">
                <p>No algorithm files found.</p>
                <p className="hint">Generated algorithms will appear here after running a simulation.</p>
              </div>
            )}

            <div className="file-list">
              {algorithms.map((algo) => (
                <div
                  key={algo.filename}
                  className={`file-item ${selectedAlgo === algo.filename ? 'selected' : ''}`}
                  onClick={() => fetchAlgorithmContent(algo.filename)}
                >
                  <div className="file-info">
                    <div className="file-name">{algo.modelName}</div>
                    <div className="file-meta">
                      {formatSize(algo.sizeBytes)} • {formatDate(algo.modifiedAt)}
                    </div>
                  </div>
                  <button
                    className="delete-file-button"
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteAlgorithm(algo.filename);
                    }}
                    title="Delete this file"
                  >
                    🗑️
                  </button>
                </div>
              ))}
            </div>
          </div>

          {/* Right: Code preview */}
          <div className="code-preview-panel">
            {selectedAlgo ? (
              <>
                <div className="preview-header">
                  <h3>{selectedAlgo}</h3>
                  <div className="preview-actions">
                    <button onClick={copyToClipboard} title="Copy to clipboard">
                      📋 Copy
                    </button>
                    <button onClick={() => downloadAlgorithm(selectedAlgo)} title="Download file">
                      💾 Download
                    </button>
                    <button onClick={closePreview} title="Close preview">
                      ✕ Close
                    </button>
                  </div>
                </div>
                {loading ? (
                  <div className="loading-content">Loading...</div>
                ) : (
                  <pre className="code-content">
                    <code>{algoContent}</code>
                  </pre>
                )}
              </>
            ) : (
              <div className="no-preview">
                <p>Select an algorithm to preview</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AlgorithmPreviewModal;
