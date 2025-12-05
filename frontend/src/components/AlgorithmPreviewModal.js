import React, { useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './AlgorithmPreviewModal.css';

const AlgorithmPreviewModal = ({ isOpen, onClose, modelName, code }) => {
  const modalRef = useRef(null);
  const previousFocusRef = useRef(null);

  // Ensure code is a string (handle cases where it might be an object or null)
  const codeString = React.useMemo(() => {
    if (!code) return '';
    if (typeof code === 'string') return code;
    // If code is an object (MongoDB nested issue), try to extract the string
    if (typeof code === 'object') {
      // Try to find the actual code string in nested object
      const extractCode = (obj) => {
        if (typeof obj === 'string') return obj;
        if (typeof obj === 'object' && obj !== null) {
          for (const key of Object.keys(obj)) {
            const result = extractCode(obj[key]);
            if (result && result.includes('def ')) return result;
          }
        }
        return '';
      };
      return extractCode(code);
    }
    return String(code);
  }, [code]);

  const normalizeModelName = (name) => {
    return String(name).split('/').pop().replace(':free', '');
  };

  // Debug: Log code length when modal opens
  useEffect(() => {
    if (isOpen && codeString) {
      console.log('Modal opened with code:', {
        lines: codeString.split('\n').length,
        chars: codeString.length,
        preview: codeString.substring(0, 100) + '...'
      });
    }
  }, [isOpen, codeString]);

  useEffect(() => {
    if (isOpen) {
      // Store the previously focused element
      previousFocusRef.current = document.activeElement;

      // Focus the modal
      if (modalRef.current) {
        modalRef.current.focus();
      }

      // Prevent body scroll
      document.body.style.overflow = 'hidden';
    } else {
      // Restore body scroll
      document.body.style.overflow = '';

      // Restore focus to previous element
      if (previousFocusRef.current) {
        previousFocusRef.current.focus();
      }
    }

    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  const handleKeyDown = (e) => {
    if (e.key === 'Escape') {
      onClose();
    }

    // Focus trap
    if (e.key === 'Tab') {
      const focusableElements = modalRef.current?.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );

      if (!focusableElements || focusableElements.length === 0) return;

      const firstElement = focusableElements[0];
      const lastElement = focusableElements[focusableElements.length - 1];

      if (e.shiftKey) {
        if (document.activeElement === firstElement) {
          e.preventDefault();
          lastElement.focus();
        }
      } else {
        if (document.activeElement === lastElement) {
          e.preventDefault();
          firstElement.focus();
        }
      }
    }
  };

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(codeString);
      // Visual feedback
      const button = document.getElementById('copy-modal-btn');
      if (button) {
        const originalText = button.textContent;
        button.textContent = '✓ Copied!';
        button.classList.add('copied');
        setTimeout(() => {
          button.textContent = originalText;
          button.classList.remove('copied');
        }, 2000);
      }
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const downloadAlgorithm = () => {
    const blob = new Blob([codeString], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${normalizeModelName(modelName).replace(/[^a-z0-9]/gi, '_')}_algorithm.py`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const overlayVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { duration: 0.2 }
    },
    exit: {
      opacity: 0,
      transition: { duration: 0.2 }
    }
  };

  const modalVariants = {
    hidden: {
      opacity: 0,
      scale: 0.9,
      y: 20
    },
    visible: {
      opacity: 1,
      scale: 1,
      y: 0,
      transition: {
        duration: 0.3,
        ease: [0.2, 0.8, 0.2, 1]
      }
    },
    exit: {
      opacity: 0,
      scale: 0.95,
      y: 10,
      transition: {
        duration: 0.2,
        ease: [0.2, 0.8, 0.2, 1]
      }
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          className="preview-modal-overlay"
          variants={overlayVariants}
          initial="hidden"
          animate="visible"
          exit="exit"
          onClick={onClose}
          role="presentation"
        >
          <motion.div
            ref={modalRef}
            className="preview-modal-content"
            variants={modalVariants}
            initial="hidden"
            animate="visible"
            exit="exit"
            onClick={(e) => e.stopPropagation()}
            onKeyDown={handleKeyDown}
            role="dialog"
            aria-modal="true"
            aria-labelledby="modal-title"
            tabIndex={-1}
          >
            <div className="preview-modal-header">
              <h2 id="modal-title" className="preview-modal-title">
                {normalizeModelName(modelName)}
              </h2>
              <div className="preview-modal-actions">
                <button
                  id="copy-modal-btn"
                  className="modal-action-btn"
                  onClick={copyToClipboard}
                  aria-label="Copy algorithm to clipboard"
                >
                  📋 Copy to Clipboard
                </button>
                <button
                  className="modal-action-btn"
                  onClick={downloadAlgorithm}
                  aria-label="Download algorithm file"
                >
                  💾 Download
                </button>
                <button
                  className="modal-close-btn"
                  onClick={onClose}
                  aria-label="Close modal"
                >
                  ✕
                </button>
              </div>
            </div>

            <div className="preview-modal-body">
              <div className="code-stats">
                {codeString && `${codeString.split('\n').length} lines • ${Math.round(codeString.length / 1024)}KB`}
              </div>
              <pre className="preview-code-block">
                <code>{codeString || '// No code available yet...'}</code>
              </pre>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

export default AlgorithmPreviewModal;
