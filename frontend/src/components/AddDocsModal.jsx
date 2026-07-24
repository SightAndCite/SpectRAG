import { useState, useRef, useEffect, useCallback } from 'react';
import * as api from '../api';

const ACCEPTED = ['.pdf', '.txt', '.md', '.html', '.docx'];

export default function AddDocsModal({ sessionId, status, onUploaded, onClose }) {
  const [files,   setFiles]   = useState([]);
  const [phase,   setPhase]   = useState('idle'); // idle | uploading | indexing | done | error
  const [errMsg,  setErrMsg]  = useState('');
  const [dragOver, setDrag]   = useState(false);
  const inputRef = useRef();

  // Detect indexing completion by polling the session's own index state — robust
  // even when a fast build finishes between the global status polls (so the
  // transient "indexing" flag is never observed).
  useEffect(() => {
    if (phase !== 'indexing') return;
    let done = false;
    const check = () => {
      if (status.index_error && status.indexing_session === sessionId) {
        setErrMsg(status.index_error);
        setPhase('error');
        done = true;
        return;
      }
      api.getSession(sessionId).then(d => {
        if (done) return;
        if (d?.has_index && !status.indexing) {
          setPhase('done');
          done = true;
          setTimeout(onClose, 1400);
        }
      }).catch(() => {});
    };
    check();
    const id = setInterval(check, 1500);
    return () => { done = true; clearInterval(id); };
  }, [phase, sessionId, status.indexing, status.index_error, status.indexing_session, onClose]);

  const addFiles = useCallback((incoming) => {
    setFiles(prev => {
      const existing = new Set(prev.map(f => f.name + f.size));
      const fresh = [...incoming].filter(f => !existing.has(f.name + f.size));
      return [...prev, ...fresh];
    });
  }, []);

  const onDrop = (e) => {
    e.preventDefault(); setDrag(false);
    addFiles(e.dataTransfer.files);
  };

  const onInputChange = (e) => addFiles(e.target.files);

  const removeFile = (i) => setFiles(prev => prev.filter((_, idx) => idx !== i));

  const handleSubmit = async () => {
    if (!files.length || phase !== 'idle') return;
    setPhase('uploading');
    try {
      await api.uploadDocuments(sessionId, files);
      onUploaded?.();  // nudge App to poll immediately
      setPhase('indexing');
    } catch (e) {
      setErrMsg(e.message);
      setPhase('error');
    }
  };

  const canClose = phase !== 'uploading' && phase !== 'indexing';

  const formatBytes = (n) => n < 1024 ? `${n} B` : n < 1048576 ? `${(n/1024).toFixed(1)} KB` : `${(n/1048576).toFixed(1)} MB`;

  return (
    <div className="modal-overlay" onClick={canClose ? onClose : undefined}>
      <div className="modal" onClick={e => e.stopPropagation()}>

        <div className="modal-header">
          <span className="modal-title">Add Documents</span>
          {canClose && (
            <button className="modal-close" onClick={onClose}>×</button>
          )}
        </div>

        {/* Uploading / Indexing states */}
        {(phase === 'uploading' || phase === 'indexing') && (
          <div className="modal-body modal-progress">
            <div className="spinner" />
            <p className="center-title">
              {phase === 'uploading' ? 'Uploading files…' : 'Building index…'}
            </p>
            {phase === 'indexing' && status.indexing_stage && (
              <p className="center-sub stage-text">{status.indexing_stage}</p>
            )}
            <p className="modal-progress-note">Do not close this window.</p>
          </div>
        )}

        {phase === 'done' && (
          <div className="modal-body modal-progress">
            <div className="done-check">✓</div>
            <p className="center-title" style={{ color: '#22c55e' }}>Index updated!</p>
          </div>
        )}

        {phase === 'error' && (
          <div className="modal-body">
            <p className="center-title error-text">Error</p>
            <p className="center-sub">{errMsg}</p>
            <button className="btn-primary" style={{ marginTop: 16 }} onClick={() => { setPhase('idle'); setErrMsg(''); }}>
              Try Again
            </button>
          </div>
        )}

        {phase === 'idle' && (
          <div className="modal-body">
            <div
              className={`dropzone ${dragOver ? 'drag-over' : ''}`}
              onClick={() => inputRef.current?.click()}
              onDragOver={e => { e.preventDefault(); setDrag(true); }}
              onDragLeave={() => setDrag(false)}
              onDrop={onDrop}
            >
              <input
                ref={inputRef}
                type="file"
                multiple
                accept={ACCEPTED.join(',')}
                style={{ display: 'none' }}
                onChange={onInputChange}
              />
              <p className="dropzone-icon">⊕</p>
              <p className="dropzone-text">Click or drag files here</p>
              <p className="dropzone-hint">{ACCEPTED.join(', ')}</p>
            </div>

            {files.length > 0 && (
              <ul className="file-list">
                {files.map((f, i) => (
                  <li key={i} className="file-item">
                    <span className="file-icon">◻</span>
                    <span className="file-name">{f.name}</span>
                    <span className="file-size">{formatBytes(f.size)}</span>
                    <button className="file-remove" onClick={() => removeFile(i)}>×</button>
                  </li>
                ))}
              </ul>
            )}

            <div className="modal-actions">
              <button className="btn-secondary" onClick={onClose}>Cancel</button>
              <button
                className="btn-primary"
                disabled={!files.length}
                onClick={handleSubmit}
              >
                Index {files.length > 0 ? `${files.length} file${files.length > 1 ? 's' : ''}` : 'Files'}
              </button>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
