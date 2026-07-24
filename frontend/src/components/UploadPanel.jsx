import { useState, useRef } from 'react';
import * as api from '../api';

const ACCEPTED = '.pdf,.txt,.md,.html,.htm,.docx';
const fmt = (bytes) => bytes < 1024 * 1024
  ? `${(bytes / 1024).toFixed(0)} KB`
  : `${(bytes / 1024 / 1024).toFixed(1)} MB`;

export default function UploadPanel({ sessionId, onUploaded }) {
  const [files, setFiles] = useState([]);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef();

  const addFiles = (incoming) => {
    const existing = new Set(files.map(f => f.name));
    setFiles(prev => [...prev, ...[...incoming].filter(f => !existing.has(f.name))]);
  };

  const removeFile = (i) => setFiles(prev => prev.filter((_, j) => j !== i));

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    addFiles(e.dataTransfer.files);
  };

  const handleSubmit = async () => {
    if (!files.length || uploading) return;
    setUploading(true);
    setError(null);
    try {
      await api.uploadDocuments(sessionId, files);
      onUploaded?.();  // nudge App to poll immediately instead of waiting
      // Status polling in App.jsx will catch the indexing state automatically
    } catch (err) {
      setError(err.message);
      setUploading(false);
    }
  };

  return (
    <div className="upload-panel">
      <div className="upload-card">
        <h2 className="upload-title">Upload Documents</h2>
        <p className="upload-subtitle">PDF · TXT · Markdown · HTML · DOCX</p>

        <div
          className={`dropzone ${dragging ? 'drag-over' : ''}`}
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
        >
          <input
            ref={inputRef}
            type="file"
            multiple
            accept={ACCEPTED}
            style={{ display: 'none' }}
            onChange={e => addFiles(e.target.files)}
          />
          <div className="dropzone-icon">↑</div>
          <p className="dropzone-text">Drop files here or <span className="link-text">browse</span></p>
          <p className="dropzone-hint">Multiple files supported</p>
        </div>

        {files.length > 0 && (
          <ul className="file-list">
            {files.map((f, i) => (
              <li key={i} className="file-row">
                <span className="file-icon">📄</span>
                <span className="file-name">{f.name}</span>
                <span className="file-size">{fmt(f.size)}</span>
                <button className="file-remove" onClick={() => removeFile(i)}>×</button>
              </li>
            ))}
          </ul>
        )}

        {error && <p className="error-banner">{error}</p>}

        <button
          className="btn-index"
          onClick={handleSubmit}
          disabled={!files.length || uploading}
        >
          {uploading
            ? 'Uploading…'
            : files.length
            ? `Index ${files.length} file${files.length > 1 ? 's' : ''}`
            : 'Select files to continue'}
        </button>
      </div>
    </div>
  );
}
