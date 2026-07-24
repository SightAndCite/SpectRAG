export default function Sidebar({ sessions, activeId, status, onSelect, onCreate, onRename, onDelete, onAddDocs }) {
  const active = sessions.find(s => s.id === activeId) || null;
  const indexingThis = status.indexing && status.indexing_session === activeId;

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <span className="brand-icon">◈</span>
        <div>
          <p className="brand-name">SpectRAG</p>
          <p className="brand-sub">Graph-Spectral</p>
        </div>
      </div>

      <div className="sidebar-section">
        <button className="btn-new-session" onClick={onCreate}>
          <span>+</span> New Session
        </button>
      </div>

      <div className="sessions-list">
        {sessions.length === 0 ? (
          <p className="sidebar-empty">No sessions yet</p>
        ) : (
          sessions.map(s => (
            <button
              key={s.id}
              className={`session-item ${s.id === activeId ? 'active' : ''}`}
              onClick={() => onSelect(s.id)}
            >
              <span className="session-dot" />
              <span className="session-name">{s.name}</span>
              <span className="session-count" title={`${s.doc_count} document${s.doc_count === 1 ? '' : 's'}`}>
                {s.doc_count}
              </span>
              <button
                className="session-rename"
                onClick={e => { e.stopPropagation(); onRename(s.id, s.name); }}
                title="Rename"
              >✎</button>
              <button
                className="session-del"
                onClick={e => { e.stopPropagation(); onDelete(s.id); }}
                title="Delete"
              >×</button>
            </button>
          ))
        )}
      </div>

      {activeId && !indexingThis && (
        <div className="sidebar-section">
          <button className="btn-add-docs" onClick={onAddDocs} title="Upload documents to this session">
            <span>⊕</span> Add Documents
          </button>
        </div>
      )}

      <div className="sidebar-footer">
        <span className={`status-indicator ${indexingThis ? 'indexing' : active?.has_index ? 'ready' : 'idle'}`} />
        <span className="status-label">
          {indexingThis ? 'Indexing…'
            : active?.has_index ? `${(active.chunk_count || 0).toLocaleString()} chunks`
            : activeId ? 'No documents yet'
            : 'No session'}
        </span>
      </div>
    </aside>
  );
}
