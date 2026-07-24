import { useState, useEffect, useCallback, useRef } from 'react';
import Sidebar from './components/Sidebar';
import ChatPanel from './components/ChatPanel';
import UploadPanel from './components/UploadPanel';
import GraphPanel from './components/GraphPanel';
import AddDocsModal from './components/AddDocsModal';
import * as api from './api';

export default function App() {
  const [sessions, setSessions] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [activeDetail, setActiveDetail] = useState(null); // per-session index info
  const [messages, setMessages] = useState([]);
  const [status, setStatus] = useState({ indexing: false, indexing_session: null, indexing_stage: '', index_error: null });
  const [queryLoading, setQueryLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dismissedError, setDismissedError] = useState(null);
  const [tab, setTab] = useState('chat');  // 'chat' | 'graph'
  const [showAddDocs, setShowAddDocs] = useState(false);

  // Poll every 2s while indexing, every 8s at rest — avoids log spam
  useEffect(() => {
    let id;
    const poll = () => {
      api.getStatus().then(s => {
        setStatus(s);
        clearInterval(id);
        id = setInterval(poll, s.indexing ? 2000 : 8000);
      }).catch(() => {});
    };
    poll();
    id = setInterval(poll, 8000);
    return () => clearInterval(id);
  }, []);

  const refreshSessions = useCallback(() =>
    api.getSessions().then(setSessions).catch(() => {}), []);

  useEffect(() => { refreshSessions(); }, [refreshSessions]);

  // Load the active session's messages + index detail whenever it changes
  const refreshActiveDetail = useCallback(() => {
    if (!activeId) { setActiveDetail(null); return; }
    api.getSession(activeId).then(setActiveDetail).catch(() => setActiveDetail(null));
  }, [activeId]);

  useEffect(() => {
    if (!activeId) { setMessages([]); setActiveDetail(null); return; }
    api.getMessages(activeId).then(setMessages).catch(() => setMessages([]));
    refreshActiveDetail();
  }, [activeId, refreshActiveDetail]);

  // While the active session has no index yet, poll its detail so the view flips
  // to chat the moment indexing finishes — even if a fast build completed between
  // the global status polls (so the transient "indexing" flag was never observed).
  useEffect(() => {
    if (!activeId || activeDetail?.has_index) return;
    const id = setInterval(refreshActiveDetail, 2000);
    return () => clearInterval(id);
  }, [activeId, activeDetail?.has_index, refreshActiveDetail]);

  // Immediately re-check status + detail (e.g. right after an upload) instead of
  // waiting for the next slow poll tick.
  const pollNow = useCallback(() => {
    api.getStatus().then(setStatus).catch(() => {});
    refreshActiveDetail();
  }, [refreshActiveDetail]);

  // When an indexing job finishes, refresh session list + the active session's detail
  const prevIndexing = usePrevious(status.indexing);
  useEffect(() => {
    if (prevIndexing && !status.indexing) {
      refreshSessions();
      refreshActiveDetail();
    }
  }, [status.indexing, prevIndexing, refreshSessions, refreshActiveDetail]);

  const handleNewSession = async () => {
    const name = window.prompt('Name this session:');
    if (name === null) return;           // cancelled
    const session = await api.createSession(name);
    await refreshSessions();
    setActiveId(session.id);
    setMessages([]);
    setTab('chat');
  };

  const handleRenameSession = async (id, currentName) => {
    const name = window.prompt('Rename session:', currentName);
    if (name === null) return;                 // cancelled
    if (!name.trim()) return;                  // empty — ignore
    await api.renameSession(id, name.trim());
    refreshSessions();
  };

  const handleDeleteSession = async (id) => {
    await api.deleteSession(id);
    if (activeId === id) { setActiveId(null); setMessages([]); setActiveDetail(null); }
    refreshSessions();
  };

  const handleQuery = async (question) => {
    setQueryLoading(true);
    setError(null);
    setMessages(prev => [...prev, { role: 'user', content: question }]);
    try {
      const result = await api.querySession(activeId, question);
      setMessages(prev => [...prev, { role: 'assistant', content: result.answer, sources: result.sources }]);
    } catch (err) {
      setError(err.message);
      setMessages(prev => prev.slice(0, -1));
    } finally {
      setQueryLoading(false);
    }
  };

  const indexingThis = status.indexing && status.indexing_session === activeId;
  const hasIndex = !!activeDetail?.has_index;

  const mainContent = () => {
    if (!activeId) {
      return (
        <div className="center-screen">
          <p className="center-title">Welcome to SpectRAG</p>
          <p className="center-sub">Create a session to upload documents and start asking questions.</p>
          <button className="btn-primary" onClick={handleNewSession}>New Session</button>
        </div>
      );
    }
    if (indexingThis) {
      return (
        <div className="center-screen">
          <div className="spinner" />
          <p className="center-title">Building index…</p>
          <p className="center-sub stage-text">{status.indexing_stage || 'Starting…'}</p>
        </div>
      );
    }
    if (status.index_error && status.index_error !== dismissedError && status.indexing_session === activeId) {
      return (
        <div className="center-screen">
          <p className="center-title error-text">Indexing failed</p>
          <p className="center-sub">{status.index_error}</p>
          <button className="btn-primary" onClick={() => setDismissedError(status.index_error)}>
            Dismiss
          </button>
        </div>
      );
    }
    if (!hasIndex) {
      return <UploadPanel sessionId={activeId} onUploaded={pollNow} />;
    }

    // Session has an index — show tab bar + content
    return (
      <div className="tabbed-main">
        <div className="tab-bar">
          <button className={`tab-btn ${tab === 'chat' ? 'active' : ''}`} onClick={() => setTab('chat')}>
            Chat
          </button>
          <button className={`tab-btn ${tab === 'graph' ? 'active' : ''}`} onClick={() => setTab('graph')}>
            Graph
          </button>
        </div>

        <div className="tab-content">
          {tab === 'chat' && (
            <ChatPanel messages={messages} onQuery={handleQuery} loading={queryLoading} error={error} />
          )}
          {tab === 'graph' && <GraphPanel sessionId={activeId} />}
        </div>
      </div>
    );
  };

  return (
    <div className="app">
      <Sidebar
        sessions={sessions}
        activeId={activeId}
        status={status}
        onSelect={setActiveId}
        onCreate={handleNewSession}
        onRename={handleRenameSession}
        onDelete={handleDeleteSession}
        onAddDocs={() => setShowAddDocs(true)}
      />
      <main className="main">{mainContent()}</main>
      {showAddDocs && activeId && (
        <AddDocsModal
          sessionId={activeId}
          status={status}
          onUploaded={pollNow}
          onClose={() => setShowAddDocs(false)}
        />
      )}
    </div>
  );
}

// Small helper: remember the previous value of a prop/state across renders.
function usePrevious(value) {
  const ref = useRef();
  useEffect(() => { ref.current = value; }, [value]);
  return ref.current;
}
