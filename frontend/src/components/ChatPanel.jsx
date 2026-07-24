import { useState, useRef, useEffect } from 'react';
import MessageBubble from './MessageBubble';

export default function ChatPanel({ messages, onQuery, loading, error }) {
  const [input, setInput] = useState('');
  const bottomRef = useRef();
  const textareaRef = useRef();

  // Auto-scroll on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  // Auto-resize textarea
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 160) + 'px';
  }, [input]);

  const submit = () => {
    const q = input.trim();
    if (!q || loading) return;
    setInput('');
    onQuery(q);
  };

  const onKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="chat-panel">
      <div className="messages-scroll">
        {messages.length === 0 && (
          <div className="chat-empty">
            <p>Ask anything about your indexed documents.</p>
            <p className="chat-empty-hint">Press Enter to send · Shift+Enter for a new line</p>
          </div>
        )}

        {messages.map((msg, i) => <MessageBubble key={i} message={msg} />)}

        {loading && (
          <div className="bubble-row assistant">
            <div className="bubble bubble-assistant thinking">
              <span className="dot" /><span className="dot" /><span className="dot" />
            </div>
          </div>
        )}

        {error && (
          <div className="error-inline">{error}</div>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="input-area">
        <div className="input-box">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Ask a question…"
            rows={1}
            disabled={loading}
          />
          <button
            className="send-btn"
            onClick={submit}
            disabled={!input.trim() || loading}
            title="Send (Enter)"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13" />
              <polygon points="22 2 15 22 11 13 2 9 22 2" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}
