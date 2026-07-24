import { useState } from 'react';

// ── Individual expandable source card ────────────────────────────────────────
function SourceCard({ source, rank, maxScore }) {
  const [open, setOpen] = useState(false);

  const filename     = String(source.source).split('/').pop();
  const scoreRatio   = maxScore > 0 ? source.score / maxScore : 0;
  const scorePercent = Math.round(scoreRatio * 100);

  // Score colour: green → yellow → red
  const scoreColor =
    scoreRatio > 0.66 ? '#22c55e' :
    scoreRatio > 0.33 ? '#f59e0b' : '#ef4444';

  return (
    <div className={`src-card${open ? ' src-card--open' : ''}`}>

      {/* ── Header (always visible, click to toggle) ── */}
      <button className="src-header" onClick={() => setOpen(o => !o)}>

        <span className="src-rank">{String(rank).padStart(2, '0')}</span>

        <div className="src-meta">
          <span className="src-file" title={source.source}>{filename}</span>
          <div className="src-tags">
            {source.page  && <span className="src-tag">p.{source.page}</span>}
            {source.position != null &&
              <span className="src-tag">chunk {source.position}</span>}
            {source.section_path?.length > 0 &&
              <span className="src-tag src-tag--section">
                {source.section_path[source.section_path.length - 1]}
              </span>}
          </div>
        </div>

        <div className="src-score-wrap">
          <div className="src-score-track">
            <div
              className="src-score-fill"
              style={{ width: `${scorePercent}%`, background: scoreColor }}
            />
          </div>
          <span className="src-score-val" style={{ color: scoreColor }}>
            {source.score.toFixed(3)}
          </span>
        </div>

        <span className="src-chevron">{open ? '▴' : '▾'}</span>
      </button>

      {/* ── Expandable body ── */}
      <div className="src-body-wrap">
        <div className="src-body">

          {source.section_path?.length > 0 && (
            <div className="src-breadcrumb">
              {source.section_path.join(' › ')}
            </div>
          )}

          <div className="src-retrieval-meta">
            <span className="src-retrieval-label">Relevance score</span>
            <span className="src-retrieval-val">{source.score.toFixed(4)}</span>
            <span className="src-retrieval-hint">
              combined spectral proximity + semantic similarity
            </span>
          </div>

          <pre className="src-full-text">{source.full_text}</pre>
        </div>
      </div>
    </div>
  );
}

// ── Message bubble ────────────────────────────────────────────────────────────
export default function MessageBubble({ message }) {
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const isUser = message.role === 'user';

  const sources  = message.sources ?? [];
  const maxScore = sources.length > 0 ? Math.max(...sources.map(s => s.score)) : 1;

  return (
    <div className={`bubble-row ${isUser ? 'user' : 'assistant'}`}>
      <div className={`bubble ${isUser ? 'bubble-user' : 'bubble-assistant'}`}>
        <p className="bubble-text">{message.content}</p>

        {!isUser && sources.length > 0 && (
          <div className="sources-section">
            <button
              className="sources-toggle"
              onClick={() => setSourcesOpen(o => !o)}
            >
              <span className="sources-toggle-icon">{sourcesOpen ? '▾' : '▸'}</span>
              <span>
                {sources.length} source{sources.length !== 1 ? 's' : ''} retrieved
              </span>
              {!sourcesOpen && (
                <span className="sources-toggle-hint">
                  · top score {sources[0]?.score.toFixed(3)}
                </span>
              )}
            </button>

            {sourcesOpen && (
              <div className="sources-list">
                {sources.map((s, i) => (
                  <SourceCard
                    key={i}
                    source={s}
                    rank={i + 1}
                    maxScore={maxScore}
                  />
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
