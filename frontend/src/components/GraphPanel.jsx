import { useEffect, useRef, useState, useCallback } from 'react';
import ForceGraph3D from '3d-force-graph';
import SpriteText from 'three-spritetext';
import * as api from '../api';

// ── Colour helpers ────────────────────────────────────────────────────────────
const _hexToRgb = (hex) => { const h = hex.replace('#', ''); return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16)); };
const _rgbToHex = (r, g, b) => '#' + [r, g, b].map(v => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0')).join('');
const lighten = (hex, amt) => { const [r, g, b] = _hexToRgb(hex); return _rgbToHex(r + (255 - r) * amt, g + (255 - g) * amt, b + (255 - b) * amt); };
const darken  = (hex, amt) => { const [r, g, b] = _hexToRgb(hex); return _rgbToHex(r * (1 - amt), g * (1 - amt), b * (1 - amt)); };

// ── Signal config ─────────────────────────────────────────────────────────────
const SIGNALS = [
  { key: 'semantic',         label: 'Semantic',         color: '#6366f1' },
  { key: 'section',          label: 'Section',          color: '#f59e0b' },
  { key: 'entity',           label: 'Entity',           color: '#ef4444' },
  { key: 'adjacency',        label: 'Adjacency',        color: '#22c55e' },
  { key: 'utility_question', label: 'Utility Question', color: '#a855f7' },
  { key: 'citation',         label: 'Citation',         color: '#14b8a6' },
];
const SIG_COLOR      = Object.fromEntries(SIGNALS.map(s => [s.key, s.color]));
const FALLBACK_COLOR = '#64748b';

// ── Per-document colour palette ───────────────────────────────────────────────
const DOC_PALETTE = ['#6366f1','#f59e0b','#22c55e','#ef4444','#14b8a6','#a855f7','#0ea5e9','#f97316'];
const docColor = (() => {
  const cache = {};
  let idx = 0;
  return (doc) => { cache[doc] ??= DOC_PALETTE[idx++ % DOC_PALETTE.length]; return cache[doc]; };
})();

// Per-theme styling: nodes rendered lighter, edges darker; dimmed colours are used
// to fade the rest of the graph while hovering a node's neighbourhood.
const THEME = {
  dark:  { bg: '#0b1020', nodeLighten: 0.42, edgeDarken: 0.10, edgeOpacity: 0.5,
           dimNode: '#28304d', dimLink: '#1b2340', label: '#e2e8f0' },
  light: { bg: '#eef2f8', nodeLighten: 0.15, edgeDarken: 0.34, edgeOpacity: 0.62,
           dimNode: '#c7cfdc', dimLink: '#d5dbe6', label: '#1e293b' },
};

const SPECTRAL_SPAN = 260;   // half-extent of the fixed spectral 3D layout (spread so nodes don't overlap)
const spanOf = (arr) => (arr.length ? Math.max(...arr) - Math.min(...arr) : 0);

// A node's section name, with fallbacks: many chunks (plain-text or
// semantically-split docs) have no heading, so fall back to the document name,
// then the chunk position — every node still shows something meaningful.
function sectionOf(n) {
  const sp = n.section_path;
  if (sp && sp.length) return sp[sp.length - 1];
  if (n.doc) return n.doc;
  return `chunk ${n.position ?? '?'}`;
}

// Text for a node's floating label, per the chosen mode.
function nodeLabelText(n, mode) {
  return mode === 'section' ? sectionOf(n) : `deg ${n.degree || 0}`;
}

// SpriteText factory. Every node gets a label (text size scales with degree so
// hubs stand out); degree is always defined, section falls back via sectionOf().
function makeNodeObject(labelColor, mode) {
  return (n) => {
    const txt = String(nodeLabelText(n, mode));
    const s = new SpriteText(txt.length > 28 ? txt.slice(0, 28) + '…' : txt);
    s.color = labelColor;
    s.textHeight = 3.5 + Math.cbrt(n.degree || 0);
    s.fontWeight = '600';
    s.position.y = 5 + Math.sqrt(n.degree || 0) * 2;
    return s;
  };
}

// ── Component ─────────────────────────────────────────────────────────────────
export default function GraphPanel({ sessionId }) {
  const mountRef = useRef(null);
  const graphRef = useRef(null);
  const [data,     setData]     = useState(null);
  const [loading,  setLoading]  = useState(true);
  const [error,    setError]    = useState(null);
  const [selected, setSelected] = useState(null);
  const [layout,   setLayout]   = useState(null); // 'spectral' | 'force'
  const [theme,    setTheme]    = useState('dark');
  const [labelMode, setLabelMode] = useState('degree'); // node label: 'degree' | 'section'

  const load = useCallback(() => {
    setLoading(true); setError(null); setSelected(null);
    api.getGraph(sessionId)
      .then(setData)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [sessionId]);

  useEffect(() => { load(); }, [load]);

  // ── Build the 3D graph whenever data or theme changes ─────────────────────────
  useEffect(() => {
    const el = mountRef.current;
    if (!data || !el) return;
    const T = THEME[theme];

    // Fresh copies — 3d-force-graph mutates node/link objects with x/y/z + refs.
    const nodes = data.nodes.map(n => ({ ...n }));
    const links = data.edges.map(e => ({ ...e }));
    const byId = Object.fromEntries(nodes.map(n => [n.id, n]));

    // Precompute each node's neighbourhood for hover highlighting.
    nodes.forEach(n => { n._nbr = new Set(); n._lnk = new Set(); });
    links.forEach(l => {
      const a = byId[l.source], b = byId[l.target];
      if (!a || !b) return;
      a._nbr.add(b); b._nbr.add(a);
      a._lnk.add(l); b._lnk.add(l);
    });

    // Fixed spectral 3D positions when coords have spread; else force layout.
    const sx = nodes.map(n => n.sx ?? 0), sy = nodes.map(n => n.sy ?? 0), sz = nodes.map(n => n.sz ?? 0);
    const useSpectral = Math.max(spanOf(sx), spanOf(sy), spanOf(sz)) > 0.02;
    if (useSpectral) {
      const scaler = (arr) => { const mn = Math.min(...arr), sp = (Math.max(...arr) - mn) || 1; return (v) => (((v - mn) / sp) - 0.5) * 2 * SPECTRAL_SPAN; };
      const fx = scaler(sx), fy = scaler(sy), fz = scaler(sz);
      nodes.forEach(n => { n.fx = fx(n.sx ?? 0); n.fy = fy(n.sy ?? 0); n.fz = fz(n.sz ?? 0); });
    }
    setLayout(useSpectral ? 'spectral' : 'force');

    // Hover state (kept outside React so the render loop reads it directly).
    const hiN = new Set(), hiL = new Set();
    const baseNode = (n) => lighten(docColor(n.doc), T.nodeLighten);
    const baseLink = (l) => darken(l.dominant ? SIG_COLOR[l.dominant] : FALLBACK_COLOR, T.edgeDarken);
    const nodeColor = (n) => (hiN.size && !hiN.has(n)) ? T.dimNode : baseNode(n);
    const linkColor = (l) => (hiL.size && !hiL.has(l)) ? T.dimLink : baseLink(l);
    const linkWidth = (l) => (0.5 + (l.weight || 0) * 2.6) * (hiL.has(l) ? 2.2 : 1);
    const linkParts = (l) => (hiL.has(l) ? 4 : 0);

    const W = el.clientWidth || 900, H = el.clientHeight || 600;

    const graph = ForceGraph3D()(el)
      .width(W).height(H)
      .backgroundColor(T.bg)
      .showNavInfo(false)
      .graphData({ nodes, links })
      .nodeLabel(n => `<div class="g3d-tip">${n.label}`
        + `<div class="g3d-tip-sub">deg ${n.degree || 0} · ${sectionOf(n)}</div></div>`)
      .nodeColor(nodeColor)
      .nodeVal(n => 1 + Math.sqrt(n.degree || 0))   // hubs bigger, but gentler (no overlap blobs)
      .nodeRelSize(3)
      .nodeOpacity(0.95)
      .nodeResolution(16)
      .nodeThreeObjectExtend(true)
      .nodeThreeObject(makeNodeObject(T.label, labelMode))
      .linkColor(linkColor)
      .linkWidth(linkWidth)
      .linkOpacity(T.edgeOpacity)
      .linkDirectionalParticles(linkParts)
      .linkDirectionalParticleWidth(1.6)
      .linkDirectionalParticleSpeed(0.012)
      .onNodeHover(node => {
        hiN.clear(); hiL.clear();
        if (node) {
          hiN.add(node);
          node._nbr.forEach(x => hiN.add(x));
          node._lnk.forEach(x => hiL.add(x));
        }
        el.style.cursor = node ? 'pointer' : 'default';
        graph.nodeColor(nodeColor).linkColor(linkColor).linkWidth(linkWidth).linkDirectionalParticles(linkParts);
      })
      .onLinkHover(link => {
        hiN.clear(); hiL.clear();
        if (link) { hiL.add(link); hiN.add(link.source); hiN.add(link.target); }
        graph.nodeColor(nodeColor).linkColor(linkColor).linkWidth(linkWidth).linkDirectionalParticles(linkParts);
      })
      .onNodeClick(node => {
        setSelected({ type: 'node', data: node });
        // Smoothly fly the camera to frame the clicked node.
        const d = 70;
        const r = 1 + d / Math.max(1, Math.hypot(node.x || 0, node.y || 0, node.z || 0));
        graph.cameraPosition({ x: (node.x || 0) * r, y: (node.y || 0) * r, z: (node.z || 0) * r }, node, 1000);
      })
      .onLinkClick(l => setSelected({ type: 'edge', data: l }))
      .onBackgroundClick(() => setSelected(null));

    // Gentle repulsion for a cleaner, less jittery force layout.
    if (graph.d3Force?.('charge')) graph.d3Force('charge').strength(-90);

    // Fit the view once the layout has settled a little.
    const fit = setTimeout(() => { try { graph.zoomToFit(700, 60); } catch { /* ignore */ } }, 400);

    graphRef.current = graph;
    const ro = new ResizeObserver(() => graph.width(el.clientWidth || W).height(el.clientHeight || H));
    ro.observe(el);

    return () => {
      clearTimeout(fit);
      ro.disconnect();
      try { graph._destructor?.(); } catch { /* ignore */ }
      graphRef.current = null;
      el.innerHTML = '';
    };
  }, [data, theme]);

  // Switch node labels (degree ↔ section) live, without rebuilding the graph.
  useEffect(() => {
    const g = graphRef.current;
    if (!g) return;
    g.nodeThreeObject(makeNodeObject(THEME[theme].label, labelMode));
  }, [labelMode, theme]);

  // ── Loading / error screens ──────────────────────────────────────────────────
  if (loading) return (
    <div className="graph-panel">
      <div className="center-screen"><div className="spinner" /><p className="center-sub">Loading graph…</p></div>
    </div>
  );
  if (error) return (
    <div className="graph-panel">
      <div className="center-screen">
        <p className="center-title error-text">Failed to load graph</p>
        <p className="center-sub">{error}</p>
        <button className="btn-primary" onClick={load}>Retry</button>
      </div>
    </div>
  );

  const docNames = [...new Set((data?.nodes ?? []).map(d => d.doc))];

  return (
    <div className={`graph-panel graph-${theme}`}>

      {/* Stale-index banner */}
      {data && !data.has_signal_data && (
        <div className="stale-banner">
          <span>⚠ Edge signal data not available — re-upload documents to see signal colours.</span>
        </div>
      )}

      {/* Stats bar */}
      <div className="graph-stats">
        <span>
          <strong>{data.nodes.length}</strong> nodes ·{' '}
          <strong>{data.edges.length}</strong> edges
          {data.total_nodes > data.nodes.length &&
            <span className="graph-stats-dim"> (top {data.nodes.length} of {data.total_nodes})</span>}
          {layout && (
            <span className="layout-badge">{layout === 'spectral' ? '◈ Spectral 3D' : '⊞ Force 3D'}</span>
          )}
        </span>
        <span className="graph-stats-right">
          <span className="graph-label-toggle" title="What the hub-node labels show">
            <span className="graph-label-caption">Labels:</span>
            <button
              className={labelMode === 'degree' ? 'active' : ''}
              onClick={() => setLabelMode('degree')}
            >Degree</button>
            <button
              className={labelMode === 'section' ? 'active' : ''}
              onClick={() => setLabelMode('section')}
            >Section</button>
          </span>
          <button
            className="graph-theme-toggle"
            onClick={() => setTheme(t => (t === 'dark' ? 'light' : 'dark'))}
            title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {theme === 'dark' ? '☀ Light' : '🌙 Dark'}
          </button>
        </span>
      </div>

      {/* Graph body */}
      <div className="graph-body">
        <div ref={mountRef} className="graph-3d" />

        {/* Signal legend */}
        <div className="graph-legend signal-legend">
          <p className="legend-title">Signals</p>
          {SIGNALS.map(s => (
            <div key={s.key} className="legend-item">
              <span className="legend-dot" style={{ background: s.color }} />
              <span>{s.label}</span>
            </div>
          ))}
        </div>

        {/* Document legend */}
        {docNames.length > 0 && (
          <div className="graph-legend doc-legend">
            <p className="legend-title">Documents</p>
            {docNames.map(doc => (
              <div key={doc} className="legend-item">
                <span className="legend-dot" style={{ background: docColor(doc) }} />
                <span className="legend-doc-name" title={doc}>{doc}</span>
              </div>
            ))}
          </div>
        )}

        {/* Detail panel */}
        {selected && (
          <div className="graph-detail">
            <div className="detail-header">
              <span className="detail-type">{selected.type === 'node' ? 'Chunk' : 'Edge'}</span>
              <button className="detail-close" onClick={() => setSelected(null)}>×</button>
            </div>
            {selected.type === 'node' && <NodeDetail node={selected.data} />}
            {selected.type === 'edge' && (
              <EdgeDetail edge={selected.data} hasSignalData={data.has_signal_data} />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Node detail ───────────────────────────────────────────────────────────────
function NodeDetail({ node }) {
  return (
    <div className="detail-body">
      <p className="detail-label">Document</p>
      <p className="detail-value doc-name">{node.doc}</p>

      {node.section_path?.length > 0 && (
        <>
          <p className="detail-label">Section</p>
          <p className="detail-value">{node.section_path.join(' › ')}</p>
        </>
      )}

      <p className="detail-label">Position · Connections</p>
      <p className="detail-value">chunk {node.position} · {node.degree} edge{node.degree !== 1 ? 's' : ''}</p>

      <p className="detail-label">Full text</p>
      <pre className="detail-text-full">{node.text}</pre>
    </div>
  );
}

// ── Edge detail ───────────────────────────────────────────────────────────────
function EdgeDetail({ edge, hasSignalData }) {
  if (!hasSignalData) {
    return (
      <div className="detail-body">
        <p className="detail-label">Combined weight</p>
        <p className="detail-value">{edge.weight.toFixed(4)}</p>
        <div className="stale-inline">
          Signal breakdown unavailable — re-index to enable.
        </div>
      </div>
    );
  }

  const sorted = SIGNALS
    .map(s => ({ ...s, val: edge.signals?.[s.key] ?? 0 }))
    .filter(s => s.val > 0)
    .sort((a, b) => b.val - a.val);

  return (
    <div className="detail-body">
      <p className="detail-label">Combined weight</p>
      <p className="detail-value">{edge.weight.toFixed(4)}</p>

      <p className="detail-label">Signal breakdown</p>
      {sorted.length === 0
        ? <p className="detail-value" style={{ color: 'var(--text-3)' }}>No active signals</p>
        : (
          <div className="signal-bars">
            {sorted.map(s => (
              <div key={s.key} className="signal-row">
                <span className="signal-name">{s.label}</span>
                <div className="signal-bar-track">
                  <div className="signal-bar-fill" style={{ width: `${s.val * 100}%`, background: s.color }} />
                </div>
                <span className="signal-val">{s.val.toFixed(3)}</span>
              </div>
            ))}
          </div>
        )}

      {edge.dominant && (
        <>
          <p className="detail-label" style={{ marginTop: 14 }}>Dominant signal</p>
          <span className="dominant-badge" style={{ background: SIG_COLOR[edge.dominant] }}>
            {SIGNALS.find(s => s.key === edge.dominant)?.label ?? edge.dominant}
          </span>
        </>
      )}
    </div>
  );
}
