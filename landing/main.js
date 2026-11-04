/* SpectRAG landing — every visualization on this page is computed live from a toy corpus.
   No build step, no dependencies. */
(() => {
'use strict';

const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;

/* ═════════════════════════ utils ═════════════════════════ */
function rng(seed) {                      // mulberry32
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
const lerp = (a, b, t) => a + (b - a) * t;
const ease = t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
const hex = h => { const n = parseInt(h.slice(1), 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; };
const rgba = (c, a) => `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a})`;
const mix = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const fmt = n => n.toLocaleString('en-US');

const HEAT_STOPS = ['#172554', '#1d4ed8', '#38bdf8', '#e0f2fe', '#fbbf24'].map(hex);
function heat(v) {
  v = clamp(v) * (HEAT_STOPS.length - 1);
  const i = Math.min(HEAT_STOPS.length - 2, Math.floor(v));
  return mix(HEAT_STOPS[i], HEAT_STOPS[i + 1], v - i);
}

/* Visibility-gated animation loop */
const animators = [];
function addAnimator(el, tick) {
  const a = { el, tick, visible: false };
  animators.push(a);
  new IntersectionObserver(([e]) => { a.visible = e.isIntersecting; }, { rootMargin: '120px' }).observe(el);
  return a;
}
let lastT = performance.now();
function loop(now) {
  const dt = Math.min(.05, (now - lastT) / 1000);
  lastT = now;
  for (const a of animators) if (a.visible) a.tick(dt, now / 1000);
  requestAnimationFrame(loop);
}

/* Canvas with DPR + resize tracking */
function fitCanvas(canvas) {
  const st = { ctx: canvas.getContext('2d'), w: 1, h: 1 };
  const resize = () => {
    const r = canvas.getBoundingClientRect();
    const dpr = Math.min(2, devicePixelRatio || 1);
    st.w = Math.max(1, r.width); st.h = Math.max(1, r.height);
    canvas.width = Math.round(st.w * dpr); canvas.height = Math.round(st.h * dpr);
    st.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };
  new ResizeObserver(resize).observe(canvas);
  resize();
  return st;
}

/* ═════════════════════════ toy corpus + graph ═════════════════════════ */
const DOCS = [
  { name: 'Supplier_Agreement.pdf', short: 'Supplier', color: '#38bdf8', topic: 0 },
  { name: 'Licence_Terms.docx',     short: 'Licence',  color: '#a78bfa', topic: 0 },
  { name: 'DPA_Addendum.pdf',       short: 'DPA',      color: '#f472b6', topic: 1 },
  { name: 'Case_Law_Digest.html',   short: 'Case law', color: '#34d399', topic: 2 },
  { name: 'Regulatory_Guidance.pdf',short: 'Guidance', color: '#fbbf24', topic: 3 },
  { name: 'Board_Minutes.docx',     short: 'Minutes',  color: '#fb923c', topic: 4 },
];
const CONCEPTS = ['survival clause', 'termination', 'confidentiality', 'data return', 'sub-processor', 'deletion',
  'liability cap', 'precedent', 'damages', 'reporting duty', 'supervisory authority', 'fines', 'resolution', 'vote', 'audit'];

// Values from config.py :: EdgeWeights
const SIGNALS = [
  { key: 'utility_question', label: 'Utility-question', w: .280, color: '#34d399', file: 'edges/utility_question.py',
    desc: 'gpt-4o-mini writes 2 questions per chunk. Chunks whose questions match (cos ≥ 0.80) answer the same kind of question.' },
  { key: 'entity', label: 'Entity / concept', w: .250, color: '#f472b6', file: 'edges/entity.py',
    desc: 'Chunks sharing an LLM-extracted concept get linked, even when they use different words. The main multi-hop link.' },
  { key: 'section', label: 'Section', w: .200, color: '#a78bfa', file: 'edges/section.py',
    desc: 'Chunks under the same heading path. A deeper shared prefix gives a stronger edge.' },
  { key: 'semantic', label: 'Semantic', w: .150, color: '#38bdf8', file: 'edges/semantic.py',
    desc: 'k-NN cosine edges (k = 7) over nomic-embed-text vectors, kept when cos ≥ 0.75.' },
  { key: 'adjacency', label: 'Adjacency', w: .100, color: '#94a3b8', file: 'edges/adjacency.py',
    desc: 'Consecutive chunks of the same document, in reading order.' },
  { key: 'citation', label: 'Citation', w: .020, color: '#fbbf24', file: 'edges/citation.py',
    desc: 'Shared bracket references, DOIs or URLs. Sparse, so its weight is small.' },
];
const SIG = Object.fromEntries(SIGNALS.map(s => [s.key, s]));
const CLUSTER_COLORS = ['#38bdf8', '#a78bfa', '#f472b6', '#34d399', '#fbbf24', '#fb923c'];

function buildGraph() {
  const R = rng(20261017);
  const chunks = [];
  DOCS.forEach((doc, d) => {
    for (let p = 0; p < 12; p++) {
      let topic = doc.topic;
      if (R() > .8) topic = Math.floor(R() * 5);
      const concepts = new Set(R() < .5 ? [topic * 3 + Math.floor(R() * 3)] : []);
      if (R() < .15) concepts.add(Math.floor(R() * 15));
      chunks.push({ id: chunks.length, doc: d, pos: p, section: Math.floor(p / 4), topic, concepts,
        tokens: 600 + Math.floor(R() * 600) });
    }
  });
  // Hand-placed bridges so the story in the mockup holds: the supplier agreement pulls in the DPA,
  // the licence and the case-law digest both talk about survival.
  const at = (d, p) => chunks[d * 12 + p];
  at(0, 7).concepts.add(3); at(2, 2).concepts.add(3); at(2, 3).concepts.add(5);
  at(1, 4).concepts.add(0); at(3, 5).concepts.add(0); at(3, 5).concepts.add(6); at(0, 9).concepts.add(6);
  for (const [a, b, cs] of [[[0, 9], [2, 2], [3, 5]], [[1, 8], [3, 5], [0, 6]], [[0, 10], [2, 3], [3, 5]]]) {
    cs.forEach(c => { at(...a).concepts.add(c); at(...b).concepts.add(c); });
  }
  at(0, 2).concepts.add(0); at(1, 8).concepts.add(2); at(0, 3).concepts.add(2);

  const N = chunks.length;
  const E = new Map();
  const add = (i, j, sig, s) => {
    if (i === j) return;
    const a = Math.min(i, j), b = Math.max(i, j), k = a * N + b;
    let e = E.get(k);
    if (!e) { e = { i: a, j: b, s: {} }; E.set(k, e); }
    e.s[sig] = Math.max(e.s[sig] || 0, s);
  };
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
    const ci = chunks[i], cj = chunks[j];
    const sameDoc = ci.doc === cj.doc, sameTopic = ci.topic === cj.topic;
    if (sameDoc && cj.pos === ci.pos + 1) add(i, j, 'adjacency', 1);
    if (sameDoc && ci.section === cj.section && cj.pos > ci.pos + 1 && R() < .5) add(i, j, 'section', 1);
    if (sameDoc && ci.section !== cj.section && R() < .05) add(i, j, 'section', .5);
    if (sameTopic && R() < .2) { const c = .72 + .28 * R(); if (c >= .75) add(i, j, 'semantic', c); }
    else if (!sameTopic && R() < .008) add(i, j, 'semantic', .75 + .05 * R());
    if (sameTopic && (!sameDoc ? R() < .12 : R() < .04)) add(i, j, 'utility_question', .8 + .2 * R());
    if (R() < .008) add(i, j, 'citation', 1);
    let shared = 0;
    for (const c of ci.concepts) if (cj.concepts.has(c)) shared++;
    if (shared) add(i, j, 'entity', shared);
  }
  // The bridges answer the same kind of question as their contract clause, so the
  // utility-question signal links them too (strong, multi-signal edges).
  for (const [a, b] of [[[0, 9], [2, 2]], [[1, 8], [3, 5]], [[0, 10], [2, 3]]]) add(at(...a).id, at(...b).id, 'utility_question', .97);
  const edges = [...E.values()];
  // min-max normalise each signal independently (graph_builder.py)
  for (const s of SIGNALS) {
    const v = edges.filter(e => e.s[s.key] != null).map(e => e.s[s.key]);
    const mn = Math.min(...v), mx = Math.max(...v);
    for (const e of edges) if (e.s[s.key] != null) {
      e.s[s.key] = mx === mn ? 1 : (e.s[s.key] - mn) / (mx - mn);
    }
  }
  for (const e of edges) {
    e.W = 0; let best = -1;
    for (const s of SIGNALS) if (e.s[s.key] != null) {
      const c = s.w * e.s[s.key];
      e.W += c;
      if (c > best) { best = c; e.dom = s.key; }
    }
    e.kept = e.W >= 0.1 - 1e-9;            // edge_sparsify_threshold
  }
  return { chunks, edges, N };
}

const G = buildGraph();
const KEPT = G.edges.filter(e => e.kept);

/* ── layouts ── */
function normalise(pts, m = .06) {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const p of pts) { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); }
  return pts.map(p => [m + (p[0] - x0) / (x1 - x0 || 1) * (1 - 2 * m), m + (p[1] - y0) / (y1 - y0 || 1) * (1 - 2 * m)]);
}
function layoutDocs() {
  return G.chunks.map(c => [(c.doc + .5) / DOCS.length, .1 + c.pos / 11 * .8]);
}
function layoutEmbedding() {
  const R = rng(3);
  const centres = [[.3, .35], [.68, .3], [.75, .7], [.32, .72], [.52, .52]];
  const g = () => (R() + R() + R() - 1.5) * .1;
  return normalise(G.chunks.map(c => [centres[c.topic][0] + g(), centres[c.topic][1] + g()]), .08);
}
function layoutForce() {
  const R = rng(11), N = G.N;
  const P = G.chunks.map(c => {
    const a = c.doc / DOCS.length * Math.PI * 2;
    return [.5 + Math.cos(a) * .3 + (R() - .5) * .1, .5 + Math.sin(a) * .3 + (R() - .5) * .1];
  });
  for (let it = 0; it < 420; it++) {
    const F = P.map(() => [0, 0]);
    const cool = 1 - it / 420;
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
      let dx = P[i][0] - P[j][0], dy = P[i][1] - P[j][1];
      const d2 = dx * dx + dy * dy + 1e-4, f = .00045 / d2;
      F[i][0] += dx * f; F[i][1] += dy * f; F[j][0] -= dx * f; F[j][1] -= dy * f;
    }
    for (const e of KEPT) {
      const a = P[e.i], b = P[e.j];
      const dx = b[0] - a[0], dy = b[1] - a[1], d = Math.hypot(dx, dy) + 1e-6;
      const f = (d - .05) * .06 * (.4 + e.W);
      F[e.i][0] += dx / d * f; F[e.i][1] += dy / d * f; F[e.j][0] -= dx / d * f; F[e.j][1] -= dy / d * f;
    }
    for (let i = 0; i < N; i++) {
      F[i][0] += (.5 - P[i][0]) * .012; F[i][1] += (.5 - P[i][1]) * .012;
      const m = Math.hypot(F[i][0], F[i][1]), cap = .02 * cool + .002;
      const s = m > cap ? cap / m : 1;
      P[i][0] += F[i][0] * s; P[i][1] += F[i][1] * s;
    }
  }
  return normalise(P, .07);
}

/* ── spectral decomposition: L = I − D^−½ W D^−½, cyclic Jacobi ── */
function jacobi(A, n) {
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p * n + q] ** 2;
    if (off < 1e-18) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      const apq = A[p * n + q];
      if (Math.abs(apq) < 1e-15) continue;
      const app = A[p * n + p], aqq = A[q * n + q];
      const th = (aqq - app) / (2 * apq);
      const t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) {
        if (k === p || k === q) continue;
        const akp = A[k * n + p], akq = A[k * n + q];
        A[k * n + p] = A[p * n + k] = c * akp - s * akq;
        A[k * n + q] = A[q * n + k] = s * akp + c * akq;
      }
      A[p * n + p] = app - t * apq; A[q * n + q] = aqq + t * apq; A[p * n + q] = A[q * n + p] = 0;
      for (let k = 0; k < n; k++) {
        const vkp = V[k * n + p], vkq = V[k * n + q];
        V[k * n + p] = c * vkp - s * vkq; V[k * n + q] = s * vkp + c * vkq;
      }
    }
  }
  const vals = [...Array(n).keys()].map(i => A[i * n + i]);
  const order = [...Array(n).keys()].sort((a, b) => vals[a] - vals[b]);
  return { vals: order.map(i => vals[i]), vec: k => { const o = order[k]; return [...Array(n).keys()].map(i => V[i * n + o]); } };
}
function spectral() {
  const n = G.N, W = new Float64Array(n * n), deg = new Float64Array(n);
  for (const e of KEPT) { W[e.i * n + e.j] = W[e.j * n + e.i] = e.W; deg[e.i] += e.W; deg[e.j] += e.W; }
  const ds = [...deg].map(d => d > 0 ? 1 / Math.sqrt(d) : 0);
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    L[i * n + j] = (i === j && deg[i] > 0 ? 1 : 0) - ds[i] * W[i * n + j] * ds[j];
  }
  const eig = jacobi(L, n);
  const K = 6;
  const U = [];                         // eigvecs 1..K (skip the trivial one)
  for (let k = 1; k <= K; k++) U.push(eig.vec(k));
  const coords = G.chunks.map((_, i) => U.map(u => u[i] * ds[i]));
  // robust 2-D view of the first two non-trivial eigenvectors
  const robust = vals => {
    const s = [...vals].sort((a, b) => a - b), lo = s[Math.floor(n * .03)], hi = s[Math.floor(n * .97)];
    return vals.map(v => .08 + clamp((v - lo) / (hi - lo || 1)) * .84);
  };
  const xs = robust(coords.map(c => c[0])), ys = robust(coords.map(c => c[1]));
  const view = xs.map((x, i) => [x, ys[i]]);
  const allVecs = [];
  for (let k = 0; k < n; k++) allVecs.push(eig.vec(k));
  return { vals: eig.vals, coords, view, allVecs };
}
function kmeans(X, k, seed = 5) {
  const R = rng(seed), n = X.length, dim = X[0].length;
  const d2 = (a, b) => a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0);
  const C = [X[Math.floor(R() * n)].slice()];
  while (C.length < k) {                 // k-means++
    const D = X.map(x => Math.min(...C.map(c => d2(x, c))));
    let r = R() * D.reduce((a, b) => a + b, 0), i = 0;
    while ((r -= D[i]) > 0 && i < n - 1) i++;
    C.push(X[i].slice());
  }
  let lab = new Array(n).fill(0);
  for (let it = 0; it < 40; it++) {
    lab = X.map(x => { let b = 0, bd = Infinity; C.forEach((c, j) => { const d = d2(x, c); if (d < bd) { bd = d; b = j; } }); return b; });
    for (let j = 0; j < k; j++) {
      const m = X.filter((_, i) => lab[i] === j);
      if (m.length) for (let t = 0; t < dim; t++) C[j][t] = m.reduce((s, x) => s + x[t], 0) / m.length;
    }
  }
  return lab;
}

// 2-D view: PCA of the row-normalised spectral coordinates (power iteration with deflation)
function pca2(X) {
  const n = X.length, d = X[0].length;
  const mu = [...Array(d).keys()].map(t => X.reduce((s, x) => s + x[t], 0) / n);
  const Y = X.map(x => x.map((v, t) => v - mu[t]));
  const C = [...Array(d)].map((_, a) => [...Array(d)].map((_, b) => Y.reduce((s, y) => s + y[a] * y[b], 0) / n));
  const comps = [];
  for (let c = 0; c < 2; c++) {
    let v = [...Array(d)].map((_, t) => 1 / (t + 1 + c));
    for (let it = 0; it < 200; it++) {
      let w = C.map(row => row.reduce((s, x, t) => s + x * v[t], 0));
      for (const u of comps) { const dot = w.reduce((s, x, t) => s + x * u[t], 0); w = w.map((x, t) => x - dot * u[t]); }
      const m = Math.hypot(...w) || 1; v = w.map(x => x / m);
    }
    comps.push(v);
  }
  return Y.map(y => comps.map(u => y.reduce((s, x, t) => s + x * u[t], 0)));
}

const POS = { docs: layoutDocs(), embed: layoutEmbedding(), force: layoutForce() };
const SPEC = spectral();
// unit-normalise spectral rows before clustering, as is usual for spectral clustering
const SPEC5 = SPEC.coords.map(c => { const v = c.slice(0, 5); const m = Math.hypot(...v) || 1; return v.map(x => x / m); });
const CLUSTER = kmeans(SPEC5, 5);
POS.spectral = normalise(pca2(SPEC5), .1);

/* ── query-time pipeline on the toy graph ── */
function runQuery() {
  const R = rng(99), N = G.N;
  const rel = G.chunks.map(c => {
    // The question names the supplier and licence agreements; their termination sections score highest.
    // Other documents that discuss surviving obligations are related but phrased differently, so dense
    // similarity alone ranks them below the rest of the two contracts.
    const contract = c.doc < 2 && c.topic === 0;
    const onPoint = c.section === 2;
    if (contract) return onPoint ? .62 + .3 * R() : .38 + .14 * R();
    if (c.concepts.has(0) || c.concepts.has(3) || c.concepts.has(6)) return .38 + .06 * R();
    return .06 + .26 * R();
  });
  const dense = rel.map(r => r + (R() - .5) * .12);
  const bm25 = rel.map(r => clamp(r + (R() - .5) * .4));
  const uq = rel.map(r => r + (R() - .5) * .2);
  const rankOf = s => { const o = [...s.keys()].sort((a, b) => s[b] - s[a]); const rk = new Array(N); o.forEach((i, r) => rk[i] = r + 1); return rk; };
  const rd = rankOf(dense), rb = rankOf(bm25), ru = rankOf(uq);
  const rrf = [...Array(N).keys()].map(i => 1 / (60 + rd[i]) + 1 / (60 + rb[i]) + 1 / (60 + ru[i]));
  const SEEDS = 6;
  const seeds = [...rrf.keys()].sort((a, b) => rrf[b] - rrf[a]).slice(0, SEEDS);
  const nz = a => { const mn = Math.min(...a), mx = Math.max(...a); return a.map(v => (v - mn) / (mx - mn || 1)); };
  const dn = nz(dense), bn = nz(bm25);
  const allScores = dn.map((v, i) => .7 * v + .3 * bn[i]);   // hybrid_lexical_weight = 0.30

  // Stage 2: 2-hop BFS, capped
  const adj = G.chunks.map(() => []);
  for (const e of KEPT) { adj[e.i].push([e.j, e.W]); adj[e.j].push([e.i, e.W]); }
  const CAP = 40;
  const hop = new Array(N).fill(-1), parent = new Array(N).fill(-1);
  seeds.forEach(s => hop[s] = 0);
  let frontier = seeds.slice(), count = seeds.length;
  for (let h = 1; h <= 2; h++) {
    const next = new Map();
    for (const u of frontier) for (const [v, w] of adj[u]) if (hop[v] < 0) {
      if (!next.has(v) || next.get(v)[1] < w) next.set(v, [u, w]);
    }
    const ordered = [...next.entries()].sort((a, b) => b[1][1] - a[1][1]);
    frontier = [];
    for (const [v, [u]] of ordered) {
      if (count >= CAP) break;
      hop[v] = h; parent[v] = u; frontier.push(v); count++;
    }
  }
  const cand = [...Array(N).keys()].filter(i => hop[i] >= 0);
  const inC = new Set(cand);

  // Stage 3: PPR over the candidate subgraph + spectral proximity
  const subAdj = new Map(cand.map(i => [i, adj[i].filter(([j]) => inC.has(j))]));
  let p = new Map(cand.map(i => [i, seeds.includes(i) ? 1 / SEEDS : 0]));
  const pers = new Map(p);
  for (let it = 0; it < 60; it++) {
    const q = new Map(cand.map(i => [i, .15 * pers.get(i)]));
    for (const i of cand) {
      const nb = subAdj.get(i), deg = nb.reduce((s, [, w]) => s + w, 0);
      if (!deg) { q.set(i, q.get(i) + .85 * p.get(i)); continue; }
      for (const [j, w] of nb) q.set(j, q.get(j) + .85 * p.get(i) * w / deg);
    }
    p = q;
  }
  const pprMax = Math.max(...p.values());
  const cos = (a, b) => { let s = 0, na = 0, nb = 0; for (let t = 0; t < a.length; t++) { s += a[t] * b[t]; na += a[t] ** 2; nb += b[t] ** 2; } return s / (Math.sqrt(na * nb) || 1); };
  const specProx = new Map(cand.map(i => [i, Math.max(...seeds.map(s => (cos(SPEC5[i], SPEC5[s]) + 1) / 2))]));
  const spMax = Math.max(...specProx.values());
  const relMax = Math.max(...cand.map(i => allScores[i]));
  const combined = new Map(), prox = new Map();
  for (const i of cand) {
    const pr = .5 * (specProx.get(i) / spMax) + .5 * (p.get(i) / pprMax);   // diffusion_mode = hybrid
    prox.set(i, pr);
    combined.set(i, .5 * (allScores[i] / relMax) + .5 * pr);              // α = 0.5
  }

  // Stage 4: token_budget selection
  const BUDGET = 12000;
  const sorted = cand.slice().sort((a, b) => combined.get(b) - combined.get(a));
  const mx = combined.get(sorted[0]);
  const head = .3 * mx, tail = .55 * mx;
  const selected = [], covered = new Set(), gated = [];
  let used = 0;
  const pool = sorted.slice();
  while (pool.length) {
    let k = 0;
    const top = combined.get(pool[0]);
    for (let t = 0; t < pool.length && combined.get(pool[t]) >= top - .02; t++) {
      if (!covered.has(G.chunks[pool[t]].doc)) { k = t; break; }
    }
    const c = pool.splice(k, 1)[0];
    const floor = selected.length < 5 ? head : tail;
    if (combined.get(c) < floor) { gated.push(c); continue; }
    if (used + G.chunks[c].tokens > BUDGET) break;
    selected.push(c); used += G.chunks[c].tokens; covered.add(G.chunks[c].doc);
  }
  for (const c of gated) {                // budget backfill
    if (combined.get(c) < head || used + G.chunks[c].tokens > BUDGET) continue;
    selected.push(c); used += G.chunks[c].tokens; covered.add(G.chunks[c].doc);
  }
  const naive = [...dense.keys()].sort((a, b) => dense[b] - dense[a]).slice(0, selected.length);
  const naiveDocs = new Set(naive.map(i => G.chunks[i].doc));
  return { rel, seeds, hop, parent, cand, combined, prox, selected, used, BUDGET, covered, naive, naiveDocs, rrf };
}
const Q = runQuery();

/* ═════════════════════════ graph scene renderer ═════════════════════════ */
const SLATE = hex('#7c8db5');
class Scene {
  constructor(canvas, { pad = 34, padTop = 34 } = {}) {
    this.c = fitCanvas(canvas); this.pad = pad; this.padTop = padTop;
    this.nodes = G.chunks.map((_, i) => {
      const [x, y] = POS.force[i];
      return { x, y, r: 4, a: 0, c: SLATE.slice(), ring: 0, tx: x, ty: y, tr: 4, ta: 0, tc: SLATE.slice(), tring: 0 };
    });
    this.edges = G.edges.map(e => ({ e, a: 0, c: SLATE.slice(), w: 1, ta: 0, tc: SLATE.slice(), tw: 1 }));
    this.speed = 5;
  }
  layout(pos) { this.nodes.forEach((n, i) => { n.tx = pos[i][0]; n.ty = pos[i][1]; }); }
  setNodes(fn) { this.nodes.forEach((n, i) => { const o = fn(i, n); if (o) { if (o.r != null) n.tr = o.r; if (o.a != null) n.ta = o.a; if (o.c) n.tc = o.c; if (o.ring != null) n.tring = o.ring; } }); }
  setEdges(fn) { this.edges.forEach((d, k) => { const o = fn(d.e, k); if (o) { if (o.a != null) d.ta = o.a; if (o.c) d.tc = o.c; if (o.w != null) d.tw = o.w; } else d.ta = 0; }); }
  snap() {
    for (const n of this.nodes) { n.x = n.tx; n.y = n.ty; n.r = n.tr; n.a = n.ta; n.c = n.tc.slice(); n.ring = n.tring; }
    for (const d of this.edges) { d.a = d.ta; d.c = d.tc.slice(); d.w = d.tw; }
  }
  px(n) {
    const { w, h } = this.c;
    return [this.pad + n.x * (w - 2 * this.pad), this.padTop + n.y * (h - this.padTop - this.pad)];
  }
  frame(dt, now) {
    const k = 1 - Math.exp(-dt * this.speed);
    for (const n of this.nodes) {
      n.x = lerp(n.x, n.tx, k); n.y = lerp(n.y, n.ty, k); n.r = lerp(n.r, n.tr, k); n.a = lerp(n.a, n.ta, k);
      n.c = mix(n.c, n.tc, k); n.ring = lerp(n.ring, n.tring, k);
    }
    for (const d of this.edges) { d.a = lerp(d.a, d.ta, k); d.c = mix(d.c, d.tc, k); d.w = lerp(d.w, d.tw, k); }
    const { ctx, w, h } = this.c;
    ctx.clearRect(0, 0, w, h);
    const P = this.nodes.map(n => this.px(n));
    ctx.lineCap = 'round';
    for (const d of this.edges) {
      if (d.a < .01) continue;
      const a = P[d.e.i], b = P[d.e.j];
      ctx.strokeStyle = rgba(d.c, d.a); ctx.lineWidth = d.w;
      ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
    }
    this.nodes.forEach((n, i) => {
      if (n.a < .01) return;
      const [x, y] = P[i];
      const g = ctx.createRadialGradient(x, y, 0, x, y, n.r * 3.2);
      g.addColorStop(0, rgba(n.c, .35 * n.a)); g.addColorStop(1, rgba(n.c, 0));
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, n.r * 3.2, 0, 7); ctx.fill();
      ctx.fillStyle = rgba(n.c, n.a); ctx.beginPath(); ctx.arc(x, y, n.r, 0, 7); ctx.fill();
      if (n.ring > .02) {
        const pulse = (now * 1.2 + i * .13) % 1;
        ctx.strokeStyle = rgba([255, 255, 255], n.ring * .9); ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.arc(x, y, n.r + 3.5, 0, 7); ctx.stroke();
        ctx.strokeStyle = rgba(n.c, n.ring * (1 - pulse) * .8); ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.arc(x, y, n.r + 4 + pulse * 12, 0, 7); ctx.stroke();
      }
    });
    if (this.overlay) this.overlay(ctx, P, now);
  }
}

/* ═════════════════════════ hero particle network ═════════════════════════ */
(function heroNet() {
  const canvas = $('#heroNet');
  const st = fitCanvas(canvas);
  const R = rng(4);
  const pts = [...Array(64)].map(() => ({ x: R(), y: R(), vx: (R() - .5) * .012, vy: (R() - .5) * .012, r: 1 + R() * 1.8 }));
  let mx = -1, my = -1;
  canvas.parentElement.addEventListener('pointermove', e => { const r = canvas.getBoundingClientRect(); mx = (e.clientX - r.left) / r.width; my = (e.clientY - r.top) / r.height; });
  const draw = dt => {
    const { ctx, w, h } = st;
    ctx.clearRect(0, 0, w, h);
    for (const p of pts) {
      p.x += p.vx * dt * 3; p.y += p.vy * dt * 3;
      if (p.x < 0 || p.x > 1) p.vx *= -1; if (p.y < 0 || p.y > 1) p.vy *= -1;
    }
    for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) {
      const a = pts[i], b = pts[j];
      const d = Math.hypot((a.x - b.x) * w, (a.y - b.y) * h);
      if (d < 150) {
        const near = mx >= 0 ? Math.max(0, 1 - Math.hypot((a.x - mx) * w, (a.y - my) * h) / 260) : 0;
        ctx.strokeStyle = `rgba(125,170,255,${(1 - d / 150) * (.12 + near * .5)})`;
        ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(a.x * w, a.y * h); ctx.lineTo(b.x * w, b.y * h); ctx.stroke();
      }
    }
    for (const p of pts) { ctx.fillStyle = 'rgba(160,200,255,.55)'; ctx.beginPath(); ctx.arc(p.x * w, p.y * h, p.r, 0, 7); ctx.fill(); }
  };
  draw(0);
  if (!REDUCED) addAnimator(canvas, draw);
})();

/* ═════════════════════════ problem: top-k vs SpectRAG (SVG animation) ═════════════════════════ */
(function compare() {
  const svg = $('#cmpSvg');
  const NS = 'http://www.w3.org/2000/svg';
  const el = (tag, attrs, parent = svg) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; };
  const DOCC = { A: '#38bdf8', B: '#a78bfa', C: '#f472b6', D: '#34d399' };
  // one shared corpus, drawn twice (coordinates inside a 480-wide panel)
  const P = [
    ['A', 212, 236], ['A', 222, 244], ['A', 206, 250], ['A', 228, 230], ['A', 186, 214], ['A', 246, 262], ['A', 170, 250], ['A', 236, 204],
    ['B', 330, 330], ['B', 352, 352], ['B', 316, 360], ['B', 366, 318], ['B', 300, 336],
    ['C', 112, 352], ['C', 92, 330], ['C', 132, 380], ['C', 84, 372],
    ['D', 368, 132], ['D', 392, 152], ['D', 350, 112], ['D', 400, 118],
  ].map(([d, x, y], i) => ({ d, x, y, i }));
  P.forEach(p => { p.name = p.d + (P.filter(q => q.d === p.d).indexOf(p) + 1); });
  const QX = 214, QY = 214;
  const dist = p => Math.hypot(p.x - QX, p.y - QY);
  const naive = P.slice().sort((a, b) => dist(a) - dist(b)).slice(0, 6).map(p => p.i);
  const dupes = new Set([0, 1, 2, 3]);         // near-identical passages of doc A
  const spect = [0, 4, 7, 8, 9, 13];            // core of A + bridge into B + C
  const links = [[0, 8, 'entity'], [4, 13, 'section'], [8, 9, 'adjacency'], [7, 8, 'utility_question'], [0, 4, 'semantic'], [4, 7, 'semantic']];

  el('defs', {}).innerHTML = `
    <radialGradient id="cmpGlow"><stop offset="0" stop-color="#38bdf8" stop-opacity=".35"/><stop offset="1" stop-color="#38bdf8" stop-opacity="0"/></radialGradient>
    <filter id="cmpBlur"><feGaussianBlur stdDeviation="3"/></filter>`;
  el('line', { x1: 480, y1: 20, x2: 480, y2: 520, stroke: 'rgba(148,170,220,.14)' });

  const panels = [0, 480].map((ox, side) => {
    const g = el('g', { transform: `translate(${ox},0)` });
    el('text', { x: 28, y: 44, fill: side ? '#7dd3fc' : '#94a0bd', 'font-size': 13, 'font-family': 'JetBrains Mono', 'letter-spacing': '.08em' }, g).textContent = side ? 'SPECTRAG' : 'NAIVE RAG';
    el('text', { x: 28, y: 66, fill: '#e9eefb', 'font-size': 17, 'font-weight': 600 }, g).textContent = side ? 'Graph-aware, token-budgeted set' : 'Top-6 by cosine similarity';
    const radius = el('circle', { cx: QX, cy: QY, r: 0, fill: 'url(#cmpGlow)', stroke: 'rgba(125,211,252,.5)', 'stroke-dasharray': '4 5' }, g);
    const edges = side ? links.map(([a, b, s]) => {
      const pa = P[a], pb = P[b], len = Math.hypot(pb.x - pa.x, pb.y - pa.y);
      return { len, line: el('line', { x1: pa.x, y1: pa.y, x2: pb.x, y2: pb.y, stroke: SIG[s].color, 'stroke-width': 2, 'stroke-dasharray': len, 'stroke-dashoffset': len, opacity: .9 }, g) };
    }) : [];
    const dots = P.map(p => el('circle', { cx: p.x, cy: p.y, r: 6, fill: DOCC[p.d], opacity: 0 }, g));
    const labels = ['A', 'B', 'C', 'D'].map(d => {
      const pts = P.filter(p => p.d === d), cx = pts.reduce((s, p) => s + p.x, 0) / pts.length, cy = Math.min(...pts.map(p => p.y)) - 18;
      const t = el('text', { x: cx, y: cy, fill: DOCC[d], 'font-size': 12, 'text-anchor': 'middle', 'font-family': 'JetBrains Mono', opacity: 0 }, g);
      t.textContent = `Doc ${d}`; return t;
    });
    const star = el('g', { transform: `translate(${QX},${QY})`, opacity: 0 }, g);
    el('circle', { r: 16, fill: '#34d399', opacity: .18, filter: 'url(#cmpBlur)' }, star);
    el('path', { d: 'M0,-9 L2.6,-2.8 9,-2.8 3.8,1.2 5.6,8 0,4 -5.6,8 -3.8,1.2 -9,-2.8 -2.6,-2.8Z', fill: '#34d399' }, star);
    el('text', { x: -14, y: 4, fill: '#34d399', 'font-size': 12, 'font-family': 'JetBrains Mono', 'text-anchor': 'end' }, star).textContent = 'query';

    el('text', { x: 45, y: 478, fill: '#8aa0c7', 'font-size': 13, 'font-weight': 600 }, g).textContent = 'Context window (6 chunks)';
    const slots = [...Array(6)].map((_, k) => {
      const x = 44 + k * 66;
      const r = el('rect', { x, y: 490, width: 58, height: 36, rx: 7, fill: '#0b1426', stroke: 'rgba(148,170,220,.22)' }, g);
      const t = el('text', { x: x + 29, y: 513, 'text-anchor': 'middle', 'font-size': 12, 'font-family': 'JetBrains Mono', fill: '#e9eefb', opacity: 0 }, g);
      return { r, t, x };
    });
    const flyers = [...Array(6)].map(() => el('circle', { r: 6, opacity: 0 }, g));
    const badge = el('g', { opacity: 0 }, g);
    el('rect', { x: 250, y: 410, width: 206, height: 40, rx: 10, fill: side ? 'rgba(52,211,153,.1)' : 'rgba(251,113,133,.1)', stroke: side ? 'rgba(52,211,153,.5)' : 'rgba(251,113,133,.5)' }, badge);
    el('text', { x: 353, y: 435, 'text-anchor': 'middle', 'font-size': 13.5, fill: side ? '#a7f3d0' : '#fecdd3', 'font-weight': 600 }, badge).textContent = side ? '✓ 3 sources · bridge reached' : '✕ 1 source · 4 near-duplicates';
    return { g, radius, edges, dots, labels, star, slots, flyers, badge, picks: side ? spect : naive, side };
  });

  const T = 11;                         // loop length (s)
  const render = t => {
    const fadeAll = t > T - 1 ? 1 - (t - (T - 1)) : clamp(t / .8);
    for (const pn of panels) {
      pn.g.setAttribute('opacity', fadeAll);
      pn.dots.forEach((d, i) => d.setAttribute('opacity', clamp((t - i * .03) / .5) * (pn.picks.includes(i) && t > 2 ? 1 : .55)));
      pn.labels.forEach(l => l.setAttribute('opacity', clamp((t - .4) / .5) * .9));
      pn.star.setAttribute('opacity', clamp((t - .6) / .4));
      const s = 1 + .08 * Math.sin(t * 5);
      pn.star.setAttribute('transform', `translate(${QX},${QY}) scale(${s})`);
      if (!pn.side) {
        const rr = Math.max(...pn.picks.map(i => dist(P[i]))) + 10;
        pn.radius.setAttribute('r', rr * ease(clamp((t - 1.1) / 1)));
      } else {
        pn.radius.setAttribute('r', 46 * ease(clamp((t - 1.1) / .8)));
        pn.edges.forEach((e, k) => e.line.setAttribute('stroke-dashoffset', e.len * (1 - ease(clamp((t - 1.8 - k * .25) / .5)))));
      }
      pn.picks.forEach((pi, k) => {
        const start = 3.4 + k * .42, p = ease(clamp((t - start) / .55));
        const f = pn.flyers[k], slot = pn.slots[k], src = P[pi];
        const tx = slot.x + 29, ty = 508;
        f.setAttribute('cx', lerp(src.x, tx, p)); f.setAttribute('cy', lerp(src.y, ty, p) - Math.sin(p * Math.PI) * 40);
        f.setAttribute('fill', DOCC[src.d]); f.setAttribute('opacity', p > 0 && p < 1 ? 1 : 0);
        const filled = p >= 1;
        slot.r.setAttribute('fill', filled ? DOCC[src.d] + '33' : '#0b1426');
        slot.r.setAttribute('stroke', filled ? DOCC[src.d] : 'rgba(148,170,220,.22)');
        const dup = !pn.side && dupes.has(pi) && pi !== 0;
        slot.t.textContent = dup ? `${src.name}·dup` : src.name;
        slot.t.setAttribute('fill', dup ? '#fda4af' : '#e9eefb');
        slot.t.setAttribute('opacity', filled ? 1 : 0);
      });
      pn.badge.setAttribute('opacity', clamp((t - 6.3) / .5));
    }
  };
  let t0 = null;
  render(REDUCED ? 8 : 0);
  if (!REDUCED) addAnimator(svg, (dt, now) => { if (t0 == null) t0 = now; render((now - t0) % T); });
})();

/* ═════════════════════════ pipeline explorer ═════════════════════════ */
(function explorer() {
  const scene = new Scene($('#exCanvas'), { pad: 30, padTop: 30 });
  const info = $('#exInfo'), legend = $('#exLegend'), stepsEl = $('#exSteps'), qEl = $('#exQuery');
  const docRGB = DOCS.map(d => hex(d.color));
  const sigRGB = Object.fromEntries(SIGNALS.map(s => [s.key, hex(s.color)]));
  const CYAN = hex('#38bdf8'), WHITE = hex('#f0f9ff'), ROSE = hex('#fb7185'), SKY = hex('#7dd3fc');
  const lbl = i => `${DOCS[G.chunks[i].doc].short} #${G.chunks[i].pos + 1}`;
  const legendDocs = () => DOCS.map(d => `<span><i style="--c:${d.color}"></i>${d.short}</span>`).join('');
  const legendSig = () => SIGNALS.map(s => `<span><i class="ln" style="--c:${s.color}"></i>${s.label}</span>`).join('');
  const kv = rows => `<div class="kv">${rows.map(([k, v, c = '']) => `<span>${k}</span><b class="${c}">${v}</b>`).join('')}</div>`;
  const head = (s) => `<div><div class="file">${s.file}</div><h3>${s.title}</h3></div>
    <div class="chips">${s.chips.map(c => `<span>${c}</span>`).join('')}</div><p>${s.desc}</p>`;
  const clusterSizes = [0, 1, 2, 3, 4].map(k => CLUSTER.filter(c => c === k).length);
  const dropped = G.edges.length - KEPT.length;
  const sigCount = k => G.edges.filter(e => e.s[k] != null).length;
  const hop1 = Q.cand.filter(i => Q.hop[i] === 1).length, hop2 = Q.cand.filter(i => Q.hop[i] === 2).length;

  const base = { r: 4.2, a: .9, ring: 0 };
  const OFFLINE = [
    { title: 'Chunking', sub: 'tiktoken · 1200 / 100', file: 'indexing/chunker.py',
      chips: ['tiktoken cl100k', 'pdfplumber', 'structural / semantic', 'tables → Markdown'],
      desc: 'Each document is split into ~1,200-token chunks with 100 tokens of sentence overlap. Headings are tracked across pages, and tables become their own chunks.',
      legend: legendDocs,
      enter() { scene.layout(POS.docs); scene.setEdges(() => null); },
      update(t) { scene.setNodes(i => ({ ...base, r: 4.6, c: docRGB[G.chunks[i].doc], a: t > (i % 12) * .06 + Math.floor(i / 12) * .12 ? .95 : 0 })); },
      extra: () => kv([['Documents', DOCS.length], ['Chunks', G.N], ['chunk_size', '1200 tok'], ['chunk_overlap', '100 tok']]) },
    { title: 'Embedding', sub: 'nomic-embed-text · FAISS', file: 'indexing/embedder.py',
      chips: ['Ollama', '768-dim', 'FAISS IndexFlatIP'],
      desc: 'Every chunk becomes a 768-dim vector, L2-normalised so inner product equals cosine. Chunks gather by topic, not by document.',
      formula: 'v̂ = v / ‖v‖₂', legend: legendDocs,
      enter() { scene.layout(POS.embed); scene.setNodes(i => ({ ...base, c: docRGB[G.chunks[i].doc] })); scene.setEdges(() => null); },
      extra: () => kv([['Vector dim', 768], ['Index', 'IndexFlatIP'], ['Batch size', 32]]) },
    { title: 'Six edge signals', sub: 'semantic → citation', file: 'indexing/edges/*.py',
      chips: SIGNALS.map(s => s.label),
      desc: 'Six extractors each propose edges with a score in [0, 1]. Watch them arrive one signal at a time.',
      legend: legendSig,
      enter() { scene.layout(POS.force); scene.setNodes(() => ({ ...base, c: SKY, a: .85 })); },
      update(t) {
        const shown = Math.floor(t / .75);
        const order = SIGNALS.map(s => s.key).reverse();    // sparse ones last
        scene.setEdges(e => {
          const first = order.findIndex(k => e.s[k] != null);
          if (first < 0 || first > shown) return null;
          const key = shown >= order.length ? e.dom : order[first];
          return { a: .55, c: sigRGB[key], w: 1 };
        });
        order.forEach((k, i) => { const r = document.getElementById('sc-' + k); if (r) r.classList.toggle('hl', i <= shown); });
      },
      extra: () => kv(SIGNALS.slice().reverse().map(s => [`<span style="color:${s.color}">●</span> ${s.label}`, `<span id="sc-${s.key}">${sigCount(s.key)}</span>`])) },
    { title: 'Combine & sparsify', sub: 'W = Σ wₖ·normₖ', file: 'indexing/graph_builder.py',
      chips: ['NetworkX', 'min-max per signal', 'threshold 0.1'],
      desc: 'Each signal is min-max normalised, then blended with its weight. Edges below W = 0.1 are dropped. Here, the red ones go.',
      formula: 'W(i,j) = Σₖ wₖ · normₖ(i,j)',
      legend: () => `<span><i class="ln" style="--c:#7dd3fc"></i>kept (width ∝ W)</span><span><i class="ln" style="--c:#fb7185"></i>W &lt; 0.1 dropped</span>`,
      enter() { scene.layout(POS.force); scene.setNodes(() => ({ ...base, c: SKY })); },
      update(t) {
        scene.setEdges(e => {
          if (e.kept) return { a: .2 + e.W * .9, c: SKY, w: .6 + e.W * 4 };
          if (t < 1) return { a: .5, c: SKY, w: .8 };
          if (t < 2) return { a: .9, c: ROSE, w: 1.4 };
          return { a: 0, c: ROSE, w: .5 };
        });
      },
      extra: () => kv([['Candidate edges', G.edges.length], ['Kept', KEPT.length, 'good'], ['Dropped (W < 0.1)', dropped, 'bad'], ['Mean degree', (2 * KEPT.length / G.N).toFixed(1)]]) },
    { title: 'Spectral decomposition', sub: 'Laplacian eigenvectors', file: 'indexing/spectral.py',
      chips: ['SciPy', 'ARPACK eigsh', 'LOBPCG > 50k'],
      desc: 'The normalised Laplacian is decomposed. The smallest non-trivial eigenvectors become coordinates. These are computed right now in your browser.',
      formula: 'L = I − D^−½ · W · D^−½',
      legend: () => `<span><i style="--c:#7dd3fc"></i>chunk in spectral space (u₂…u₆)</span>`,
      enter() { scene.layout(POS.spectral); scene.setNodes(() => ({ ...base, c: SKY })); scene.setEdges(e => e.kept ? { a: .08, c: SKY, w: .6 } : null); },
      extra: () => kv(SPEC.vals.slice(0, 5).map((v, k) => [`λ${k + 1}`, v.toFixed(4), k === 0 ? '' : 'hl'])) },
    { title: 'Offline clusters', sub: 'k-means · k = 5', file: 'indexing/cluster_assigner.py',
      chips: ['scikit-learn KMeans', 'step 5b', 'stable labels'],
      desc: 'K-means on the spectral coordinates gives every chunk a community label. Stage 4 reuses these labels at query time.',
      legend: () => [0, 1, 2, 3, 4].map(k => `<span><i style="--c:${CLUSTER_COLORS[k]}"></i>c${k} · ${clusterSizes[k]}</span>`).join(''),
      enter() { scene.layout(POS.spectral); scene.setNodes(i => ({ ...base, r: 4.8, c: hex(CLUSTER_COLORS[CLUSTER[i]]) })); scene.setEdges(e => e.kept ? { a: .06, c: SKY, w: .6 } : null); },
      extra: () => kv(clusterSizes.map((s, k) => [`cluster ${k}`, `${s} chunks`])) },
    { title: 'Write & persist', sub: 'Neo4j + disk', file: 'store/neo4j_client.py',
      chips: ['Neo4j Bolt', 'Cypher UNWIND', 'chunks.pkl', 'faiss.index', 'spectral.npy'],
      desc: 'The graph is written to Neo4j as :Chunk nodes and :CONNECTED edges with every per-signal score. Chunks, the FAISS index and spectral coordinates go to disk.',
      legend: () => [0, 1, 2, 3, 4].map(k => `<span><i style="--c:${CLUSTER_COLORS[k]}"></i>c${k}</span>`).join(''),
      enter() { scene.layout(POS.force); scene.setNodes(i => ({ ...base, c: hex(CLUSTER_COLORS[CLUSTER[i]]) })); scene.setEdges(e => e.kept ? { a: .18 + e.W * .4, c: SKY, w: .6 + e.W * 2 } : null); },
      extra: () => kv([[':Chunk nodes', G.N], [':CONNECTED edges', KEPT.length], ['Batch size', 500], ['Store', './rag_index/']]) },
  ];

  const dim = hex('#33415c');
  const ONLINE = [
    { title: 'Hybrid seed retrieval', sub: 'RRF(dense, BM25, UQ)', file: 'query/seed_retrieval.py',
      chips: ['FAISS cosine', 'BM25Okapi', 'utility questions', 'RRF k = 60'],
      desc: 'Three ranked lists (dense, lexical and question-to-question) are fused with Reciprocal Rank Fusion. The top chunks become seeds.',
      formula: 'RRF(c) = Σ_lists 1 / (60 + rank(c))',
      legend: () => `<span><i style="--c:#38bdf8"></i>seed</span><span><i style="--c:#33415c"></i>other chunks</span>`,
      enter() { scene.layout(POS.force); scene.setEdges(e => e.kept ? { a: .07, c: SKY, w: .6 } : null); },
      update(t) {
        scene.setNodes(i => {
          const k = Q.seeds.indexOf(i);
          if (k >= 0 && t > .4 + k * .25) return { r: 6.5, a: 1, c: CYAN, ring: 1 };
          return { r: 3.6, a: .55, c: dim, ring: 0 };
        });
      },
      extra: () => kv([['Seeds (toy)', Q.seeds.length, 'hl'], ['seed_k in production', 20], ['hybrid_lexical_weight', '0.30']]) +
        `<div class="kv">${Q.seeds.map((s, k) => `<span>${k + 1}. ${lbl(s)}</span><b>${(Q.rrf[s] * 1000).toFixed(1)}e-3</b>`).join('')}</div>` },
    { title: 'Graph expansion', sub: 'Neo4j BFS · 2 hops', file: 'query/graph_expansion.py',
      chips: ['Cypher', 'undirected', 'max_candidates cap'],
      desc: 'From the seeds, the graph is walked two hops in Neo4j. Chunks that are linked but were never similar enough to rank now enter the pool.',
      legend: () => `<span><i style="--c:#38bdf8"></i>seed</span><span><i style="--c:#818cf8"></i>hop 1</span><span><i style="--c:#c4b5fd"></i>hop 2</span>`,
      enter() { scene.layout(POS.force); },
      update(t) {
        const H = [CYAN, hex('#818cf8'), hex('#c4b5fd')];
        const on = h => h === 0 || (h === 1 && t > .5) || (h === 2 && t > 1.5);
        scene.setNodes(i => {
          const h = Q.hop[i];
          if (h >= 0 && on(h)) return { r: h ? 5 : 6.5, a: 1, c: H[h], ring: h ? 0 : 1 };
          return { r: 3.4, a: .4, c: dim, ring: 0 };
        });
        scene.setEdges(e => {
          const tree = (Q.parent[e.j] === e.i && on(Q.hop[e.j])) || (Q.parent[e.i] === e.j && on(Q.hop[e.i]));
          if (tree) return { a: .8, c: H[Math.max(Q.hop[e.i], Q.hop[e.j])], w: 1.6 };
          return e.kept ? { a: .05, c: SKY, w: .6 } : null;
        });
      },
      extra: () => kv([['Seeds', Q.seeds.length], ['+ hop 1', hop1], ['+ hop 2', hop2], ['Candidates (toy cap 40)', Q.cand.length, 'hl'], ['Production cap', 200]]) },
    { title: 'Proximity re-scoring', sub: '½ spectral + ½ PPR', file: 'query/diffusion.py',
      chips: ['Personalized PageRank', 'damping 0.85', 'spectral cosine', 'α = 0.5'],
      desc: 'Each candidate is re-scored by relevance and by how close it sits to the seeds, both along real edges (PPR) and in spectral space.',
      formula: 'combined = ½·relevance + ½·(½ spectral + ½ PPR)',
      legend: () => `<span><i style="--c:#1d4ed8"></i>low</span><span><i style="--c:#38bdf8"></i>mid</span><span><i style="--c:#fbbf24"></i>high</span>`,
      enter() {
        scene.layout(POS.force);
        const vals = [...Q.combined.values()], mn = Math.min(...vals), mx = Math.max(...vals);
        scene.setNodes(i => Q.combined.has(i)
          ? { r: 3.5 + 4 * (Q.combined.get(i) - mn) / (mx - mn), a: 1, c: heat((Q.combined.get(i) - mn) / (mx - mn)), ring: 0 }
          : { r: 3, a: .25, c: dim, ring: 0 });
        scene.setEdges(e => Q.combined.has(e.i) && Q.combined.has(e.j) && e.kept ? { a: .25, c: SKY, w: .5 + e.W * 2 } : null);
      },
      extra: () => {
        const top = [...Q.combined.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6);
        return `<div class="kv">${top.map(([i, v]) => `<span>${lbl(i)}</span><b class="hl">${v.toFixed(3)}</b>`).join('')}</div>`;
      } },
    { title: 'Token-budget selection', sub: '≤ 12,000 tokens', file: 'query/cluster_selection.py',
      chips: ['precision-gated', 'doc-coverage tie-break', 'backfill'],
      desc: 'Chunks are added in score order until the token budget is full. The top 5 pass a lenient floor, the rest a stricter one, and chunks from uncovered documents win ties.',
      legend: () => `<span><i style="--c:#f0f9ff"></i>selected</span><span><i style="--c:#33415c"></i>gated / over budget</span>`,
      enter() { scene.layout(POS.force); },
      update(t) {
        const n = Math.min(Q.selected.length, Math.floor(t / .22));
        const chosen = new Set(Q.selected.slice(0, n));
        scene.setNodes(i => chosen.has(i) ? { r: 6, a: 1, c: hex(DOCS[G.chunks[i].doc].color), ring: .9 }
          : Q.combined.has(i) ? { r: 3.6, a: .5, c: hex('#475569'), ring: 0 } : { r: 3, a: .18, c: dim, ring: 0 });
        scene.setEdges(e => chosen.has(e.i) && chosen.has(e.j) && e.kept ? { a: .5, c: WHITE, w: 1.2 } : null);
        const used = Q.selected.slice(0, n).reduce((s, i) => s + G.chunks[i].tokens, 0);
        const bar = $('#tbBar'), lab = $('#tbLab'), cnt = $('#tbCnt'), docs = $('#tbDocs');
        if (bar) { bar.style.width = (used / Q.BUDGET * 100) + '%'; lab.textContent = `${fmt(used)} / ${fmt(Q.BUDGET)}`; cnt.textContent = n;
          docs.textContent = new Set(Q.selected.slice(0, n).map(i => G.chunks[i].doc)).size; }
      },
      extra: () => `<div class="kv"><span>Tokens</span><b id="tbLab">0 / 12,000</b></div>
        <div class="mini-bar"><i id="tbBar"></i></div>` +
        kv([['Chunks selected', '<span id="tbCnt">0</span>', 'hl'], ['Documents covered', '<span id="tbDocs">0</span>', 'good'],
          [`Naive top-${Q.selected.length} would cover`, `${Q.naiveDocs.size} docs`, 'bad']]) },
    { title: 'Grounded generation', sub: 'gpt-4o-mini · temp 0', file: 'generation/generator.py',
      chips: ['langdetect', 'grouped by source', 'max_tokens 2048'],
      desc: 'The context is grouped under === Source: X === headers so the model can combine evidence across documents. The answer comes back in the question’s language.',
      legend: () => [...Q.covered].map(d => `<span><i style="--c:${DOCS[d].color}"></i>${DOCS[d].short}</span>`).join(''),
      enter() {
        scene.layout(POS.force);
        const S = new Set(Q.selected);
        scene.setNodes(i => S.has(i) ? { r: 6, a: 1, c: hex(DOCS[G.chunks[i].doc].color), ring: 0 } : { r: 3, a: .15, c: dim, ring: 0 });
        scene.setEdges(e => S.has(e.i) && S.has(e.j) && e.kept ? { a: .55, c: WHITE, w: 1.2 } : null);
      },
      extra: () => {
        const byDoc = {};
        Q.selected.forEach(i => { const d = G.chunks[i].doc; byDoc[d] = (byDoc[d] || 0) + 1; });
        return `<div class="answer">Confidentiality, data return and liability caps survive termination. The supplier agreement pulls the DPA in by reference, and case law narrows the caps.</div>
          <div class="kv">${Object.entries(byDoc).map(([d, n]) => `<span><span style="color:${DOCS[d].color}">●</span> ${DOCS[d].name}</span><b>${n} chunks</b>`).join('')}</div>`;
      } },
  ];

  const TRACKS = { offline: OFFLINE, online: ONLINE };
  let track = 'offline', idx = 0, tStep = 0, playing = !REDUCED;
  const STEP_LEN = 5.2;

  function renderSteps() {
    const list = TRACKS[track];
    stepsEl.innerHTML = list.map((s, i) => `<li data-i="${i}" class="${i === idx ? 'on' : i < idx ? 'done' : ''}"><span class="k">${i + 1}</span><span>${s.title}<small>${s.sub}</small></span></li>`).join('');
    // on narrow screens the steps become a horizontal strip: keep the active one centred
    if (stepsEl.scrollWidth > stepsEl.clientWidth) {
      const li = stepsEl.children[idx];
      stepsEl.scrollTo({ left: li.offsetLeft - (stepsEl.clientWidth - li.offsetWidth) / 2, behavior: REDUCED ? 'auto' : 'smooth' });
    }
  }
  function go(tr, i) {
    track = tr; idx = i; tStep = 0;
    $$('.explorer .seg-btn').forEach(b => b.classList.toggle('on', b.dataset.track === track));
    const s = TRACKS[track][idx];
    qEl.hidden = track !== 'online';
    scene.padTop = track === 'online' ? 70 : 30;
    s.enter();
    if (s.update) s.update(0);
    info.innerHTML = head(s) + (s.formula ? `<div class="formula">${s.formula}</div>` : '') + (s.extra ? s.extra() : '');
    legend.innerHTML = s.legend ? s.legend() : '';
    renderSteps();
  }
  stepsEl.addEventListener('click', e => { const li = e.target.closest('li'); if (li) go(track, +li.dataset.i); });
  $$('.explorer .seg-btn').forEach(b => b.addEventListener('click', () => go(b.dataset.track, 0)));
  const next = () => { const L = TRACKS[track]; if (idx < L.length - 1) go(track, idx + 1); else go(track === 'offline' ? 'online' : 'offline', 0); };
  const prev = () => { if (idx > 0) go(track, idx - 1); else { const t = track === 'offline' ? 'online' : 'offline'; go(t, TRACKS[t].length - 1); } };
  $('#exNext').onclick = next; $('#exPrev').onclick = prev;
  const playBtn = $('#exPlay');
  const syncPlay = () => { playBtn.textContent = playing ? '❚❚' : '▶'; };
  playBtn.onclick = () => { playing = !playing; syncPlay(); };
  syncPlay();

  go('offline', 0);
  scene.snap();
  const prog = $('#exProgress');
  addAnimator($('#exCanvas'), (dt, now) => {
    tStep += dt;
    const s = TRACKS[track][idx];
    if (s.update) s.update(tStep);
    scene.frame(dt, now);
    prog.style.width = (playing ? clamp(tStep / STEP_LEN) * 100 : 0) + '%';
    if (playing && tStep > STEP_LEN) next();
  });
})();

/* ═════════════════════════ signals ═════════════════════════ */
(function signals() {
  const scene = new Scene($('#sigCanvas'), { pad: 30, padTop: 30 });
  const SKY = hex('#9fb2d8');
  const stack = $('#weightStack'), cards = $('#sigCards'), cap = $('#sigCaption');
  stack.innerHTML = SIGNALS.map(s => `<div data-k="${s.key}" style="--c:${s.color};flex:${s.w}" title="${s.label} ${s.w.toFixed(3)}">${s.w >= .1 ? s.w.toFixed(2) : ''}</div>`).join('');
  cards.innerHTML = SIGNALS.map(s => {
    const n = G.edges.filter(e => e.s[s.key] != null).length;
    return `<div class="sig" data-k="${s.key}" style="--c:${s.color}" tabindex="0">
      <div class="sig-top"><b><i></i>${s.label}</b><span>${s.w.toFixed(3)}</span></div>
      <p>${s.desc}</p><small>${s.file} · ${n} edges in the toy graph</small></div>`;
  }).join('');
  let focus = null;
  const apply = () => {
    if (!focus) {
      scene.setNodes(() => ({ r: 4.2, a: .9, c: SKY }));
      scene.setEdges(e => e.kept ? { a: .35 + e.W * .5, c: hex(SIG[e.dom].color), w: .6 + e.W * 2.4 } : null);
      cap.textContent = `All signals · ${KEPT.length} edges after sparsification · colored by strongest contributor`;
    } else {
      const touch = new Set();
      G.edges.forEach(e => { if (e.s[focus] != null) { touch.add(e.i); touch.add(e.j); } });
      scene.setNodes(i => touch.has(i) ? { r: 4.8, a: 1, c: hex(SIG[focus].color) } : { r: 3.2, a: .3, c: hex('#3b4a6b') });
      scene.setEdges(e => e.s[focus] != null ? { a: .3 + e.s[focus] * .6, c: hex(SIG[focus].color), w: .8 + e.s[focus] * 2 } : null);
      cap.textContent = `${SIG[focus].label} · ${G.edges.filter(e => e.s[focus] != null).length} edges · weight ${SIG[focus].w.toFixed(3)}`;
    }
    stack.classList.toggle('focus', !!focus);
    $$('div', stack).forEach(d => d.classList.toggle('on', d.dataset.k === focus));
    $$('.sig', cards).forEach(d => d.classList.toggle('on', d.dataset.k === focus));
  };
  const set = k => { focus = k; apply(); };
  $$('.sig', cards).forEach(c => {
    c.addEventListener('mouseenter', () => set(c.dataset.k));
    c.addEventListener('focus', () => set(c.dataset.k));
    c.addEventListener('click', () => set(focus === c.dataset.k ? null : c.dataset.k));
  });
  cards.addEventListener('mouseleave', () => set(null));
  apply(); scene.snap();
  addAnimator($('#sigCanvas'), (dt, now) => scene.frame(dt, now));
})();

/* ═════════════════════════ spectral morph ═════════════════════════ */
(function spectralView() {
  const scene = new Scene($('#specCanvas'), { pad: 36, padTop: 36 });
  scene.speed = 2.6;
  const SKY = hex('#7dd3fc');
  let view = 'graph', auto = !REDUCED, tAuto = 0;
  const apply = () => {
    scene.layout(view === 'graph' ? POS.force : POS.spectral);
    scene.setNodes(i => ({ r: 4.6, a: 1, c: hex(CLUSTER_COLORS[CLUSTER[i]]) }));
    scene.setEdges(e => e.kept ? { a: view === 'graph' ? .22 : .07, c: SKY, w: .6 + e.W * 1.6 } : null);
    $$('#spectral .seg-btn').forEach(b => b.classList.toggle('on', b.dataset.view === view));
  };
  $$('#spectral .seg-btn').forEach(b => b.addEventListener('click', () => { auto = false; view = b.dataset.view; apply(); }));
  $('#eigReadout').innerHTML = `Computed live · ${G.N}×${G.N} L<sub>sym</sub><br>` +
    SPEC.vals.slice(0, 5).map((v, k) => `λ${k + 1} = <b>${v.toFixed(4)}</b>${k === 0 ? ' (trivial, dropped)' : ''}`).join('<br>') +
    `<br>view: PCA of u₂…u₆ · colour: k-means (k = 5)`;
  apply(); scene.snap();
  addAnimator($('#specCanvas'), (dt, now) => {
    if (auto) { tAuto += dt; if (tAuto > 4.2) { tAuto = 0; view = view === 'graph' ? 'spectral' : 'graph'; apply(); } }
    scene.frame(dt, now);
  });
})();

/* ═════════════════════════ diffusion: heat kernel e^{−tL} ═════════════════════════ */
(function diffusion() {
  const scene = new Scene($('#diffCanvas'), { pad: 40, padTop: 40 });
  scene.speed = 12;
  const slider = $('#diffT'), label = $('#diffTLabel'), playBtn = $('#diffPlay');
  const n = G.N, V = SPEC.allVecs, L = SPEC.vals;
  // project the seed indicator onto the eigenbasis once
  const x0 = new Array(n).fill(0); Q.seeds.forEach(s => x0[s] = 1);
  const coef = V.map(v => v.reduce((s, vi, i) => s + vi * x0[i], 0));
  const heatAt = t => {
    const h = new Array(n).fill(0);
    for (let k = 0; k < n; k++) { const c = coef[k] * Math.exp(-t * L[k]); if (Math.abs(c) < 1e-9) continue; const v = V[k]; for (let i = 0; i < n; i++) h[i] += c * v[i]; }
    const mx = Math.max(...h.map(Math.abs)) || 1;
    return h.map(v => clamp(v / mx));
  };
  let t = 0, playing = !REDUCED, hold = 0;
  const seedSet = new Set(Q.seeds);
  const SKY = hex('#7dd3fc');
  const apply = () => {
    const h = heatAt(t);
    const reached = h.filter(v => v > .25).length;
    scene.setNodes(i => ({ r: 3.4 + h[i] * 5, a: .35 + h[i] * .65, c: heat(Math.pow(h[i], .7)), ring: seedSet.has(i) ? 1 : 0 }));
    scene.setEdges(e => e.kept ? { a: .04 + (h[e.i] + h[e.j]) * .35, c: mix(SKY, hex('#fde68a'), (h[e.i] + h[e.j]) / 2), w: .5 + (h[e.i] + h[e.j]) * 1.4 } : null);
    label.textContent = `t = ${t.toFixed(2)}`;
    scene.reached = reached;
  };
  scene.overlay = (ctx) => {
    ctx.font = '500 12px "JetBrains Mono", monospace'; ctx.fillStyle = 'rgba(148,160,189,.9)';
    ctx.fillText(`seeds: ${Q.seeds.length}   chunks with heat > 0.25: ${scene.reached}`, 18, 26);
  };
  slider.addEventListener('input', () => { t = +slider.value; playing = false; playBtn.textContent = '▶'; apply(); });
  playBtn.onclick = () => { playing = !playing; playBtn.textContent = playing ? '❚❚' : '▶'; if (playing && t >= 3.99) t = 0; };
  playBtn.textContent = playing ? '❚❚' : '▶';
  if (REDUCED) { t = 1.2; slider.value = t; }
  apply(); scene.snap();
  addAnimator($('#diffCanvas'), (dt, now) => {
    if (playing) {
      if (t >= 4) { hold += dt; if (hold > 1.6) { hold = 0; t = 0; } }
      else t = Math.min(4, t + dt * .5);
      slider.value = t; apply();
    }
    scene.frame(dt, now);
  });
})();

/* ═════════════════════════ results charts (paper tables) ═════════════════════════ */
(function charts() {
  const NS = 'http://www.w3.org/2000/svg';
  const SYS = { SpectRAG: '#3b8bef', Hybrid: '#1fa37a', LightRAG: '#e0622a', NaiveRAG: '#d9517e', HyDE: '#d4900a' };
  $('#sysLegend').innerHTML = Object.entries(SYS).map(([k, c]) => `<span><i style="--c:${c}"></i>${k}</span>`).join('');
  const mk = (svg, tag, a, txt) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); if (txt != null) e.textContent = txt; svg.appendChild(e); return e; };
  const AX = '#8a96b3', GRID = 'rgba(148,170,220,.12)';
  const tweens = [];
  const tween = (dur, fn, delay = 0) => tweens.push({ dur, fn, delay });

  const builders = {
    em(svg) {          // Table 7
      const data = [['SpectRAG', 71.2], ['Hybrid', 70.6], ['LightRAG', 70.5], ['NaiveRAG', 67.8], ['HyDE', 65.2]];
      const x0 = 110, x1 = 600, sx = v => x0 + v / 80 * (x1 - x0);
      for (let v = 0; v <= 80; v += 10) {
        mk(svg, 'line', { x1: sx(v), x2: sx(v), y1: 10, y2: 292, stroke: GRID });
        mk(svg, 'text', { x: sx(v), y: 314, fill: AX, 'font-size': 13, 'text-anchor': 'middle', 'font-weight': 600 }, v);
      }
      data.forEach(([k, v], i) => {
        const y = 20 + i * 55;
        mk(svg, 'text', { x: x0 - 12, y: y + 25, fill: '#e9eefb', 'font-size': 15, 'font-weight': 700, 'text-anchor': 'end' }, k);
        const r = mk(svg, 'rect', { x: x0, y, height: 38, width: 0, rx: 4, fill: SYS[k] });
        const t = mk(svg, 'text', { x: x0, y: y + 25, fill: '#fff', 'font-size': 15, 'font-weight': 700, opacity: 0 }, v.toFixed(1));
        mk(r, 'title', {}, `${k}: ${v} EM`);
        tween(900, p => { const w = (sx(v) - x0) * ease(p); r.setAttribute('width', w); t.setAttribute('x', x0 + w + 8); t.setAttribute('opacity', p); }, i * 90);
      });
    },
    recall(svg) {      // Table 6 — "All" values are from the table; earlier points are read off the paper's figure
      const cats = ['R@2', 'R@5', 'R@10', 'All'];
      const data = { SpectRAG: [39.1, 64.8, 84.1, 91.8], Hybrid: [46.3, 68.1, 76.2, 76.1], NaiveRAG: [41.1, 62.6, 69.6, 69.5], LightRAG: [31.4, 49.3, 61.9, 65.6], HyDE: [36.2, 58.3, 64.2, 64.1] };
      const x0 = 50, x1 = 470, y0 = 20, y1 = 290;
      const sx = i => x0 + i / 3 * (x1 - x0), sy = v => y1 - (v - 25) / 75 * (y1 - y0);
      [25, 50, 75, 100].forEach(v => { mk(svg, 'line', { x1: x0 - 10, x2: x1 + 10, y1: sy(v), y2: sy(v), stroke: GRID }); mk(svg, 'text', { x: x0 - 16, y: sy(v) + 5, fill: AX, 'font-size': 13, 'text-anchor': 'end', 'font-weight': 600 }, v); });
      cats.forEach((c, i) => mk(svg, 'text', { x: sx(i), y: 316, fill: AX, 'font-size': 13, 'text-anchor': 'middle', 'font-weight': 600 }, c));
      const labelY = { SpectRAG: 91.8, Hybrid: 76.1, NaiveRAG: 69.5, LightRAG: 65.2, HyDE: 60.6 };
      Object.entries(data).reverse().forEach(([k, vals], si) => {
        const d = vals.map((v, i) => `${i ? 'L' : 'M'}${sx(i)},${sy(v)}`).join(' ');
        const main = k === 'SpectRAG';
        const path = mk(svg, 'path', { d, fill: 'none', stroke: SYS[k], 'stroke-width': main ? 4 : 2.2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' });
        const len = path.getTotalLength ? path.getTotalLength() : 600;
        path.setAttribute('stroke-dasharray', len); path.setAttribute('stroke-dashoffset', len);
        const dots = vals.map((v, i) => { const c = mk(svg, 'circle', { cx: sx(i), cy: sy(v), r: main ? 6 : 4, fill: SYS[k], stroke: '#04060c', 'stroke-width': 2, opacity: 0 }); mk(c, 'title', {}, `${k} ${cats[i]}: ${i === 3 ? '' : '≈ '}${v}`); return c; });
        const lab = mk(svg, 'text', { x: x1 + 18, y: sy(labelY[k]) + 5, fill: main ? '#fff' : '#c5cee3', 'font-size': main ? 15 : 13.5, 'font-weight': 700, opacity: 0 }, `${k}  ${vals[3]}`);
        tween(1200, p => { path.setAttribute('stroke-dashoffset', len * (1 - ease(p))); dots.forEach((c, i) => c.setAttribute('opacity', p * 3 > i ? 1 : 0)); lab.setAttribute('opacity', clamp(p * 2 - 1)); }, si * 120);
      });
      const ann = mk(svg, 'text', { x: sx(2.2), y: sy(97), fill: '#fff', 'font-size': 13, 'font-weight': 700, 'text-anchor': 'middle', opacity: 0 }, '+15.7 pts over the next best');
      tween(500, p => ann.setAttribute('opacity', p), 1300);
    },
    qtype(svg) {       // Table 8 — SpectRAG values from the table; others read off the paper's figure
      const rows = [
        ['Inference', { HyDE: 86.9, NaiveRAG: 87.8, Hybrid: 92.3, SpectRAG: 95.1 }],
        ['Null', { LightRAG: 72.8, Hybrid: 87.4, HyDE: 89.7, NaiveRAG: 89.7, SpectRAG: 83.1 }],
        ['Comparison', { HyDE: 55.7, NaiveRAG: 59.8, LightRAG: 66.4, SpectRAG: 62.7 }],
        ['Temporal', { HyDE: 35.8, NaiveRAG: 40.1, Hybrid: 42.6, LightRAG: 45.2, SpectRAG: 43.9 }],
      ];
      const x0 = 120, x1 = 610, sx = v => x0 + (v - 30) / 70 * (x1 - x0);
      for (let v = 30; v <= 100; v += 10) {
        mk(svg, 'line', { x1: sx(v), x2: sx(v), y1: 10, y2: 290, stroke: GRID });
        mk(svg, 'text', { x: sx(v), y: 314, fill: AX, 'font-size': 13, 'text-anchor': 'middle', 'font-weight': 600 }, v);
      }
      rows.forEach(([name, vals], ri) => {
        const y = 52 + ri * 68;
        mk(svg, 'text', { x: x0 - 14, y: y + 5, fill: '#e9eefb', 'font-size': 15, 'font-weight': 700, 'text-anchor': 'end' }, name);
        const vs = Object.values(vals), lo = Math.min(...vs), hi = Math.max(...vs);
        const bar = mk(svg, 'rect', { x: sx(lo), y: y - 6, width: 0, height: 12, rx: 6, fill: 'rgba(59,130,246,.18)' });
        const entries = Object.entries(vals).sort((a, b) => (a[0] === 'SpectRAG') - (b[0] === 'SpectRAG'));
        const dots = entries.map(([k, v]) => {
          const main = k === 'SpectRAG';
          const off = name === 'Null' && k === 'HyDE' ? -12 : name === 'Null' && k === 'NaiveRAG' ? 12 : 0;
          const c = mk(svg, 'circle', { cx: sx(lo), cy: y + off, r: main ? 11 : 6, fill: SYS[k], stroke: '#04060c', 'stroke-width': 2, opacity: 0 });
          mk(c, 'title', {}, `${k}: ${main ? '' : '≈ '}${v}`);
          return [c, v, main];
        });
        const lab = mk(svg, 'text', { x: sx(vals.SpectRAG), y: y - 22, fill: '#fff', 'font-size': 14, 'font-weight': 700, 'text-anchor': 'middle', opacity: 0 }, vals.SpectRAG.toFixed(1));
        tween(1000, p => {
          const e = ease(p);
          bar.setAttribute('width', (sx(hi) - sx(lo)) * e);
          dots.forEach(([c, v]) => { c.setAttribute('cx', lerp(sx(lo), sx(v), e)); c.setAttribute('opacity', clamp(p * 3)); });
          lab.setAttribute('opacity', clamp(p * 2 - 1));
        }, ri * 140);
      });
    },
    index(svg) {       // Table 4
      const base = 290, top = 40, h = v => (base - top) * v / 274;
      mk(svg, 'line', { x1: 20, x2: 240, y1: base, y2: base, stroke: 'rgba(148,170,220,.4)' });
      [['LightRAG', 274, 40], ['SpectRAG', 46, 140]].forEach(([k, v, x], i) => {
        const r = mk(svg, 'rect', { x, y: base, width: 80, height: 0, fill: SYS[k] });
        const t = mk(svg, 'text', { x: x + 40, y: base - 8, fill: '#fff', 'font-size': 15, 'font-weight': 700, 'text-anchor': 'middle', opacity: 0 }, `${v} MB`);
        mk(svg, 'text', { x: x + 40, y: base + 22, fill: AX, 'font-size': 14, 'font-weight': 700, 'text-anchor': 'middle' }, k);
        tween(1000, p => { const hh = h(v) * ease(p); r.setAttribute('y', base - hh); r.setAttribute('height', hh); t.setAttribute('y', base - hh - 8); t.setAttribute('opacity', p); }, i * 150);
      });
      const pct = mk(svg, 'text', { x: 180, y: base - h(46) - 40, fill: '#fff', 'font-size': 22, 'font-weight': 800, 'text-anchor': 'middle', opacity: 0 }, '−83%');
      tween(400, p => pct.setAttribute('opacity', p), 1000);
      const corp = [['Agriculture', 1236, 239, '5.2×'], ['CS', 1327, 461, '2.9×'], ['Legal', 5615, 1989, '2.8×'], ['Mix', 7428, 1127, '6.6×'], ['MultiHopRAG', 15803, 5022, '3.2×']];
      const x0 = 380, x1 = 560, sx = v => (x1 - x0) * v / 15803;
      mk(svg, 'line', { x1: x0, x2: 630, y1: base + 4, y2: base + 4, stroke: 'rgba(148,170,220,.3)' });
      corp.forEach(([name, l, s, f], i) => {
        const y = 20 + i * 54;
        mk(svg, 'text', { x: x0 - 10, y: y + 22, fill: '#c5cee3', 'font-size': 14, 'font-weight': 700, 'text-anchor': 'end' }, name);
        const r1 = mk(svg, 'rect', { x: x0, y, width: 0, height: 18, fill: SYS.LightRAG });
        const r2 = mk(svg, 'rect', { x: x0, y: y + 21, width: 0, height: 18, fill: SYS.SpectRAG });
        const t1 = mk(svg, 'text', { x: x0, y: y + 14, fill: '#c5cee3', 'font-size': 12, 'font-weight': 700, opacity: 0 }, fmt(l));
        const t2 = mk(svg, 'text', { x: x0, y: y + 35, fill: '#fff', 'font-size': 12, 'font-weight': 700, opacity: 0 }, `${fmt(s)} · ${f} fewer`);
        tween(900, p => {
          const e = ease(p);
          r1.setAttribute('width', Math.max(2, sx(l) * e)); r2.setAttribute('width', Math.max(2, sx(s) * e));
          t1.setAttribute('x', x0 + Math.max(2, sx(l) * e) + 6); t2.setAttribute('x', x0 + Math.max(2, sx(s) * e) + 6);
          t1.setAttribute('opacity', p); t2.setAttribute('opacity', p);
        }, 300 + i * 110);
      });
    },
  };

  $$('.chart').forEach(fig => {
    const svg = $('svg', fig);
    const start = tweens.length;
    builders[fig.dataset.chart](svg);
    const mine = tweens.splice(start);
    const run = () => {
      const t0 = performance.now();
      const step = now => {
        let done = true;
        for (const tw of mine) { const p = clamp((now - t0 - tw.delay) / tw.dur); tw.fn(p); if (p < 1) done = false; }
        if (!done) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    };
    if (REDUCED) { mine.forEach(tw => tw.fn(1)); return; }
    mine.forEach(tw => tw.fn(0));
    const io = new IntersectionObserver(([e]) => { if (e.isIntersecting) { io.disconnect(); run(); } }, { threshold: .35 });
    io.observe(fig);
  });
  $$('.chart[data-chart="recall"] .src, .chart[data-chart="qtype"] .src').forEach(s => s.textContent += ' Hover a point for its value; points marked ≈ are read off the paper’s figure.');
})();

/* ═════════════════════════ ablation (Stage-4 study) ═════════════════════════ */
(function ablation() {
  const rows = [['token_budget', 60], ['protected_mmr', 55], ['cross_encoder', 55], ['merge_dups', 55], ['mmr_adaptive', 55],
    ['small_pool_all', 55], ['relevance_floor', 50], ['mmr_threshold', 50], ['dedup_only', 50], ['protected_core', 50],
    ['facility_location', 47.5], ['two_tier', 47.5], ['llm_listwise', 45], ['cluster_summary', 40]];
  const el = $('#resChart');
  el.innerHTML = rows.map(([n, v]) => `<div class="res-row ${v > 50 ? 'win' : v < 50 ? 'lose' : ''}"><span class="name">${n}</span>
    <span class="track"><span class="fill" data-v="${v}" style="display:block"></span></span><span class="val">${v}</span></div>`).join('') +
    `<div class="res-axis"><span></span><div>${[0, 10, 20, 30, 40, 50, 60, 70].map(v => `<span style="left:${v / 70 * 100}%">${v}</span>`).join('')}</div><span></span></div>`;
  const fill = () => $$('.fill', el).forEach(f => f.style.width = (f.dataset.v / 70 * 100) + '%');
  $('.ablation').addEventListener('toggle', e => { if (e.target.open) requestAnimationFrame(fill); });
})();

/* ═════════════════════════ terminal ═════════════════════════ */
(function terminal() {
  const TABS = {
    setup: [
      ['c', '# 1 · local models'], ['p', 'ollama pull nomic-embed-text'], ['p', 'ollama pull llama3.1:8b'],
      ['c', '# 2 · python deps'], ['p', 'python3.11 -m venv .venv && source .venv/bin/activate'],
      ['p', 'pip install -r requirements.txt'], ['p', 'python -m spacy download en_core_web_lg'],
      ['c', '# 3 · frontend'], ['p', 'cd frontend && npm install && npm run build && cd ..'],
    ],
    cli: [
      ['c', '# index a folder (chunk → embed → 6 signals → spectral → Neo4j)'], ['p', '.venv/bin/python3 main.py index papers/'],
      ['c', '# ask a multi-hop question'], ['p', '.venv/bin/python3 main.py query "What methods improve cross-document reasoning?"'],
      ['c', '# custom index location'], ['p', '.venv/bin/python3 main.py index docs/ --store ./my_index'],
    ],
    server: [
      ['c', '# default mode needs OPENAI_API_KEY in .env'], ['p', 'echo "OPENAI_API_KEY=sk-..." > .env'],
      ['c', '# Neo4j starts automatically on first run'], ['p', '.venv/bin/python3 server.py'],
      ['o', '→ open http://localhost:8000'],
      ['c', '# fully local: entity_extraction_mode="ner", utility_question_llm_backend="ollama"'],
    ],
  };
  const pre = $('#term');
  let tab = 'setup', timer = null;
  const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
  const lineHTML = ([k, s]) => k === 'p' ? `<span class="p">$</span> ${esc(s)}` : `<span class="${k}">${esc(s)}</span>`;
  const show = name => {
    tab = name; clearInterval(timer);
    $$('#termTabs .seg-btn').forEach(b => b.classList.toggle('on', b.dataset.tab === tab));
    const lines = TABS[tab];
    if (REDUCED) { pre.innerHTML = lines.map(lineHTML).join('\n'); return; }
    let li = 0, ch = 0;
    timer = setInterval(() => {
      const done = lines.slice(0, li).map(lineHTML);
      if (li >= lines.length) { pre.innerHTML = done.join('\n') + '\n<span class="p">$</span> <span class="cursor"></span>'; clearInterval(timer); return; }
      const [k, s] = lines[li];
      if (k !== 'p') { li++; ch = 0; pre.innerHTML = lines.slice(0, li).map(lineHTML).join('\n'); return; }
      ch += 3;
      const partial = `<span class="p">$</span> ${esc(s.slice(0, ch))}<span class="cursor"></span>`;
      pre.innerHTML = [...done, partial].join('\n');
      if (ch >= s.length) { li++; ch = 0; }
    }, 28);
  };
  $$('#termTabs .seg-btn').forEach(b => b.addEventListener('click', () => show(b.dataset.tab)));
  const io = new IntersectionObserver(([e]) => { if (e.isIntersecting) { io.disconnect(); show('setup'); } }, { threshold: .3 });
  io.observe(pre);
  pre.innerHTML = TABS.setup.map(lineHTML).join('\n');
  $$('.copy').forEach(b => b.addEventListener('click', async () => {
    const text = b.dataset.copy === 'term' ? TABS[tab].filter(l => l[0] === 'p').map(l => l[1]).join('\n') : $('#bibtex').textContent;
    try { await navigator.clipboard.writeText(text); b.textContent = 'Copied ✓'; } catch { b.textContent = 'Select & copy'; }
    setTimeout(() => b.textContent = b.dataset.copy === 'term' ? 'Copy' : 'Copy BibTeX', 1600);
  }));
})();

/* ═════════════════════════ scroll reveal ═════════════════════════ */
(function reveal() {
  const els = $$('.section-head, .section > .card, .charts > .card, .compare-anim, .stack-grid .tile, .quick, .cite-card, .sys-legend');
  els.forEach(e => e.classList.add('reveal'));
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { threshold: .12 });
  els.forEach(e => io.observe(e));
})();

const nav = $('.nav');
const onScroll = () => nav.classList.toggle('scrolled', scrollY > 40);
addEventListener('scroll', onScroll, { passive: true }); onScroll();

requestAnimationFrame(loop);
})();
