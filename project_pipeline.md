# SpectRAG — Project Pipeline

**SpectRAG (Spectral Graph-Augmented Retrieval)** improves context selection in RAG by augmenting dense vector retrieval with a weighted multi-signal chunk graph and a spectral (Laplacian-eigenspace) representation of that graph. Instead of returning the top-k chunks by cosine similarity, it selects a context set that is jointly *relevant*, *non-redundant*, *structurally connected*, and *topically diverse*.

This document walks through the two pipelines step by step, naming the technology used at each step and writing out the exact formula where the code implements one.

---

## Global Technology Stack

| Concern | Technology | Where |
|---|---|---|
| Embeddings | `nomic-embed-text` (768-dim) via **Ollama** | `indexing/embedder.py` |
| Vector index / ANN | **FAISS** `IndexFlatIP` (exact inner product = cosine on L2-normalised vectors) | `embedder.py`, `seed_retrieval.py` |
| Graph (transient, for spectral) | **NetworkX** undirected graph | `indexing/graph_builder.py` |
| Graph (persistent) | **Neo4j** (Bolt `bolt://localhost:7687`), Cypher + `UNWIND` batches | `store/neo4j_client.py` |
| Spectral decomposition | **SciPy** sparse Laplacian + ARPACK (`eigsh`) / LOBPCG | `indexing/spectral.py` |
| Clustering | **scikit-learn** `KMeans` | `query/cluster_selection.py` |
| Weight learning (Stage B) | **scikit-learn** `LogisticRegression` + ROC-AUC | `indexing/weight_calibrator.py` |
| Chunking / token sizing | **tiktoken** (`cl100k_base`) + **langchain** `RecursiveCharacterTextSplitter` | `indexing/chunker.py` |
| Entity/concept extraction | **OpenAI** `gpt-4o-mini` (`entity_extraction_mode="llm_concepts"`); or spaCy/GLiNER NER | `edges/llm_concept.py`, `ner/extractor.py` |
| Utility-question generation | **OpenAI** `gpt-4o-mini` (`utility_question_llm_backend="openai"`); or Ollama `llama3.1:8b` | `edges/utility_question.py` |
| Lexical retrieval (BM25) | **rank-bm25** `BM25Okapi` | `query/seed_retrieval.py` |
| Cross-encoder rerank (Stage 4 `anchor_expand_ce`) | **sentence-transformers** `ms-marco-MiniLM-L-6-v2` | `query/cluster_selection.py` |
| Language detection | **langdetect** | `language/detector.py` |
| Final answer generation | **OpenAI** `gpt-4o-mini` (chat completions) | `generation/generator.py` |
| Document parsing | `pdfplumber` (+ table extraction), `BeautifulSoup`/`lxml`, `python-docx`, plain text | `indexing/chunker.py` |
| Backend / Frontend | FastAPI + uvicorn / React + Vite + D3 v7 | `server.py`, `frontend/` |

> **Index-time reasoning runs on OpenAI `gpt-4o-mini`** (utility questions + concept extraction); Ollama is used only for embeddings. Both are config-toggleable back to local `llama3.1:8b` / spaCy NER.

---

# OFFLINE PIPELINE — Indexing

Orchestrated by `rag_system/indexing/pipeline.py :: IndexingPipeline.index()`.
Flow: **chunk → embed → 6 edge signals → (optional) weight learning → graph build → spectral decomposition → Neo4j write → persist**.

---

## Step 1 — Chunking
**File:** `indexing/chunker.py` · **Tech:** tiktoken (`cl100k_base`), `pdfplumber`, langchain `RecursiveCharacterTextSplitter`, `python-docx`, `BeautifulSoup`.

- **Real token counts.** Chunk sizes are measured with **tiktoken** (`_token_len`, encoding `cl100k_base`), not a character heuristic. Targets: `chunk_size = 1200` tokens, `chunk_overlap = 100` tokens, tail chunks below `min_chunk_tokens = 32` merged into their neighbour (backward, or forward for a small leading piece).
- **Strategy selection** (`chunking_strategy`, default `"auto"`, via `_select_strategy`):
  - **structural** — heading-aware. A heading stack (`_segment_by_heading`) persists **across pages**, so a section that spans a page break stays one unit; used when a document exposes headings.
  - **semantic** — embedding-based breakpoints (`_chunk_semantic`): split where consecutive-sentence similarity drops past `semantic_breakpoint_percentile = 90`; used when there are no headings and an embedder is available.
  - `"auto"` picks structural when headings exist, else semantic.
- **Cross-boundary overlap.** `_sentence_tail` carries the trailing sentences of one chunk into the next so ~`chunk_overlap` tokens of context bridge every boundary; never cuts mid-sentence.
- **Table extraction** (`table_extraction = True`): PDF/DOCX tables are pulled out as dedicated Markdown chunks (`_read_with_tables`, `_table_to_md`, `_split_table`) instead of being flattened into prose.
- **Multi-format reading:** `.pdf` → **pdfplumber** (text + tables), `.html/.htm` → BeautifulSoup (`lxml`), `.docx` → python-docx (heading-style detection via `_docx_prose`), `.txt/.md` → raw UTF-8.
- **Section path, reference extraction, per-page `langdetect`, and the stable `sha256` chunk-id hash** work as before.

> **Import-order guard:** `chunker.py` imports **faiss before** `langchain_text_splitters` on purpose — the reverse order segfaults on macOS x86 (OpenMP clash between faiss and pydantic-core/orjson). Do not reorder.

**Output:** `Chunk` objects (`text`, `position`, `section_path`, `references`, `language`, `metadata`).

---

## Step 2 — Embedding & FAISS Index
**File:** `indexing/embedder.py` · **Tech:** Ollama `nomic-embed-text`, FAISS.

- Each chunk text → 768-dim vector, batched (`embed_batch_size = 32`).
- **L2 normalisation** so inner product = cosine similarity:
  `v̂ = v / ‖v‖₂` (zero-norm vectors guarded to 1.0).
- **FAISS `IndexFlatIP`** built over the normalised matrix — exact (brute-force) inner-product search.

**Output:** `chunk.embedding` per chunk + an in-memory `faiss.IndexFlatIP`.

---

## Step 3 — Six Edge Signals
Each extractor returns `RawEdge(i, j, score)` with `score ∈ [0,1]` (`edges/base.py`). They run sequentially in the pipeline.

### 3a. Semantic — `edges/semantic.py`
k-NN cosine edges via a temporary FAISS `IndexFlatIP`. For each chunk take `k = semantic_k_neighbors = 7` nearest neighbours; keep an edge only if `cosine ≥ semantic_threshold = 0.75`. Edge weight = the cosine similarity. Symmetrised by keeping the max score per unordered pair.

### 3b. Adjacency — `edges/adjacency.py`
Connects consecutive chunks of the **same document** (`cj.position == ci.position + 1`) with a constant **raw score `1.0`** (sequential reading order).

### 3c. Section — `edges/section.py`
Chunks sharing a `section_path` prefix are linked; **deeper shared prefix ⇒ stronger**:
```
score(i, j) = shared_prefix_depth / max(depth(i), depth(j), 1)
```
Members per prefix capped at `max_section_members = 100` to avoid O(N²) blow-up.

### 3d. Entity / Concept — `edges/entity.py`, `edges/llm_concept.py`
The primary multi-hop link. Extracts per-chunk terms, maps each term → chunk list; chunks sharing a term (appearing in `≥ entity_min_freq = 2` chunks) get an edge weighted by shared count / global max:
```
score(i, j) = shared_term_count(i, j) / max_count
```
Term source is set by `entity_extraction_mode`:
- **`"llm_concepts"` (default)** — `gpt-4o-mini` extracts 5–10 canonical domain **concepts** per chunk (`CONCEPT_EXTRACTION_PROMPT`). Two chunks discussing the same idea in *different words* (which cosine misses) share a concept key and get linked — the cross-document signal **orthogonal to the embedding space**, which is the graph's reason to exist. Results are disk-cached by content hash (`.concept_cache/`), so re-indexing the same corpus is free.
- **`"ner"`** — spaCy/GLiNER/LLM named-entity recognition (generic entities; the original behaviour).

This changed *how the entity signal is computed*, not the equation — it is still one of the six edge signals at the same weight.

### 3e. Utility-Question — `edges/utility_question.py`
Functional similarity: `gpt-4o-mini` (default; `utility_question_llm_backend="openai"`) generates `utility_questions_per_chunk = 2` questions per chunk (`UTILITY_QUESTION_PROMPT`). All questions are embedded with `nomic-embed-text`, indexed in FAISS, and searched (`utility_question_search_k = 20`). Two chunks whose questions have `cosine ≥ utility_question_threshold = 0.80` get an edge; weight = that cosine. (Chunks that answer the *same* kind of question are linked even without lexical overlap.) The questions are also **stored on each chunk** and reused at query time as query-shaped surrogates (Stage 1 UQ fusion).

### 3f. Citation — `edges/citation.py`
Shared citation keys: bracket refs `\[(\d+(?:[,;]\s*\d+)*)\]` (expanded per number → `ref:N`), DOIs → `doi:...`, URLs → `url:...`. Co-occurrence counted and normalised exactly like entity edges:
```
score(i, j) = shared_citation_count(i, j) / max_count
```

---

## Step 4 — Edge-Weight Selection (automation)
**File:** `indexing/weight_calibrator.py` · controlled by `IndexingConfig.edge_weight_mode`.

Default per-signal priors (`config.py :: EdgeWeights`, must sum to 1.0):

| Signal | Prior weight |
|---|---|
| semantic | 0.300 |
| section | 0.250 |
| utility_question | 0.200 |
| adjacency | 0.084 |
| entity | 0.083 |
| citation | 0.083 |

Three modes:

### `"fixed"` (baseline)
Use the hardcoded priors above.

### `"calibrate"` — Stage A (label-free heuristic)
Weight each signal by how much information it carries on *this* corpus:
```
importance(signal) = sqrt(n_edges) · dispersion
```
- `sqrt(n_edges)` — sub-linear coverage (sparse signals shrink toward 0; dense signals don't dominate).
- `dispersion` — population std (`statistics.pstdev`) of the signal's min-max-normalised scores (≈0 for constant signals).

`fixed = ("adjacency",)` keeps its prior (its raw score is a constant 1.0 → zero dispersion, so it must not calibrate to zero). The remaining `budget = 1 − Σ fixed_priors` is split across the other signals proportional to `importance`, with a `floor = 0.01` for live-but-weak signals, then renormalised so the full vector sums to 1.0. Falls back to priors if no signal is informative.

### `"fit"` — Stage B (self-supervised logistic regression)
Learns feature weights against a corpus-derived relatedness target:
- **Positives** = chunk pairs linked by *label* signals `("adjacency", "section")`.
- **Negatives** = randomly sampled distant pairs not in the positive set (`neg_ratio = 1.0`).
- **Features** = min-max-normalised scores of `("semantic", "entity", "utility_question", "citation")` — label signals are excluded from features (no leakage) and keep their priors.
- Fit `sklearn LogisticRegression(class_weight="balanced")` on a 75/25 split; log held-out **ROC-AUC**.
- Coefficients clamped `≥ 0` (`np.clip`), normalised to fill `budget = 1 − Σ label_priors`, floored at `0.01`, renormalised.
- Falls back to priors if `< min_positives = 20` positive pairs or no feature has positive predictive weight.

---

## Step 5 — Graph Build (NetworkX)
**File:** `indexing/graph_builder.py` · **Tech:** NetworkX.

Each signal's raw edges are **min-max normalised** independently:
```
norm(e) = (score(e) − min) / (max − min)      # if max == min, all → 1.0
```
The combined weight of an edge is the weighted sum across the signals that fired on it:
```
W(i, j) = Σ_signal  weight_signal · norm_signal(i, j)
```
Per-signal normalised scores are stored as edge attributes (for visualisation). **Sparsification:** edges with `W < edge_sparsify_threshold = 0.1` are dropped. Result is an undirected `nx.Graph` `G = (V, E, W)`.

---

## Step 6 — Spectral Decomposition
**File:** `indexing/spectral.py` · **Tech:** SciPy sparse Laplacian, ARPACK `eigsh` / LOBPCG.

Each chunk gets a `n_spectral_components = 32`-dim coordinate encoding its position in the graph's community structure. Two modes (`IndexingConfig.spectral_mode`):

### `"combined"` (baseline)
Build adjacency `A` from `G`'s edge weights, then the **normalised Laplacian**:
```
L = I − D^(−1/2) · W · D^(−1/2)     (scipy.sparse.csgraph.laplacian(A, normed=True))
```
Take the eigenvectors of the `k+1` **smallest** eigenvalues and **drop the trivial constant eigenvector** (smallest), keeping 32. Solver auto-switches:
- `n ≤ spectral_large_graph_threshold = 50_000` → **ARPACK** `eigsh(L, k, which="SM")`
- larger → **LOBPCG** (random QR-orthonormalised init, `maxiter=300`, `tol=1e-4`), with ARPACK fallback on failure.

### `"multiplex"` (Option 4)
Build a per-signal normalised Laplacian `Lₖ` and aggregate, scaling by the chosen edge weights `βₖ`:
```
L = Σ_k  βₖ · Lₖ        (each layer degree-normalised on its own)
```
This keeps sparse-but-precise signals from being drowned by dense ones. Same eigensolve afterwards.

**Isolated chunks** (not in the graph, all-zero coords) are given position-interpolated coords inside the valid range, so they don't collapse to one point in the visualisation:
```
coords(i) = v_min + (position_i / (n−1)) · (v_max − v_min)
```

**Output:** `chunk.spectral_coords` (32-dim) per chunk.

> **Update:** `indexing/cluster_assigner.py :: SpectralClusterAssigner` (offline K-Means on spectral coords → `cluster_label`, `n_offline_clusters = 5`) is now wired into `IndexingPipeline` (**step 5b**). Stage 4 reuses these stable global labels; query-time K-Means is only a fallback for legacy indexes.

---

## Step 7 — Neo4j Write
**File:** `store/neo4j_client.py` · **Tech:** Neo4j, Cypher `UNWIND` batches (`write_batch_size = 500`).

`write_graph()` atomically replaces the graph: `MATCH (n:Chunk) DETACH DELETE n`, then batch-creates `:Chunk` nodes (`id`, `doc_id`, `position`), then `[:CONNECTED]` relationships carrying `weight` plus all six per-signal scores as properties. A `CREATE INDEX chunk_id ... ON (n.id)` is ensured on connect.

---

## Step 8 — Persist to Disk
**File:** `store/index_store.py` · **Tech:** pickle, FAISS writer, NumPy.

Written to `./rag_index/`:
- `chunks.pkl` — chunk objects (with embeddings)
- `faiss.index` — the FAISS index
- `spectral.npy` — stacked spectral coords

(The graph itself lives only in Neo4j; legacy `graph.pkl` is migrated on startup.)

---

# ONLINE PIPELINE — Retrieval & Generation

Orchestrated by `rag_system/query/pipeline.py :: QueryPipeline.query()`.
Five stages: **seed retrieval → graph expansion → spectral re-scoring → cluster-aware selection → generation.**

---

## Stage 1 — Seed Retrieval (hybrid)
**File:** `query/seed_retrieval.py` · **Tech:** Ollama embedding + FAISS ANN + **BM25** + Reciprocal Rank Fusion.

Three ranked lists are fused, so seeds come from complementary signals:
1. **Dense** — FAISS cosine over `nomic-embed-text` query embedding.
2. **Utility-question** — the query matched against each chunk's stored index-time utility questions (question↔question), closing the abstractive-query ↔ passage gap.
3. **BM25** (`bm25_seed_enabled`) — lexical scoring (`rank-bm25`), a signal **orthogonal to the embedding space** — strongest where exact terms decide relevance (statute/API names).

The three are combined by **Reciprocal Rank Fusion** (`rrf_k = 60`) → seed set **S₀** (top `seed_k = 20`).

The **full relevance vector** consumed by later stages is now **hybrid** (`hybrid_lexical_weight = 0.30`), folding normalized BM25 into the dense cosine so lexical precision reaches Stage 3/4, not just seed choice:
```
all_scores = (1 − w)·cosine_norm  +  w·bm25_norm        # (N,), w = 0.30
```

**Output:** seed indices (20) + hybrid `all_scores` over all chunks. *(The NaiveRAG / Hybrid baselines use pure dense / dense+BM25 respectively — this hybrid vector is SpectRAG-only.)*

---

## Stage 2 — Graph Expansion (BFS)
**File:** `query/graph_expansion.py` + `neo4j_client.bfs_expand()` · **Tech:** Neo4j Cypher BFS.

Iterative breadth-first traversal from S₀ in Neo4j, `graph_expansion_hops = 2` hops, one Cypher round-trip per hop, **undirected** (`-[:CONNECTED]-`), capped at `max_candidates = 200` nodes. Keeps the full graph out of Python memory.

**Output:** candidate set **S₁** (≤ 200 chunk indices). The candidate subgraph is then fetched once via `neo4j_client.subgraph()` as an `nx.Graph` for Stages 3–4.

---

## Stage 3 — Proximity Re-scoring (spectral / PPR / hybrid)
**File:** `query/diffusion.py` · **Tech:** NumPy on spectral coords + **NetworkX Personalized PageRank**.

Re-score each candidate by its proximity to the seeds, blended with the (hybrid) relevance:
```
combined(c) = (1 − α) · relevance(c)  +  α · proximity(c)        # α = diffusion_spectral_alpha = 0.5
```
`diffusion_mode` chooses the proximity term:
- **`"spectral"`** — cosine proximity in Laplacian eigenspace (offline, query-independent coords), aggregated over seeds (`diffusion_seed_aggregation`: `max` default, or `topk_mean` / `relevance_weighted`).
- **`"ppr"`** — **Personalized PageRank** from the seeds over the real weighted **candidate subgraph** (`nx.pagerank`, `ppr_damping = 0.85`): relevance flows along *actual* edges (concept, section, citation…), query-specifically — the graph does work at query time.
- **`"hybrid"` (default)** — mean of the spectral and PPR proximity terms. Each source degrades gracefully to the other (or to pure cosine) when unavailable.

**Output:** `diffused_scores` — `{candidate_idx: combined_score}` (set **S₂**).

---

## Stage 4 — Context Selection
**File:** `query/cluster_selection.py` · **Tech:** tiktoken budgeting / MMR / scikit-learn `KMeans`.

Selects the final context from the diffused candidates. The algorithm is chosen by `RetrievalConfig.selection_mode`. After a Stage-4 A/B study (see **Experiments** below) the default is **`token_budget`** — the original greedy selector was dropping ~35% of the relevant chunks to enforce diversity, which capped SpectRAG at parity with NaiveRAG.

### `"token_budget"` (default)
Fill a **token budget** with relevant chunks — no fixed k. Three parts:
1. **Precision-gated fill.** The top `token_budget_head_k = 5` chunks (answer-bearing core) use a lenient floor (`0.30 · max_diffused`); every chunk beyond that must clear a stricter tail floor (`0.55 · max_diffused`), so graph-expanded-but-tangential material can't dilute the answer. Add in score order until `Σ tiktoken_len > token_budget = 12000`.
2. **Document-coverage tie-break** (`token_budget_tie_band = 0.02`): within a small score band, chunks from not-yet-covered source documents come first — multi-source context at ~zero relevance cost.
3. **Budget backfill** (`token_budget_backfill`): if the gate left the budget under-used, fill the rest with the next-best gated-out chunks **that still clear the head floor** — so the gate *re-orders* quality without *shrinking* quantity (this fixed the loss cases where a starved, short context lost to the baseline).

### `"anchor_expand_ce"` (graph-native alternative)
Protected top-`anchor_core_k = 10` core → expand along the **candidate subgraph** to each core chunk's unchosen neighbors (`anchor_neighbors = 4`) → **cross-encoder** (`ms-marco-MiniLM`) ranks the neighbors → add relevant full chunks to the token budget. Graph-anchored + cross-encoder-validated + adaptive size.

### Other modes (`selection_mode`)
`protected_mmr` (protected core + MMR tail), `mmr`, `topk_diffused` (≈ NaiveRAG), and `greedy` (the original cluster-aware selector — `KMeans(5)` + marginal-utility loop; kept for ablation, this is the one that dropped ~1/3 of relevant chunks).

**Output:** final context set **C** (`token_budget`/`anchor_expand_ce`: adaptive, ~10–15 chunks; others: `final_context_k`).

---

## Stage 5 — Generation
**File:** `generation/generator.py` · **Tech:** OpenAI `gpt-4o-mini` chat completions.

- Detect the **query language** (`langdetect`) so the answer matches the question's language.
- **Group the context by source document** — chunks are emitted under `=== Source: X ===` headers (numbered blocks within), so the generator can see and synthesize *across* sources. Applied identically to SpectRAG and the baselines (shared generator), so it stays fair — it just reflects each system's real source spread (SpectRAG's graph expansion typically spans more documents).
- System prompt (`GENERATION_SYSTEM`) is **synthesis-oriented**: combine partial evidence across the provided passages into a coherent, multi-perspective answer in `{language}`; only declare the context insufficient when the passages are truly unrelated to the question. (An earlier strict-extractive prompt caused mass refusals in the eval — see Experiments.)
- Call `gpt-4o-mini` (`temperature = 0.0` → reproducible, `max_tokens = 2048`).

**Output:** `QueryResult(chunks, answer, scores)` — the answer plus per-chunk diffused scores as ranked source metadata.

---

# End-to-End Summary

```
OFFLINE:  docs ─► chunk (tiktoken, 1200/100 tok, structural/semantic + tables)
          ─► embed (nomic-embed-text, FAISS IP)
          ─► 6 signals {semantic, adjacency, section, entity=gpt-4o-mini concepts,
                        utility-Q=gpt-4o-mini, citation}
          ─► weight selection {fixed | calibrate(√n·std) | fit(logreg)}
          ─► combine W = Σ wₖ·normₖ, sparsify ≥0.1 (NetworkX)
          ─► Laplacian L = I − D^−½ W D^−½ → 32 eigvecs (ARPACK/LOBPCG)
          ─► offline KMeans cluster labels (step 5b)
          ─► Neo4j write + persist (chunks.pkl, faiss.index, spectral.npy)

ONLINE:   query ─► hybrid seeds: RRF(dense, utility-Q, BM25); all_scores = dense⊕BM25 (S₀)
          ─► Neo4j 2-hop BFS, ≤200 candidates (S₁)
          ─► rescore (1−α)·rel + α·proximity, proximity = ½spectral + ½PPR, α=0.5 (S₂)
          ─► token_budget: precision-gated + backfill to ≤12000 tok (C)
          ─► gpt-4o-mini synthesis answer, grouped by source (temp 0)
```

---

# Recent Updates & Experiments (`enhancement_test` branch)

Logs the production-hardening and the Stage-4 study done on top of the pipeline above.

## Production hardening
- **Chunker rewrite** — real tiktoken token counts, `pdfplumber` + table extraction, heading-stack structural chunking that persists across pages, embedding-based semantic fallback, cross-boundary sentence overlap. Chunk target moved to **1200 tokens / 100 overlap** (see *Chunk size* below).
- **Embedder** — pooled Ollama client, tenacity retries, response-count/dim validation, guarded FAISS build.
- **Cluster assigner wired** — offline spectral K-Means (`n_offline_clusters = 5`) now runs in the pipeline (step 5b); Stage 4 reuses the stable labels.
- **Spectral robustness** — deterministic ARPACK init + shift-invert with fallback chain; small-graph (n ≤ 2) guards.
- **Reproducibility** — generation and utility-question temperatures set to **0**. Given a fixed index, retrieval is fully deterministic; the only run-to-run variation had been LLM temperature.

## The Stage-4 problem
A per-stage tracer (`generation_eval/trace_stages.py`) showed Stages 1–3 retained ~100% of the relevant chunks, but the original **greedy** Stage-4 selector kept only ~65% — it traded away ~35% of the most-relevant chunks for diversity/anti-redundancy. That capped SpectRAG at parity with NaiveRAG.

## The experiment — Stage-4 A/B on Legal
`generation_eval/legal.py` builds + caches the Legal index/graph **once**, then A/B-tests Stage-4 algorithms without reindexing (Stages 1–3 and the NaiveRAG answers are computed once and reused across all modes). 13 variants across 4 strategy families (`generation_eval/stage4_variants.py`) were each judged pairwise vs NaiveRAG (gpt-4o judge, answer-order swapped), 30 docs / 20 questions.

**Results — SpectRAG win-rate % (>50 = SpectRAG better), sorted by Overall:**

| Variant | Group | Overall | Comp. | Diversity | Emp. |
|---|---|---|---|---|---|
| **token_budget** | A | **60.0** | **60.0** | **60.0** | **60.0** |
| protected_mmr | A | 55.0 | 55.0 | 55.0 | 55.0 |
| cross_encoder | C | 55.0 | 55.0 | 60.0 | 55.0 |
| merge_dups | D | 55.0 | 55.0 | 60.0 | 55.0 |
| mmr_adaptive | B | 55.0 | 55.0 | 50.0 | 55.0 |
| small_pool_all | D | 55.0 | 55.0 | 47.5 | 55.0 |
| relevance_floor | A | 50.0 | 50.0 | 52.5 | 50.0 |
| mmr_threshold | B | 50.0 | 50.0 | 40.0 | 50.0 |
| dedup_only | B | 50.0 | 50.0 | 42.5 | 50.0 |
| protected_core | A | 50.0 | 47.5 | 45.0 | 50.0 |
| facility_location | C | 47.5 | 47.5 | 52.5 | 47.5 |
| two_tier | D | 47.5 | 47.5 | 40.0 | 47.5 |
| llm_listwise | C | 45.0 | 47.5 | 42.5 | 45.0 |
| cluster_summary | D | 40.0 | 40.0 | 32.5 | 40.0 |

**How it went / what we learned**
- **More raw relevant text wins.** Every top variant (token_budget, protected_mmr, cross_encoder, merge_dups) simply gives the generator more relevant passages. Every variant that **compresses or restricts** the evidence (cluster_summary, llm_listwise, two_tier) lost — summaries/digests replace the detail the judge rewards.
- **`token_budget` won** and is now the production default: the purest form of "keep everything relevant, cap by tokens, filter nothing."
- **Caveat:** n = 20, so one question = 5 points — the 60-vs-55 lead is weak evidence (protected_mmr vs protected_core, the same algorithm, differ by 5 pts purely from judge non-determinism at temp 0). Confirm at `--questions 40`, and note this was a single domain (Legal); the four-domain `run.py` is the real test.
- Cross-encoder reranking (`ms-marco-MiniLM`) tied the leaders — a solid precision fallback if needed.

## Config changes shipped
- `selection_mode = "token_budget"`, `token_budget = 12000`, `token_budget_min_rel_ratio = 0.30`
- `chunk_size = 1200`, `chunk_overlap = 100`
- generation & utility-question temperatures = `0.0`

## Chunk size (256 vs 1200)
Settled on **1200 / 100** for the headline evaluation: it matches LightRAG's UltraDomain default (defensible comparison) and suits UltraDomain's abstractive, whole-book sensemaking questions, which need coherent passages rather than fragments. Trade-off: bigger chunks → fewer graph nodes → a sparser graph, which weakens SpectRAG's spectral engine. If density is needed at 1200, the fix is to **enrich the graph** (raise `utility_questions_per_chunk`, loosen `semantic_threshold`), not to shrink chunks. 256 is kept as an **ablation** demonstrating "SpectRAG's advantage grows with graph density / smaller chunks."

## New eval tooling
- `generation_eval/legal.py` — cached-index Stage-4 A/B harness. Group flags `--A/--B/--C/--D` or `--stage4 <name>`; auto-rebuilds when corpus **or chunking** settings change; appends every run to `results/legal_stage4_comparison.csv` and reprints the cumulative table.
- `generation_eval/stage4_variants.py` — the 13 experimental selectors (production selector stays clean).
- `generation_eval/trace_stages.py` — per-stage relevant-chunk retention tracer.

## Open next steps
1. Confirm `token_budget` at `--questions 40` and across all four domains (`run.py`).
2. Stage 5 prompt tuning to exploit the larger, richer context.
3. Sweep diffusion `α` (Stage 3) — cheap, directly affects what `token_budget` consumes.
4. Enrich the graph at 1200-token chunks (`utility_questions_per_chunk` ↑) — most expensive, compounds with the above.
