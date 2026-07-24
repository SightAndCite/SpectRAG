from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from a .env file sitting next to this module,
# so OPENAI_API_KEY (and friends) are available without exporting them manually.
load_dotenv(Path(__file__).with_name(".env"))


@dataclass
class LanguageConfig:
    auto_detect: bool = True           # detect language per document page at index time
    fallback_language: str = "en"      # ISO 639-1 code used when detection is uncertain
    use_llm_ner_fallback: bool = True  # use LLM-based NER when no spaCy model is installed for the language
    llm_ner_max_text_chars: int = 1000
    llm_ner_temperature: float = 0.0
    llm_ner_max_tokens: int = 300


@dataclass
class EdgeWeights:
    """Per-signal weights for the graph edge combiner. Must sum to 1.0."""
    semantic:          float = 0.150   # was 0.300 — de-emphasize the embedding-only signal
    section:           float = 0.200   # was 0.250 — modest trim, still meaningful structure
    entity:            float = 0.250   # was 0.083 — the orthogonal cross-doc concept signal
    adjacency:         float = 0.100   # was 0.084 — cheap local-structure prior, kept modest
    utility_question:  float = 0.280   # was 0.200 — strongest structural signal, boost it
    citation:          float = 0.020   # was 0.083 — near-zero: ~0% measured coverage on this corpus

    def __post_init__(self) -> None:
        total = (
            self.semantic + self.section + self.entity
            + self.adjacency + self.utility_question + self.citation
        )
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"EdgeWeights must sum to 1.0, got {total:.6f}")


@dataclass
class IndexingConfig:
    # Chunking (real token counts via tiktoken)
    chunk_size: int              = 1200   # max tokens per chunk (measured with tiktoken)
    chunk_overlap: int           = 100    # token overlap between adjacent chunks
    min_chunk_tokens: int        = 32     # merge tail chunks smaller than this into the previous
    tiktoken_encoding: str       = "cl100k_base"  # tokenizer used for sizing
    chunk_id_hash_prefix: int    = 64     # chars of text included in chunk-id hash
    # Strategy: "auto" picks structural (heading-aware) when a doc has headings,
    # else semantic (embedding-based) if an embedder is available; "structural" or
    # "semantic" force one. Tables (PDF/DOCX) are extracted as dedicated chunks.
    chunking_strategy: str          = "auto"
    table_extraction: bool          = True
    semantic_breakpoint_percentile: float = 90.0  # sentence-distance percentile → split

    # Semantic edges
    semantic_k_neighbors: int    = 7     # k-NN per chunk
    semantic_threshold: float    = 0.75   # cosine similarity floor

    # Graph sparsification
    edge_sparsify_threshold: float = 0.1 # drop combined edges below this weight

    # How per-signal edge weights are chosen:
    #   "fixed"     — use the hardcoded EdgeWeights (baseline).
    #   "calibrate" — Stage A: weight by signal informativeness (variance/coverage),
    #                 label-free heuristic.
    #   "fit"       — Stage B: self-supervised logistic regression. Learns weights
    #                 for semantic/entity/utility_question/citation against labels
    #                 derived from adjacency + section pairs (which keep fixed
    #                 priors). Reports a held-out AUC.
    edge_weight_mode: str = "fixed"

    # Spectral decomposition
    # spectral_mode:
    #   "combined"  — Laplacian of the single combined-weight graph (baseline).
    #   "multiplex" — Option 4: aggregate per-signal normalized Laplacians
    #                 L = Σ βₖ·Lₖ, with βₖ = the chosen edge weights. Each layer
    #                 is degree-normalized on its own, so sparse precise signals
    #                 aren't drowned by dense ones.
    spectral_mode: str                  = "combined"
    n_spectral_components: int          = 32
    spectral_large_graph_threshold: int = 50_000  # switch ARPACK → LOBPCG above this
    spectral_lobpcg_max_iter: int       = 300
    spectral_lobpcg_tol: float          = 1e-4
    # Offline spectral clustering: K over ALL chunks, stored as chunk.cluster_label
    # and reused at query time for the Stage-4 cluster-novelty bonus (avoids
    # per-query K-Means). Keep aligned with retrieval.n_clusters.
    n_offline_clusters: int             = 5

    # Utility-question edges
    # Index-time LLM for utility-question generation:
    #   "openai" — gpt-4o-mini (sharper, more specific questions → better UQ edges
    #              AND better query-side seeds). "ollama" — local llama3.1:8b.
    utility_question_llm_backend: str       = "openai"
    utility_questions_per_chunk: int        = 2      # LLM calls per chunk; richer graph + more query-side seeds
    utility_question_threshold: float       = 0.80   # cosine sim floor for UQ edges
    utility_question_search_k: int          = 20     # k-NN over question embeddings
    utility_question_max_text_chars: int    = 2000   # chars of chunk text sent to LLM
    utility_question_llm_temperature: float = 0.0    # 0 → reproducible UQ edges across rebuilds
    utility_question_llm_max_tokens: int    = 200
    utility_question_log_interval: int      = 20     # log progress every N chunks
    # Disk cache for generated utility questions, keyed by chunk-text hash (like
    # the concept cache). Re-indexing a corpus after adding docs only pays the LLM
    # for genuinely new chunks; unchanged chunks are free.
    utility_question_cache_dir: str         = "./.uq_cache"

    # Entity / concept edges — how the shared-link terms are extracted:
    #   "ner"          — spaCy/GLiNER/LLM named-entity recognition (generic entities).
    #   "llm_concepts" — gpt-4o-mini extracts domain CONCEPTS/topics per chunk, so
    #                    chunks that discuss the same idea in different words (which
    #                    cosine misses) share a concept key and get linked. The
    #                    cross-document signal ORTHOGONAL to the embedding space —
    #                    the graph's reason to exist. One LLM call per chunk (cached
    #                    on disk, so re-indexing the same corpus is free).
    entity_extraction_mode: str = "llm_concepts"
    concept_max_text_chars: int = 2000   # chars of chunk text sent to the LLM
    concept_llm_max_tokens: int = 200
    concept_cache_dir: str      = "./.concept_cache"
    entity_min_freq: int       = 2    # entity/concept must appear in ≥ N chunks for an edge
    entity_log_interval: int   = 50   # log progress every N chunks

    # Section edges
    max_section_members: int   = 100  # cap members per section prefix (avoids O(N²))

    # Index-time LLM concurrency: utility-question + concept extraction issue one
    # small OpenAI call per chunk; running them in a thread pool cuts indexing
    # wall-clock ~Nx. Failed calls retry with exponential backoff (and are NOT
    # cached), so transient 429/5xx errors don't poison the graph or the cache.
    llm_parallel_workers: int  = 8
    llm_max_retries: int       = 3


@dataclass
class RetrievalConfig:
    seed_k: int                = 20    # initial ANN hits (Stage 1)
    graph_expansion_hops: int  = 2     # BFS depth (Stage 2)
    max_candidates: int        = 200   # cap on candidate pool after expansion

    # Utility-question seed augmentation (Stage 1)
    # Match the query against each chunk's index-time utility questions
    # (question↔question, semantically aligned) and fuse those hits into the
    # dense seeds via Reciprocal Rank Fusion. Closes the abstractive-query ↔
    # passage gap, reusing the LLM questions already generated for edges. No
    # query-time LLM cost. Disabled → dense-only seeds.
    uq_seed_enabled: bool      = True
    uq_seed_k:       int       = 20    # top utility-question hits merged into seeds
    rrf_k:           int       = 60    # RRF damping constant (standard default)

    # BM25 lexical seed augmentation (Stage 1). A lexical signal ORTHOGONAL to the
    # embedding space that dense seeds, utility-questions, and spectral coords all
    # share — so it makes different mistakes. Strongest where exact terms decide
    # relevance (Legal statute/case names, CS API/algorithm names) which dense
    # cosine blurs. Fused as a third Reciprocal-Rank-Fusion list; NaiveRAG stays
    # pure-dense, so this is genuine SpectRAG machinery. No LLM cost, no reindex.
    bm25_seed_enabled: bool    = True
    bm25_seed_k:       int     = 20    # top BM25 hits merged into seeds
    # Hybrid lexical relevance: blend normalized BM25 into the dense relevance
    # VECTOR (all_scores), so lexical precision reaches Stage-3 diffusion and
    # Stage-4 token-budget selection — not just the Stage-1 seed choice. The A/B
    # showed BM25 is the signal that helps Legal/CS precision, but it was being
    # washed out downstream by the all-embedding path. Naive stays pure-dense.
    #   relevance = (1 − w)·dense_cosine_norm + w·bm25_norm
    # w = 0 → pure dense (original behaviour).
    hybrid_lexical_weight: float = 0.30

    # Spectral diffusion (Stage 3)
    # Re-scores candidates by proximity to seeds, blended with cosine:
    # combined = (1 − α) · cosine_sim  +  α · proximity
    diffusion_spectral_alpha: float = 0.5
    # How the proximity term is computed:
    #   "spectral" — cosine proximity in Laplacian eigenspace (query-independent
    #                coords; the graph does no work at query time).
    #   "ppr"      — Personalized PageRank from the seeds over the real weighted
    #                candidate graph: relevance flows along ACTUAL edges
    #                (co-citation, shared entities, section structure),
    #                query-specifically. Makes the graph do work at query time.
    #   "hybrid"   — mean of the spectral and PPR proximity terms (default; keeps
    #                the spectral signal while adding true graph diffusion).
    diffusion_mode:  str            = "hybrid"
    ppr_damping:     float          = 0.85   # PageRank damping (restart = 1 − damping)
    # How a candidate's per-seed spectral proximities combine into one score.
    # The spectral term is SpectRAG's main advantage over naive cosine, so this
    # aggregation directly shapes retrieval quality.
    #   "topk_mean"          — mean of the diffusion_seed_topk highest proximities.
    #                          Smooths the single-stray-seed noise of "max" while
    #                          keeping scale, so alpha stays comparable. Default.
    #   "relevance_weighted" — weight each seed's proximity by that seed's own
    #                          query cosine, so proximity to STRONG seeds counts
    #                          more than proximity to weak ones (lower scale — may
    #                          pair with a higher alpha).
    #   "max"                — max over all seeds (original; one marginal seed near
    #                          a tangential chunk can inflate the score).
    # Default back to "max": the topk_mean/relevance_weighted variants showed no
    # gain on Legal at n=40 (47.5–52.5 ≈ parity band). Kept as options for ablation.
    diffusion_seed_aggregation: str = "max"
    diffusion_seed_topk: int        = 3

    n_clusters: int            = 5     # fallback clusters if offline labels are missing
    # NaiveRAG's context size (chunks). Kept at ~token_budget / chunk_size so the
    # baseline gets the SAME token budget as SpectRAG (fair comparison): at
    # chunk_size=1200 and token_budget=12000, 10 chunks ≈ 12k tokens each side.
    final_context_k: int       = 10    # chunks handed to the LLM (Stage 5)

    # Stage 4 selection mode:
    #   "token_budget"  — keep EVERY meaningfully-relevant chunk (diffused score
    #                     ≥ token_budget_min_rel_ratio × the top score) in
    #                     descending relevance order until token_budget tokens are
    #                     filled. No diversity filtering, no k-cap: relevant chunks
    #                     are never dropped for redundancy — the token budget is
    #                     the only limit. Best-performing mode in the Legal
    #                     Stage-4 A/B (generation_eval/legal.py). This is the mode.
    #   "protected_mmr" — Protected-core + MMR tail. The top `protected_core_k`
    #                     chunks by spectral-diffused score are ALWAYS kept, the
    #                     remaining slots up to final_context_k filled by MMR.
    #   "mmr"           — Maximal Marginal Relevance across the whole pool: keep
    #                     relevant chunks but skip near-duplicates. Can still
    #                     demote a related-but-distinct relevant chunk. Tuned by
    #                     mmr_lambda.
    #   "topk_diffused" — take the final_context_k highest spectral-diffused
    #                     scores. Keeps the most relevant chunks but wastes slots
    #                     on duplicates (≈ NaiveRAG's behaviour).
    #   "greedy"        — cluster-aware greedy marginal-utility selection using
    #                     the sel_* weights below (adds diversity/anti-redundancy
    #                     but drops ~1/3 of the most-relevant chunks).
    #   "anchor_expand_ce" — keep a protected top-K core (answer-bearing), expand
    #                     along GRAPH edges to each core chunk's unchosen
    #                     neighbors, cross-encoder-rank those neighbors, and add
    #                     the relevant ones (full chunks) up to the token budget.
    #                     Graph-anchored + cross-encoder-validated + adaptive.
    selection_mode: str        = "token_budget"
    # anchor_expand_ce knobs
    anchor_core_k:       int   = 10    # protected top-diffused core, always kept
    anchor_neighbors:    int   = 4     # graph neighbors expanded per core chunk
    # Cross-encoder logit floor for an added neighbor. Default is very negative so
    # the cross-encoder RANKS the neighbors (best first) and the budget fills —
    # growing context, which the judge rewards. Raise toward 0 to instead GATE
    # (add only neighbors the cross-encoder deems query-relevant), trading volume
    # for precision. -10 ≈ no gate (ms-marco logits rarely fall below ~-11).
    anchor_ce_threshold: float = -10.0
    token_budget: int          = 12000 # max context tokens (token_budget mode)
    token_budget_min_rel_ratio: float = 0.30  # HEAD floor: keep chunks scoring ≥ ratio × top diffused
    # Precision gate on the token-budget TAIL. The first token_budget_head_k
    # chunks (the answer-bearing core) use the lenient head floor above; every
    # chunk beyond that must clear the stricter tail floor. This keeps the head
    # that wins most domains, while stopping graph-expanded but tangential
    # material (the Legal failure mode: varied-but-imprecise context) from
    # diluting the answer. In domains where the tail is genuinely relevant it
    # clears the higher bar anyway, so the gate binds only when it should.
    # Set tail ratio == head ratio to disable the gate (single-floor behaviour).
    token_budget_head_k: int   = 5     # chunks exempt from the stricter tail floor
    token_budget_tail_min_rel_ratio: float = 0.55  # stricter floor for tail chunks
    # Backfill the token budget when the tail gate leaves it under-used: fill the
    # remaining space with the next-best gated-out chunks. Stops the gate from
    # starving SpectRAG's context below naive's (shorter answers were losing). The
    # gate then re-orders quality without shrinking quantity.
    token_budget_backfill: bool = True
    # Document-coverage tie-break: within this fractional score band, promote
    # chunks from not-yet-covered source documents. Diversity at ~zero relevance
    # cost (it only reorders near-ties), so multi-document context — which the
    # judge rewards as multi-perspective — surfaces without dropping any relevant
    # chunk. 0.0 disables (exact-score ties only).
    token_budget_tie_band: float = 0.02
    protected_core_k: int      = 10    # top-diffused chunks always kept (protected_mmr)
    mmr_lambda: float          = 0.75  # MMR relevance↔diversity trade (→1 = pure relevance)

    # Greedy selection weights (only used when selection_mode == "greedy")
    sel_diffused_relevance: float = 0.35  # heat-diffused score
    sel_raw_similarity:     float = 0.15  # direct query–chunk cosine similarity
    sel_anti_redundancy:    float = 0.30  # penalty for overlap with already-selected
    sel_connectivity:       float = 0.10  # graph edges to already-selected chunks
    sel_diversity:          float = 0.10  # bonus for uncovered cluster


@dataclass
class OllamaConfig:
    base_url:        str   = "http://localhost:11434"
    embedding_model: str   = "nomic-embed-text"
    llm_model:       str   = "llama3.1:8b"
    embedding_dim:   int   = 768   # nomic-embed-text output dimension
    request_timeout: int   = 120
    llm_temperature: float = 0.1
    llm_max_tokens:  int   = 2048
    embed_batch_size: int  = 32
    # Disk cache for chunk embeddings, keyed by text hash + model. Unchanged chunks
    # are not re-embedded on a re-index (helps the additive per-session add flow).
    embedding_cache_dir: str = "./.embedding_cache"


@dataclass
class OpenAIConfig:
    """Final-answer generator (Stage 5). Utility-question generation stays on Ollama."""
    api_key:         str        = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    base_url:        str | None = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL") or None)
    model:           str        = "gpt-4o-mini"
    temperature:     float      = 0.0    # 0 → reproducible generation across runs
    max_tokens:      int        = 2048
    request_timeout: int        = 120


@dataclass
class NERConfig:
    spacy_model: str   = "xx_core_web_sm"  # multilingual fallback; detector overrides per language
    # GLiNER is opt-in: requires PyTorch + ~1.5 GB model download.
    use_gliner: bool   = False
    gliner_model: str  = "urchade/gliner_medium-v2.1"
    gliner_entity_types: list[str] = field(default_factory=lambda: [
        "person", "organization", "location", "product", "date",
        "event", "technology", "method", "dataset", "concept",
    ])
    gliner_threshold: float = 0.40
    gliner_flat_ner:  bool  = True


@dataclass
class Neo4jConfig:
    uri:              str = "bolt://localhost:7687"
    user:             str = "neo4j"
    password:         str = "password123"
    database:         str = "neo4j"
    write_batch_size: int = 500   # nodes / edges per UNWIND transaction


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: [
        "http://localhost:5173",
        "http://localhost:3000",
    ])
    max_viz_nodes: int       = 200   # node cap for the graph visualization endpoint
    node_label_chars: int    = 80    # chars of chunk text used as the node label
    source_preview_chars: int = 200  # chars shown in the sources panel
    session_id_chars: int    = 8     # UUID prefix length for session IDs
    log_suppress_paths: list[str] = field(default_factory=lambda: ["/api/status"])


@dataclass
class Config:
    edge_weights: EdgeWeights    = field(default_factory=EdgeWeights)
    indexing:     IndexingConfig = field(default_factory=IndexingConfig)
    retrieval:    RetrievalConfig = field(default_factory=RetrievalConfig)
    ollama:       OllamaConfig   = field(default_factory=OllamaConfig)
    openai:       OpenAIConfig   = field(default_factory=OpenAIConfig)
    ner:          NERConfig      = field(default_factory=NERConfig)
    neo4j:        Neo4jConfig    = field(default_factory=Neo4jConfig)
    server:       ServerConfig   = field(default_factory=ServerConfig)
    language:     LanguageConfig = field(default_factory=LanguageConfig)
    store_path:   Path           = field(default_factory=lambda: Path("./rag_index"))
