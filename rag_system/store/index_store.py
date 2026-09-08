from __future__ import annotations
import logging
import pickle
from pathlib import Path

import numpy as np
import faiss
from rag_system.models import Chunk
from rag_system.store.lexical_index import LexicalIndex

logger = logging.getLogger(__name__)

_CHUNKS_FILE  = "chunks.pkl"
_FAISS_FILE   = "faiss.index"
_VECTORS_FILE = "vectors.npy"


class IndexStore:
    """
    Persists the offline artefacts that live on disk: chunk payloads, their
    vectors, and the FAISS index. The graph is stored separately (see
    Neo4jGraphClient / InMemoryGraphStore).

    Vectors are kept OUT of chunks.pkl and written as one float32 array. They
    were previously pickled per chunk as well as copied into FAISS, so a 1M-chunk
    corpus stored 3.07 GB of vectors twice, and unpickling made every one of them
    a resident Python object before a single query ran.

    On load the array is memory-mapped and each chunk points at its row. NumPy row
    indexing on a memmap yields a *view*, so nothing is read until a vector is
    actually touched — and retrieval only touches its candidates, not the corpus.
    """

    def __init__(self, store_path: Path | str) -> None:
        self.path = Path(store_path)

    def save(self, chunks: list[Chunk], faiss_index: faiss.Index) -> None:
        self.path.mkdir(parents=True, exist_ok=True)

        missing = [c.chunk_id for c in chunks if c.embedding is None]
        if missing:
            raise ValueError(
                f"{len(missing)} chunk(s) have no embedding (e.g. {missing[:3]}) — "
                "embed before saving."
            )

        matrix = np.stack([np.asarray(c.embedding) for c in chunks]).astype(np.float32)
        np.save(self.path / _VECTORS_FILE, matrix)

        # Detach the vectors for the duration of the pickle so they are stored
        # once, in vectors.npy. Mutate-and-restore rather than copying, because
        # deep-copying a million chunks to drop one field defeats the purpose.
        detached = [c.embedding for c in chunks]
        try:
            for c in chunks:
                c.embedding = None
            with open(self.path / _CHUNKS_FILE, "wb") as fh:
                pickle.dump(chunks, fh, protocol=5)
        finally:
            for c, emb in zip(chunks, detached):
                c.embedding = emb

        faiss.write_index(faiss_index, str(self.path / _FAISS_FILE))

        # Inverted index for BM25, built once here rather than reconstructed from
        # the whole corpus inside the first query after every restart.
        LexicalIndex.build([c.text for c in chunks]).save(
            self.path / LexicalIndex.directory_name())

        # spectral_coords stay a Chunk field inside chunks.pkl. The old side file
        # was reloaded by absolute position, which silently misaligned coords onto
        # the wrong chunks if any chunk lacked coords (see CASE-08).

        logger.info(
            "IndexStore saved: %d chunks, vectors %s → %s",
            len(chunks), "x".join(map(str, matrix.shape)), self.path,
        )

    def load(self) -> tuple[list[Chunk], faiss.Index]:
        for fname in (_CHUNKS_FILE, _FAISS_FILE):
            if not (self.path / fname).exists():
                raise FileNotFoundError(
                    f"Index artefact missing: {self.path / fname}\n"
                    "Upload documents to build the index."
                )

        with open(self.path / _CHUNKS_FILE, "rb") as fh:
            chunks: list[Chunk] = pickle.load(fh)

        faiss_index = faiss.read_index(str(self.path / _FAISS_FILE))

        vectors_path = self.path / _VECTORS_FILE
        if vectors_path.exists():
            vectors = np.load(vectors_path, mmap_mode="r")
            if len(vectors) != len(chunks):
                raise ValueError(
                    f"vectors.npy holds {len(vectors)} rows but chunks.pkl holds "
                    f"{len(chunks)} chunks — the index is inconsistent, rebuild it."
                )
            for i, chunk in enumerate(chunks):
                chunk.embedding = vectors[i]      # view into the memmap, not a copy
        elif any(c.embedding is None for c in chunks):
            raise FileNotFoundError(
                f"Index artefact missing: {vectors_path}\n"
                "Chunks carry no embeddings and there is no vector file — rebuild."
            )
        else:
            # Index written before vectors were split out; embeddings are inside
            # the pickle and already loaded. Rebuilding moves it to the new layout.
            logger.info("Legacy index without %s — using pickled embeddings", _VECTORS_FILE)

        logger.info("IndexStore loaded: %d chunks from %s", len(chunks), self.path)
        return chunks, faiss_index

    def load_lexical(self) -> LexicalIndex | None:
        """Persisted inverted index, or None for an index built before F2."""
        return LexicalIndex.load(self.path / LexicalIndex.directory_name())

    def chunk_count(self) -> int | None:
        """Number of chunks, read from the vector file's header.

        `np.load(mmap_mode="r")` reads only the header, so this costs no vector
        or payload I/O. Counting used to load the whole index and throw it away.
        Returns None for a legacy index with no vector file.
        """
        vectors_path = self.path / _VECTORS_FILE
        if not vectors_path.exists():
            return None
        return int(np.load(vectors_path, mmap_mode="r").shape[0])

    @property
    def legacy_graph_pkl(self) -> Path:
        """Path to old graph.pkl if it exists (for migration)."""
        return self.path / "graph.pkl"
