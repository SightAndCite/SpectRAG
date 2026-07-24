from __future__ import annotations
import logging
import pickle
from pathlib import Path
import faiss
from rag_system.models import Chunk

logger = logging.getLogger(__name__)

_CHUNKS_FILE  = "chunks.pkl"
_FAISS_FILE   = "faiss.index"


class IndexStore:
    """
    Persists the two offline artefacts that live on disk: chunks and FAISS index.
    The graph is now stored in Neo4j (see Neo4jGraphClient) — graph.pkl is no
    longer written. Legacy graph.pkl files are migrated on server startup.
    """

    def __init__(self, store_path: Path | str) -> None:
        self.path = Path(store_path)

    def save(self, chunks: list[Chunk], faiss_index: faiss.Index) -> None:
        self.path.mkdir(parents=True, exist_ok=True)

        with open(self.path / _CHUNKS_FILE, "wb") as fh:
            pickle.dump(chunks, fh, protocol=5)

        faiss.write_index(faiss_index, str(self.path / _FAISS_FILE))

        # spectral_coords are a Chunk field and are already persisted inside
        # chunks.pkl, so no separate spectral.npy is written. (The old side file
        # was reloaded by absolute position, which silently misaligned coords onto
        # the wrong chunks if any chunk lacked coords — see possible_fixes CASE-08.)

        logger.info("IndexStore saved: %d chunks → %s", len(chunks), self.path)

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

        # spectral_coords come back with the chunks from the pickle above; any
        # legacy spectral.npy side file is intentionally ignored (see CASE-08).

        logger.info("IndexStore loaded: %d chunks from %s", len(chunks), self.path)
        return chunks, faiss_index

    @property
    def legacy_graph_pkl(self) -> Path:
        """Path to old graph.pkl if it exists (for migration)."""
        return self.path / "graph.pkl"
