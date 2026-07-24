from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path

from config import Config
from rag_system.indexing.pipeline import IndexingPipeline
from rag_system.query.pipeline import QueryPipeline
from rag_system.store.index_store import IndexStore
from rag_system.store.neo4j_client import Neo4jGraphClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class CLI:
    """Command-line interface for the Graph-Spectral RAG system."""

    _SEPARATOR = "=" * 64

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    # Public commands

    def index(self, paths: list[Path]) -> None:
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            logger.error("Files not found: %s", ", ".join(missing))
            sys.exit(1)

        neo4j = self._connect_neo4j()
        try:
            IndexingPipeline(self._cfg, neo4j).index(paths)
            print(f"\nIndex saved to: {self._cfg.store_path}")
        finally:
            neo4j.close()

    def query(self, question: str) -> None:
        chunks, faiss_index = IndexStore(self._cfg.store_path).load()
        neo4j    = self._connect_neo4j()
        pipeline = QueryPipeline(self._cfg)
        try:
            result = pipeline.query(question, chunks, faiss_index, neo4j)
        finally:
            pipeline.close()
            neo4j.close()

        sep = self._SEPARATOR
        print(f"\n{sep}\nANSWER\n{sep}")
        print(result.answer)

        print(f"\n{sep}\nSOURCES ({len(result.chunks)} chunks)\n{sep}")
        for chunk in sorted(result.chunks, key=lambda c: result.scores.get(c.chunk_id, 0.0), reverse=True):
            score    = result.scores.get(chunk.chunk_id, 0.0)
            src      = chunk.metadata.get("source", chunk.doc_id)
            page     = chunk.metadata.get("page", "")
            page_str = f"  p.{page}" if page else ""
            print(f"  [{score:.4f}]  {src}{page_str}  (pos {chunk.position})")

    # Internal helpers

    def _connect_neo4j(self) -> Neo4jGraphClient:
        client = Neo4jGraphClient(self._cfg.neo4j)
        try:
            client.connect()
        except Exception as exc:
            logger.error("Cannot connect to Neo4j at %s: %s", self._cfg.neo4j.uri, exc)
            sys.exit(1)
        return client


# Entry point

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Graph-Spectral RAG — graph-diffusion retrieval with spectral clustering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py index papers/*.pdf\n"
            "  python main.py query 'What methods improve cross-document reasoning?'\n"
            "  python main.py index docs/ --store ./my_index\n"
            "  python main.py query 'Summarize the key findings' --store ./my_index\n"
        ),
    )
    parser.add_argument(
        "--store", default=None, metavar="PATH",
        help="Override the index store directory (default: ./rag_index)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    idx_p = sub.add_parser("index", help="Build index from documents")
    idx_p.add_argument(
        "files", nargs="+",
        help="Paths to documents (.pdf .txt .md .html .docx)",
    )

    qry_p = sub.add_parser("query", help="Answer a question from the index")
    qry_p.add_argument("question", help="Natural-language question")

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    cfg  = Config()
    if args.store:
        cfg.store_path = Path(args.store)

    cli = CLI(cfg)
    if args.command == "index":
        cli.index([Path(p) for p in args.files])
    elif args.command == "query":
        cli.query(args.question)


if __name__ == "__main__":
    main()
