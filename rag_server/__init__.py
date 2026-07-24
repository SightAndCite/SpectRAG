"""SpectRAG web server package.

Splits the FastAPI server into focused modules:
  runtime.py       — shared Config + on-disk path layout
  schemas.py       — request models
  session_store.py — SessionStore (persisted chat sessions)
  services.py      — ActiveIndex (in-memory index cache) + IndexingService
  graph_view.py    — /graph payload serialization

The FastAPI app, lifespan, and routes live in the top-level ``server.py``.
"""
