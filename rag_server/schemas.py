"""Request body models for the server routes."""
from __future__ import annotations

from pydantic import BaseModel


class CreateSessionBody(BaseModel):
    name: str | None = None


class QueryBody(BaseModel):
    question: str
