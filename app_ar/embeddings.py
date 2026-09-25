"""Embedding helper for the Arabic triage agent.

Thin re-export of shared/embeddings.py — the actual model is loaded once
there and shared across app/, app_ar/, and the precedent/ProofLayer stores.
Do not load a separate model here; see shared/embeddings.py's docstring.

Both agents share the same faq_articles / knowledge_base_chunks tables and
must use the same embedding model as app/embeddings.py so vector spaces
match — that's exactly why this re-exports the same shared singleton
instead of loading its own.
"""

from __future__ import annotations

from langfuse import observe

from shared.embeddings import EMBEDDING_DIM, MODEL_NAME, embed_batch
from shared.embeddings import embed as _embed

__all__ = ["MODEL_NAME", "EMBEDDING_DIM", "embed", "embed_batch"]


@observe(name="embed_ar")
def embed(text: str) -> list[float]:
    return _embed(text)
