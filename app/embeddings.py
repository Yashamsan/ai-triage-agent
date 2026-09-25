"""Embedding helper for the English triage agent.

Thin re-export of shared/embeddings.py — the actual model is loaded once
there and shared across app/, app_ar/, and the precedent/ProofLayer stores.
Do not load a separate model here; see shared/embeddings.py's docstring.
"""

from __future__ import annotations

from langfuse import observe

from shared.embeddings import EMBEDDING_DIM, MODEL_NAME, embed_batch
from shared.embeddings import embed as _embed

__all__ = ["MODEL_NAME", "EMBEDDING_DIM", "embed", "embed_batch"]


@observe(name="embed")
def embed(text: str) -> list[float]:
    """Return an EMBEDDING_DIM-dim embedding for text."""
    return _embed(text)
