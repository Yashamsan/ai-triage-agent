"""Embedding helper via sentence-transformers.

Uses all-MiniLM-L6-v2 (384-dim, ~80MB). Model is loaded once and cached
as a module-level singleton to avoid re-loading on every call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langfuse import observe

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer as _ST

_model: _ST | None = None
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def _get_model() -> _ST:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


@observe(name="embed")
def embed(text: str) -> list[float]:
    """Return a 384-dim embedding for text."""
    return _get_model().encode(text, convert_to_numpy=True).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts in one forward pass (faster for seeding)."""
    return _get_model().encode(texts, convert_to_numpy=True).tolist()
