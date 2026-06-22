"""Arabic-capable embedding via sentence-transformers (multilingual)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langfuse import observe

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer as _ST

_model: _ST | None = None
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384


def _get_model() -> _ST:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


@observe(name="embed_ar")
def embed(text: str) -> list[float]:
    return _get_model().encode(text, convert_to_numpy=True).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    return _get_model().encode(texts, convert_to_numpy=True).tolist()
