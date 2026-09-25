"""Single source of truth for the sentence-embedding model used across the
whole app — KB/FAQ retrieval (app/embeddings.py, app_ar/embeddings.py),
precedent memory (shared/precedent_store.py), and ProofLayer node similarity
(app/prooflayer_graph.py).

Previously each of those four call sites loaded its own `SentenceTransformer`
instance independently (same model, four copies in memory). That was
survivable at ~80MB (all-MiniLM-L6-v2) but would not be at this model's
size — load the model ONCE here and import embed()/embed_batch() from this
module; do not add another SentenceTransformer(...) call site.

Model: BAAI/bge-m3 — a real multilingual embedding model (Arabic included),
replacing all-MiniLM-L6-v2 (English-only training; see app_ar/database.py's
former keyword-overlap workaround, no longer needed with real Arabic-aware
similarity scores). 1024-dim dense output, cosine similarity.

Changing EMBEDDING_DIM requires a matching schema migration for every
`vector(N)` column (see app/schema.sql, app_ar/schema.sql,
app/schema_prooflayer.sql) — the column type must match this constant or
inserts fail outright.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer as _ST

MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# Set to "cuda" or "cpu" to force a device; unset auto-detects (GPU if the
# container has one passed through and torch was built with CUDA support,
# else CPU). See docker-compose.yml's GPU reservation for the runtime side
# of this.
_DEVICE_OVERRIDE = os.getenv("EMBEDDING_DEVICE")

_model: _ST | None = None


def _select_device() -> str:
    if _DEVICE_OVERRIDE:
        return _DEVICE_OVERRIDE
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _get_model() -> _ST:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        device = _select_device()
        model = SentenceTransformer(MODEL_NAME, device=device)
        if device == "cuda":
            # A 4-6GB card can't comfortably hold bge-m3 in fp32 (~2.2GB
            # weights + CUDA context overhead, easily 3GB+ under real
            # inference) alongside whatever else is using the GPU. fp16
            # roughly halves that with no meaningful quality loss for
            # cosine-similarity search, and every CUDA GPU this app is
            # likely to run on (Turing/GTX 16xx or newer) supports it
            # natively.
            model = model.half()
        print(f"[shared.embeddings] {MODEL_NAME} loaded on device={device}"
              + (" (fp16)" if device == "cuda" else ""))
        _model = model
    return _model


def embed(text: str) -> list[float]:
    """Return a single EMBEDDING_DIM-dim embedding for text."""
    return _get_model().encode(
        text, convert_to_numpy=True, normalize_embeddings=True
    ).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts in one forward pass (faster for seeding/ingestion)."""
    return _get_model().encode(
        texts, convert_to_numpy=True, normalize_embeddings=True
    ).tolist()
