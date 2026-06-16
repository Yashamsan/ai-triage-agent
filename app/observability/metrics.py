"""Retrieval observability — scores and child spans for FAQ lookup and ticket creation.

All public functions and the RetrievalMetricsLogger silently no-op when no
Langfuse trace is active so they never break the app on import or at runtime.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

from langfuse import get_client, observe


def _score(name: str, value: float) -> None:
    """Score the current active span; silently no-ops when no trace is active."""
    try:
        get_client().score_current_span(name=name, value=value)
    except Exception:
        pass


class RetrievalMetricsLogger:
    """Attach retrieval quality scores to the current Langfuse span."""

    def score_hit(self, hit: bool) -> None:
        """1.0 if the vector search returned a result, 0.0 otherwise."""
        _score("retrieval_hit_rate", 1.0 if hit else 0.0)

    def score_mrr(self, rank: int | None) -> None:
        """Mean Reciprocal Rank — 1/rank for first relevant result (0.0 on miss)."""
        _score("retrieval_mrr", (1.0 / rank) if rank and rank > 0 else 0.0)

    def score_precision_at_k(self, relevant: int, k: int) -> None:
        """Fraction of top-k results that were relevant."""
        _score("retrieval_precision_at_k", relevant / k if k > 0 else 0.0)

    def score_ticket_created(self, success: bool) -> None:
        _score("ticket_creation_success", 1.0 if success else 0.0)

    def log_latency(self, name: str, ms: float) -> None:
        _score(f"latency_ms_{name}", ms)

    @contextmanager
    def trace_latency(self, name: str):
        """Context manager that measures wall-clock ms and logs it as a score."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.log_latency(name, elapsed_ms)


# ── Instrumented wrappers ────────────────────────────────────────────────────
# Each creates a child span under the active parent observation.

@observe(name="trace-embedding")
def trace_embedding(text: str) -> list[float]:
    """Embed text and record latency as a child span."""
    from app.embeddings import embed
    return embed(text)


@observe(name="trace-vector-search")
def trace_vector_search(intent: str, embedding: list[float]) -> dict[str, Any] | None:
    """Run pgvector cosine search and return the top FAQ row (or None)."""
    from app.database import find_faq
    return find_faq(intent, embedding)


@observe(name="trace-ticket-creation")
def trace_ticket_creation(user_message: str, intent: str, embedding: list[float]) -> int:
    """Insert a support ticket and return its ID."""
    from app.database import create_ticket
    return create_ticket(user_message, intent, embedding)
