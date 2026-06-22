"""Retrieval observability — scores and child spans for Arabic FAQ lookup."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

from langfuse import get_client, observe


def _score(name: str, value: float) -> None:
    try:
        get_client().score_current_span(name=name, value=value)
    except Exception:
        pass


class RetrievalMetricsLogger:
    def score_hit(self, hit: bool) -> None:
        _score("retrieval_hit_rate", 1.0 if hit else 0.0)

    def score_mrr(self, rank: int | None) -> None:
        _score("retrieval_mrr", (1.0 / rank) if rank and rank > 0 else 0.0)

    def score_precision_at_k(self, relevant: int, k: int) -> None:
        _score("retrieval_precision_at_k", relevant / k if k > 0 else 0.0)

    def score_ticket_created(self, success: bool) -> None:
        _score("ticket_creation_success", 1.0 if success else 0.0)

    def log_latency(self, name: str, ms: float) -> None:
        _score(f"latency_ms_{name}", ms)

    @contextmanager
    def trace_latency(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.log_latency(name, elapsed_ms)


@observe(name="trace-embedding-ar")
def trace_embedding(text: str) -> list[float]:
    from app_ar.embeddings import embed

    return embed(text)


@observe(name="trace-vector-search-ar")
def trace_vector_search(intent: str, embedding: list[float]) -> dict[str, Any] | None:
    from app_ar.database import find_faq

    return find_faq(intent, embedding)


@observe(name="trace-ticket-creation-ar")
def trace_ticket_creation(user_message: str, intent: str, embedding: list[float]) -> int:
    from app_ar.database import create_ticket

    return create_ticket(user_message, intent, embedding)
