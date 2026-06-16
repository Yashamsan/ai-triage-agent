#!/usr/bin/env python3
"""Offline retrieval eval harness.

Sends 7 known-answer queries through the real retrieval pipeline,
logs hit-rate and MRR scores to Langfuse, and prints an aggregate summary.

Usage (from repo root, with the DB seeded and .env loaded):
    python scripts/eval_retrieval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langfuse import get_client, observe

from app.observability import RetrievalMetricsLogger, trace_embedding, trace_vector_search

EVAL_CASES = [
    {"query": "I forgot my password and can't log in", "intent": "password_reset", "expect_hit": True},
    {"query": "How do I reset my password?", "intent": "password_reset", "expect_hit": True},
    {"query": "My invoice looks wrong", "intent": "billing", "expect_hit": True},
    {"query": "How do I update my billing information?", "intent": "billing", "expect_hit": True},
    {"query": "The app keeps crashing on startup", "intent": "technical_support", "expect_hit": True},
    {"query": "I can't connect to the service", "intent": "technical_support", "expect_hit": True},
    {"query": "Lorem ipsum dolor sit amet", "intent": "password_reset", "expect_hit": False},
]


@observe(name="eval-retrieval-run")
def run_eval() -> None:
    metrics = RetrievalMetricsLogger()
    hits = 0
    mrr_total = 0.0
    n = len(EVAL_CASES)

    print(f"Running {n} eval cases...\n")

    for i, case in enumerate(EVAL_CASES, start=1):
        query: str = case["query"]
        intent: str = case["intent"]
        expect_hit: bool = case["expect_hit"]

        embedding = trace_embedding(query)
        row = trace_vector_search(intent, embedding)

        hit = row is not None
        metrics.score_hit(hit)
        metrics.score_mrr(1 if hit else None)

        if hit:
            hits += 1
            mrr_total += 1.0

        status = "HIT " if hit else "MISS"
        expected = "HIT " if expect_hit else "MISS"
        correct = "v" if hit == expect_hit else "x"
        sim = f"{row['similarity']:.3f}" if row else "  —  "
        print(f"[{i}/{n}] {correct} {status} (expected {expected}) sim={sim} | {query[:55]}")

    hit_rate = hits / n
    mrr = mrr_total / n

    print(f"\nAggregate  hit_rate={hit_rate:.3f}  MRR={mrr:.3f}  ({hits}/{n} hits)")

    get_client().score_current_span(name="eval_hit_rate", value=hit_rate)
    get_client().score_current_span(name="eval_mrr", value=mrr)


if __name__ == "__main__":
    run_eval()
    print("\nScores flushed to Langfuse.")
