"""Export formats — chain verification, CSV, summary."""

import csv
import io
import os

from .decision import DecisionTransaction
from .hasher import verify_chain
from .ledger import JSONLLedger


def chain_summary() -> dict:
    ledger = JSONLLedger()
    txs = ledger.read_all()
    violations = verify_chain(txs) if txs else []

    type_counts = {}
    for tx in txs:
        t = tx.decision_type
        type_counts[t] = type_counts.get(t, 0) + 1

    overrides = sum(1 for tx in txs if tx.human_override)

    return {
        "status": "INTACT" if not violations else "COMPROMISED",
        "total_transactions": len(txs),
        "violations": violations,
        "time_range": (
            f"{txs[0].timestamp[:10] if txs else 'N/A'} to "
            f"{txs[-1].timestamp[:10] if txs else 'N/A'}"
        ),
        "decisions_by_type": type_counts,
        "human_overrides": overrides,
    }


def export_csv(output_path: str | None = None) -> str:
    ledger = JSONLLedger()
    txs = ledger.read_all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "transaction_id", "timestamp", "agent_id", "decision_type",
        "confidence", "human_override", "input_preview", "chain_hash",
    ])
    for tx in txs:
        writer.writerow([
            tx.transaction_id, tx.timestamp, tx.agent_id, tx.decision_type,
            tx.confidence, tx.human_override, tx.input_preview[:80], tx.chain_hash,
        ])

    if output_path:
        with open(output_path, "w", newline="") as f:
            f.write(output.getvalue())

    return output.getvalue()
