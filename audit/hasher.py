"""Hash chain — tamper evidence."""

import hashlib
import json

from .decision import DecisionTransaction


def compute_chain_hash(
    previous_hash: str | None, tx: DecisionTransaction
) -> str:
    payload = (
        (previous_hash or "")
        + tx.transaction_id
        + tx.timestamp
        + json.dumps(tx.output, sort_keys=True, ensure_ascii=False)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_chain(
    transactions: list[DecisionTransaction],
) -> list[int]:
    violations = []
    for i, tx in enumerate(transactions):
        expected_prev = transactions[i - 1].chain_hash if i > 0 else None
        expected_hash = compute_chain_hash(expected_prev, tx)
        if tx.chain_hash != expected_hash:
            violations.append(i)
    return violations
