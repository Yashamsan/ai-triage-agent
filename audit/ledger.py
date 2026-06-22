"""Append-only JSONL ledger."""

import json
import os


class JSONLLedger:
    def __init__(self, path: str = "audit_data/ledger.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def append(self, tx: "DecisionTransaction") -> None:
        from .decision import DecisionTransaction
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(tx.to_json() + "\n")
            f.flush()
            os.fsync(f.fileno())

    def read_all(self) -> list["DecisionTransaction"]:
        from .decision import DecisionTransaction
        if not os.path.exists(self.path):
            return []
        txs = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    txs.append(DecisionTransaction.from_json(line))
        return txs

    def query(
        self,
        session_id: str | None = None,
        decision_type: str | None = None,
        limit: int = 100,
    ) -> list["DecisionTransaction"]:
        from .decision import DecisionTransaction
        txs = self.read_all()
        if session_id:
            txs = [t for t in txs if t.session_id == session_id]
        if decision_type:
            txs = [t for t in txs if t.decision_type == decision_type]
        return txs[-limit:]
