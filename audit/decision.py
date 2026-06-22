"""Core data model — DecisionTransaction dataclass."""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


@dataclass
class DecisionTransaction:
    transaction_id: str = ""
    agent_id: str = "ai-triage-agent"
    session_id: str = ""
    timestamp: str = ""
    decision_type: str = "classification"
    input_hash: str = ""
    input_preview: str = ""
    output: dict = field(default_factory=dict)
    confidence: float | None = None
    policies_applied: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    model_id: str = "unknown"
    previous_hash: str | None = None
    chain_hash: str = ""
    human_override: bool = False
    human_correction: dict | None = None
    version: int = 1

    def __post_init__(self):
        if not self.transaction_id:
            self.transaction_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.input_hash and self.input_preview:
            import hashlib
            self.input_hash = hashlib.sha256(
                self.input_preview.encode("utf-8")
            ).hexdigest()

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=str)

    @classmethod
    def from_json(cls, line: str) -> "DecisionTransaction":
        data = json.loads(line)
        return cls(**data)
