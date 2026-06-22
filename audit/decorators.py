"""@audit_record decorator — instrument any agent function."""

import hashlib
import inspect
import uuid
from datetime import datetime, timezone
from functools import wraps

from .decision import DecisionTransaction
from .hasher import compute_chain_hash
from .ledger import JSONLLedger

ledger = JSONLLedger()


def audit_record(agent_id: str = "ai-triage-agent", model_id: str = "unknown"):
    def decorator(func):
        def _extract_input(*args, **kwargs) -> tuple[str, str]:
            input_text = ""
            for arg in args:
                if isinstance(arg, str):
                    input_text = arg
                    break
            if not input_text and "user_message" in kwargs:
                input_text = kwargs["user_message"]
            # Also look for Pydantic request objects (e.g. TriageRequest)
            if not input_text:
                for arg in args:
                    msg = getattr(arg, "message", None)
                    if isinstance(msg, str):
                        input_text = msg
                        break
            input_preview = input_text[:200]
            input_hash = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
            return input_preview, input_hash

        def _build_and_store(result, input_preview: str, input_hash: str, kwargs: dict):
            prev_txs = ledger.read_all()
            previous_hash = prev_txs[-1].chain_hash if prev_txs else None

            if isinstance(result, dict):
                output = result
                decision_type = result.get("decision_type", result.get("action", "unknown"))
                confidence = result.get("confidence")
            else:
                # Pydantic model or other object
                output = {
                    k: getattr(result, k)
                    for k in ("intent", "confidence", "needs_escalation", "interrupted")
                    if hasattr(result, k)
                }
                decision_type = "classification"
                confidence = getattr(result, "confidence", None)

            # session_id from kwargs or first request arg
            session_id = kwargs.get("session_id", "unknown")

            tx = DecisionTransaction(
                transaction_id=str(uuid.uuid4()),
                agent_id=agent_id,
                session_id=session_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                decision_type=decision_type,
                input_hash=input_hash,
                input_preview=input_preview,
                output=output,
                confidence=confidence,
                policies_applied=kwargs.get("policies_applied", []),
                evidence_refs=kwargs.get("evidence_refs", []),
                model_id=model_id,
                previous_hash=previous_hash,
                chain_hash="",
                human_override=kwargs.get("human_override", False),
                human_correction=kwargs.get("human_correction"),
                version=1,
            )
            tx.chain_hash = compute_chain_hash(previous_hash, tx)
            ledger.append(tx)

        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                input_preview, input_hash = _extract_input(*args, **kwargs)
                result = await func(*args, **kwargs)
                try:
                    _build_and_store(result, input_preview, input_hash, kwargs)
                except Exception as e:
                    print(f"[Audit] Failed to record decision: {e}")
                return result
            return async_wrapper
        else:
            @wraps(func)
            def wrapper(*args, **kwargs):
                input_preview, input_hash = _extract_input(*args, **kwargs)
                result = func(*args, **kwargs)
                try:
                    _build_and_store(result, input_preview, input_hash, kwargs)
                except Exception as e:
                    print(f"[Audit] Failed to record decision: {e}")
                return result
            return wrapper

    return decorator
