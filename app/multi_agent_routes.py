"""FastAPI routes for the multi-agent orchestrator.

Adds a POST /triage/multi endpoint that routes through RouterAgent.
Initializes the orchestrator lazily on first request so the app starts
quickly even if orchestrator imports are slow.
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.security.guard_classifier import guard_classify
from app.security.input_sanitizer import InputSanitizer
from app.security.output_filter import OutputFilter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/triage", tags=["multi-agent"])

# ── Lazy singleton ─────────────────────────────────────────────────────────────

_orchestrator: dict[str, Any] | None = None


def _get_orchestrator() -> dict[str, Any]:
    """Initialize and cache the orchestrator on first call."""
    global _orchestrator
    if _orchestrator is not None:
        return _orchestrator

    from orchestrator.agents import (
        BillingAgent,
        ComplaintsAgent,
        FraudDetectionAgent,
        LoyaltyAgent,
        RetentionAgent,
        SalesAgent,
        TechnicalAgent,
    )
    from orchestrator.config import load_config
    from orchestrator.memory_bridge import MemoryBridge
    from orchestrator.prooflayer_gateway import ProofLayerGateway
    from orchestrator.router import RouterAgent

    cfg = load_config()
    # This route runs inside the same app/container as the real ProofLayer
    # API (app/prooflayer_api.py) and its Postgres-backed pl_agents/pl_nodes
    # tables, so default to real recording (local_mode=False) — that's what
    # makes specialist agents and their decisions show up in the ProofLayer
    # UI. Set ORCHESTRATOR_LOCAL_MODE=true to fall back to the in-memory
    # store (e.g. running this route without the Postgres stack available).
    local_mode = os.getenv("ORCHESTRATOR_LOCAL_MODE", "false").lower() == "true"
    gateway = ProofLayerGateway(cfg, local_mode=local_mode)
    memory  = MemoryBridge()

    # Each agent gets the shared gateway so decisions land in the same store
    agents = [
        TechnicalAgent(config=cfg, gateway=gateway),
        BillingAgent(config=cfg, gateway=gateway),
        ComplaintsAgent(config=cfg, gateway=gateway),
        SalesAgent(config=cfg, gateway=gateway),
        LoyaltyAgent(config=cfg, gateway=gateway),
        RetentionAgent(config=cfg, gateway=gateway),
        FraudDetectionAgent(config=cfg, gateway=gateway),
    ]

    router_agent = RouterAgent(cfg, gateway=gateway, memory=memory)
    router_agent.load_agents(agents)

    _orchestrator = {
        "gateway":  gateway,
        "memory":   memory,
        "agents":   {a.name: a for a in agents},
        "router":   router_agent,
    }
    logger.info(
        "Multi-agent orchestrator initialized: %d agents, %d routing rules",
        len(agents),
        len(router_agent.routing_table),
    )
    return _orchestrator


def init_orchestrator() -> None:
    """Eagerly initialize the orchestrator so all specialist agents register
    themselves with ProofLayer (pl_agents) at app startup, instead of only
    on first /triage/multi request. Non-fatal — a failure here just means
    agents register lazily on first use instead."""
    try:
        orch = _get_orchestrator()
        logger.info("Orchestrator pre-warmed: %d agents registered", len(orch["agents"]))
    except Exception as exc:
        logger.warning("Orchestrator eager init skipped: %s", exc)


# ── Pydantic models ────────────────────────────────────────────────────────────


class MultiTriageRequest(BaseModel):
    message:    str
    session_id: str | None = None


class MultiTriageResponse(BaseModel):
    intent:           str
    response:         str
    confidence:       float
    agent:            str
    agent_group:      str
    needs_escalation: bool
    trace:            dict[str, Any] | None = None


# ── Security helpers (module-level singletons) ─────────────────────────────────

_sanitizer     = InputSanitizer()
_output_filter = OutputFilter()


# ── Endpoint ───────────────────────────────────────────────────────────────────


@router.post("/multi", response_model=MultiTriageResponse)
def triage_multi(request: MultiTriageRequest) -> MultiTriageResponse:
    """Route a customer message through the multi-agent orchestrator.

    Security pipeline: sanitize → guard → route → filter PII → return.
    """
    if not request.message.strip():
        raise HTTPException(status_code=422, detail="message cannot be empty")

    # ── Phase 1: Input sanitization ───────────────────────────────────────
    sanitized = _sanitizer.sanitize(request.message)
    if sanitized.blocked:
        raise HTTPException(
            status_code=422,
            detail=f"Message rejected: {sanitized.block_reason}",
        )
    safe_message = sanitized.sanitized_message

    # ── Phase 2: Guard classifier ─────────────────────────────────────────
    guard = guard_classify(safe_message)
    if guard.is_injection and guard.confidence > 0.7:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Message rejected: suspected prompt injection "
                f"(confidence={guard.confidence:.2f})"
            ),
        )

    # ── Phase 3: Route via orchestrator ───────────────────────────────────
    orch         = _get_orchestrator()
    memory       = orch["memory"]
    router_agent = orch["router"]
    agents_map   = orch["agents"]

    session_id = request.session_id or str(uuid.uuid4())
    memory.start_session(session_id)

    result = router_agent.route_sync(
        message=safe_message,
        session_id=session_id,
    )

    agent_name    = result.get("target_agent") or "unknown"
    response_text = result.get("response") or ""

    # Derive agent group from the registered agent instance
    agent_obj   = agents_map.get(agent_name)
    agent_group = agent_obj.group if agent_obj else ""

    # Prefer agent-level confidence; fall back to router classification confidence.
    # Must check "is not None", not truthiness -- 0.0 is router.py's deliberate
    # signal for a dispatch failure/escalation and must not be masked by a
    # fallback to the (possibly high) router classification confidence.
    agent_confidence = result.get("agent_confidence")
    confidence = float(agent_confidence if agent_confidence is not None else result.get("confidence", 0.0))
    needs_escalation = bool(result.get("needs_escalation", False))

    # ── Phase 4: PII output filter ────────────────────────────────────────
    pii_result = _output_filter.filter_pii(response_text)

    return MultiTriageResponse(
        intent=result.get("intent") or "unknown",
        response=pii_result.filtered_text,
        confidence=confidence,
        agent=agent_name,
        agent_group=agent_group,
        needs_escalation=needs_escalation,
        trace={
            "session_id":        session_id,
            "tier":              result.get("tier"),
            "route_decision_id": result.get("route_decision_id"),
            "agent_decision_id": result.get("agent_decision_id"),
            "reasoning":         result.get("reasoning"),
            "reflection_notes":  result.get("reflection_notes"),
        },
    )
