"""Multi-agent orchestration for the AI Triage platform.

Architecture:
    RouterAgent              — two-stage dispatch (classify → route)
    BaseSpecialistAgent      — abstract base all agents inherit
    ProofLayerGateway        — cross-agent governance bridge
    MemoryBridge             — shared cross-agent session memory
    load_config              — YAML + built-in defaults loader

Specialist agents (orchestrator.agents):
    TechnicalAgent           — outages, connectivity, API, device config
    BillingAgent             — invoices, refunds, payment failures, disputes
    ComplaintsAgent          — escalations, regulatory complaints, grievances
    SalesAgent               — upgrades, promos, cross-sell, lead capture
    LoyaltyAgent             — points, redemptions, tier status, benefits
    RetentionAgent           — churn, cancellations, save offers, win-back
    FraudDetectionAgent      — risk scoring, fraud reports, account security

Quick start:
    from orchestrator import RouterAgent, load_config
    from orchestrator.agents import (
        TechnicalAgent, BillingAgent, ComplaintsAgent,
        SalesAgent, LoyaltyAgent, RetentionAgent, FraudDetectionAgent,
    )

    cfg    = load_config()
    router = RouterAgent(cfg)
    router.load_agents([
        TechnicalAgent(cfg), BillingAgent(cfg), ComplaintsAgent(cfg),
        SalesAgent(cfg),     LoyaltyAgent(cfg), RetentionAgent(cfg),
        FraudDetectionAgent(cfg),
    ])
    result = router.route_sync("I was charged twice", session_id="s1")
    print(result["response"])
"""

from orchestrator.agent_base import AgentResponse, BaseSpecialistAgent
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

__all__ = [
    "RouterAgent",
    "BaseSpecialistAgent",
    "AgentResponse",
    "ProofLayerGateway",
    "MemoryBridge",
    "load_config",
    # Specialist agents
    "TechnicalAgent",
    "BillingAgent",
    "ComplaintsAgent",
    "SalesAgent",
    "LoyaltyAgent",
    "RetentionAgent",
    "FraudDetectionAgent",
]
