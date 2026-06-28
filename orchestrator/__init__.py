"""Multi-agent orchestration for the AI Triage platform.

Architecture:
    RouterAgent              — two-stage dispatch (classify → route)
    BaseSpecialistAgent      — abstract base all 6 agents inherit
    ProofLayerGateway        — cross-agent governance bridge
    MemoryBridge             — shared cross-agent session memory
    load_config              — YAML + built-in defaults loader

Quick start:
    from orchestrator import RouterAgent, load_config
    from orchestrator.prooflayer_gateway import ProofLayerGateway
    from orchestrator.memory_bridge import MemoryBridge

    cfg    = load_config()
    gw     = ProofLayerGateway(cfg)
    mem    = MemoryBridge()
    router = RouterAgent(cfg, gateway=gw, memory=mem)
    router.load_agents([billing_agent, technical_agent, ...])

    result = router.route_sync("I was charged twice", session_id="s1")
    print(result["response"])
"""

from orchestrator.agent_base import AgentResponse, BaseSpecialistAgent
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
]
