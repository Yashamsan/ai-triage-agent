"""Tests for the multi-agent orchestrator module.

All tests are offline — no real LLM calls are made.

- Specialist agents use keyword-based _classify_node: no mock needed.
- RouterAgent._classify_intent_node calls litellm: mocked via patch.
- ProofLayerGateway is always used in local_mode=True.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.agents.billing import BillingAgent
from orchestrator.agents.complaints import ComplaintsAgent
from orchestrator.agents.fraud_detection import FraudDetectionAgent
from orchestrator.agents.loyalty import LoyaltyAgent
from orchestrator.agents.retention import RetentionAgent
from orchestrator.agents.sales import SalesAgent
from orchestrator.agents.technical import TechnicalAgent
from orchestrator.config import load_config
from orchestrator.memory_bridge import MemoryBridge
from orchestrator.prooflayer_gateway import ProofLayerGateway
from orchestrator.router import RouterAgent

# ═══════════════════════════════════════════════════════════════════════
# Shared fixtures
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture
def gateway(cfg):
    return ProofLayerGateway(cfg, local_mode=True)


@pytest.fixture
def memory():
    return MemoryBridge()


@pytest.fixture
def all_agents(cfg, gateway):
    return [
        TechnicalAgent(config=cfg, gateway=gateway),
        BillingAgent(config=cfg, gateway=gateway),
        ComplaintsAgent(config=cfg, gateway=gateway),
        SalesAgent(config=cfg, gateway=gateway),
        LoyaltyAgent(config=cfg, gateway=gateway),
        RetentionAgent(config=cfg, gateway=gateway),
        FraudDetectionAgent(config=cfg, gateway=gateway),
    ]


@pytest.fixture
def router(cfg, gateway, memory, all_agents):
    r = RouterAgent(cfg, gateway=gateway, memory=memory)
    r.load_agents(all_agents)
    return r


def _mock_llm(intent: str, agent: str, confidence: float = 0.93) -> MagicMock:
    """Build a litellm completion mock that returns a known routing JSON."""
    mock = MagicMock()
    mock.choices[0].message.content = json.dumps({
        "intent":      intent,
        "agent":       agent,
        "confidence":  confidence,
        "reasoning":   "test",
        "contains_pii": False,
    })
    return mock


# ═══════════════════════════════════════════════════════════════════════
# ProofLayerGateway — local-mode store
# ═══════════════════════════════════════════════════════════════════════


class TestProofLayerGateway:

    def test_local_mode_flag(self, cfg):
        gw = ProofLayerGateway(cfg, local_mode=True)
        assert gw.local_mode is True

    def test_local_mode_from_config(self):
        gw = ProofLayerGateway({"orchestrator": {"local_mode": True}})
        assert gw.local_mode is True

    def test_register_agent_stores_record(self, gateway):
        result = gateway.register_agent(
            name="test-agent", group="test", model_id="m1",
            intents=["billing_inquiry"], contains_pii=True,
        )
        assert result["agent_name"] == "test-agent"
        assert "agent_id" in result
        store = gateway.local_store()
        assert any(a["agent_name"] == "test-agent" for a in store["agents"])

    def test_record_decision_returns_id(self, gateway):
        result = gateway.record_decision(
            agent_name="billing-agent",
            agent_group="finance",
            model_id="m1",
            decision_value="refund_request",
            confidence=0.92,
            session_id="sess-001",
        )
        assert "decision_id" in result
        assert result["decision_id"] is not None

    def test_record_decision_stores_in_local(self, gateway):
        gateway.record_decision(
            agent_name="technical-agent",
            agent_group="support",
            model_id="m1",
            decision_value="connectivity_issue",
            confidence=0.88,
            session_id="sess-002",
        )
        store = gateway.local_store()
        assert any(
            d["agent_name"] == "technical-agent" and d["session_id"] == "sess-002"
            for d in store["decisions"]
        )

    def test_record_exception_stores(self, gateway):
        # Record a decision first so we have an ID
        dec = gateway.record_decision(
            agent_name="billing-agent", agent_group="finance",
            model_id="m1", decision_value="escalate",
            confidence=0.6, session_id="sess-003",
        )
        exc = gateway.record_exception(
            decision_id=dec["decision_id"],
            reason="Customer threatened legal action",
            severity="high",
        )
        assert "exception_id" in exc
        store = gateway.local_store()
        assert len(store["exceptions"]) >= 1
        assert any(e["severity"] == "high" for e in store["exceptions"])

    def test_record_cross_agent_edge(self, gateway):
        gateway.record_cross_agent_edge(
            from_decision_id="from-uuid",
            to_decision_id="to-uuid",
            relationship="ROUTER_DISPATCHED_TO",
        )
        store = gateway.local_store()
        assert any(
            e["relationship"] == "ROUTER_DISPATCHED_TO"
            for e in store["edges"]
        )

    def test_query_overview_empty(self, cfg):
        gw = ProofLayerGateway(cfg, local_mode=True)
        overview = gw.query_overview()
        assert overview["total_decisions"] == 0
        assert overview["avg_confidence"] == 0.0

    def test_query_overview_after_decisions(self, cfg):
        gw = ProofLayerGateway(cfg, local_mode=True)
        gw.record_decision(
            agent_name="a", agent_group="g", model_id="m",
            decision_value="x", confidence=0.8, session_id="s",
        )
        gw.record_decision(
            agent_name="a", agent_group="g", model_id="m",
            decision_value="y", confidence=0.6, session_id="s",
        )
        overview = gw.query_overview()
        assert overview["total_decisions"] == 2
        assert abs(overview["avg_confidence"] - 0.7) < 0.01

    def test_query_decisions_filter_by_agent(self, cfg):
        gw = ProofLayerGateway(cfg, local_mode=True)
        gw.record_decision(
            agent_name="billing-agent", agent_group="finance",
            model_id="m", decision_value="x", confidence=0.9, session_id="s1",
        )
        gw.record_decision(
            agent_name="technical-agent", agent_group="support",
            model_id="m", decision_value="y", confidence=0.8, session_id="s2",
        )
        results = gw.query_decisions(agent_name="billing-agent")
        assert all(d["agent_name"] == "billing-agent" for d in results)

    def test_local_store_returns_all_collections(self, gateway):
        store = gateway.local_store()
        assert set(store.keys()) == {"agents", "decisions", "exceptions", "edges"}

    def test_no_network_call_in_local_mode(self, cfg):
        gw = ProofLayerGateway(cfg, local_mode=True)
        # Must not raise ConnectionError or similar
        result = gw.record_decision(
            agent_name="test", agent_group="test",
            model_id="m", decision_value="test",
            confidence=0.5, session_id="offline-test",
        )
        assert "decision_id" in result


# ═══════════════════════════════════════════════════════════════════════
# MemoryBridge
# ═══════════════════════════════════════════════════════════════════════


class TestMemoryBridge:

    def test_start_session_creates_session(self, memory):
        sess = memory.start_session("m-sess-1")
        assert sess.session_id == "m-sess-1"
        assert sess.decisions == []

    def test_session_exists(self, memory):
        memory.start_session("m-exists")
        assert memory.session_exists("m-exists")
        assert not memory.session_exists("m-missing")

    def test_get_session_returns_none_for_missing(self, memory):
        assert memory.get_session("m-no-such") is None

    def test_add_agent_decision(self, memory):
        memory.start_session("m-sess-2")
        memory.add_agent_decision(
            session_id="m-sess-2",
            agent_name="billing-agent",
            agent_group="finance",
            intent="refund_request",
            confidence=0.91,
            decision_id="dec-abc",
        )
        sess = memory.get_session("m-sess-2")
        assert len(sess.decisions) == 1
        assert sess.decisions[0].agent_name == "billing-agent"
        assert sess.decisions[0].intent == "refund_request"

    def test_has_agent_handled(self, memory):
        memory.start_session("m-sess-3")
        memory.add_agent_decision(
            "m-sess-3", "technical-agent", "support",
            "connectivity_issue", 0.85, "dec-001",
        )
        assert memory.has_agent_handled("m-sess-3", "technical-agent")
        assert not memory.has_agent_handled("m-sess-3", "billing-agent")

    def test_get_agents_involved(self, memory):
        memory.start_session("m-sess-4")
        memory.add_agent_decision("m-sess-4", "billing-agent", "finance", "i", 0.9, "d1")
        memory.add_agent_decision("m-sess-4", "technical-agent", "support", "i", 0.8, "d2")
        involved = memory.get_agents_involved("m-sess-4")
        assert "billing-agent" in involved
        assert "technical-agent" in involved

    def test_get_cross_agent_context_excludes_requesting_agent(self, memory):
        memory.start_session("m-sess-5")
        memory.add_agent_decision("m-sess-5", "billing-agent", "finance", "refund", 0.9, "d1")
        memory.add_agent_decision("m-sess-5", "retention-agent", "retention", "cancel", 0.8, "d2")
        ctx = memory.get_cross_agent_context("m-sess-5", "retention-agent")
        assert "billing-agent" in ctx
        assert "retention-agent" not in ctx

    def test_get_cross_agent_context_empty_for_no_prior(self, memory):
        memory.start_session("m-sess-6")
        ctx = memory.get_cross_agent_context("m-sess-6", "billing-agent")
        assert ctx == ""

    def test_get_session_summary_shape(self, memory):
        memory.start_session("m-sess-7", customer_id="C001", language="en")
        memory.add_agent_decision("m-sess-7", "sales-agent", "growth", "upgrade", 0.88, "d1")
        summary = memory.get_session_summary("m-sess-7")
        assert summary["session_id"] == "m-sess-7"
        assert summary["total_decisions"] == 1
        assert "sales-agent" in summary["agents_involved"]
        assert "duration_s" in summary

    def test_clear_session(self, memory):
        memory.start_session("m-clear")
        memory.clear_session("m-clear")
        assert not memory.session_exists("m-clear")

    def test_add_decision_auto_creates_session(self, memory):
        memory.add_agent_decision(
            "m-auto-create", "billing-agent", "finance", "billing_inquiry", 0.7, "d1"
        )
        assert memory.session_exists("m-auto-create")


# ═══════════════════════════════════════════════════════════════════════
# RouterAgent
# ═══════════════════════════════════════════════════════════════════════


class TestRouterAgent:

    def test_load_agents_registers_all(self, router):
        assert len(router.agents) == 7

    def test_routing_table_covers_specialist_agents(self, router):
        table_agents = set(router.routing_table.values())
        assert "billing-agent" in table_agents
        assert "technical-agent" in table_agents
        assert "complaints-agent" in table_agents
        assert "fraud-detection-agent" in table_agents
        assert "retention-agent" in table_agents
        assert "sales-agent" in table_agents
        assert "loyalty-agent" in table_agents

    def test_routing_table_key_intents(self, router):
        assert "refund_request" in router.routing_table
        assert "service_outage" in router.routing_table
        assert "cancellation_request" in router.routing_table
        assert "fraud_report" in router.routing_table
        assert "points_inquiry" in router.routing_table

    def test_keyword_classify_returns_5tuple(self, router):
        result = router._keyword_classify("my internet is down")
        assert len(result) == 5
        intent, conf, reasoning, pii, agent_hint = result
        assert isinstance(intent, str)
        assert isinstance(conf, float)
        assert isinstance(pii, bool)
        assert agent_hint is None  # keyword fallback never sets agent_hint

    def test_keyword_classify_billing(self, router):
        intent, conf, *_ = router._keyword_classify("i need a refund for double charge")
        assert intent in ("billing_error", "refund_request")
        assert conf >= 0.70

    def test_keyword_classify_fraud(self, router):
        intent, conf, _, pii, _ = router._keyword_classify("my account was hacked and stolen")
        assert intent == "fraud_report"
        assert pii is True

    def test_keyword_classify_cancellation(self, router):
        intent, conf, *_ = router._keyword_classify("i want to cancel my service")
        assert intent == "cancellation_request"

    def test_keyword_classify_fallback(self, router):
        intent, conf, _, _, _ = router._keyword_classify("xylophone quantum nebula")
        assert intent == "general_inquiry"
        assert conf < 0.70

    def test_resolve_agent_known_intent(self, router):
        assert router._resolve_agent("refund_request") == "billing-agent"
        assert router._resolve_agent("service_outage") == "technical-agent"
        assert router._resolve_agent("cancellation_request") == "retention-agent"
        assert router._resolve_agent("fraud_report") == "fraud-detection-agent"

    def test_resolve_agent_unknown_falls_back(self, router):
        result = router._resolve_agent("totally_unknown_intent_xyz")
        assert isinstance(result, str)

    def test_system_prompt_includes_agent_block(self, router):
        assert "Available agents" in router._system_prompt
        assert "billing-agent" in router._system_prompt
        assert "technical-agent" in router._system_prompt

    def test_route_sync_tier1_direct_dispatch(self, router):
        """High confidence (>=90%) → tier 1, direct dispatch."""
        with patch("orchestrator.router.litellm.completion",
                   return_value=_mock_llm("connectivity_issue", "technical-agent", 0.95)):
            result = router.route_sync("my internet keeps dropping", session_id="rt-1")
        assert result["tier"] == 1
        assert result["target_agent"] == "technical-agent"
        assert result["intent"] == "connectivity_issue"
        assert result["needs_escalation"] is False

    def test_route_sync_uses_keyword_fallback_on_llm_error(self, router):
        """When litellm raises, _keyword_classify must be used and route succeeds."""
        with patch("orchestrator.router.litellm.completion",
                   side_effect=RuntimeError("API unavailable")):
            result = router.route_sync(
                "i need a refund for double charge", session_id="rt-2"
            )
        assert result["intent"] in ("billing_error", "refund_request")
        assert result["target_agent"] == "billing-agent"

    def test_route_sync_records_decision(self, router):
        with patch("orchestrator.router.litellm.completion",
                   return_value=_mock_llm("fraud_report", "fraud-detection-agent", 0.91)):
            result = router.route_sync("suspicious transaction on my account", session_id="rt-3")
        assert result["target_agent"] == "fraud-detection-agent"

    def test_route_sync_escalation_tier3(self, router):
        """Very low confidence → tier 3 escalation."""
        with patch("orchestrator.router.litellm.completion",
                   return_value=_mock_llm("general_inquiry", "triage-agent-en", 0.45)):
            result = router.route_sync("zxcvbnm asdfgh", session_id="rt-4")
        assert result["tier"] == 3
        assert result["needs_escalation"] is True

    def test_route_sync_state_has_all_keys(self, router):
        with patch("orchestrator.router.litellm.completion",
                   return_value=_mock_llm("billing_inquiry", "billing-agent", 0.92)):
            result = router.route_sync("what is my balance?", session_id="rt-5")
        for key in ("intent", "confidence", "tier", "target_agent",
                    "response", "needs_escalation", "agent_confidence"):
            assert key in result, f"missing key: {key}"


# ═══════════════════════════════════════════════════════════════════════
# Specialist agents — behavior_profile + process
# ═══════════════════════════════════════════════════════════════════════


class TestTechnicalAgent:

    def test_name_and_group(self, cfg):
        agent = TechnicalAgent(config=cfg)
        assert agent.name == "technical-agent"
        assert agent.group == "support"

    def test_behavior_profile_shape(self, cfg):
        agent = TechnicalAgent(config=cfg)
        profile = agent.behavior_profile()
        assert "intents" in profile
        assert len(profile["intents"]) >= 5
        assert "service_outage" in profile["intents"]
        assert profile["contains_pii"] is False
        assert profile["data_classification"] == "internal"
        assert "policies" in profile

    def test_behavior_profile_policies(self, cfg):
        agent = TechnicalAgent(config=cfg)
        policies = agent.behavior_profile()["policies"]
        assert policies.get("p1_auto_incident_ticket") is True
        assert "sev_levels" in policies

    def test_process_connectivity(self, cfg, gateway):
        agent = TechnicalAgent(config=cfg, gateway=gateway)
        result = agent.process("my internet keeps disconnecting", session_id="tech-1")
        assert result.intent == "connectivity_issue"
        assert result.confidence > 0.70
        assert result.response

    def test_process_outage(self, cfg, gateway):
        agent = TechnicalAgent(config=cfg, gateway=gateway)
        result = agent.process("service is completely down for all users", session_id="tech-2")
        assert result.intent == "service_outage"

    def test_process_returns_agent_response(self, cfg, gateway):
        agent = TechnicalAgent(config=cfg, gateway=gateway)
        result = agent.process("api is returning 503 errors", session_id="tech-3")
        assert result.intent is not None
        assert isinstance(result.confidence, float)
        assert isinstance(result.response, str)
        assert len(result.response) > 0


class TestBillingAgent:

    def test_name_and_group(self, cfg):
        agent = BillingAgent(config=cfg)
        assert agent.name == "billing-agent"
        assert agent.group == "finance"

    def test_behavior_profile_pii(self, cfg):
        agent = BillingAgent(config=cfg)
        profile = agent.behavior_profile()
        assert profile["contains_pii"] is True
        assert profile["data_classification"] == "restricted"

    def test_behavior_profile_policies(self, cfg):
        agent = BillingAgent(config=cfg)
        policies = agent.behavior_profile()["policies"]
        assert "refund_auto_approve_max_sar" in policies
        assert "fraud_signal_after_n_failures" in policies

    def test_intents_include_refund(self, cfg):
        agent = BillingAgent(config=cfg)
        assert "refund_request" in agent.behavior_profile()["intents"]

    def test_process_refund(self, cfg, gateway):
        agent = BillingAgent(config=cfg, gateway=gateway)
        result = agent.process("I need a refund for my last payment", session_id="bill-1")
        assert result.intent == "refund_request"
        assert result.contains_pii is True

    def test_process_dispute(self, cfg, gateway):
        agent = BillingAgent(config=cfg, gateway=gateway)
        result = agent.process("I was charged twice on my invoice", session_id="bill-2")
        assert result.intent in ("billing_error", "invoice_dispute", "refund_request")


class TestComplaintsAgent:

    def test_name_and_group(self, cfg):
        agent = ComplaintsAgent(config=cfg)
        assert agent.name == "complaints-agent"
        assert agent.group == "escalation"

    def test_intents_include_speak_to_manager(self, cfg):
        agent = ComplaintsAgent(config=cfg)
        intents = agent.behavior_profile()["intents"]
        assert "speak_to_manager" in intents or "escalation" in intents

    def test_policies_include_sla(self, cfg):
        agent = ComplaintsAgent(config=cfg)
        policies = agent.behavior_profile()["policies"]
        assert "sla" in policies

    def test_process_escalation(self, cfg, gateway):
        agent = ComplaintsAgent(config=cfg, gateway=gateway)
        result = agent.process("this is completely unacceptable, I want a manager", session_id="comp-1")
        assert result.intent is not None
        assert result.response


class TestSalesAgent:

    def test_name_and_group(self, cfg):
        agent = SalesAgent(config=cfg)
        assert agent.name == "sales-agent"
        assert agent.group == "growth"

    def test_intents_include_upgrade(self, cfg):
        agent = SalesAgent(config=cfg)
        assert "upgrade_request" in agent.behavior_profile()["intents"]

    def test_process_plan_inquiry(self, cfg, gateway):
        agent = SalesAgent(config=cfg, gateway=gateway)
        result = agent.process("what plans do you have?", session_id="sales-1")
        assert result.intent is not None
        assert result.response

    def test_process_upgrade(self, cfg, gateway):
        agent = SalesAgent(config=cfg, gateway=gateway)
        result = agent.process("I want to upgrade to the premium plan", session_id="sales-2")
        assert "upgrade" in result.intent


class TestLoyaltyAgent:

    def test_name_and_group(self, cfg):
        agent = LoyaltyAgent(config=cfg)
        assert agent.name == "loyalty-agent"
        assert agent.group == "growth"

    def test_intents_include_points(self, cfg):
        agent = LoyaltyAgent(config=cfg)
        assert "points_inquiry" in agent.behavior_profile()["intents"]

    def test_behavior_profile_policies(self, cfg):
        agent = LoyaltyAgent(config=cfg)
        policies = agent.behavior_profile()["policies"]
        assert "points_to_sar_rate" in policies
        assert "tiers" in policies

    def test_process_points(self, cfg, gateway):
        agent = LoyaltyAgent(config=cfg, gateway=gateway)
        result = agent.process("how many points do I have?", session_id="loyal-1")
        assert result.intent is not None
        assert result.response


class TestRetentionAgent:

    def test_name_and_group(self, cfg):
        agent = RetentionAgent(config=cfg)
        assert agent.name == "retention-agent"
        assert agent.group == "retention"

    def test_behavior_profile_pii(self, cfg):
        agent = RetentionAgent(config=cfg)
        profile = agent.behavior_profile()
        assert profile["contains_pii"] is True

    def test_intents_include_cancellation(self, cfg):
        agent = RetentionAgent(config=cfg)
        assert "cancellation_request" in agent.behavior_profile()["intents"]

    def test_process_cancellation(self, cfg, gateway):
        agent = RetentionAgent(config=cfg, gateway=gateway)
        result = agent.process("I want to cancel my subscription", session_id="ret-1")
        assert result.intent == "cancellation_request"
        assert result.response

    def test_process_competitor_mention(self, cfg, gateway):
        agent = RetentionAgent(config=cfg, gateway=gateway)
        result = agent.process("I'm thinking of switching to a competitor", session_id="ret-2")
        assert result.intent is not None


class TestFraudDetectionAgent:

    def test_name_and_group(self, cfg):
        agent = FraudDetectionAgent(config=cfg)
        assert agent.name == "fraud-detection-agent"
        assert agent.group == "risk"

    def test_behavior_profile_phi(self, cfg):
        agent = FraudDetectionAgent(config=cfg)
        profile = agent.behavior_profile()
        assert profile["contains_pii"] is True
        assert profile["data_classification"] == "phi"

    def test_policies_include_thresholds(self, cfg):
        agent = FraudDetectionAgent(config=cfg)
        policies = agent.behavior_profile()["policies"]
        assert "block_threshold_pct" in policies
        assert "review_threshold_pct" in policies
        assert "known_patterns" in policies

    def test_intents_include_fraud_alert(self, cfg):
        agent = FraudDetectionAgent(config=cfg)
        intents = agent.behavior_profile()["intents"]
        assert "fraud_alert" in intents or "account_compromise" in intents

    def test_process_account_compromise(self, cfg, gateway):
        agent = FraudDetectionAgent(config=cfg, gateway=gateway)
        result = agent.process("my account was hacked and transactions were made", session_id="fraud-1")
        assert result.intent in ("account_compromise", "fraud_alert", "fraud_report")
        assert result.contains_pii is True

    def test_process_high_value_transaction(self, cfg, gateway):
        agent = FraudDetectionAgent(config=cfg, gateway=gateway)
        result = agent.process(
            "why was my transaction for 12,450 SAR blocked?", session_id="fraud-2"
        )
        assert result.intent is not None
        assert result.response

    def test_process_suspicious_transaction(self, cfg, gateway):
        agent = FraudDetectionAgent(config=cfg, gateway=gateway)
        result = agent.process(
            "there are unauthorized charges on my account", session_id="fraud-3"
        )
        assert result.intent is not None


# ═══════════════════════════════════════════════════════════════════════
# Cross-agent integration
# ═══════════════════════════════════════════════════════════════════════


class TestCrossAgentIntegration:
    """Verify memory bridge + router work together across multiple hops."""

    def test_billing_then_retention_sees_context(self, router):
        """Billing decision is visible to retention in the same session."""
        sid = "cross-1"

        with patch("orchestrator.router.litellm.completion",
                   return_value=_mock_llm("refund_request", "billing-agent", 0.93)):
            router.route_sync("I need a refund", session_id=sid)

        ctx = router.memory.get_cross_agent_context(sid, "retention-agent")
        assert "billing-agent" in ctx

    def test_session_summary_after_two_hops(self, router):
        sid = "cross-2"
        with patch("orchestrator.router.litellm.completion",
                   return_value=_mock_llm("fraud_report", "fraud-detection-agent", 0.91)):
            router.route_sync("my card was stolen", session_id=sid)

        summary = router.memory.get_session_summary(sid)
        assert summary["total_decisions"] >= 1
        assert "fraud-detection-agent" in summary["agents_involved"]

    def test_prooflayer_gateway_collects_both_decisions(self, router):
        sid = "cross-3"
        initial_count = len(router.gateway.local_store()["decisions"])
        with patch("orchestrator.router.litellm.completion",
                   return_value=_mock_llm("points_inquiry", "loyalty-agent", 0.92)):
            router.route_sync("how many points do I have?", session_id=sid)
        final_count = len(router.gateway.local_store()["decisions"])
        # Router decision + agent decision
        assert final_count > initial_count
