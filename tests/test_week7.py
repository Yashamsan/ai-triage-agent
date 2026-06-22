"""Tests for Week 7 — Memory + Reflection."""

from app.agent_graph import (
    build_triage_agent,
    route_after_classifier,
    route_after_memory,
    route_after_reflection,
    route_after_tool,
)
from app.memory import clear_session, get_session, session_exists


class TestSessionMemory:
    def test_create_session(self):
        clear_session("test-session")
        session = get_session("test-session")
        assert session is not None
        assert session.session_id == "test-session"
        assert len(session.turns) == 0

    def test_add_turn(self):
        clear_session("test-session-2")
        session = get_session("test-session-2")
        session.add_turn("user", "Hello")
        assert len(session.turns) == 1
        assert session.turns[0].role == "user"
        assert session.turns[0].content == "Hello"

    def test_recent_context_returns_last_n(self):
        clear_session("test-context")
        session = get_session("test-context")
        session.add_turn("user", "Turn 1")
        session.add_turn("assistant", "Reply 1")
        session.add_turn("user", "Turn 2")
        session.add_turn("assistant", "Reply 2")
        context = session.recent_context(n=2)
        assert "Turn 1" not in context
        assert "Turn 2" in context
        assert "Reply 2" in context

    def test_to_context_block(self):
        clear_session("test-block")
        session = get_session("test-block")
        session.add_turn("user", "I need a password reset")
        block = session.to_context_block()
        assert "Conversation History" in block
        assert "password reset" in block

    def test_session_exists(self):
        clear_session("test-exists")
        assert not session_exists("test-exists")
        session = get_session("test-exists")
        session.add_turn("user", "Hello")
        assert session_exists("test-exists")


class TestRouting:
    def test_route_after_classifier_always_reflect(self):
        state = {"intent": "password_reset"}
        assert route_after_classifier(state) == "reflect"

    def test_route_after_reflection_always_tool_runner(self):
        # All intents go through tool_runner — no shortcuts to responder
        for intent in ("unknown", "billing", "greeting", "password_reset"):
            state = {"intent": intent}
            assert route_after_reflection(state) == "tool_runner"

    def test_route_after_tool_always_memory(self):
        state = {"resolved": True}
        assert route_after_tool(state) == "store_memory"

    def test_route_after_memory_resolved_no_escalation(self):
        state = {"needs_escalation": False, "resolved": True}
        assert route_after_memory(state) == "responder"

    def test_route_after_memory_unresolved_not_flagged_goes_responder(self):
        # unresolved but not escalation-flagged → responder (best-effort answer)
        state = {"needs_escalation": False, "resolved": False}
        assert route_after_memory(state) == "responder"

    def test_route_after_memory_escalation_flag_escalates(self):
        state = {"needs_escalation": True, "resolved": False}
        assert route_after_memory(state) == "escalation"


class TestGraphStructure:
    def test_graph_builds(self):
        graph = build_triage_agent()
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        graph = build_triage_agent()
        drawable = graph.get_graph()
        node_names = set(n for n in drawable.nodes)
        expected = {"classifier", "reflect", "tool_runner",
                    "store_memory", "responder", "escalation"}
        assert expected.issubset(node_names), (
            f"Missing nodes: {expected - node_names}"
        )
