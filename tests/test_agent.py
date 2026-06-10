"""Tests for the LangGraph triage agent."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent_graph import build_triage_agent, route_after_tool


def test_graph_builds():
    graph = build_triage_agent()
    assert graph is not None


def test_graph_has_expected_nodes():
    graph = build_triage_agent()
    drawable = graph.get_graph()
    node_names = set(drawable.nodes)
    assert "classifier" in node_names
    assert "tool_runner" in node_names
    assert "responder" in node_names
    assert "escalation" in node_names


def test_graph_valid_traces():
    graph = build_triage_agent()
    drawable = graph.get_graph()
    assert len(drawable.nodes) >= 4


@pytest.mark.parametrize(
    "state_overrides,expected_last_node",
    [
        ({"resolved": True, "needs_escalation": False}, "responder"),
        ({"resolved": False, "needs_escalation": True}, "escalation"),
        ({"resolved": False, "needs_escalation": False}, "responder"),
    ],
)
def test_routing_logic(state_overrides, expected_last_node):
    state = {
        "message": "test",
        "intent": "password_reset",
        "confidence": 0.95,
        "needs_escalation": False,
        "tool_output": "helpful info",
        "resolved": True,
        "response_text": "",
        **state_overrides,
    }
    result = route_after_tool(state)
    assert result == expected_last_node
