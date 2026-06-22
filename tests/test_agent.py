"""Tests for the LangGraph triage agent."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent_graph import build_triage_agent, route_after_memory, route_after_tool


def test_graph_builds():
    graph = build_triage_agent()
    assert graph is not None


def test_graph_has_expected_nodes():
    graph = build_triage_agent()
    drawable = graph.get_graph()
    node_names = set(drawable.nodes)
    # 6-node graph
    assert {"classifier", "reflect", "tool_runner",
            "store_memory", "responder", "escalation"}.issubset(node_names)


def test_graph_valid_traces():
    graph = build_triage_agent()
    drawable = graph.get_graph()
    assert len(drawable.nodes) >= 6


def test_route_after_tool_always_store_memory():
    # tool_runner always hands off to store_memory in the 6-node graph
    for state in [
        {"resolved": True, "needs_escalation": False},
        {"resolved": False, "needs_escalation": True},
        {"resolved": False, "needs_escalation": False},
    ]:
        assert route_after_tool(state) == "store_memory"


@pytest.mark.parametrize(
    "state,expected",
    [
        ({"needs_escalation": False, "resolved": True}, "responder"),
        ({"needs_escalation": True, "resolved": True}, "escalation"),
        # unresolved but NOT escalation-flagged → responder (best-effort answer)
        ({"needs_escalation": False, "resolved": False}, "responder"),
    ],
)
def test_route_after_memory(state, expected):
    assert route_after_memory(state) == expected
