"""ProofLayer HTTP gateway — all cross-agent governance goes through here.

Wraps the ProofLayer REST API (running at prooflayer_url, typically
http://localhost:8000) so every router decision and every agent decision
lands in the single CISO dashboard without agents depending on each other.

All methods are fire-and-log: a ProofLayer outage never crashes the agents.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class ProofLayerGateway:
    """HTTP client for the ProofLayer v3 API.

    Usage:
        gw = ProofLayerGateway(config)
        result = gw.record_decision(agent_name="billing-agent", ...)
        gw.record_cross_agent_edge(from_id=router_did, to_id=agent_did)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        base = config.get("orchestrator", {}).get("prooflayer_url", "http://localhost:8000")
        self.base_url = base.rstrip("/") + "/api/v1"
        self.timeout  = 8

    # ── Low-level helpers ─────────────────────────────────────────────────

    def _post(self, path: str, payload: dict) -> dict:
        url  = f"{self.base_url}{path}"
        body = json.dumps(payload).encode()
        req  = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode(errors="replace")
            raise RuntimeError(f"ProofLayer {path} HTTP {exc.code}: {raw}") from exc

    def _get(self, path: str, params: dict | None = None) -> dict | list:
        qs  = ""
        if params:
            qs = "?" + "&".join(f"{k}={v}" for k, v in params.items() if v is not None)
        url = f"{self.base_url}{path}{qs}"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode(errors="replace")
            raise RuntimeError(f"ProofLayer {path} HTTP {exc.code}: {raw}") from exc

    # ── Agent registration ────────────────────────────────────────────────

    def register_agent(
        self,
        name: str,
        group: str,
        model_id: str,
        version: str = "1.0",
        intents: list[str] | None = None,
        contains_pii: bool = False,
        data_classification: str = "internal",
        policies: dict | None = None,
        description: str = "",
    ) -> dict:
        """Register an agent with its full behavioral contract.

        All behavioral metadata (intents, policies, PII flag) is stored in
        `pl_agents.metadata` so the governance dashboard can display what each
        agent can do without reading source code.
        """
        return self._post("/agents", {
            "agent_name":         name,
            "agent_version":      version,
            "model_id":           model_id,
            "agent_group":        group,
            "description":        description or f"{name} ({group} group)",
            "data_classification": data_classification,
            "metadata": {
                "intents":      intents or [],
                "contains_pii": contains_pii,
                "policies":     policies or {},
            },
        })

    # ── Decision recording ────────────────────────────────────────────────

    def record_decision(
        self,
        agent_name: str,
        agent_group: str,
        model_id: str,
        decision_value: str,
        confidence: float,
        session_id: str,
        contains_pii: bool = False,
        trace_steps: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Record a specialist agent's decision as a ProofLayer node."""
        payload: dict[str, Any] = {
            "agent_name":       agent_name,
            "agent_group":      agent_group,
            "model_id":         model_id,
            "decision":         decision_value,
            "confidence":       confidence,
            "session_id":       session_id,
            "contains_pii":     contains_pii,
        }
        if trace_steps:
            payload["trace_steps"] = trace_steps
        if metadata:
            payload["metadata"] = metadata

        result = self._post("/decisions", payload)

        # Record trace steps if decision was created successfully
        decision_id = result.get("decision_id")
        if decision_id and trace_steps:
            for step in trace_steps:
                try:
                    self._post("/trace-steps", {
                        "decision_node_id": decision_id,
                        "node_type":        step.get("node_type", "unknown"),
                        "thought":          step.get("thought", ""),
                        "action":           step.get("action", ""),
                        "observation":      step.get("observation", ""),
                        "confidence":       step.get("confidence"),
                        "latency_ms":       step.get("latency_ms"),
                    })
                except Exception:
                    pass  # trace steps are best-effort

        return result

    def record_route(
        self,
        intent: str,
        confidence: float,
        target_agent: str,
        session_id: str,
        contains_pii: bool = False,
        tier: int = 1,
        agent_decision_id: str | None = None,
        reflection_notes: str | None = None,
    ) -> dict:
        """Record the router's routing decision as a ProofLayer node."""
        decision_value = f"route:{intent}→{target_agent}(tier{tier})"
        metadata: dict[str, Any] = {
            "tier":             tier,
            "target_agent":     target_agent,
            "router":           "orchestrator-router",
        }
        if reflection_notes:
            metadata["reflection_notes"] = reflection_notes

        result = self._post("/decisions", {
            "agent_name":   "orchestrator-router",
            "agent_group":  "orchestrator",
            "model_id":     "router-v1",
            "decision":     decision_value,
            "confidence":   confidence,
            "session_id":   session_id,
            "contains_pii": contains_pii,
            "metadata":     metadata,
        })

        # Link router decision → agent decision as a cross-agent edge
        route_id = result.get("decision_id")
        if route_id and agent_decision_id:
            try:
                self.record_cross_agent_edge(
                    from_decision_id=route_id,
                    to_decision_id=agent_decision_id,
                    relationship="ROUTER_DISPATCHED_TO",
                )
            except Exception:
                pass

        return result

    # ── Exception / override recording ────────────────────────────────────

    def record_exception(
        self,
        decision_id: str,
        reason: str,
        severity: str = "medium",
        raised_by: str = "orchestrator",
    ) -> dict:
        """Record a human override or policy exception on a decision."""
        return self._post("/exceptions", {
            "decision_id": decision_id,
            "reason":      reason,
            "severity":    severity,
            "raised_by":   raised_by,
        })

    # ── Cross-agent edge ──────────────────────────────────────────────────

    def record_cross_agent_edge(
        self,
        from_decision_id: str,
        to_decision_id: str,
        relationship: str = "CROSS_AGENT_REFERENCE",
        metadata: dict | None = None,
    ) -> dict:
        """Create a directed edge between two decisions in the governance graph."""
        return self._post("/cross-agent-edge", {
            "from_decision_id": from_decision_id,
            "to_decision_id":   to_decision_id,
            "relationship":     relationship,
            "metadata":         metadata or {},
        })

    # ── Query helpers ─────────────────────────────────────────────────────

    def query_overview(self) -> dict:
        """Return the governance overview (KPI cards)."""
        return self._get("/overview")  # type: ignore[return-value]

    def query_decisions(
        self,
        agent_name: str | None = None,
        agent_group: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Return recent decisions, optionally filtered by agent."""
        params: dict[str, Any] = {"limit": limit}
        if agent_name:
            params["agent_name"] = agent_name
        if agent_group:
            params["agent_group"] = agent_group
        return self._get("/decisions", params)  # type: ignore[return-value]

    def query_exceptions(self, hours: int = 24) -> list[dict]:
        return self._get("/exceptions", {"hours": hours})  # type: ignore[return-value]

    def query_cross_agent_queries(self) -> list[dict]:
        return self._get("/cross-agent-queries")  # type: ignore[return-value]

    # ── Governance dashboard shortcut ─────────────────────────────────────

    def governance_report(self) -> dict[str, Any]:
        """Aggregate ProofLayer data into a single governance snapshot."""
        try:
            overview   = self.query_overview()
        except Exception as exc:
            overview   = {"error": str(exc)}
        try:
            exceptions = self.query_exceptions(hours=24)
        except Exception:
            exceptions = []
        try:
            cross_edges = self.query_cross_agent_queries()
        except Exception:
            cross_edges = []

        return {
            "overview":        overview,
            "open_exceptions": len(exceptions),
            "cross_agent_edges": len(cross_edges),
        }

    def __repr__(self) -> str:
        return f"<ProofLayerGateway base={self.base_url}>"
