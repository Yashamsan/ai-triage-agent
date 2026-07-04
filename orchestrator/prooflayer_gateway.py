"""ProofLayer HTTP gateway — all cross-agent governance goes through here.

Wraps the ProofLayer REST API (running at prooflayer_url, typically
http://localhost:8000) so every router decision and every agent decision
lands in the single CISO dashboard without agents depending on each other.

All public methods are fire-and-log: a ProofLayer outage never crashes the
agents.

local_mode=True
    No HTTP calls are made. All decisions, agents, edges, and exceptions are
    stored in-process (lists on the gateway instance). Useful for demos,
    integration tests, and development without Docker.

    Enable via config:
        orchestrator:
          local_mode: true

    Or at construction:
        gw = ProofLayerGateway(cfg, local_mode=True)

    Inspect the in-memory store:
        gw.local_store()  → {"agents": [...], "decisions": [...], ...}
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from datetime import UTC, datetime
from typing import Any


class ProofLayerGateway:
    """HTTP client for the ProofLayer v3 API.

    Usage:
        gw = ProofLayerGateway(config)
        result = gw.record_decision(agent_name="billing-agent", ...)
        gw.record_cross_agent_edge(from_id=router_did, to_id=agent_did)

    Local mode (no Docker required):
        gw = ProofLayerGateway(config, local_mode=True)
    """

    def __init__(
        self,
        config: dict[str, Any],
        local_mode: bool | None = None,
    ) -> None:
        orch = config.get("orchestrator", {})
        base = orch.get("prooflayer_url", "http://localhost:8000")
        self.base_url = base.rstrip("/") + "/api/v1"
        self.timeout  = 8

        # local_mode: constructor kwarg wins, then config, then False
        if local_mode is None:
            local_mode = bool(orch.get("local_mode", False))
        self.local_mode = local_mode

        # In-memory store (only populated when local_mode=True)
        self._local_agents:    dict[str, dict] = {}
        self._local_decisions: list[dict]      = []
        self._local_exceptions: list[dict]     = []
        self._local_edges:     list[dict]      = []

    # ── Low-level HTTP helpers ────────────────────────────────────────────

    def _post(self, path: str, payload: dict) -> dict:
        if self.local_mode:
            return self._local_post(path, payload)
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
        if self.local_mode:
            return self._local_get(path, params)
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

    # ── Local-mode in-memory store ────────────────────────────────────────

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()

    def _uid(self) -> str:
        return str(uuid.uuid4())

    def _local_post(self, path: str, payload: dict) -> dict:
        if path == "/agents":
            record = {
                **payload,
                "agent_id":       self._uid(),
                "agent_version":  payload.get("agent_version", "1.0"),
                "registered_at":  self._now(),
                "last_seen":      self._now(),
            }
            self._local_agents[payload.get("agent_name", self._uid())] = record
            return record

        if path == "/decisions":
            decision_id = self._uid()
            record = {
                **payload,
                "decision_id":  decision_id,
                "snapshot_id":  self._uid(),
                "policy_edges": 0,
                "trace_steps":  len(payload.get("trace_steps", [])),
                "timestamp":    self._now(),
            }
            self._local_decisions.append(record)
            return record

        if path == "/trace-steps":
            return {"step_id": self._uid(), "created": True}

        if path == "/exceptions":
            record = {
                **payload,
                "exception_id": self._uid(),
                "exc_node_id":  self._uid(),
                "created_at":   self._now(),
            }
            self._local_exceptions.append(record)
            return record

        if path == "/cross-agent-edge":
            record = {**payload, "edge_id": self._uid(), "created_at": self._now()}
            self._local_edges.append(record)
            return record

        return {"status": "ok", "path": path}

    def _local_get(self, path: str, params: dict | None = None) -> dict | list:
        p = params or {}

        if path == "/overview":
            confs = [float(d["confidence"]) for d in self._local_decisions
                     if "confidence" in d]
            pii_count = sum(1 for d in self._local_decisions if d.get("contains_pii"))
            groups    = {d.get("agent_group") for d in self._local_decisions
                         if d.get("agent_group")}
            trace_count = sum(
                int(d.get("trace_steps", 0)) for d in self._local_decisions
            )
            return {
                "total_decisions":    len(self._local_decisions),
                "human_overrides":    0,
                "escalations":        0,
                "avg_confidence":     round(sum(confs) / len(confs), 3) if confs else 0.0,
                "exception_count":    len(self._local_exceptions),
                "trace_step_count":   trace_count,
                "pii_decision_count": pii_count,
                "agent_group_count":  len(groups),
            }

        if path in ("/decisions", "/search"):
            results = list(self._local_decisions)
            if p.get("agent_name"):
                results = [d for d in results if d.get("agent_name") == p["agent_name"]]
            if p.get("agent_group"):
                results = [d for d in results if d.get("agent_group") == p["agent_group"]]
            limit = int(p.get("limit", 50))
            return results[-limit:]

        if path == "/agents":
            return list(self._local_agents.values())

        if path == "/exceptions":
            return list(self._local_exceptions)

        if path == "/cross-agent-queries":
            return list(self._local_edges)

        return []

    def local_store(self) -> dict[str, Any]:
        """Return a snapshot of the in-memory store (local_mode only)."""
        return {
            "agents":     list(self._local_agents.values()),
            "decisions":  list(self._local_decisions),
            "exceptions": list(self._local_exceptions),
            "edges":      list(self._local_edges),
        }

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
