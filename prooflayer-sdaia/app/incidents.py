"""Incident reporting for the SDAIA compliance module (illustrative scaffold).

CRITICAL-severity incidents are auto-flagged as reported to the regulator
(a mocked report reference is generated — no real network call is made).
"""
from __future__ import annotations

from app.sdaia_risk import LEVELS as SEVERITY_LEVELS  # LOW|MEDIUM|HIGH|CRITICAL


def log_incident(storage, agent_id: str, severity: str, category: str, description: str) -> dict:
    if severity not in SEVERITY_LEVELS:
        raise ValueError(f"unknown severity: {severity}")

    incident = storage.log_incident(agent_id, severity, category, description)

    if severity == "CRITICAL":
        incident = storage.mark_incident_reported(incident["id"])

    return incident


def resolve_incident(storage, incident_id: int, root_cause: str, corrective_action: str) -> dict:
    return storage.resolve_incident(incident_id, root_cause, corrective_action)


def list_open_incidents(storage, agent_id: str | None = None) -> list[dict]:
    incidents = storage.list_incidents(agent_id=agent_id)
    return [i for i in incidents if not i.get("resolved_at")]


def list_critical_unreported(storage, agent_id: str | None = None) -> list[dict]:
    incidents = storage.list_incidents(agent_id=agent_id, severity="CRITICAL")
    return [i for i in incidents if not i.get("reported_to_regulator")]
