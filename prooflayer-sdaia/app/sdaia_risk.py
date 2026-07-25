"""SDAIA-style risk classification engine (illustrative scaffold).

Assesses an agent across 7 risk categories and derives an overall level.
This is a heuristic governance model, not a certified regulatory scoring
system — calibrate the thresholds/weights to your own compliance mapping
before relying on it for real decisions.
"""
from __future__ import annotations

from typing import Any

RISK_CATEGORIES = [
    "political",
    "religious",
    "financial",
    "health",
    "safety",
    "legal",
    "environmental",
]

LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
_LEVEL_RANK = {lvl: i for i, lvl in enumerate(LEVELS)}

HIGH_RISK_SECTORS = {"telecom", "banking", "healthcare", "government", "energy"}

# Categories where PII exposure or an autonomous decision loop matters most.
_PII_SENSITIVE_CATEGORIES = {"financial", "legal", "health"}


def _max_level(*levels: str) -> str:
    return max(levels, key=lambda lvl: _LEVEL_RANK[lvl])


def classify_category(category: str, factors: dict[str, Any]) -> tuple[str, str]:
    """Classify a single risk category. Returns (level, rationale)."""
    if category not in RISK_CATEGORIES:
        raise ValueError(f"unknown risk category: {category}")

    sector = factors.get("sector", "")
    handles_pii = bool(factors.get("handles_pii", False))
    autonomy = factors.get("autonomy_level", "assisted")
    affected_population = factors.get("affected_population", "individual")  # individual|group|public

    level = "LOW"
    reasons = []

    if sector in HIGH_RISK_SECTORS:
        level = _max_level(level, "MEDIUM")
        reasons.append(f"regulated sector '{sector}'")

    if handles_pii and category in _PII_SENSITIVE_CATEGORIES:
        level = _max_level(level, "HIGH")
        reasons.append("handles personal data in a PII-sensitive category")

    if autonomy == "autonomous":
        level = _max_level(level, "HIGH")
        reasons.append("acts autonomously without human-in-the-loop")

    if affected_population == "public":
        level = _max_level(level, "HIGH")
        reasons.append("decisions affect the general public")

    if handles_pii and sector in HIGH_RISK_SECTORS and autonomy == "autonomous":
        level = "CRITICAL"
        reasons.append("combination of regulated sector + PII + autonomous decisioning")

    if not reasons:
        reasons.append("no elevated risk factors identified")

    rationale = f"[{category}] " + "; ".join(reasons)
    return level, rationale


def classify_agent(factors: dict[str, Any]) -> dict[str, Any]:
    """Assess all 7 risk categories for an agent.

    factors: dict with keys like sector, handles_pii, autonomy_level,
    affected_population.

    Returns {"categories": {cat: {"level":..., "rationale":...}}, "overall": level}
    """
    categories = {}
    for cat in RISK_CATEGORIES:
        level, rationale = classify_category(cat, factors)
        categories[cat] = {"level": level, "rationale": rationale}

    overall = _max_level(*(c["level"] for c in categories.values()))
    return {"categories": categories, "overall": overall}


def assess_and_store(storage, agent_id: str, factors: dict[str, Any]) -> dict[str, Any]:
    """Run classify_agent and persist each category assessment via storage."""
    result = classify_agent(factors)
    for cat, info in result["categories"].items():
        storage.save_risk_assessment(agent_id, cat, info["level"], info["rationale"], factors)
    return result


def assess_agent_risk(
    store,
    agent_id: str,
    sector: str,
    has_pii: bool = False,
    is_autonomous: bool = False,
    has_human_oversight: bool = False,
    affected_population: str = "individual",
) -> dict[str, Any]:
    """Named-argument convenience wrapper around assess_and_store, for
    callers integrating an existing agent registry (e.g. pl_agents) that
    already track these booleans individually rather than building a
    factors dict by hand.

    has_human_oversight de-escalates autonomy risk: an agent that acts
    autonomously but under human oversight is treated as "supervised"
    rather than "autonomous" for classification purposes.
    """
    autonomy_level = "assisted"
    if is_autonomous:
        autonomy_level = "supervised" if has_human_oversight else "autonomous"

    factors = {
        "sector": sector,
        "handles_pii": has_pii,
        "autonomy_level": autonomy_level,
        "affected_population": affected_population,
        "has_human_oversight": has_human_oversight,
    }
    return assess_and_store(store, agent_id, factors)
