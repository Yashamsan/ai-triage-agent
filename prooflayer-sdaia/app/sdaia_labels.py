"""Ethics label system for the SDAIA compliance module (illustrative scaffold).

Five governance-maturity tiers, scored from an agent's safety report and
incident-handling history. Inherent risk level (from sdaia_risk.py) is
tracked separately from this label: a CRITICAL-risk agent can still earn
a high label if it is well-governed (incidents resolved and reported,
human oversight in place, safety report complete).

Tier names/thresholds are an illustrative scoring scheme, not a verified
transcription of a specific SDAIA document section.
"""
from __future__ import annotations

TIERS = [
    (1, "غير ملتزم", "Non-Compliant"),
    (2, "تحت المراقبة", "Under Monitoring"),
    (3, "متوافق جزئياً", "Partially Compliant"),
    (4, "ملتزم", "Committed"),
    (5, "متميز", "Exemplary"),
]


def _tier_for_score(score: int) -> tuple[int, str, str]:
    if score >= 90:
        return TIERS[4]
    if score >= 75:
        return TIERS[3]
    if score >= 50:
        return TIERS[2]
    if score >= 25:
        return TIERS[1]
    return TIERS[0]


def compute_score(storage, agent_id: str) -> int:
    """Score governance maturity, not inherent risk.

    A clean record (no incidents at all) can reach the top "Exemplary"
    tier. An agent that *had* incidents — even a critical one — but
    resolved and reported them properly lands in "Committed": good
    governance, but a track record that isn't spotless.
    """
    safety_report = storage.get_latest_safety_report(agent_id)
    incidents = storage.list_incidents(agent_id=agent_id)
    decisions = storage.list_decisions(agent_id)

    score = 0

    if safety_report:
        score += 30

    if incidents:
        resolved = [i for i in incidents if i.get("resolved_at")]
        with_root_cause = [i for i in incidents if i.get("root_cause") and i.get("corrective_action")]
        score += round(15 * (len(resolved) / len(incidents)))
        score += round(10 * (len(with_root_cause) / len(incidents)))
    else:
        # Clean record bonus — nothing to remediate, at least as good as
        # remediating everything, so it earns the same base points...
        score += 25

    critical_incidents = [i for i in incidents if i["severity"] == "CRITICAL"]
    if critical_incidents:
        reported = [i for i in critical_incidents if i.get("reported_to_regulator")]
        score += round(15 * (len(reported) / len(critical_incidents)))
        # ...but ever having had a critical incident caps how "exemplary"
        # the record can look, even once it's fully remediated.
        score -= 5
    else:
        score += 15
        score += 10  # clean-record bonus, only reachable with zero critical incidents ever

    unresolved_critical = [
        i for i in incidents if i["severity"] == "CRITICAL" and not i.get("resolved_at")
    ]
    if unresolved_critical:
        score -= 30

    if decisions and any(d.get("requires_human_review") for d in decisions):
        score += 10

    return max(0, min(100, score))


def compute_ethics_label(storage, agent_id: str) -> dict:
    score = compute_score(storage, agent_id)
    tier, name_ar, name_en = _tier_for_score(score)
    storage.save_ethics_label(agent_id, tier, name_ar, name_en, score)
    return {
        "tier": tier,
        "tier_name_ar": name_ar,
        "tier_name_en": name_en,
        "score": score,
    }
