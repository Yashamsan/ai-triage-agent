"""SDAIA-P145 five-stage risk cycle -- pure business logic (no DB).

Ported from the standalone prototype at
prooflayer-sdaia/P145/prooflayer-rmf/app/sdaia_rmf.py (9/9 tests passing
before this port) -- kept as its own DB-free module, mirroring how
prooflayer-sdaia/app/sdaia_risk.py separates pure classification logic from
its Postgres-backed API layer, so the risk math stays independently testable.

Five stages: context & scope -> risk identification -> risk assessment
(likelihood x impact, 4x4) -> risk treatment (avoid/mitigate/transfer/accept)
-> monitoring & review.

Risk level = likelihood * impact on a 4x4 matrix.
Bands: 1-2 low, 3-6 medium, 8-12 high, 16 catastrophic.
7, 13, 14, 15 are impossible products of 1-4 x 1-4 and are not valid levels.
"""
from __future__ import annotations

from dataclasses import dataclass

# Framework's seven-category taxonomy (SDAIA-P145, risk identification stage)
TAXONOMY = [
    "bias_discrimination_abuse",
    "privacy_security",
    "misinformation",
    "malicious_use",
    "human_machine_interaction",
    "social_economic_environmental",
    "safety_limitations",
]

TREATMENT_STRATEGIES = ["avoid", "mitigate", "transfer", "accept"]

BANDS = [
    ("low", 1, 2),
    ("medium", 3, 6),
    ("high", 8, 12),
    ("catastrophic", 16, 16),
]


def band_for(level: int) -> str:
    for name, lo, hi in BANDS:
        if lo <= level <= hi:
            return name
    raise ValueError(f"risk level {level} is not a valid product of 1-4 x 1-4")


@dataclass
class AssessmentResult:
    likelihood: int
    impact: int
    risk_level: int
    band: str


def assess(likelihood: int, impact: int) -> AssessmentResult:
    """Compute the 4x4 matrix result. Both axes are 1-4."""
    if likelihood not in (1, 2, 3, 4):
        raise ValueError("likelihood must be 1-4 (rare, unlikely, likely, almost certain)")
    if impact not in (1, 2, 3, 4):
        raise ValueError("impact must be 1-4 (low, medium, high, catastrophic)")
    level = likelihood * impact
    return AssessmentResult(
        likelihood=likelihood,
        impact=impact,
        risk_level=level,
        band=band_for(level),
    )


def validate_category(category: str) -> None:
    if category not in TAXONOMY:
        raise ValueError(f"category must be one of {TAXONOMY}")


def validate_treatment(strategy: str, rationale: str, approver: str) -> None:
    if strategy not in TREATMENT_STRATEGIES:
        raise ValueError(f"strategy must be one of {TREATMENT_STRATEGIES}")
    if not rationale.strip():
        raise ValueError("rationale is required: the framework demands documented decisions")
    if not approver.strip():
        raise ValueError(
            "approver is required: SDAIA-P145 requires identified responsibilities "
            "and approval mechanisms for acceptance decisions"
        )
