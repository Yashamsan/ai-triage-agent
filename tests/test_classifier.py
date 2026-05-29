import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import classify

TEST_DATA = json.loads((Path(__file__).parent / "test_data.json").read_text())

_results: dict[str, list[bool]] = defaultdict(list)


@pytest.mark.parametrize("case", TEST_DATA, ids=lambda c: c["message"][:50])
def test_classify(case):
    result = classify(case["message"])
    expected = case["expected_intent"]
    correct = result.intent == expected
    _results[expected].append(correct)
    assert correct, (
        f"Expected '{expected}', got '{result.intent}' "
        f"(confidence={result.confidence:.2f})"
    )


def pytest_sessionfinish(session, exitstatus):
    if not _results:
        return
    print("\n\n── Accuracy by Intent ─────────────────────────")
    total_correct = total = 0
    for intent in sorted(_results):
        hits = _results[intent]
        c, t = sum(hits), len(hits)
        total_correct += c
        total += t
        bar = ("█" * c) + ("░" * (t - c))
        print(f"  {intent:<20} {bar}  {c}/{t}")
    print(f"  {'TOTAL':<20} {total_correct}/{total}  ({100*total_correct//total}%)")
    print("────────────────────────────────────────────────")
