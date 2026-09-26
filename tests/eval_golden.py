"""Golden dataset evaluator — LM-as-Judge for Metrics-Driven Development.

Loads tests/test_data_golden.json, runs each scenario through the
appropriate classifier (EN or AR), scores intent + parameter extraction
+ tool_query formulation against expected values.

Exit code: 0 if pass_rate >= threshold, 1 otherwise.
Designed to run in CI/CD as a deployment gate.
"""
import json
import sys
from pathlib import Path

# This runs as a standalone script (python tests/eval_golden.py), not via
# pytest -- pytest's rootdir insertion doesn't apply here, so without this
# `import app...`/`import app_ar...` inside classify_wrapper() fail with
# ModuleNotFoundError. Matches the sys.path bootstrap every other test file
# in tests/ already does (e.g. tests/test_classifier.py).
sys.path.insert(0, str(Path(__file__).parent.parent))

# Thresholds — adjust as your baseline stabilizes
PASS_THRESHOLD = 0.80      # 80% overall pass rate required
INTENT_WEIGHT = 0.5         # intent accuracy is 50% of score
PARAM_WEIGHT = 0.3          # parameter extraction is 30%
TOOL_WEIGHT = 0.2           # tool_query formulation is 20%

DATA_PATH = Path(__file__).parent / "test_data_golden.json"


def load_scenarios() -> list[dict]:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return data["scenarios"]


def classify_wrapper(scenario: dict) -> dict:
    """Route to the correct classifier based on language."""
    lang = scenario["language"]
    msg = scenario["message"]

    if lang == "ar":
        from app_ar.classifier import classify_ar
        result = classify_ar(msg)
    else:
        from app.classifier import classify
        result = classify(msg)

    return {
        "intent": result.intent,
        "confidence": result.confidence,
        "parameters": result.parameters or {},
        "tool_query": result.tool_query,
    }


def score_intent(actual: str, expected: str) -> tuple[bool, str]:
    passed = actual == expected
    detail = f"intent: expected={expected}, got={actual} {'✅' if passed else '❌'}"
    return passed, detail


def score_parameters(actual: dict, expected: dict) -> tuple[bool, str]:
    """Score parameter extraction. Missing both sides = pass."""
    if not expected and not actual:
        return True, "params: none expected, none extracted ✅"
    if not expected and actual:
        return False, f"params: none expected, but got {actual} ❌"
    if expected and not actual:
        return False, f"params: expected {expected}, got none ❌"

    matches = 0
    total = len(expected)
    mismatches = []
    for key, exp_val in expected.items():
        act_val = actual.get(key)
        if act_val == exp_val:
            matches += 1
        else:
            mismatches.append(f"{key}={exp_val} vs {act_val}")

    passed = matches == total
    detail = f"params: {matches}/{total} matched"
    if mismatches:
        detail += f" | mismatches: {mismatches}"
    detail += " ✅" if passed else " ❌"
    return passed, detail


def score_tool_query(actual: str | None, expected_tool: str | None) -> tuple[bool, str]:
    """Score tool_query by comparing the formulated tool name against
    expected_tool -- the golden dataset's field is a bare tool-name string
    (e.g. "get_invoice"), not a JSON blob, so this compares directly rather
    than parsing a second JSON payload that doesn't exist in the schema."""
    if expected_tool is None and actual is None:
        return True, "tool_query: none expected, none formulated ✅"

    if expected_tool is not None and actual is None:
        return False, f"tool_query: expected tool={expected_tool}, got none ❌"

    if expected_tool is None and actual is not None:
        return False, f"tool_query: none expected, got {actual} ❌"

    try:
        actual_dict = json.loads(actual)
        actual_tool = actual_dict.get("tool")
        tool_ok = actual_tool == expected_tool
        detail = f"tool: {actual_tool} vs {expected_tool}"
        detail += " ✅" if tool_ok else " ❌"
        return tool_ok, detail
    except (json.JSONDecodeError, TypeError, AttributeError):
        return False, f"tool_query: parse error on actual={actual}"


def evaluate() -> dict:
    scenarios = load_scenarios()
    results = []
    intents_ok = params_ok = tools_ok = 0

    for s in scenarios:
        actual = classify_wrapper(s)
        exp_params = s.get("expected_parameters", {})

        i_ok, i_detail = score_intent(actual["intent"], s["expected_intent"])
        p_ok, p_detail = score_parameters(actual["parameters"], exp_params)
        t_ok, t_detail = score_tool_query(actual["tool_query"], s.get("expected_tool"))

        if i_ok:
            intents_ok += 1
        if p_ok:
            params_ok += 1
        if t_ok:
            tools_ok += 1

        results.append({
            "id": s["id"],
            "language": s["language"],
            "intent_pass": i_ok,
            "params_pass": p_ok,
            "tool_pass": t_ok,
            "details": f"{i_detail} | {p_detail} | {t_detail}",
        })

    n = len(scenarios)
    pass_rate = (
        INTENT_WEIGHT * (intents_ok / n)
        + PARAM_WEIGHT * (params_ok / n)
        + TOOL_WEIGHT * (tools_ok / n)
    )

    return {
        "total": n,
        "intent_accuracy": f"{intents_ok}/{n}",
        "param_accuracy": f"{params_ok}/{n}",
        "tool_accuracy": f"{tools_ok}/{n}",
        "weighted_pass_rate": round(pass_rate, 3),
        "threshold": PASS_THRESHOLD,
        "passed": pass_rate >= PASS_THRESHOLD,
        "results": results,
    }


if __name__ == "__main__":
    report = evaluate()

    print("\n=== GOLDEN DATASET EVALUATION ===\n")
    print(f"  Scenarios:        {report['total']}")
    print(f"  Intent accuracy:  {report['intent_accuracy']}")
    print(f"  Param extraction: {report['param_accuracy']}")
    print(f"  Tool query:       {report['tool_accuracy']}")
    print(f"  Weighted score:   {report['weighted_pass_rate']:.1%}")
    print(f"  Threshold:        {report['threshold']:.0%}")
    print(f"  RESULT:           {'✅ PASS' if report['passed'] else '❌ FAIL'}")

    print("\n--- Details ---")
    for r in report["results"]:
        status = "✅" if r["intent_pass"] and r["params_pass"] and r["tool_pass"] else "❌"
        print(f"  {r['id']} [{r['language']}] {status}")
        print(f"    {r['details']}")

    sys.exit(0 if report["passed"] else 1)
