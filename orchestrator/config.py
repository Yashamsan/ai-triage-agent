"""Config loader — YAML file merged over built-in defaults.

Usage:
    from orchestrator.config import load_config
    cfg = load_config()                        # reads config/contact_center.yaml
    cfg = load_config("config/banking.yaml")   # swap domain without changing code
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

try:
    import yaml as _yaml
    _YAML = True
except ImportError:
    _YAML = False

# ── Built-in defaults (YAML overrides any key) ─────────────────────────────

_DEFAULTS: dict[str, Any] = {
    "orchestrator": {
        "name": "contact-center-orchestrator",
        "version": "1.0",
        "prooflayer_url": "http://localhost:8000",
        "classify_model": "deepseek/deepseek-chat",
        "tier1_threshold": 0.90,
        "tier2_threshold": 0.70,
        "escalation_threshold": 0.40,
        "default_language": "en",
        "max_hops": 3,
    },
    "agents": [
        {
            "name": "triage-agent-en", "group": "contact-center",
            "language": "en", "model": "deepseek/deepseek-chat",
            "version": "2.1", "data_classification": "internal",
            "intents": ["greeting", "password_reset", "billing_inquiry",
                        "technical_issue", "general_inquiry", "service_status"],
        },
        {
            "name": "triage-agent-ar", "group": "contact-center",
            "language": "ar", "model": "openrouter/qwen/qwen3-235b-a22b",
            "version": "1.2", "data_classification": "internal",
            "intents": ["ar_greeting", "ar_password_reset", "ar_billing_inquiry",
                        "ar_technical_issue", "ar_general_inquiry", "ar_service_status"],
        },
        {
            "name": "billing-agent", "group": "finance",
            "language": "en", "model": "deepseek/deepseek-chat",
            "version": "1.0", "data_classification": "confidential",
            "contains_pii": True,
            "intents": ["payment_failed", "refund_request", "invoice_dispute",
                        "subscription_change", "pricing_inquiry", "billing_error"],
        },
        {
            "name": "fraud-detection-agent", "group": "risk",
            "language": "en", "model": "deepseek/deepseek-chat",
            "version": "1.0", "data_classification": "restricted",
            "contains_pii": True,
            "intents": ["suspicious_transaction", "fraud_report", "account_compromise",
                        "unusual_activity", "block_request", "verification_request"],
        },
        {
            "name": "technical-agent", "group": "support",
            "language": "en", "model": "deepseek/deepseek-chat",
            "version": "1.0", "data_classification": "internal",
            "intents": ["service_outage", "connectivity_issue", "device_config",
                        "api_error", "performance_issue", "feature_request"],
        },
        {
            "name": "retention-agent", "group": "retention",
            "language": "en", "model": "deepseek/deepseek-chat",
            "version": "1.0", "data_classification": "confidential",
            "contains_pii": True,
            "intents": ["cancellation_request", "downgrade_request", "competitor_mention",
                        "dissatisfaction", "win_back", "loyalty_inquiry"],
        },
        {
            "name": "complaints-agent", "group": "escalation",
            "language": "en", "model": "deepseek/deepseek-chat",
            "version": "1.0", "data_classification": "internal",
            "intents": ["escalation", "regulatory_complaint", "service_grievance",
                        "agent_complaint", "billing_complaint", "safety_concern"],
        },
        {
            "name": "sales-agent", "group": "growth",
            "language": "en", "model": "deepseek/deepseek-chat",
            "version": "1.0", "data_classification": "internal",
            "intents": ["upgrade_request", "promo_inquiry", "cross_sell",
                        "plan_comparison", "lead_capture", "upsell_opportunity"],
        },
        {
            "name": "loyalty-agent", "group": "growth",
            "language": "en", "model": "deepseek/deepseek-chat",
            "version": "1.0", "data_classification": "internal",
            "intents": ["points_inquiry", "redemption_request", "tier_status",
                        "benefit_inquiry", "points_transfer", "loyalty_complaint"],
        },
    ],
    "routing": {
        "rules": [
            # contact-center
            {"intent": "greeting",           "agent": "triage-agent-en",       "min_confidence": 0.95},
            {"intent": "password_reset",     "agent": "triage-agent-en",       "min_confidence": 0.92},
            {"intent": "billing_inquiry",    "agent": "billing-agent",          "min_confidence": 0.85},
            {"intent": "technical_issue",    "agent": "technical-agent",        "min_confidence": 0.85},
            {"intent": "general_inquiry",    "agent": "triage-agent-en",       "min_confidence": 0.75},
            {"intent": "service_status",     "agent": "technical-agent",        "min_confidence": 0.88},
            # finance
            {"intent": "payment_failed",     "agent": "billing-agent",          "min_confidence": 0.92},
            {"intent": "refund_request",     "agent": "billing-agent",          "min_confidence": 0.90},
            {"intent": "invoice_dispute",    "agent": "billing-agent",          "min_confidence": 0.88},
            {"intent": "subscription_change","agent": "billing-agent",          "min_confidence": 0.85},
            {"intent": "pricing_inquiry",    "agent": "billing-agent",          "min_confidence": 0.82},
            {"intent": "billing_error",      "agent": "billing-agent",          "min_confidence": 0.90},
            # risk
            {"intent": "suspicious_transaction","agent": "fraud-detection-agent","min_confidence": 0.94},
            {"intent": "fraud_report",       "agent": "fraud-detection-agent",  "min_confidence": 0.96},
            {"intent": "account_compromise", "agent": "fraud-detection-agent",  "min_confidence": 0.95},
            {"intent": "unusual_activity",   "agent": "fraud-detection-agent",  "min_confidence": 0.88},
            {"intent": "block_request",      "agent": "fraud-detection-agent",  "min_confidence": 0.90},
            {"intent": "verification_request","agent": "fraud-detection-agent", "min_confidence": 0.85},
            # support
            {"intent": "service_outage",     "agent": "technical-agent",        "min_confidence": 0.93},
            {"intent": "connectivity_issue", "agent": "technical-agent",        "min_confidence": 0.90},
            {"intent": "device_config",      "agent": "technical-agent",        "min_confidence": 0.85},
            {"intent": "api_error",          "agent": "technical-agent",        "min_confidence": 0.88},
            {"intent": "performance_issue",  "agent": "technical-agent",        "min_confidence": 0.87},
            {"intent": "feature_request",    "agent": "technical-agent",        "min_confidence": 0.80},
            # retention
            {"intent": "cancellation_request","agent": "retention-agent",       "min_confidence": 0.94},
            {"intent": "downgrade_request",  "agent": "retention-agent",        "min_confidence": 0.90},
            {"intent": "competitor_mention", "agent": "retention-agent",        "min_confidence": 0.88},
            {"intent": "dissatisfaction",    "agent": "retention-agent",        "min_confidence": 0.85},
            {"intent": "win_back",           "agent": "retention-agent",        "min_confidence": 0.92},
            {"intent": "loyalty_inquiry",    "agent": "retention-agent",        "min_confidence": 0.87},
            # escalation
            {"intent": "escalation",             "agent": "complaints-agent",   "min_confidence": 0.88},
            {"intent": "regulatory_complaint",   "agent": "complaints-agent",   "min_confidence": 0.92},
            {"intent": "service_grievance",      "agent": "complaints-agent",   "min_confidence": 0.85},
            {"intent": "agent_complaint",        "agent": "complaints-agent",   "min_confidence": 0.90},
            {"intent": "billing_complaint",      "agent": "complaints-agent",   "min_confidence": 0.87},
            {"intent": "safety_concern",         "agent": "complaints-agent",   "min_confidence": 0.96},
            # growth — sales
            {"intent": "upgrade_request",        "agent": "sales-agent",        "min_confidence": 0.88},
            {"intent": "promo_inquiry",          "agent": "sales-agent",        "min_confidence": 0.90},
            {"intent": "cross_sell",             "agent": "sales-agent",        "min_confidence": 0.83},
            {"intent": "plan_comparison",        "agent": "sales-agent",        "min_confidence": 0.85},
            {"intent": "lead_capture",           "agent": "sales-agent",        "min_confidence": 0.88},
            {"intent": "upsell_opportunity",     "agent": "sales-agent",        "min_confidence": 0.82},
            # growth — loyalty
            {"intent": "points_inquiry",         "agent": "loyalty-agent",      "min_confidence": 0.90},
            {"intent": "redemption_request",     "agent": "loyalty-agent",      "min_confidence": 0.92},
            {"intent": "tier_status",            "agent": "loyalty-agent",      "min_confidence": 0.90},
            {"intent": "benefit_inquiry",        "agent": "loyalty-agent",      "min_confidence": 0.87},
            {"intent": "points_transfer",        "agent": "loyalty-agent",      "min_confidence": 0.90},
            {"intent": "loyalty_complaint",      "agent": "loyalty-agent",      "min_confidence": 0.88},
            # Arabic contact-center
            {"intent": "ar_greeting",        "agent": "triage-agent-ar",        "min_confidence": 0.95},
            {"intent": "ar_password_reset",  "agent": "triage-agent-ar",        "min_confidence": 0.92},
            {"intent": "ar_billing_inquiry", "agent": "billing-agent",          "min_confidence": 0.85},
            {"intent": "ar_technical_issue", "agent": "technical-agent",        "min_confidence": 0.85},
            {"intent": "ar_general_inquiry", "agent": "triage-agent-ar",        "min_confidence": 0.75},
            {"intent": "ar_service_status",  "agent": "technical-agent",        "min_confidence": 0.88},
        ],
    },
    "intent_descriptions": {
        "greeting":              "hello, hi, good morning, how are you, conversational opener",
        "password_reset":        "can't log in, forgot password, account locked, reset credentials",
        "billing_inquiry":       "question about my bill, invoice amount, what am I being charged",
        "technical_issue":       "something not working, bug, error, feature broken",
        "general_inquiry":       "general question not fitting other categories",
        "service_status":        "is the service down, outage, status page",
        "payment_failed":        "payment declined, card not charged, transaction failed",
        "refund_request":        "I want my money back, please refund, charged incorrectly",
        "invoice_dispute":       "this charge is wrong, I don't recognise this amount",
        "subscription_change":   "upgrade, downgrade, change plan",
        "pricing_inquiry":       "how much does it cost, pricing, plans",
        "billing_error":         "double charged, overcharged, wrong amount on bill",
        "suspicious_transaction":"suspicious activity, I didn't make this payment",
        "fraud_report":          "I've been defrauded, unauthorised transaction",
        "account_compromise":    "my account was hacked, someone else using my account",
        "unusual_activity":      "strange activity, unusual login, odd transaction",
        "block_request":         "please block my card, freeze my account",
        "verification_request":  "need to verify identity, confirm transaction",
        "service_outage":        "service is down, can't connect, internet outage",
        "connectivity_issue":    "internet not working, connection dropped",
        "device_config":         "how do I configure, setup help, device settings",
        "api_error":             "API not working, 500 error, integration failing",
        "performance_issue":     "slow, laggy, poor performance",
        "feature_request":       "please add this feature, suggestion, improvement",
        "cancellation_request":  "I want to cancel, close my account, stop service",
        "downgrade_request":     "switch to basic plan, downgrade my subscription",
        "competitor_mention":    "switching to competitor, better offer elsewhere",
        "dissatisfaction":       "unhappy with service, disappointed, frustrated",
        "win_back":              "want to come back, reconsidering cancellation",
        "loyalty_inquiry":       "loyalty program, rewards, how long am I a customer",
        "ar_greeting":           "Arabic greeting: مرحبا، أهلاً، السلام عليكم",
        "ar_password_reset":     "Arabic: مشكلة تسجيل الدخول، نسيت كلمة المرور",
        "ar_billing_inquiry":    "Arabic: استفسار عن الفاتورة",
        "ar_technical_issue":    "Arabic: مشكلة تقنية",
        "ar_general_inquiry":    "Arabic: استفسار عام",
        "ar_service_status":     "Arabic: هل الخدمة متوقفة؟",
    },
    "policies": {
        "pii_agents": ["billing-agent", "fraud-detection-agent", "retention-agent"],
        "require_human_above_sar": 10000,
        "max_routing_hops": 3,
    },
}


# ── Public API ─────────────────────────────────────────────────────────────

def load_config(path: str | None = None) -> dict[str, Any]:
    """Load config from YAML, falling back to built-in defaults for any missing key."""
    if path is None:
        path = os.getenv(
            "ORCHESTRATOR_CONFIG",
            str(Path(__file__).parent.parent / "config" / "contact_center.yaml"),
        )

    cfg = copy.deepcopy(_DEFAULTS)

    if _YAML and Path(path).exists():
        with open(path, encoding="utf-8") as fh:
            override = _yaml.safe_load(fh) or {}
        cfg = _deep_merge(cfg, override)

    return cfg


def get_routing_table(config: dict[str, Any]) -> dict[str, str]:
    """Return {intent: agent_name} from routing rules."""
    return {r["intent"]: r["agent"] for r in config["routing"]["rules"]}


def get_agent_config(config: dict[str, Any], name: str) -> dict[str, Any] | None:
    for a in config["agents"]:
        if a["name"] == name:
            return a
    return None


def get_all_intents(config: dict[str, Any]) -> list[str]:
    return list(get_routing_table(config).keys())


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
