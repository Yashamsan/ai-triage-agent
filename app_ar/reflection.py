"""Reflection: LLM-as-Judge self-review of Arabic triage decisions.

Handles Arabic user queries while judging against English intent labels.
Mirrors app/reflection.py; uses the same model routing.
"""

import json
import os
import re
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from litellm import completion

VALID_INTENTS = {"greeting", "password_reset", "billing", "technical_support", "escalation", "unknown"}

# English prompt used deliberately — DeepSeek produces reliable JSON with English
# system prompts even when the user query is in Arabic. The intent labels are
# English regardless of the query language, so English evaluation is correct.
REFLECTION_SYSTEM_PROMPT = """You are a senior support triage reviewer. Your job is
to review classification decisions for SAFETY and ACCURACY.

You receive:
- The customer's query (may be in Arabic — evaluate it correctly)
- The agent's classification (intent) — always an English label
- Confidence score

You evaluate:
1. **Accuracy** — Is the intent correct for this query?
2. **Escalation need** — Does this query need a human agent even if classified correctly?
3. **Policy compliance** — Does the handling align with standard support protocols?

Valid intents: password_reset | billing | technical_support | escalation | unknown

Respond ONLY in JSON format:
{
    "needs_revision": true/false,
    "issues": ["issue 1"],
    "suggested_intent": "password_reset" | "billing" | "technical_support" | "escalation" | null,
    "suggested_routing": "responder" | "escalation",
    "confidence_adjustment": -0.1,
    "critique": "Brief 1-2 sentence explanation in Arabic (the customer's language)"
}

Set needs_revision=false if the classification is accurate and appropriate.
"""


def _resolve_model() -> tuple[str, dict]:
    api_base = os.getenv("LITELLM_PROXY_URL")
    api_key = os.getenv("LITELLM_MASTER_KEY")
    default = "cheap-classifier" if api_base else "deepseek/deepseek-chat"
    model = os.getenv("LLM_MODEL", default)
    kwargs: dict = {}
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    elif os.getenv("OPENROUTER_API_KEY"):
        kwargs["api_key"] = os.getenv("OPENROUTER_API_KEY")
    return model, kwargs


def _extract_json(text: str) -> dict:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


def reflect(
    query: str,
    classification: str,
    confidence: float,
    context: str = "",
    reasoning: str = "",
    model: Optional[str] = None,
) -> Optional[dict]:
    """Review an Arabic triage classification.

    Returns a revision dict when needs_revision=True, None otherwise.
    """
    resolved_model, extra_kwargs = _resolve_model()
    model = model or resolved_model

    user_prompt = f"""## Customer Query
{query}

## Agent Classification
Intent: {classification}
Confidence: {confidence:.2f}

## Context
{context[:1500] if context else "None"}"""

    if reasoning:
        user_prompt += f"\n\n## Agent Reasoning\n{reasoning}"

    try:
        response = completion(
            model=model,
            messages=[
                {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=500,
            **extra_kwargs,
        )
        raw = response.choices[0].message.content or ""
        result = _extract_json(raw)

        suggested = result.get("suggested_intent")
        if suggested and suggested not in VALID_INTENTS:
            result["suggested_intent"] = None

        if result.get("needs_revision"):
            return result

    except (json.JSONDecodeError, KeyError, AttributeError) as e:
        pass
    except Exception as e:
        pass

    return None
