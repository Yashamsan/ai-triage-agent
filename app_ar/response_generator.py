"""LLM-powered Arabic response synthesis for the Arabic triage agent.

Mirrors app/response_generator.py with an Arabic-first system prompt.
English system prompt used deliberately for reliable JSON-free output
from DeepSeek — the instruction to respond in Arabic is explicit.
"""

import os

from dotenv import load_dotenv

load_dotenv()

from litellm import completion

_SYSTEM_PROMPT = """You are a professional, warm Arabic-speaking customer support agent for a SaaS company.

Your task is to write the body of a response to a customer IN ARABIC using the context below.

Rules by intent:
- **greeting (تحية)**: Mirror the customer's greeting style. If they said "السلام عليكم", respond with "وعليكم السلام". Welcome them warmly and briefly mention 2-3 areas you can help with. 2-3 sentences maximum.
- **password_reset / billing / technical_support**: Synthesize the retrieved information into a direct, clear Arabic answer. Do NOT paste instructions verbatim — explain naturally. Reference the customer's words ("كما ذكرت...", "بما أنك تواجه مشكلة في...").
- **unknown**: Acknowledge what they said in Arabic, then ask exactly ONE targeted clarifying question. Offer 2-3 examples of what you can help with. Keep it brief.
- **escalation**: Confirm the ticket is created in warm Arabic. Reassure them a senior agent is personally handling it.
- **Multi-turn**: If conversation history shows prior context, reference it naturally ("بالمتابعة مع استفسارك السابق...", "كما أشرنا سابقاً...").

Tone rules:
- Warm, professional Modern Standard Arabic — accessible but not colloquial
- First person ("يمكنني المساعدة", "سنحل هذا معاً")
- Avoid filler phrases
- Match formality to the customer's message

Output ONLY the Arabic response body — no headers, no "---", no confidence scores, no metadata."""


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


def generate_response_ar(
    message: str,
    intent: str,
    tool_output: str,
    context_history: str = "",
    precedent_context: str = "",
    confidence: float = 1.0,
    critique: str = "",
) -> str:
    """Synthesize a natural, personalized Arabic support response.

    Falls back to raw tool_output if the LLM call fails.
    """
    model, extra_kwargs = _resolve_model()

    context_parts: list[str] = []
    if tool_output:
        context_parts.append(f"## المعلومات المستردة\n{tool_output}")
    if context_history:
        context_parts.append(f"## سجل المحادثة\n{context_history}")
    if precedent_context:
        context_parts.append(f"## حالات مشابهة سابقة\n{precedent_context}")
    if critique:
        context_parts.append(f"## ملاحظة المراجع (داخلية)\n{critique}")

    context_block = "\n\n".join(context_parts) or "لا يوجد سياق إضافي."

    user_prompt = (
        f"## رسالة العميل\n{message}\n\n"
        f"## النية المكتشفة: {intent} (الثقة: {confidence:.0%})\n\n"
        f"{context_block}"
    )

    try:
        response = completion(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.45,
            max_tokens=500,
            **extra_kwargs,
        )
        generated = (response.choices[0].message.content or "").strip()
        return generated if generated else tool_output
    except Exception:
        return tool_output
