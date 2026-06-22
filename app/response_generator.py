"""LLM-powered response synthesis for the English triage agent.

Implements the RAG generation step: takes the classified intent, retrieved
tool output, and conversation context, then produces a warm, personalized
response rather than echoing raw content verbatim.
"""

import os

from dotenv import load_dotenv

load_dotenv()

from litellm import completion

_SYSTEM_PROMPT = """You are a professional, warm customer support agent for a SaaS company.

Your task is to write the body of a response to a customer using the context below.

Rules by intent:
- **greeting**: Mirror the customer's tone and energy. Warmly welcome them and briefly mention 2-3 areas you can help with. 2-3 sentences maximum.
- **password_reset / billing / technical_support**: Synthesize the retrieved information into a direct, clear answer to their specific question. Do NOT paste instructions verbatim — explain them naturally and helpfully. Reference the customer's words ("you mentioned...", "since you're having trouble with...").
- **unknown**: Acknowledge what they said, then ask exactly ONE targeted clarifying question. Offer 2-3 examples of what you can help with. Keep it brief.
- **escalation**: Confirm the ticket is created, reassure them a senior agent is handling it personally. Empathetic tone.
- **Multi-turn**: If conversation history shows prior context, reference it naturally ("Following up on your earlier question...", "Since we were discussing...").

Tone rules:
- Warm, clear, professional — never robotic or corporate
- First person ("I can help", "We'll get this sorted")
- No filler phrases ("Of course!", "Absolutely!", "Great question!")
- Match formality to the customer's message

Output ONLY the response body — no headers, no "---", no confidence scores, no metadata."""


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


def generate_response(
    message: str,
    intent: str,
    tool_output: str,
    context_history: str = "",
    precedent_context: str = "",
    confidence: float = 1.0,
    critique: str = "",
) -> str:
    """Synthesize a natural, personalized support response.

    Falls back to raw tool_output if the LLM call fails, so the graph
    always produces a usable response.
    """
    model, extra_kwargs = _resolve_model()

    context_parts: list[str] = []
    if tool_output:
        context_parts.append(f"## Retrieved Information\n{tool_output}")
    if context_history:
        context_parts.append(f"## Conversation History\n{context_history}")
    if precedent_context:
        context_parts.append(f"## Similar Past Cases\n{precedent_context}")
    if critique:
        context_parts.append(f"## Reviewer Note (internal)\n{critique}")

    context_block = "\n\n".join(context_parts) or "No additional context available."

    user_prompt = (
        f"## Customer Message\n{message}\n\n"
        f"## Detected Intent: {intent} ({confidence:.0%} confidence)\n\n"
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
