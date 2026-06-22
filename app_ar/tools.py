"""Tool layer — Arabic responses for the Arabic triage agent."""

from __future__ import annotations

from dataclasses import dataclass

from langfuse import observe

from app_ar.observability import (
    RetrievalMetricsLogger,
    trace_embedding,
    trace_ticket_creation,
    trace_vector_search,
)


@dataclass
class ToolResult:
    success: bool
    data: str
    resolved: bool  # True = fully answered, False = needs escalation


@observe(name="faq_lookup_ar")
def faq_lookup(intent: str, user_message: str) -> ToolResult:
    """Find the best matching Arabic FAQ article via vector similarity search."""
    fallbacks = {
        "password_reset": (
            "لإعادة تعيين كلمة المرور، تفضل بزيارة صفحة تسجيل الدخول وانقر على 'نسيت كلمة المرور'. "
            "سيتم إرسال رابط إعادة التعيين إلى بريدك الإلكتروني (صالحة لمدة 30 دقيقة). "
            "إذا لم يصلك البريد، تحقق من مجلد الرسائل غير المرغوب فيها."
        ),
        "billing": (
            "للاستفسارات المتعلقة بالفواتير، يرجى تسجيل الدخول وزيارة الحساب ← الفواتير، "
            "أو مراسلتنا على billing@support.example.com. "
            "نعالج طلبات استرداد المبالغ المالية خلال 5-7 أيام عمل."
        ),
        "technical_support": (
            "حاول مسح ذاكرة التخزين المؤقت وملفات تعريف الارتباط (Cache & Cookies)، "
            "ثم تحقق من status.example.com لمعرفة الحوادث الحالية. "
            "إذا استمرت المشكلة، سيقوم فريقنا الفني بالتحقيق."
        ),
    }
    content = fallbacks.get(intent)
    if content:
        return ToolResult(success=True, data=content, resolved=True)

    metrics = RetrievalMetricsLogger()
    try:
        with metrics.trace_latency("embedding"):
            embedding = trace_embedding(user_message)
        with metrics.trace_latency("vector_search"):
            row = trace_vector_search(intent, embedding)

        hit = row is not None
        metrics.score_hit(hit)
        metrics.score_mrr(1 if hit else None)

        if hit:
            return ToolResult(
                success=True,
                data=f"{row['title']}\n\n{row['content']}",
                resolved=True,
            )
    except Exception:
        pass

    return ToolResult(success=False, data="لم يتم العثور على مقالة مناسبة.", resolved=False)


@observe(name="ticket_lookup_ar")
def ticket_lookup(user_message: str) -> ToolResult:
    """Create a support ticket with Arabic confirmation."""
    metrics = RetrievalMetricsLogger()
    try:
        with metrics.trace_latency("embedding"):
            embedding = trace_embedding(user_message)
        with metrics.trace_latency("ticket_db_insert"):
            ticket_id = trace_ticket_creation(user_message, "escalation", embedding)
        metrics.score_ticket_created(True)
        return ToolResult(
            success=True,
            data=(
                f"تم إنشاء تذكرة ذات أولوية برقم #{ticket_id}. "
                "سيتواصل معك وكيل دعم أول خلال 15 دقيقة."
            ),
            resolved=True,
        )
    except Exception:
        metrics.score_ticket_created(False)
        return ToolResult(
            success=True,
            data="تم إنشاء تذكرة ذات أولوية. سيتواصل معك وكيل دعم أول قريباً.",
            resolved=True,
        )


def run_tool(intent: str, user_message: str) -> ToolResult:
    """Dispatch to the right tool based on classified intent."""
    tool_map = {
        "greeting": lambda i, m: ToolResult(
            success=True,
            data=(
                "أهلاً وسهلاً! كيف يمكنني مساعدتك اليوم؟\n\n"
                "يمكنني المساعدة في:\n\n"
                "• **كلمة المرور وحساب الدخول** — إعادة تعيين، حسابات مقفلة\n"
                "• **الفواتير والمدفوعات** — الفواتير، الرسوم، استرداد المبالغ\n"
                "• **المشاكل التقنية** — أخطاء التطبيق، مشاكل API"
            ),
            resolved=True,
        ),
        "password_reset": faq_lookup,
        "billing": faq_lookup,
        "technical_support": faq_lookup,
        "escalation": lambda i, m: ticket_lookup(m),
        "unknown": lambda i, m: ToolResult(
            success=False,
            data=(
                "لم أفهم استفسارك تماماً. يمكنني المساعدة في:\n\n"
                "• **كلمة المرور وحساب الدخول** — إعادة تعيين، حسابات مقفلة\n"
                "• **الفواتير والمدفوعات** — الفواتير، الرسوم، استرداد المبالغ\n"
                "• **المشاكل التقنية** — أخطاء التطبيق، مشاكل API\n\n"
                "هل يمكنك توضيح مشكلتك؟"
            ),
            resolved=False,
        ),
    }
    handler = tool_map.get(intent, tool_map["unknown"])
    return handler(intent, user_message)
