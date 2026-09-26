"""Context Engineer — Level 2 (Strategic Problem-Solver).

Transforms raw intent + message into structured, precise tool queries.
The output of one step feeds the next step's input, not raw context.
"""
import json
import re

# Saudi PII / account patterns
SA_ACCOUNT = re.compile(r'(?:ACC|acc|حساب)[-\s]?(\d{2,8})', re.IGNORECASE)
SA_TICKET  = re.compile(r'(?:TK|TKT|ticket|تذكرة)[-\s]?(\d{5,10})', re.IGNORECASE)
SA_PHONE   = re.compile(r'(?:05|5)(\d{8})')
SA_INVOICE = re.compile(r'(?:INV|inv|فاتورة)[-\s]?(\d{3,10})', re.IGNORECASE)
SA_AMOUNT  = re.compile(r'(\d+[.,]?\d*)\s*(?:ريال|SAR|sr|رس)\b', re.IGNORECASE)

# Month/period patterns
AR_MONTHS = {
    'يناير': '01', 'فبراير': '02', 'مارس': '03', 'أبريل': '04',
    'مايو': '05', 'يونيو': '06', 'يوليو': '07', 'أغسطس': '08',
    'سبتمبر': '09', 'أكتوبر': '10', 'نوفمبر': '11', 'ديسمبر': '12',
    'السابق': None, 'الماضي': None, 'الحالي': 'current',
}
EN_MONTHS = {
    'january': '01', 'february': '02', 'march': '03', 'april': '04',
    'may': '05', 'june': '06', 'july': '07', 'august': '08',
    'september': '09', 'october': '10', 'november': '11', 'december': '12',
    'last': None, 'previous': None, 'current': 'current',
}


class ExtractedParams:
    """Structured parameters extracted from a user message."""
    def __init__(self,
                 account_id: str | None = None,
                 ticket_id: str | None = None,
                 phone: str | None = None,
                 invoice_id: str | None = None,
                 amount: float | None = None,
                 period: str | None = None,
                 raw_text: str = ""):
        self.account_id = account_id
        self.ticket_id = ticket_id
        self.phone = phone
        self.invoice_id = invoice_id
        self.amount = amount
        self.period = period
        self.raw_text = raw_text

    def has_any(self) -> bool:
        return any([self.account_id, self.ticket_id, self.phone,
                    self.invoice_id, self.amount, self.period])

    def __repr__(self):
        return (f"ExtractedParams(account={self.account_id}, "
                f"ticket={self.ticket_id}, phone={self.phone}, "
                f"invoice={self.invoice_id}, amount={self.amount}, "
                f"period={self.period})")


def extract_parameters(message: str) -> ExtractedParams:
    """Extract structured parameters from a user message.

    Uses regex patterns for Saudi account/ticket/invoice formats.
    Returns ExtractedParams — empty if nothing found.
    """
    params = ExtractedParams(raw_text=message[:200])

    m = SA_ACCOUNT.search(message)
    if m:
        params.account_id = f"ACC-{m.group(1)}"

    m = SA_TICKET.search(message)
    if m:
        params.ticket_id = f"TK-{m.group(1)}"

    m = SA_PHONE.search(message)
    if m:
        params.phone = f"05{m.group(1)}"

    m = SA_INVOICE.search(message)
    if m:
        params.invoice_id = f"INV-{m.group(1)}"

    m = SA_AMOUNT.search(message)
    if m:
        params.amount = float(m.group(1).replace(',', ''))

    msg_lower = message.lower()
    for name, code in {**AR_MONTHS, **EN_MONTHS}.items():
        if name in msg_lower or name in message:
            # False-positive guard: 'الحالي'/'current' both mean "current" in
            # the generic adjective sense too (e.g. "رصيدي الحالي" = "my
            # CURRENT balance", nothing to do with a month) -- only treat it
            # as a period reference when it's actually adjacent to a
            # month/year word. Checks both word orders: Arabic puts the
            # temporal word first ("الشهر الحالي"), English puts it last
            # ("current month") -- a one-directional check would silently
            # break whichever language it didn't cover.
            if code == 'current':
                temporal_context = (
                    r'(?:الشهر|شهر|عام|month|year)\s*' + re.escape(name)
                    + r'|' + re.escape(name) + r'\s*(?:month|year|الشهر|شهر|عام)'
                )
                if not re.search(temporal_context, message, re.IGNORECASE):
                    continue

            year_m = re.search(r'(20\d{2})', message)
            year = year_m.group(1) if year_m else '2026'
            if code == 'current':
                params.period = year
            elif code is None:
                params.period = 'last'
            else:
                params.period = f"{year}-{code}"
            break

    return params


def formulate_query(intent: str, params: ExtractedParams) -> str | None:
    """Formulate a precise tool query from intent + parameters.

    Returns a JSON string the tool can parse directly.
    Example: {"tool": "get_invoice", "account_id": "ACC-55", "invoice_id": "INV-12345"}
    """
    base = {"intent": intent}

    if intent == "billing":
        base["tool"] = "get_invoice"
        if params.account_id:
            base["account_id"] = params.account_id
        if params.invoice_id:
            base["invoice_id"] = params.invoice_id
        if params.period:
            base["period"] = params.period

    elif intent in ("technical", "technical_support"):
        base["tool"] = "get_ticket_status"
        if params.ticket_id:
            base["ticket_id"] = params.ticket_id
        if params.account_id:
            base["account_id"] = params.account_id

    elif intent == "escalation":
        base["tool"] = "escalate_ticket"
        if params.ticket_id:
            base["ticket_id"] = params.ticket_id
        if params.amount:
            base["compensation_requested"] = params.amount

    elif intent == "password_reset":
        base["tool"] = "reset_password"
        if params.account_id:
            base["account_id"] = params.account_id
        if params.phone:
            base["phone"] = params.phone

    elif intent == "product_inquiry":
        base["tool"] = "get_quote"
        if params.amount:
            base["budget"] = params.amount

    elif intent in ("greeting", "unknown"):
        return None

    else:
        return None

    # Don't return a hollow query with no extracted identifiers
    if not params.has_any():
        return None

    return json.dumps(base, ensure_ascii=False)
