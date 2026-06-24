"""Server-side PDF generator for ISO 42001 compliance reports.

Uses fpdf2 to produce a 4-section PDF:
  1. Cover page
  2. Scorecard / KPI summary
  3. Requirement-by-requirement breakdown
  4. Remediation checklist (quick wins + documentation gaps)

Call generate_compliance_pdf(agent_name=None) -> bytes to get the PDF bytes.
"""

from __future__ import annotations

import io
from datetime import date

from fpdf import FPDF

from app.compliance import (
    MET,
    NOT_MET,
    PARTIAL,
    generate_iso_42001_report,
    generate_remediation_checklist,
)


def _safe(text: str, maxlen: int = 0) -> str:
    “””Encode to latin-1, replacing any non-latin-1 character with ‘?’.”””
    text = text.encode(“latin-1”, errors=”replace”).decode(“latin-1”)
    if maxlen:
        text = text[:maxlen]
    return text

# ── Brand colours (RGB) ───────────────────────────────────────────────────────
C_PRIMARY = (10,  40,  80)
C_ACCENT  = (40, 120, 220)
C_GREEN   = (63, 185,  80)
C_ORANGE  = (210, 153,  34)
C_RED     = (248,  81,  73)
C_GRAY    = (140, 150, 160)
C_LIGHT   = (240, 242, 245)


def _status_colour(status: str) -> tuple[int, int, int]:
    if status == MET:
        return C_GREEN
    if status == PARTIAL:
        return C_ORANGE
    if status == NOT_MET:
        return C_RED
    return C_GRAY


class CompliancePDF(FPDF):
    def __init__(self, title: str = "ISO 42001:2023 Compliance Report"):
        super().__init__(orientation="P", unit="mm", format="A4")
        self._title = title
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(20, 15, 20)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*C_GRAY)
        self.cell(0, 6, self._title, align="L")
        self.set_text_color(*C_GRAY)
        self.cell(0, 6, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*C_ACCENT)
        self.line(20, self.get_y(), 190, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*C_GRAY)
        self.cell(0, 5, f"ProofLayer AI Governance | {date.today().isoformat()} | Confidential", align="C")

    # ── Layout helpers ────────────────────────────────────────────────────────

    def _section_header(self, text: str):
        self.set_fill_color(*C_PRIMARY)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 9, f"  {_safe(text)}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def _sub_header(self, text: str):
        self.set_text_color(*C_ACCENT)
        self.set_font("Helvetica", "B", 9)
        self.cell(0, 7, _safe(text), new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)

    def _req_row(self, req: dict):
        status  = req["status"]
        colour  = _status_colour(status)

        self.set_fill_color(*C_LIGHT)
        self.set_font("Helvetica", "B", 8)

        # ID badge
        self.set_fill_color(*colour)
        self.set_text_color(255, 255, 255)
        self.cell(16, 7, _safe(req["req_id"]), fill=True, border=0)

        # Title
        self.set_text_color(0, 0, 0)
        self.set_fill_color(*C_LIGHT)
        self.set_font("Helvetica", "B", 8)
        self.cell(114, 7, _safe(req["title"], 70), fill=True, border=0)

        # Status badge
        self.set_fill_color(*colour)
        self.set_text_color(255, 255, 255)
        self.cell(20, 7, _safe(status), fill=True, border=0, align="C")
        self.ln(7)

        # Evidence / recommendation
        self.set_text_color(*C_GRAY)
        self.set_font("Helvetica", "", 7)
        evidence = _safe((req.get("evidence") or ""), 200)
        recommendation = _safe((req.get("recommendation") or ""), 200)
        self.multi_cell(150, 4, f"Evidence: {evidence}", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*C_ACCENT)
        self.multi_cell(150, 4, f"  -> {recommendation}", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def _score_bar(self, label: str, pct: float, colour: tuple[int, int, int] = C_ACCENT):
        bar_w = 100
        fill_w = int(bar_w * pct / 100)
        self.set_font("Helvetica", "", 8)
        self.cell(60, 6, label)
        self.set_fill_color(*C_LIGHT)
        x = self.get_x()
        y = self.get_y()
        self.cell(bar_w, 6, "", fill=True, border=1)
        self.set_fill_color(*colour)
        self.rect(x, y + 1, fill_w, 4, style="F")
        self.set_font("Helvetica", "B", 8)
        self.cell(15, 6, f"{pct:.1f}%", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)


# ═════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═════════════════════════════════════════════════════════════════════════════


def generate_compliance_pdf(agent_name: str | None = None) -> bytes:
    """Run the compliance engine and render a PDF report; return raw bytes."""
    report     = generate_iso_42001_report(agent_name)
    remediation = generate_remediation_checklist(agent_name)

    agent_label  = agent_name or "All Agents"
    pct          = report["overall_score_pct"]
    grade        = report["score_grade"]
    counts       = report["counts"]
    requirements = report["requirements"]

    pdf = CompliancePDF(title=f"ISO 42001:2023 Compliance - {agent_label}")
    pdf.set_author("ProofLayer AI Governance")

    # ── Page 1: Cover ─────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_fill_color(*C_PRIMARY)
    pdf.rect(0, 0, 210, 80, style="F")
    pdf.set_y(20)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 12, "ISO/IEC 42001:2023", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "AI Management System - Compliance Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(180, 200, 240)
    pdf.cell(0, 8, f"Agent Scope: {agent_label}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Report Date: {date.today().isoformat()}", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_y(90)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 48)
    grade_colour = C_GREEN if grade in ("A", "B") else (C_ORANGE if grade == "C" else C_RED)
    pdf.set_text_color(*grade_colour)
    pdf.cell(0, 20, f"{grade}  {pct:.1f}%", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*C_GRAY)
    pdf.cell(0, 6, "Overall Compliance Score", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(10)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Executive Summary", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    summary_text = (
        f"This report evaluates {report['total_requirements']} requirements across "
        f"ISO 42001:2023 Clauses 4-10 and Annex A controls. "
        f"{counts['met']} requirements are fully met, {counts['partial']} show partial "
        f"evidence, and {counts['not_met']} have identified gaps. "
        f"Core clause compliance stands at {report['core_clauses_pct']:.1f}% and "
        f"Annex A control compliance at {report['annex_controls_pct']:.1f}%."
    )
    pdf.multi_cell(0, 5, summary_text, align="C")

    # ── Page 2: Scorecard ─────────────────────────────────────────────────────
    pdf.add_page()
    pdf._section_header("Compliance Scorecard")

    pdf.set_font("Helvetica", "B", 9)
    for label, val, colour in [
        ("Met (fully evidenced)",     counts["met"],     C_GREEN),
        ("Partial (gap action req.)", counts["partial"], C_ORANGE),
        ("Not Met (no evidence)",     counts["not_met"], C_RED),
        ("Not Applicable",            counts["na"],      C_GRAY),
    ]:
        pdf.set_text_color(*colour)
        pdf.cell(80, 7, label)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 7, str(val), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf._sub_header("Compliance by Axis")
    pdf._score_bar("Core Clauses (4-10)",   report["core_clauses_pct"],   C_ACCENT)
    pdf._score_bar("Annex A Controls",       report["annex_controls_pct"], C_PRIMARY)
    pdf.ln(4)

    # Section breakdown
    pdf._sub_header("Compliance by Section")
    pdf.set_font("Helvetica", "B", 7)
    for sec, stats in report["section_breakdown"].items():
        m = stats.get("met", 0)
        p = stats.get("partial", 0)
        nm = stats.get("not_met", 0)
        total_sec = m + p + nm
        sec_pct = round((m + p * 0.5) / total_sec * 100) if total_sec else 0
        c = C_GREEN if sec_pct >= 75 else (C_ORANGE if sec_pct >= 40 else C_RED)
        pdf.set_text_color(*c)
        pdf.cell(10, 5, f"{sec_pct}%")
        pdf.set_text_color(0, 0, 0)
        pdf.cell(100, 5, _safe(sec, 60))
        pdf.set_text_color(*C_GRAY)
        pdf.cell(0, 5, f"OK={m}  ~={p}  X={nm}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    # ── Page 3: Requirements ──────────────────────────────────────────────────
    pdf.add_page()
    pdf._section_header("Requirement-by-Requirement Analysis")

    current_section = None
    for req in requirements:
        sec = req["section"]
        if sec != current_section:
            pdf.ln(2)
            pdf._sub_header(sec)
            current_section = sec
        pdf._req_row(req)

    # ── Page 4: Remediation Checklist ─────────────────────────────────────────
    pdf.add_page()
    pdf._section_header("Prioritised Remediation Checklist")

    def _checklist_block(items: list[dict], heading: str, colour: tuple[int, int, int]):
        pdf._sub_header(heading)
        if not items:
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(*C_GRAY)
            pdf.cell(0, 6, "None", new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
            return
        for item in items:
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*colour)
            pdf.cell(18, 5, _safe(item["req_id"]))
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 5, _safe(item["title"], 80), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(*C_GRAY)
            rec = _safe((item.get("recommendation") or ""), 200)
            pdf.multi_cell(0, 4, f"    -> {rec}", new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
            pdf.ln(1)
        pdf.ln(3)

    _checklist_block(remediation["quick_wins"],           "Quick Wins (PARTIAL - evidence exists)",     C_ORANGE)
    _checklist_block(remediation["documentation_needed"], "Documentation Needed (NOT MET - no evidence)", C_RED)
    _checklist_block(remediation["already_compliant"],    "Already Compliant (MET)",                    C_GREEN)

    buffer = io.BytesIO()
    pdf.output(buffer)
    return buffer.getvalue()
