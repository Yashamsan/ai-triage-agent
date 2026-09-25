#!/usr/bin/env python3
"""Retrieval-quality eval harness for the Arabic product knowledge base.

Built after tracing several "RAG gives wrong/irrelevant answers" reports
back to real find_kb_chunk() misses and near-misses on realistic customer
phrasing — see the golden set below, which encodes exactly those cases
plus deliberate near-duplicate discrimination tests (three different
"backup service" KB entries that are easy to confuse) and negative
controls (queries nothing in the KB should answer).

This calls the ACTUAL app_ar.database.find_kb_chunk() used by the live
triage agent, so it's a real regression test for retrieval logic changes,
not a reimplementation. Run after any change to embeddings, chunking, or
the retrieval/ranking logic — before this script existed, every such
change was validated by testing one query at a time by hand, which is how
a real regression slipped through unnoticed for a full session.

Usage:
    python scripts/eval_kb_retrieval.py            # full report
    python scripts/eval_kb_retrieval.py --quiet     # summary line only
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app_ar.database import find_kb_chunk  # noqa: E402
from app_ar.embeddings import embed  # noqa: E402

# (query, expected_row_ref or None for "should not resolve to anything")
# row_refs verified against the live knowledge_base_chunks table at the
# time this set was built — re-verify with
#   SELECT row_ref, title FROM knowledge_base_chunks WHERE lang='ar'
# if the source spreadsheet changes.
GOLDEN_SET: list[tuple[str, str | None]] = [
    ("خدمة موجود اكسترا", "All Knowledge Assistant!row9"),
    ("اريد الغاء الرسائل الدعائية", "All Knowledge Assistant!row71"),
    ("كيف افعل خدمة تحويل المكالمات", "All Knowledge Assistant!row6"),
    ("ابي اعرف رصيد الانترنت والمكالمات", "All Knowledge Assistant!row68"),
    ("وش هي باقة اكسبرو كويك نت", "All Knowledge Assistant!row100"),
    ("عندي مشكلة في تفعيل eSIM", "All Knowledge Assistant!row128"),
    ("كيف اعدل اعدادات الانترنت على الجوال APN", "All Knowledge Assistant!row11"),
    ("كيف افعل البريد الصوتي بجوالي", "All Knowledge Assistant!row3"),
    ("كيف افعل البريد الصوتي للخط الارضي", "All Knowledge Assistant!row2"),
    ("ودي اسوي اعداد الرسائل القصيرة", "All Knowledge Assistant!row7"),
    ("كيف احول من دفع لاحق لدفع مسبق", "All Knowledge Assistant!row14"),
    ("وين اقرب مكتب اشتراكات STC في جدة", "All Knowledge Assistant!row155"),
    ("كيف اسوي تبديل الشريحة مع اكثر من جوال", "All Knowledge Assistant!row13"),
    ("ما هي باقات تجوال دول الخليج", "All Knowledge Assistant!row104"),
    ("كيف اعرف نوع التغطية عندي بالخريطة", "All Knowledge Assistant!row17"),
    ("وين اقرب متجر لكم", "All Knowledge Assistant!row15"),
    # Near-duplicate discrimination: three distinct "backup service" entries.
    ("كيف اسوي باك اب لخدمة IP VPN", "All Knowledge Assistant!row27"),
    ("كيف اسوي باك اب ل VPLS", "All Knowledge Assistant!row28"),
    ("كيف اسوي باك اب للدوائر الصوتية SIP", "All Knowledge Assistant!row37"),
    # Negative controls: nothing in this KB should confidently answer these.
    ("ابي اطلب بيتزا", None),
    ("كم الساعة الحين", None),
]


def run(quiet: bool = False) -> tuple[int, int]:
    hits = 0
    misses: list[tuple[str, str | None, str | None]] = []

    for query, expected in GOLDEN_SET:
        embedding = embed(query)
        row = find_kb_chunk(embedding, query_text=query, lang="ar")
        got = row["row_ref"] if row else None
        ok = got == expected
        hits += ok
        if not ok:
            misses.append((query, expected, got))
        if not quiet:
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] {query!r}")
            print(f"       expected={expected}  got={got}")

    total = len(GOLDEN_SET)
    print(f"\n{hits}/{total} correct ({hits / total:.0%})")
    if misses:
        print("\nMisses:")
        for query, expected, got in misses:
            print(f"  {query!r}: expected {expected}, got {got}")
    return hits, total


if __name__ == "__main__":
    quiet = "--quiet" in sys.argv
    hits, total = run(quiet=quiet)
    sys.exit(0 if hits == total else 1)
