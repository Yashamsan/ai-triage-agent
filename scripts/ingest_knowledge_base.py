#!/usr/bin/env python3
"""Ingest a private company knowledge base (Excel) into knowledge_base_chunks.

Reads every .xlsx/.xlsm file in knowledge_base/raw/ — a gitignored folder,
see .gitignore — and embeds rows for semantic search via the product_inquiry
intent (app/tools.py:kb_lookup, app_ar/tools.py:kb_lookup).

Two ingestion paths:

1. Bilingual schema (preferred): a sheet with "English Query", "English
   Content", "Arabic Query", "Arabic Content" columns produces TWO chunks
   per row — one lang='en', one lang='ar' — each queried only by its own
   language's agent (see find_kb_chunk()'s lang filter in app/database.py
   and app_ar/database.py). Optional "Keywords / Tags", "Confidence
   Triggers", and "Expected Prompts (English|Arabic)" columns are folded
   into the *embedded* text (to widen what phrasings match) but never into
   the *displayed* answer, which stays just the clean Content column. A
   "Metadata" column containing a JSON object is parsed and merged in;
   "Section / Category" / "Entry Type" / row number are captured too.

   When a workbook has multiple sheets, only the first bilingual-schema
   sheet is ingested — later sheets whose "#" column is a subset of an
   already-ingested sheet's "#" values are skipped as duplicate views
   (this workbook format ships the same entries again as "Expected
   Prompts" and "English Only" convenience exports).

2. Generic fallback: any other sheet shape. No assumptions about column
   names — every non-empty cell becomes "Header: value" text, and the
   whole row is embedded as one lang='en' chunk.

Safe to re-run: each file's previously ingested chunks are deleted before
its rows are re-inserted, so editing the spreadsheet and re-running this
script keeps the DB in sync instead of piling up duplicates.

Requires pandas + openpyxl (NOT in the main requirements.txt / Docker image
— only this offline script needs them):
    pip install -r requirements-kb.txt

Usage:
    python scripts/ingest_knowledge_base.py                  # every file in knowledge_base/raw/
    python scripts/ingest_knowledge_base.py path/to/one.xlsx  # a single file
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
except ImportError:
    print("Missing dependency: pandas (and openpyxl for .xlsx support).")
    print("Install with:  pip install -r requirements-kb.txt")
    sys.exit(1)

from app.database import apply_schema, delete_kb_chunks_for_file, insert_kb_chunk
from app.embeddings import embed_batch as embed_batch_en
from app_ar.embeddings import embed_batch as embed_batch_ar

RAW_DIR = Path(__file__).parent.parent / "knowledge_base" / "raw"

# Column names (case-insensitive) preferred as a chunk's title, in the
# generic fallback path.
_TITLE_COLUMNS = {"title", "question", "topic", "feature", "name"}

_BILINGUAL_REQUIRED = ["English Query", "English Content", "Arabic Query", "Arabic Content"]
_SIGNAL_COLUMNS = ["Keywords / Tags", "Confidence Triggers"]


def _cell(row: pd.Series, col: str) -> str:
    val = row.get(col)
    return str(val).strip() if pd.notna(val) else ""


def _parse_metadata_cell(raw) -> dict:
    if pd.isna(raw) or not str(raw).strip():
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except Exception:
        return {"raw_metadata": str(raw)}


def _is_bilingual_schema(df: pd.DataFrame) -> bool:
    return all(col in df.columns for col in _BILINGUAL_REQUIRED)


class Chunk:
    __slots__ = ("lang", "title", "content", "embed_text", "row_ref", "metadata")

    def __init__(self, lang, title, content, embed_text, row_ref, metadata):
        self.lang = lang
        self.title = title
        self.content = content
        self.embed_text = embed_text
        self.row_ref = row_ref
        self.metadata = metadata


def _bilingual_chunks(sheet_name: str, df: pd.DataFrame) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i, row in df.iterrows():
        row_ref = f"{sheet_name}!row{i + 2}"  # +2: header row + 0-index
        section = _cell(row, "Section / Category") or _cell(row, "Section")
        entry_type = _cell(row, "Entry Type")
        meta = {
            **_parse_metadata_cell(row.get("Metadata")),
            "row_number": _cell(row, "#") or None,
            "section": section or None,
            "entry_type": entry_type or None,
        }
        signal = " ".join(filter(None, (_cell(row, c) for c in _SIGNAL_COLUMNS)))

        en_content = _cell(row, "English Content")
        if en_content:
            en_query = _cell(row, "English Query")
            embed_text = " ".join(filter(None, [
                en_query, en_content, signal, _cell(row, "Expected Prompts (English)"),
            ]))
            chunks.append(Chunk(
                "en", en_query or section or "Untitled", en_content, embed_text, row_ref, meta,
            ))

        ar_content = _cell(row, "Arabic Content")
        if ar_content:
            ar_query = _cell(row, "Arabic Query")
            embed_text = " ".join(filter(None, [
                ar_query, ar_content, signal, _cell(row, "Expected Prompts (Arabic)"),
            ]))
            chunks.append(Chunk(
                "ar", ar_query or section or "غير مصنف", ar_content, embed_text, row_ref, meta,
            ))
    return chunks


def _generic_chunks(sheet_name: str, df: pd.DataFrame) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i, row in df.iterrows():
        parts: list[str] = []
        title: str | None = None
        for col, value in row.items():
            if pd.isna(value) or str(value).strip() == "":
                continue
            text_value = str(value).strip()
            if title is None and str(col).strip().lower() in _TITLE_COLUMNS:
                title = text_value
            parts.append(f"{col}: {text_value}")
        if not parts:
            continue
        content = "\n".join(parts)
        title = title or parts[0].split(": ", 1)[-1][:80]
        chunks.append(Chunk("en", title, content, content, f"{sheet_name}!row{i + 2}", {"sheet": sheet_name}))
    return chunks


def ingest_file(path: Path) -> int:
    print(f"\n--- {path.name} ---")
    sheets = pd.read_excel(path, sheet_name=None, dtype=str)  # {sheet_name: DataFrame}

    deleted = delete_kb_chunks_for_file(path.name)
    if deleted:
        print(f"  removed {deleted} old chunk(s) from a previous run of this file")

    seen_ids: set[str] = set()
    all_chunks: list[Chunk] = []

    for sheet_name, df in sheets.items():
        if "#" in df.columns:
            sheet_ids = {str(v) for v in df["#"].dropna()}
            if sheet_ids and sheet_ids.issubset(seen_ids):
                print(f"  {sheet_name}: skipped — same entries already ingested from an earlier sheet")
                continue
            seen_ids |= sheet_ids

        if _is_bilingual_schema(df):
            chunks = _bilingual_chunks(sheet_name, df)
            print(f"  {sheet_name}: bilingual schema — {len(chunks)} chunk(s) (EN + AR per row)")
        else:
            chunks = _generic_chunks(sheet_name, df)
            print(f"  {sheet_name}: generic schema — {len(chunks)} chunk(s)")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("  nothing to ingest")
        return 0

    en_chunks = [c for c in all_chunks if c.lang != "ar"]
    ar_chunks = [c for c in all_chunks if c.lang == "ar"]

    if en_chunks:
        print(f"  embedding {len(en_chunks)} English chunk(s)...")
        embeddings = embed_batch_en([c.embed_text for c in en_chunks])
        for c, embedding in zip(en_chunks, embeddings):
            insert_kb_chunk(
                source_file=path.name, row_ref=c.row_ref, title=c.title,
                content=c.content, embedding=embedding, lang="en", metadata=c.metadata,
            )

    if ar_chunks:
        print(f"  embedding {len(ar_chunks)} Arabic chunk(s)...")
        embeddings = embed_batch_ar([c.embed_text for c in ar_chunks])
        for c, embedding in zip(ar_chunks, embeddings):
            insert_kb_chunk(
                source_file=path.name, row_ref=c.row_ref, title=c.title,
                content=c.content, embedding=embedding, lang="ar", metadata=c.metadata,
            )

    total = len(all_chunks)
    print(f"  ingested {total} chunk(s) from this file ({len(en_chunks)} en / {len(ar_chunks)} ar)")
    return total


def main() -> None:
    apply_schema()  # idempotent — ensures knowledge_base_chunks (+ lang column) exists

    if len(sys.argv) > 1:
        files = [Path(sys.argv[1])]
    else:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(RAW_DIR.glob("*.xlsx")) + sorted(RAW_DIR.glob("*.xlsm"))

    if not files:
        print(f"No Excel files found in {RAW_DIR}")
        print("Drop your company knowledge-base spreadsheet(s) there and re-run this script.")
        return

    grand_total = 0
    for path in files:
        if not path.exists():
            print(f"Skipping {path} — not found")
            continue
        grand_total += ingest_file(path)

    print(f"\nDone. {grand_total} chunk(s) across {len(files)} file(s) now searchable via product_inquiry.")


if __name__ == "__main__":
    main()
