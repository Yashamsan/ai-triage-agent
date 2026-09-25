"""Standalone FastAPI app for the SDAIA-P145 RMF module.

Run with:  uvicorn app.main:app --port 8090
or:        python run.py

Zero impact on the existing Responsible AI Policy module: this app binds its
own port (8090), uses its own SQLite database (rmf_data.db), and serves its
own dashboard at /admin-rmf.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from . import storage
from .api import router

app = FastAPI(
    title="ProofLayer - SDAIA National AI Risk Management Framework (P145)",
    version="0.1.0",
    description=(
        "Standalone implementation of the five-stage risk cycle "
        "(context, identification, assessment, treatment, monitoring/review). "
        "Does not touch the Responsible AI Policy (Mar 2026) module."
    ),
)

DASHBOARD = Path(__file__).resolve().parent / "dashboard.html"


@app.on_event("startup")
def _startup():
    storage.init_db()


app.include_router(router)


@app.get("/admin-rmf", include_in_schema=False)
def admin_rmf():
    return FileResponse(DASHBOARD)


@app.get("/health")
def health():
    return {"status": "ok", "module": "sdaia-rmf-p145"}
