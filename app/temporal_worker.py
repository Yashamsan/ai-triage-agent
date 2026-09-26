"""Temporal worker entry point — durable-execution demo.

    python -m app.temporal_worker          (or, with the project venv:
    .venv/Scripts/python -m app.temporal_worker   on Windows)

Connects to the Temporal server (docker/docker-compose.temporal.yml) and
polls the 'triage-queue' task queue for TriageAgentWorkflow work. Kill this
process (Ctrl+C) mid-workflow and restart it — Temporal Server already
persisted every completed step, so the workflow resumes from where it left
off instead of restarting from scratch.

Runs either on the host (TEMPORAL_ADDRESS defaults to "localhost:7233",
matching that compose file's port-forward) or as the temporal-worker
service in that same compose file (which sets TEMPORAL_ADDRESS=temporal:7233
and DATABASE_URL to reach triage-db over the shared docker_default network —
both stacks' containers are on it, confirmed via `docker network inspect`).
Needs the same DEEPSEEK_API_KEY / DATABASE_URL as the main app, since its
activities (app/temporal_activities.py) call the exact same classify() /
run_tool() / generate_response() functions.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from temporalio.client import Client
from temporalio.worker import Worker

from app.temporal_activities import (
    classify_activity,
    generate_response_activity,
    run_tool_activity,
)
from app.temporal_workflows import TriageAgentWorkflow

TASK_QUEUE = "triage-queue"
TEMPORAL_ADDRESS = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")


async def _connect_with_retry(address: str, attempts: int = 10, delay_seconds: float = 3.0) -> Client:
    """Retry the initial connection -- when this runs as a container started
    alongside the Temporal server (docker-compose.temporal.yml's
    temporal-worker service), there's no healthcheck to depend_on before the
    server is actually accepting connections, and Client.connect() doesn't
    retry on its own."""
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await Client.connect(address)
        except Exception as exc:
            last_exc = exc
            print(f"  attempt {attempt}/{attempts} failed ({exc}); retrying in {delay_seconds:.0f}s ...")
            await asyncio.sleep(delay_seconds)
    raise RuntimeError(f"Could not connect to Temporal Server at {address} after {attempts} attempts") from last_exc


async def main() -> None:
    print(f"🔄 Connecting to Temporal Server at {TEMPORAL_ADDRESS} ...")
    client = await _connect_with_retry(TEMPORAL_ADDRESS)
    print("✅ Connected.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as activity_executor:
        worker = Worker(
            client,
            task_queue=TASK_QUEUE,
            workflows=[TriageAgentWorkflow],
            activities=[classify_activity, run_tool_activity, generate_response_activity],
            activity_executor=activity_executor,
        )
        print(f"🚀 Worker listening on task queue '{TASK_QUEUE}'.")
        await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
