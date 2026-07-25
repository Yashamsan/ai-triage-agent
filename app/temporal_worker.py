"""Temporal worker entry point — durable-execution demo.

    python -m app.temporal_worker          (or, with the project venv:
    .venv/Scripts/python -m app.temporal_worker   on Windows)

Connects to the local Temporal server (docker/docker-compose.temporal.yml)
and polls the 'triage-queue' task queue for TriageAgentWorkflow work. Kill
this process (Ctrl+C) mid-workflow and restart it — Temporal Server already
persisted every completed step, so the workflow resumes from where it left
off instead of restarting from scratch.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
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
TEMPORAL_ADDRESS = "localhost:7233"


async def main() -> None:
    print(f"🔄 Connecting to Temporal Server at {TEMPORAL_ADDRESS} ...")
    client = await Client.connect(TEMPORAL_ADDRESS)
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
