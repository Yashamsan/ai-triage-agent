FROM python:3.13-slim AS builder
WORKDIR /app

# CUDA-enabled torch so shared/embeddings.py can put bge-m3 on a
# passed-through GPU (docker-compose.yml's device reservation) instead of
# this CPU-only box's slow, highly-variable inference latency. Falls back
# to CPU on its own at runtime if no GPU is visible (see
# shared/embeddings.py's _select_device()), so this image still works
# without a GPU, just slowly. Installed before -r requirements.txt so
# sentence-transformers doesn't pull a redundant/mismatched torch wheel.
#
# Pinned to the cu124 index, not plain "torch" (which now resolves to a
# CUDA 13.0 build): that failed at runtime with "CUDA initialization: The
# NVIDIA driver on your system is too old (found version 12090)" on the
# target GPU's driver, which only supports up to CUDA 12.9. A 12.x-compiled
# wheel is forward-compatible with any driver supporting that version or
# newer, so this covers the current host (and most others) without needing
# to track the exact driver version.
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cu124

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.13-slim AS runtime

# tini becomes PID 1 (via ENTRYPOINT below) so orphaned child processes get
# reaped properly. Without it, the app itself is PID 1, which never calls
# wait() on children it didn't directly spawn — those become zombies and
# accumulate for the life of the container (observed: ~1 per real request
# over normal operation, eventually enough to make new work hang).
RUN apt-get update && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home appuser
WORKDIR /home/appuser/app
COPY --from=builder /root/.local /home/appuser/.local
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser app_ar/ ./app_ar/
COPY --chown=appuser:appuser audit/ ./audit/
COPY --chown=appuser:appuser shared/ ./shared/
COPY --chown=appuser:appuser orchestrator/ ./orchestrator/
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser ui/ ./ui/
# Pre-create audit_data owned by appuser so the named volume mounted here
# (docker-compose.yml: triage_audit_data) inherits correct ownership on
# first creation instead of defaulting to root, which appuser can't write to.
RUN mkdir -p audit_data && chown appuser:appuser audit_data
ENV PATH="/home/appuser/.local/bin:$PATH"
ENV PYTHONPATH="/home/appuser/app"
USER appuser

# Pre-download the embedding model into the image so a fresh container
# doesn't spend its startup (or the healthcheck's start_period) downloading
# ~2GB from the Hugging Face Hub cold — this was this session's biggest
# recurring pain point (multi-minute container-start waits, easy to mistake
# for a hang). Caches to appuser's default ~/.cache/huggingface, the same
# place runtime would look, so this is a pure no-op cache warm at runtime.
#
# HF_HUB_DISABLE_XET=1: the Hugging Face Hub's newer "Xet" download backend
# (hf_xet) hit a hard, reproducible failure on this connection --
# "CAS Client Error: Format error: I/O error: error decoding response body"
# -- partway through this exact download. Forces the older plain-HTTP
# download path instead, which is slower but doesn't have that failure mode.
ENV HF_HUB_DISABLE_XET=1
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
