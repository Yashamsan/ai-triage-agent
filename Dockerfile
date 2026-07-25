FROM python:3.13-slim AS builder
WORKDIR /app

# Install CPU-only torch first so sentence-transformers doesn't pull the
# 900 MB CUDA wheel. This must happen before -r requirements.txt.
RUN pip install --no-cache-dir --user \
    torch --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.13-slim AS runtime
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
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
