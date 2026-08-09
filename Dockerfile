# syntax=docker/dockerfile:1.7
#
# skin-metrics HTTP API container.
#
# Two targets share one dependency build:
#   api  (default) - Phase 1 pipeline + FastAPI. No torch. Deployable image.
#   full           - the same, plus the Phase 2 'dl' extra so `skin-metrics
#                    train` works exactly as it does locally (~2GB larger).
#
#   docker build -t skin-metrics-api .
#   docker build --target full -t skin-metrics-api:full .
#
# The FaceLandmarker model is baked in at build time, so the container needs no
# network at runtime (the build does). Only the image URL fetch talks out.
#
# The build context is an allow-list (.dockerignore): pyproject/uv.lock/README
# and skin_metrics/ only. No local photos, reports, tests or venv are shipped.

ARG PYTHON_VERSION=3.11
ARG UV_VERSION=0.12.0

# --------------------------------------------------------------------------
# builder - resolve the locked dependency set into /app/.venv
#
# uv is installed from PyPI rather than pulled from ghcr.io/astral-sh/uv so the
# whole build depends on Docker Hub only (some networks block anonymous ghcr).
# --------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder
ARG UV_VERSION

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir "uv==${UV_VERSION}"

WORKDIR /app

# Dependencies first: this layer is reused whenever only source files change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra api --extra detection

# README.md is required by pyproject's `readme` field when building the wheel.
COPY README.md ./
COPY skin_metrics ./skin_metrics
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra api --extra detection

# --------------------------------------------------------------------------
# builder-full - adds Phase 2 (torch / timm / albumentations / pandas)
# --------------------------------------------------------------------------
FROM builder AS builder-full
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra api --extra detection --extra dl

# --------------------------------------------------------------------------
# runtime-base - OS deps, app user, source. No Python packages yet.
# --------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime-base

# Lets redeploy.sh clean up *only* this project's leftover images.
LABEL org.opencontainers.image.title="skin-metrics-api"
LABEL org.opencontainers.image.source="https://github.com/-/skin-metrics"

# mediapipe depends on opencv-contrib-python (not the headless build), whose
# cv2 shared object links libGL / glib / libxcb at import time; sounddevice
# needs PortAudio. Verified with `ldd .../cv2/cv2.abi3.so | grep 'not found'`.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
        libgomp1 \
        libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    SKIN_METRICS_FACE_MODEL=/opt/skin-metrics/face_landmarker.task

WORKDIR /app

# uv installs the project as an editable pointing at /app, so the source tree
# has to be here at runtime. pyproject/README are build-time only.
COPY skin_metrics ./skin_metrics

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /opt/skin-metrics \
    && chown -R appuser:appuser /app /opt/skin-metrics
USER appuser

EXPOSE 8000

# The pipeline is CPU-bound; a single 12MP image can take tens of seconds, so
# the health check must not be tied to analysis latency.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"]

CMD ["uvicorn", "skin_metrics.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# --------------------------------------------------------------------------
# full - API + Phase 2 training in the same container
#
# The model fetch is repeated per target rather than shared: it needs both the
# venv (for skin_metrics) and the OS libs (cv2 imports at module level), which
# only exist together in a final stage.
# --------------------------------------------------------------------------
FROM runtime-base AS full
COPY --from=builder-full /app/.venv /app/.venv
RUN python -c "from skin_metrics.detection.face import ensure_face_model; \
               print(ensure_face_model('$SKIN_METRICS_FACE_MODEL'))"

# --------------------------------------------------------------------------
# api - default target: lean deployment image
# --------------------------------------------------------------------------
FROM runtime-base AS api
COPY --from=builder /app/.venv /app/.venv
RUN python -c "from skin_metrics.detection.face import ensure_face_model; \
               print(ensure_face_model('$SKIN_METRICS_FACE_MODEL'))"
