"""FastAPI application exposing the Phase 1 pipeline over HTTP.

Endpoints
---------
``GET  /healthz``          liveness + whether face detection is actually usable.
``POST /analyze``          ``{"image_url": ...}`` -> 0-100 ``pigmentation`` / ``erythema`` / ``hydration``.
``POST /analyze/simple``   same body -> 0-10 ``skin_tone`` / ``dryness`` / ``redness``.

Both responses share the same flat envelope: scores + ``confidence`` +
``warnings`` + ``elapsed_ms`` + ``version`` + ``disclaimer``. The full
:class:`~skin_metrics.scoring.schema.SkinReport` is CLI-only.

Run it with::

    uv run skin-metrics serve --download-model
    # or: uvicorn skin_metrics.api.app:app --host 127.0.0.1 --port 8000

Every error response uses the ``{"error": {"code", "message"}}`` envelope.
The analysis itself is CPU-bound and synchronous, so it runs in a worker thread
behind a semaphore (``SKIN_METRICS_API_MAX_CONCURRENCY``).
"""

from __future__ import annotations

import functools
import time
from contextlib import asynccontextmanager

import anyio
import anyio.to_thread
import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .. import DISCLAIMER, __version__
from ..config import load_config
from ..pipeline import analyze as run_analyze
from .fetch import ImageFetchError, fetch_image
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ErrorResponse,
    HealthResponse,
    SimpleAnalyzeResponse,
)
from .settings import ApiSettings


class AnalysisError(Exception):
    """Pipeline failure carrying the HTTP status/code it should map to."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message

_ERROR_RESPONSES: dict[int | str, dict] = {
    400: {"model": ErrorResponse, "description": "Invalid URL or undecodable image."},
    403: {"model": ErrorResponse, "description": "URL points at a non-public address."},
    413: {"model": ErrorResponse, "description": "Image exceeds the byte/pixel limit."},
    422: {"model": ErrorResponse, "description": "No face found, or the image cannot be scored."},
    502: {"model": ErrorResponse, "description": "Upstream image host failed."},
    503: {"model": ErrorResponse, "description": "Face detection is unavailable on the server."},
    504: {"model": ErrorResponse, "description": "Fetching the image timed out."},
}


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    """Build a JSON error response in the standard envelope."""
    return JSONResponse(
        status_code=status_code, content={"error": {"code": code, "message": message}}
    )


def _detection_available() -> bool:
    """Return ``True`` if the ``detection`` extra (MediaPipe) is importable."""
    from importlib.util import find_spec

    try:
        return find_spec("mediapipe") is not None
    except (ImportError, ValueError):  # pragma: no cover - broken install
        return False


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    """Create the FastAPI app.

    Parameters
    ----------
    settings : ApiSettings, optional
        Runtime settings; defaults to :meth:`ApiSettings.from_env`.

    Returns
    -------
    fastapi.FastAPI
        App with ``/healthz`` and ``/analyze`` mounted.
    """
    cfg = settings or ApiSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Load config, resolve the face model, and hold one shared HTTP client."""
        from ..detection.face import ensure_face_model, resolve_face_model

        app.state.config = load_config(cfg.config_path)
        model_path = cfg.face_model_path
        if cfg.download_model:
            model_path = str(ensure_face_model(model_path))
        elif model_path is None:
            resolved = resolve_face_model()
            model_path = str(resolved) if resolved is not None else None
        app.state.face_model_path = model_path
        app.state.limiter = anyio.Semaphore(cfg.max_concurrency)
        async with httpx.AsyncClient(
            timeout=cfg.fetch_timeout,
            follow_redirects=False,
            headers={"user-agent": f"skin-metrics/{__version__}"},
        ) as client:
            app.state.http = client
            yield

    app = FastAPI(
        title="skin-metrics API",
        version=__version__,
        description=(
            "Pigmentation / erythema / hydration-proxy scoring from a single face "
            "image URL.\n\n**NOT a medical device.** " + DISCLAIMER
        ),
        lifespan=lifespan,
    )
    app.state.settings = cfg

    async def _run_report(payload: AnalyzeRequest, request: Request):
        """Fetch the image and score it; shared by both analyze endpoints.

        Parameters
        ----------
        payload : AnalyzeRequest
            Validated request body.
        request : fastapi.Request
            Incoming request (for app state).

        Returns
        -------
        SkinReport
            The scored report.

        Raises
        ------
        ImageFetchError
            Propagated from :func:`fetch_image` (handled app-wide).
        AnalysisError
            When the pipeline cannot score the image.
        """
        state = request.app.state
        image, _meta = await fetch_image(str(payload.image_url), cfg, client=state.http)
        bbox = tuple(payload.reference_bbox) if payload.reference_bbox else None

        work = functools.partial(
            run_analyze,
            image,
            ref_bbox=bbox,
            model_path=state.face_model_path,
            config=state.config,
        )
        try:
            async with state.limiter:
                report = await anyio.to_thread.run_sync(work)
        except ValueError as exc:
            # Pipeline refuses to score: no face, or every ROI failed the gate.
            raise AnalysisError(422, "analysis_failed", str(exc)) from exc
        except FileNotFoundError as exc:
            raise AnalysisError(503, "face_model_missing", str(exc)) from exc
        except ImportError as exc:
            raise AnalysisError(503, "detection_unavailable", str(exc)) from exc
        return report

    @app.exception_handler(ImageFetchError)
    async def _fetch_error_handler(request: Request, exc: ImageFetchError):
        """Map fetch/decode failures onto the error envelope."""
        return _error(exc.status_code, exc.code, exc.message)

    @app.exception_handler(AnalysisError)
    async def _analysis_error_handler(request: Request, exc: AnalysisError):
        """Map pipeline failures onto the error envelope."""
        return _error(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError):
        """Return body-validation failures in the same envelope as everything else."""
        return _error(422, "invalid_request", str(exc.errors()))

    @app.get("/healthz", response_model=HealthResponse, tags=["meta"])
    async def healthz(request: Request) -> HealthResponse:
        """Report liveness and whether an image URL could actually be analyzed."""
        return HealthResponse(
            face_model_available=request.app.state.face_model_path is not None,
            detection_available=_detection_available(),
        )

    @app.post(
        "/analyze",
        response_model=AnalyzeResponse,
        responses=_ERROR_RESPONSES,
        tags=["analysis"],
    )
    async def analyze(payload: AnalyzeRequest, request: Request):
        """Fetch ``image_url`` and return its three 0-100 scores.

        The image is downloaded server-side under the SSRF/size limits in
        :mod:`skin_metrics.api.fetch`, then scored by
        :func:`skin_metrics.pipeline.analyze` in a worker thread. The full
        report is flattened to the same envelope shape as ``/analyze/simple``.
        """
        started = time.perf_counter()
        report = await _run_report(payload, request)
        return AnalyzeResponse.from_report(
            report, elapsed_ms=round((time.perf_counter() - started) * 1000.0, 2)
        )

    @app.post(
        "/analyze/simple",
        response_model=SimpleAnalyzeResponse,
        responses=_ERROR_RESPONSES,
        tags=["analysis"],
    )
    async def analyze_simple(payload: AnalyzeRequest, request: Request):
        """Fetch ``image_url`` and return three consumer 0-10 scores.

        Same fetch/limits/pipeline as ``/analyze``; the full report is then
        collapsed by :meth:`SimpleAnalyzeResponse.from_report` into
        ``skin_tone`` / ``dryness`` (당김·건조함) / ``redness``.
        """
        started = time.perf_counter()
        report = await _run_report(payload, request)
        return SimpleAnalyzeResponse.from_report(
            report, elapsed_ms=round((time.perf_counter() - started) * 1000.0, 2)
        )

    return app


app = create_app()
