"""FastAPI application exposing the Phase 1 pipeline over HTTP.

Endpoints
---------
``GET  /healthz``          liveness + face detection + result-store status.
``POST /analyze``          submit an image URL -> 202 ``{"request_id", "redis_key"}``.
``POST /analyze/diary``    same, scored on the 0-10 diary axes.
``GET  /results/{key}``    read back a stored result document (debugging).

The analysis itself is **asynchronous**: a POST validates the URL, returns a
``request_id`` immediately, and runs fetch + scoring in a background task.
When it finishes -- success or failure -- the outcome is written to Redis under
``{request_id}:analyze`` / ``{request_id}:diary`` (see
:mod:`skin_metrics.api.results` for the document shape), where the consuming
service (Spring Boot) picks it up directly.

Run it with::

    uv run skin-metrics serve --download-model
    # or: uvicorn skin_metrics.api.app:app --host 127.0.0.1 --port 8000

Every error response uses the ``{"error": {"code", "message"}}`` envelope.
The scoring is CPU-bound and synchronous, so it runs in a worker thread behind
a semaphore (``SKIN_METRICS_API_MAX_CONCURRENCY``); queued submissions simply
wait their turn.
"""

from __future__ import annotations

import asyncio
import functools
import uuid
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
from .fetch import ImageFetchError, fetch_image, validate_url
from .results import make_store
from .schemas import (
    AcceptedResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    DiaryAnalyzeResponse,
    ErrorResponse,
    HealthResponse,
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
    400: {"model": ErrorResponse, "description": "Invalid URL."},
    403: {"model": ErrorResponse, "description": "URL points at a non-public address."},
    422: {"model": ErrorResponse, "description": "Request body failed validation."},
    503: {"model": ErrorResponse, "description": "Result store is unreachable."},
}

# Maps a metric kind to the response model the background worker stores.
_RESULT_MODELS = {"analyze": AnalyzeResponse, "diary": DiaryAnalyzeResponse}


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
        App with ``/healthz``, ``/analyze``, ``/analyze/diary`` and
        ``/results/{key}`` mounted.
    """
    cfg = settings or ApiSettings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Load config, resolve the face model, connect the result store."""
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
        app.state.results = make_store(cfg.redis_url, cfg.result_ttl)
        # Strong references to in-flight background tasks: asyncio only keeps
        # weak ones, so an unreferenced task can be garbage-collected mid-run.
        app.state.tasks = set()
        async with httpx.AsyncClient(
            timeout=cfg.fetch_timeout,
            follow_redirects=False,
            headers={"user-agent": f"skin-metrics/{__version__}"},
        ) as client:
            app.state.http = client
            try:
                yield
            finally:
                # Let queued analyses finish writing their results before the
                # store goes away; cap the wait so shutdown cannot hang.
                if app.state.tasks:
                    await asyncio.wait(app.state.tasks, timeout=30)
                await app.state.results.close()

    app = FastAPI(
        title="skin-metrics API",
        version=__version__,
        description=(
            "Pigmentation / erythema / hydration-proxy scoring from a single face "
            "image URL. Submissions are asynchronous: POST returns a request_id "
            "and the result lands in Redis as {request_id}:analyze / "
            "{request_id}:diary.\n\n**NOT a medical device.** " + DISCLAIMER
        ),
        lifespan=lifespan,
    )
    app.state.settings = cfg

    async def _run_report(payload: AnalyzeRequest, app: FastAPI):
        """Fetch the image and score it; runs inside a background task.

        Parameters
        ----------
        payload : AnalyzeRequest
            Validated request body.
        app : fastapi.FastAPI
            The application (for config, model path, limiter, HTTP client).

        Returns
        -------
        SkinReport
            The scored report.

        Raises
        ------
        ImageFetchError
            From :func:`fetch_image` (bad host, oversized, undecodable...).
        AnalysisError
            When the pipeline cannot score the image.
        """
        state = app.state
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

    async def _process(request_id: str, kind: str, payload: AnalyzeRequest) -> None:
        """Background worker: analyze and store the outcome under its key."""
        store = app.state.results
        try:
            report = await _run_report(payload, app)
            body = _RESULT_MODELS[kind].from_report(report)
            await store.finish(request_id, kind, body.model_dump())
        except (ImageFetchError, AnalysisError) as exc:
            await store.fail(request_id, kind, exc.code, exc.message)
        except Exception as exc:  # noqa: BLE001 - must never lose a result
            await store.fail(request_id, kind, "internal_error", str(exc))

    async def _submit(payload: AnalyzeRequest, request: Request, kind: str) -> JSONResponse:
        """Accept a submission: validate cheaply, mark processing, spawn work."""
        # Cheap failures (bad scheme, blocked host) should be synchronous 4xx,
        # not a "failed" document the consumer has to poll for.
        validate_url(str(payload.image_url), allow_private_hosts=cfg.allow_private_hosts)

        request_id = uuid.uuid4().hex
        state = request.app.state
        try:
            await state.results.start(request_id, kind)
        except Exception as exc:  # noqa: BLE001 - store down = can't accept work
            return _error(503, "result_store_unavailable", str(exc))

        task = asyncio.create_task(_process(request_id, kind, payload))
        state.tasks.add(task)
        task.add_done_callback(state.tasks.discard)

        body = AcceptedResponse(request_id=request_id, redis_key=f"{request_id}:{kind}")
        return JSONResponse(status_code=202, content=body.model_dump())

    @app.exception_handler(ImageFetchError)
    async def _fetch_error_handler(request: Request, exc: ImageFetchError):
        """Map fetch/validation failures onto the error envelope."""
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
        """Report liveness, detection readiness, and the result-store state."""
        store = request.app.state.results
        if store.backend == "redis":
            store_status = "redis" if await store.ping() else "redis_unreachable"
        else:
            store_status = "memory"
        return HealthResponse(
            face_model_available=request.app.state.face_model_path is not None,
            detection_available=_detection_available(),
            result_store=store_status,
        )

    @app.post(
        "/analyze",
        status_code=202,
        response_model=AcceptedResponse,
        responses=_ERROR_RESPONSES,
        tags=["analysis"],
    )
    async def analyze(payload: AnalyzeRequest, request: Request):
        """Submit ``image_url`` for 0-100 scoring.

        Returns 202 with a ``request_id`` immediately. The result document
        (shape: :class:`AnalyzeResponse` inside the store envelope) appears in
        Redis under ``{request_id}:analyze`` when the analysis completes.
        """
        return await _submit(payload, request, "analyze")

    @app.post(
        "/analyze/diary",
        status_code=202,
        response_model=AcceptedResponse,
        responses=_ERROR_RESPONSES,
        tags=["analysis"],
    )
    async def analyze_diary(payload: AnalyzeRequest, request: Request):
        """Submit ``image_url`` for the 0-10 diary scores.

        Same flow as ``/analyze``; the result document (shape:
        :class:`DiaryAnalyzeResponse` -- ``skin_tone`` / ``dryness`` (당김·건조함)
        / ``redness``) lands under ``{request_id}:diary``.
        """
        return await _submit(payload, request, "diary")

    @app.get("/results/{key}", tags=["analysis"])
    async def get_result(key: str, request: Request):
        """Read a stored result document by its full ``{request_id}:{kind}`` key.

        The Spring Boot consumer should read Redis directly; this endpoint
        exists so a result can be checked with nothing but curl.
        """
        doc = await request.app.state.results.get(key)
        if doc is None:
            return _error(404, "result_not_found", f"No result under key '{key}'.")
        return doc

    return app


app = create_app()
