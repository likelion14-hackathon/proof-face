"""Runtime settings for the HTTP API, read from environment variables.

All variables are prefixed ``SKIN_METRICS_API_`` except the pre-existing
``SKIN_METRICS_FACE_MODEL`` (shared with
:func:`skin_metrics.detection.face.resolve_face_model`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_TRUTHY = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


def _env_int(name: str, default: int) -> int:
    """Read an int environment variable, falling back on parse failure."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable, falling back on parse failure."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class ApiSettings:
    """Configuration for the HTTP API.

    Attributes
    ----------
    config_path : str or None
        Path to a ``config.yaml``; ``None`` uses the packaged default.
    face_model_path : str or None
        Explicit ``face_landmarker.task`` path (MediaPipe Tasks API).
    download_model : bool
        Download the FaceLandmarker model at startup if missing.
    max_bytes : int
        Hard cap on downloaded image bytes.
    max_pixels : int
        Hard cap on decoded pixel count (decompression-bomb guard). Images over
        this are rejected outright.
    analysis_max_pixels : int
        Pixel budget handed to the pipeline. Anything larger is downscaled to
        fit rather than rejected, because peak memory scales with the decoded
        size (~63 MB per megapixel: a 40 MP image peaks at 2.5 GB, and
        ``max_concurrency`` of those at once will OOM a 4 GB host). Resolution
        above this budget buys no accuracy either -- the pipeline normalises
        every face to ``normalization.target_eye_span_px`` before extracting
        features. A face that ends up too small after the downscale is flagged
        ``under_resolved`` by the pipeline exactly as it would be natively.
    fetch_timeout : float
        Per-request timeout (seconds) for the image download.
    max_redirects : int
        Redirect hops to follow; every hop is re-validated against the SSRF rules.
    allow_private_hosts : bool
        Permit loopback/private/link-local targets. **Development only** — this
        disables the SSRF guard.
    max_concurrency : int
        Concurrent analyses; the pipeline is CPU-bound and runs in a thread pool.
    redis_url : str or None
        ``redis://user:password@host:port/db`` for the asynchronous result
        hand-off. Unset selects an in-process store (dev/tests only -- results
        are then invisible to any other service).
    result_ttl : int
        Seconds an analysis result stays readable in the store.
    """

    config_path: str | None = None
    face_model_path: str | None = None
    download_model: bool = False
    max_bytes: int = 20 * 1024 * 1024
    max_pixels: int = 40_000_000
    analysis_max_pixels: int = 16_000_000
    fetch_timeout: float = 10.0
    max_redirects: int = 3
    allow_private_hosts: bool = False
    max_concurrency: int = 2
    redis_url: str | None = None
    result_ttl: int = 3600

    @classmethod
    def from_env(cls) -> "ApiSettings":
        """Build settings from ``SKIN_METRICS_API_*`` environment variables.

        Returns
        -------
        ApiSettings
            Settings with environment overrides applied over the defaults.
        """
        return cls(
            config_path=os.environ.get("SKIN_METRICS_API_CONFIG") or None,
            face_model_path=os.environ.get("SKIN_METRICS_FACE_MODEL") or None,
            download_model=_env_bool("SKIN_METRICS_API_DOWNLOAD_MODEL", False),
            max_bytes=_env_int("SKIN_METRICS_API_MAX_BYTES", 20 * 1024 * 1024),
            max_pixels=_env_int("SKIN_METRICS_API_MAX_PIXELS", 40_000_000),
            analysis_max_pixels=max(
                1, _env_int("SKIN_METRICS_API_ANALYSIS_MAX_PIXELS", 16_000_000)
            ),
            fetch_timeout=_env_float("SKIN_METRICS_API_FETCH_TIMEOUT", 10.0),
            max_redirects=_env_int("SKIN_METRICS_API_MAX_REDIRECTS", 3),
            allow_private_hosts=_env_bool("SKIN_METRICS_API_ALLOW_PRIVATE_HOSTS", False),
            max_concurrency=max(1, _env_int("SKIN_METRICS_API_MAX_CONCURRENCY", 2)),
            redis_url=os.environ.get("SKIN_METRICS_REDIS_URL") or None,
            result_ttl=max(1, _env_int("SKIN_METRICS_RESULT_TTL", 3600)),
        )
