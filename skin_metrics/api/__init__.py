"""HTTP API for skin_metrics (optional ``api`` extra).

Exposes the Phase 1 pipeline over HTTP: give it an image URL, get a
:class:`~skin_metrics.scoring.schema.SkinReport` back.

Install and run::

    uv sync --extra api --extra detection
    uv run skin-metrics serve --download-model

FastAPI/uvicorn are imported lazily by :func:`create_app` so that importing
``skin_metrics`` never requires the ``api`` extra.
"""

from __future__ import annotations

from .settings import ApiSettings

__all__ = ["ApiSettings", "create_app"]


def create_app(settings: ApiSettings | None = None):
    """Build the FastAPI application (lazy re-export of :mod:`skin_metrics.api.app`).

    Parameters
    ----------
    settings : ApiSettings, optional
        Runtime settings; defaults to :meth:`ApiSettings.from_env`.

    Returns
    -------
    fastapi.FastAPI
        The configured application.
    """
    from .app import create_app as _create_app

    return _create_app(settings)
