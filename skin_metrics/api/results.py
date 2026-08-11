"""Redis result store for the asynchronous analysis flow.

``POST /analyze`` and ``POST /analyze/diary`` return a ``request_id``
immediately; the analysis runs in the background and its outcome is written
under the key ``{request_id}:{kind}`` (``abc123:analyze`` / ``abc123:diary``)
so another service (the Spring Boot backend) can pick it up straight from
Redis without calling this API again.

Stored document (JSON string)::

    {"status": "processing", "request_id": ..., "kind": ..., "submitted_at": ...}
    {"status": "done",   ..., "completed_at": ..., "result": {<response body>}}
    {"status": "failed", ..., "completed_at": ..., "error": {"code", "message"}}

Redis is **required**: it is the hand-off itself, not a cache. Every entry
carries a TTL so an abandoned result cannot fill the (small) Redis Cloud
instance.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    """UTC timestamp in ISO-8601, seconds precision."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _processing(request_id: str, kind: str) -> dict[str, Any]:
    """The document written the moment a request is accepted."""
    return {
        "status": "processing",
        "request_id": request_id,
        "kind": kind,
        "submitted_at": _now(),
    }


def _connect(url: str):
    """Open an asyncio Redis client.

    Deferred import: ``redis`` ships with the ``api`` extra only. Kept as a
    module-level function so tests can substitute a stand-in client and still
    exercise the document logic in :class:`ResultStore`.

    Parameters
    ----------
    url : str
        ``redis://user:password@host:port/db``.

    Returns
    -------
    redis.asyncio.Redis
        Client decoding responses to ``str``.
    """
    import redis.asyncio as aioredis

    return aioredis.from_url(
        url,
        decode_responses=True,
        socket_timeout=5.0,
        socket_connect_timeout=5.0,
    )


class ResultStore:
    """Redis-backed store; keys are ``{request_id}:{kind}`` with a TTL.

    Parameters
    ----------
    url : str
        ``redis://user:password@host:port/db``.
    ttl : int
        Seconds a stored document stays readable.
    """

    def __init__(self, url: str, ttl: int) -> None:
        self.ttl = ttl
        self._client = _connect(url)

    async def ping(self) -> bool:
        """Return ``True`` when Redis answers."""
        try:
            return bool(await self._client.ping())
        except Exception:  # noqa: BLE001 - health probe, any failure = down
            return False

    async def start(self, request_id: str, kind: str) -> None:
        """Record a just-accepted request as processing."""
        await self._write(f"{request_id}:{kind}", _processing(request_id, kind))

    async def finish(self, request_id: str, kind: str, result: dict[str, Any]) -> None:
        """Store a successful result."""
        await self._complete(request_id, kind, "done", {"result": result})

    async def fail(self, request_id: str, kind: str, code: str, message: str) -> None:
        """Store a failure with its stable error code."""
        await self._complete(
            request_id, kind, "failed", {"error": {"code": code, "message": message}}
        )

    async def get(self, key: str) -> dict[str, Any] | None:
        """Fetch a stored document by its full ``{request_id}:{kind}`` key."""
        raw = await self._client.get(key)
        return None if raw is None else json.loads(raw)

    async def close(self) -> None:
        """Release the connection pool."""
        await self._client.aclose()

    async def _complete(
        self, request_id: str, kind: str, status: str, extra: dict[str, Any]
    ) -> None:
        """Overwrite the processing document, keeping its ``submitted_at``."""
        key = f"{request_id}:{kind}"
        base = await self.get(key) or _processing(request_id, kind)
        await self._write(key, base | {"status": status, "completed_at": _now()} | extra)

    async def _write(self, key: str, doc: dict[str, Any]) -> None:
        """Serialise a document and store it with the configured TTL."""
        await self._client.set(key, json.dumps(doc, ensure_ascii=False), ex=self.ttl)
