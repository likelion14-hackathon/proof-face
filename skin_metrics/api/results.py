"""Result stores for the asynchronous analysis flow.

``POST /analyze`` and ``POST /analyze/diary`` return a ``request_id``
immediately; the analysis runs in the background and its outcome is written to
a store under the key ``{request_id}:{kind}`` (``abc123:analyze`` /
``abc123:diary``) so another service (the Spring Boot backend) can pick it up
straight from Redis without calling this API again.

Stored document (JSON string)::

    {"status": "processing", "request_id": ..., "kind": ..., "submitted_at": ...}
    {"status": "done",   ..., "completed_at": ..., "result": {<response body>}}
    {"status": "failed", ..., "completed_at": ..., "error": {"code", "message"}}

Two implementations:

* :class:`RedisResultStore` -- production; every entry carries a TTL so an
  abandoned result cannot fill the (small) Redis Cloud instance.
* :class:`MemoryResultStore` -- fallback when ``SKIN_METRICS_REDIS_URL`` is not
  set. Results only live inside this process; fine for tests and local
  development, useless for the Spring hand-off (which is why ``/healthz``
  reports which store is active).
"""

from __future__ import annotations

import json
import time
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


class MemoryResultStore:
    """In-process dict store with the same interface as the Redis one."""

    backend = "memory"

    def __init__(self, ttl: int) -> None:
        self.ttl = ttl
        self._data: dict[str, tuple[float, str]] = {}  # key -> (expiry, json)

    async def ping(self) -> bool:
        """Always healthy; there is nothing to reach."""
        return True

    async def start(self, request_id: str, kind: str) -> None:
        """Record a just-accepted request as processing."""
        key = f"{request_id}:{kind}"
        self._set(key, _processing(request_id, kind))

    async def finish(self, request_id: str, kind: str, result: dict[str, Any]) -> None:
        """Store a successful result, keeping the original ``submitted_at``."""
        key = f"{request_id}:{kind}"
        base = await self.get(key) or _processing(request_id, kind)
        doc = base | {"status": "done", "completed_at": _now(), "result": result}
        self._set(key, doc)

    async def fail(self, request_id: str, kind: str, code: str, message: str) -> None:
        """Store a failure, keeping the original ``submitted_at``."""
        key = f"{request_id}:{kind}"
        base = await self.get(key) or _processing(request_id, kind)
        doc = base | {
            "status": "failed",
            "completed_at": _now(),
            "error": {"code": code, "message": message},
        }
        self._set(key, doc)

    async def get(self, key: str) -> dict[str, Any] | None:
        """Fetch a stored document, honouring the TTL."""
        entry = self._data.get(key)
        if entry is None:
            return None
        expiry, raw = entry
        if time.monotonic() > expiry:
            del self._data[key]
            return None
        return json.loads(raw)

    async def close(self) -> None:
        """Nothing to release."""

    def _set(self, key: str, doc: dict[str, Any]) -> None:
        self._data[key] = (time.monotonic() + self.ttl, json.dumps(doc))


class RedisResultStore:
    """Redis-backed store; keys are ``{request_id}:{kind}`` with a TTL."""

    backend = "redis"

    def __init__(self, url: str, ttl: int) -> None:
        # Deferred import: `redis` ships with the `api` extra only.
        import redis.asyncio as aioredis

        self.ttl = ttl
        self._client = aioredis.from_url(
            url,
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )

    async def ping(self) -> bool:
        """Return ``True`` when Redis answers."""
        try:
            return bool(await self._client.ping())
        except Exception:  # noqa: BLE001 - health probe, any failure = down
            return False

    async def start(self, request_id: str, kind: str) -> None:
        """Record a just-accepted request as processing."""
        key = f"{request_id}:{kind}"
        await self._client.set(key, json.dumps(_processing(request_id, kind)), ex=self.ttl)

    async def finish(self, request_id: str, kind: str, result: dict[str, Any]) -> None:
        """Store a successful result, keeping the original ``submitted_at``."""
        key = f"{request_id}:{kind}"
        base = await self.get(key) or _processing(request_id, kind)
        doc = base | {"status": "done", "completed_at": _now(), "result": result}
        await self._client.set(key, json.dumps(doc, ensure_ascii=False), ex=self.ttl)

    async def fail(self, request_id: str, kind: str, code: str, message: str) -> None:
        """Store a failure, keeping the original ``submitted_at``."""
        key = f"{request_id}:{kind}"
        base = await self.get(key) or _processing(request_id, kind)
        doc = base | {
            "status": "failed",
            "completed_at": _now(),
            "error": {"code": code, "message": message},
        }
        await self._client.set(key, json.dumps(doc, ensure_ascii=False), ex=self.ttl)

    async def get(self, key: str) -> dict[str, Any] | None:
        """Fetch a stored document by its full ``{request_id}:{kind}`` key."""
        raw = await self._client.get(key)
        return None if raw is None else json.loads(raw)

    async def close(self) -> None:
        """Release the connection pool."""
        await self._client.aclose()


def make_store(url: str | None, ttl: int):
    """Build the store the settings ask for.

    Parameters
    ----------
    url : str or None
        ``SKIN_METRICS_REDIS_URL``; ``None`` selects the in-memory fallback.
    ttl : int
        Seconds a stored result stays readable.

    Returns
    -------
    MemoryResultStore or RedisResultStore
        The configured store.
    """
    if url:
        return RedisResultStore(url, ttl)
    return MemoryResultStore(ttl)
