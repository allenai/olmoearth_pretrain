"""Staff only: the caller's Studio API key or token, checked against Studio's users/me.

A copy of gea's embedding_heads/tileserver/auth.py (the Embedding Maps tile server).

Answers are cached per key (by its SHA-256, never the key itself) for ten minutes, refusals for
one. Keys are never logged.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time

import aiohttp

STUDIO_API = os.environ.get("STUDIO_API", "https://olmoearth.allenai.org/api/v1")
OK_TTL = 600.0
NO_TTL = 60.0


class AuthError(Exception):
    """A refusal with its HTTP status."""

    def __init__(self, status: int, message: str):
        """Status and message for the response."""
        super().__init__(message)
        self.status = status
        self.message = message


class Auth:
    """Staff checks against Studio, cached per key."""

    def __init__(self):
        """No session until open()."""
        self.session: aiohttp.ClientSession | None = None
        self.cache: dict[str, tuple[int, float]] = {}  # digest -> (status, expires)
        self.pending: dict[str, asyncio.Future] = {}

    async def open(self):
        """Start the HTTP session."""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=20)
            )

    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def _ask(self, key: str) -> int:
        """200 for staff, 403 for another role, 401 for a key Studio refuses."""
        assert self.session is not None
        async with self.session.get(
            f"{STUDIO_API}/users/me", headers={"Authorization": f"Bearer {key}"}
        ) as r:
            if r.status in (401, 403):
                return 401
            if r.status != 200:
                raise AuthError(502, f"Studio answered {r.status}")
            body = await r.json()
        records = body.get("records") or []
        return 200 if records and records[0].get("role") == "staff" else 403

    async def check(self, authorization: str | None) -> None:
        """Raise AuthError unless the bearer key belongs to a staff user."""
        if not authorization or not authorization.lower().startswith("bearer "):
            raise AuthError(401, "Sign in with a Studio API key")
        key = authorization[7:].strip()
        digest = hashlib.sha256(key.encode()).hexdigest()
        now = time.time()
        hit = self.cache.get(digest)
        if hit is None or hit[1] < now:
            task = self.pending.get(digest)
            if task is None:
                task = asyncio.ensure_future(self._ask(key))
                self.pending[digest] = task
                task.add_done_callback(lambda _: self.pending.pop(digest, None))
            status = await asyncio.shield(task)
            hit = (status, time.time() + (OK_TTL if status == 200 else NO_TTL))
            self.cache[digest] = hit
            if len(self.cache) > 10_000:
                self.cache = {k: v for k, v in self.cache.items() if v[1] > now}
        if hit[0] == 401:
            raise AuthError(401, "Studio did not accept this key")
        if hit[0] == 403:
            raise AuthError(403, "ClipEarth is for Ai2 staff")
