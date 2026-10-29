"""ClipEarth text encoder: the lab's query encoding (EmbeddingGemma) as a service.

The browser skips the 216 MB model download and ~2 s of WASM per query. The lab applies its aligner heads to
the vectors itself.

    POST /encode   {"model": MODEL_ID/MODEL_FILE, "texts": [...]} -> {"model", "dims", "vectors"}:
                   vectors are base64 little-endian float32, one row of `dims` per text, in order.
                   Texts carry the head's prefix already.
    GET /health    stats (no auth)

/encode needs `Authorization: Bearer <Studio API key or token>` of a staff user.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from contextlib import asynccontextmanager

from auth import Auth, AuthError
from encoder import DIMS, MODEL_FILE, MODEL_ID, Encoder
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
log = logging.getLogger("encoder")

MODEL = f"{MODEL_ID}/{MODEL_FILE}"
MODEL_DIR = os.environ.get("MODEL_DIR", "/model")
ORIGINS = [o for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o]
ORT_THREADS = int(os.environ.get("ORT_THREADS", "0"))
MAX_TEXTS = int(os.environ.get("MAX_TEXTS", "64"))
MAX_CHARS = int(os.environ.get("MAX_CHARS", "1000"))
# Local runs only (bin/ scripts, tests): deploy.sh never sets it.
NO_AUTH = os.environ.get("CLIPEARTH_ENCODER_NO_AUTH") == "1"

auth = Auth()
state: dict = {"encoder": None, "at": time.time()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once, before the port opens."""
    await auth.open()
    if state["encoder"] is None:
        state["encoder"] = Encoder(MODEL_DIR, threads=ORT_THREADS)
    yield
    await auth.close()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
if ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
        max_age=7200,
    )


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        {"error": message}, status_code=status, headers={"Cache-Control": "no-store"}
    )


@app.get("/health")
async def health():
    """Request, cache and timing stats."""
    encoder = state["encoder"]
    return {
        "ok": True,
        "model": MODEL,
        "up_s": round(time.time() - state["at"]),
        "stats": encoder.stats if encoder else None,
    }


@app.post("/encode")
async def encode(request: Request):
    """Texts to base64 float32 vectors, one row per text."""
    if not NO_AUTH:
        try:
            await auth.check(request.headers.get("authorization"))
        except AuthError as e:
            return _error(e.status, e.message)
    try:
        body = await request.json()
    except ValueError:
        return _error(400, "Body is not JSON")
    if not isinstance(body, dict):
        return _error(400, "Body is not an object")
    if body.get("model") != MODEL:
        return _error(409, f"This server encodes with {MODEL}")
    texts = body.get("texts")
    if (
        not isinstance(texts, list)
        or not texts
        or not all(isinstance(t, str) for t in texts)
    ):
        return _error(400, "texts must be a non-empty list of strings")
    if len(texts) > MAX_TEXTS:
        return _error(400, f"At most {MAX_TEXTS} texts per request")
    if any(len(t) > MAX_CHARS for t in texts):
        return _error(400, f"Each text is at most {MAX_CHARS} characters")
    vectors = await asyncio.to_thread(state["encoder"].encode, texts)
    return JSONResponse(
        {
            "model": MODEL,
            "dims": DIMS,
            "vectors": base64.b64encode(vectors.astype("<f4").tobytes()).decode(),
        },
        headers={"Cache-Control": "no-store"},
    )
