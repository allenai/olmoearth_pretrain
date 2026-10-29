# ClipEarth text encoder

The text-search lab's query encoding (EmbeddingGemma-300m, q4 ONNX) as a Cloud Run service, in `olmoearth_pretrain/clipearth/encoder_service/`. The lab
(`ui/labs/text-search` in allenai/olmoearth_studio) calls it when built with
`VITE_CLIPEARTH_ENCODER_URL`, so the browser skips the 216 MB model and ~2 s of WASM per query, and
applies its aligner heads to the vectors itself. If the service fails, the lab encodes in the browser.

- `POST /encode` `{"model": "onnx-community/embeddinggemma-300m-ONNX/model_no_gather_q4", "texts": [...]}`
  → `{"model", "dims": 768, "vectors": <base64 little-endian float32, one row per text>}`. Texts carry
  the head's prefix. Staff only: `Authorization: Bearer <Studio API key or token>` (`auth.py`, a copy of
  the tile server's).
- `GET /health`: request, cache and timing stats.

## Deploy

```bash
./deploy.sh   # earthsystem-dev-c3po / us-central1 / clipearth-encoder; 4 vCPU, 4 GiB, 1-3 instances
```

Then build Studio with `VITE_CLIPEARTH_ENCODER_URL=<the service URL>`.

## Local run and tests

```bash
python -m pytest --noconftest tests/unit/clipearth/test_encoder_service.py   # from the repo root, with requirements.txt installed
CLIPEARTH_ENCODER_NO_AUTH=1 MODEL_DIR=<snapshot of the model> ALLOWED_ORIGINS=http://localhost:3000 \
  uvicorn app:app --port 8090
```

`CLIPEARTH_ENCODER_NO_AUTH=1` is for local runs only; `deploy.sh` never sets it.

## Measured (2026-10-29, hammond, 4 ONNX threads)

- 60-100 ms per new query, ~1 ms for a cached one; 5 texts in one request 0.28 s.
- Against transformers.js 4.3.1 in Chromium on 6 queries: cosine >= 0.9998, max element difference 0.003.
- Lab, headless Chromium, 10 m search: first search 11.3 s → 4.4 s (no model download), later
  searches 2.6 s → 1.1-1.4 s.
