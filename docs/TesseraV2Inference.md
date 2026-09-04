# Running TESSERA v2 inference ourselves

Status (2026-07-27): TESSERA v2 **weights + inference code are public**
(github.com/ucam-eo/tessera, branch `v2`; HF org `geotessera`), but **v2
precomputed embeddings are not** — the distribution bucket
(`s3://tessera-embeddings/`) holds only `v1/` and `v1.1/`, and v2 embeddings
are pre-request-only while their infra ramps up. To eval PASTIS against v2 we
must run their inference ourselves. This doc is the plan plus time / storage /
FLOPs estimates. It was written for PASTIS, which is where the pipeline was
built; the same pipeline now runs on any rslearn eval dataset — see
"Other datasets" below for `africa_crop_mask_year_aligned` and
`ethiopia_crops_year_aligned`.

All Tessera facts below were read from their released code (`tessera_infer_v2/`,
`student/{model,infer}.py`, `teacher/model.py`, Rust stackers), model cards, and
the v2 paper (arXiv:2607.03949). OlmoEarth numbers were measured from this repo
(`all_evals.py` ws16/ps1 PASTIS tasks, `base_shallow_decoder` = 768d/12L/12H,
encoder ≈ 89M params exercised at eval).

## What v2 inference needs

One forward pass = **one pixel's year** of observations, dual-branch temporal
transformer (S2 branch + merged-S1 branch, attention-pooled over time, concat +
MLP → 128-d Matryoshka embedding; teacher → 1024-d, not truncatable):

| Model  | Params | Layers/branch | d_model | FFN  | ckpt size |
|--------|--------|---------------|---------|------|-----------|
| nano   | 1.07M  | 2             | 144     | 384  | 4 MB      |
| small  | 7.11M  | 4             | 256     | 1024 | 28 MB     |
| medium | 21.0M  | 4             | 440     | 1792 | 84 MB     |
| large  | 43.8M  | 4             | 640     | 2560 | 175 MB    |
| teacher| 2.06B  | 4 (+2 fusion) | 4096    | 16384| 8.26 GB   |

Inputs per pixel: **all** valid acquisitions in the calendar year —
S2 L2A 10 bands (order B04,B02,B03,B08,B8A,B05,B06,B07,B11,B12, raw DN) with
per-pixel SCL-derived cloud mask; S1 RTC VV/VH, ascending + descending kept
separate. Per-pixel valid counts are bucketized to bins {8,16,…,256} and
batched by bin. For PASTIS France 2019 expect T_s2(valid) ≈ 24–48 and
T_s1 ≈ 128–136 (S1A+S1B, multiple orbits) → ~160–190 tokens per pixel total.

Their pipeline: `s1_s2_downloader.sh` (data source **must be MPC for 2019** —
the AWS path uses OPERA RTC-S1 which only exists from ~2021) → Rust
`s2_stack`/`s1_stack` → `dpixel_retiler.py` → per-tile "d-pixel" npy dirs →
`python infer_v2.py --model medium --data-root … --out-dir …` → per-tile
`(H,W,128)` float32 npy (or `--int8` + per-pixel scales).

## Plan — IMPLEMENTED (rslearn route)

Scope: **PASTIS patch footprints only** (2,433 128×128 windows ≈ 3,990 km² ≈
8% of the 4-tile area). Instead of their bash + Rust preprocessing we reuse
rslearn against Planetary Computer — the fetch is expressed as three
all-scenes layers in `config_tessera_v2_fetch.json` (`sentinel2_l2a_all` with
an SCL band set, `sentinel1_ascending_all`, `sentinel1_descending_all`;
`space_mode=INTERSECTS`, `sort_by=datetime`, one item group per acquisition;
declared inline in `config_pastis_rslearn.json`, which is where they were
first written and which a unit test holds byte-identical to the shared file),
and inference is `olmoearth_pretrain/evals/datasets/tessera_v2_export.py`
running the vendored v2 student
(`olmoearth_pretrain/evals/models/tessera/tessera_v2_{model,infer}.py`,
vendored from their repo; param counts reproduce the model cards exactly).
Their preprocessing is replicated faithfully: SCL classes {0,1,2,3,8,9} →
cloud-invalid, S1 stored as `clip((20*log10(raw_MPC_power) + 50) * 200, 0,
32767)` int16 with 0 as the missing sentinel (verified against their
`s1_fast_processor.py`/Rust stacker sources), per-pixel valid-count
bucketization to {8,…,256} via their own vendored `encode_tile`.

Runbook (details in the script docstring):

```
export DS_PATH=/weka/dfive-default/rslearn-eai/datasets/pastis_rslearn
# 1. Calendar-2019 fetch windows mirroring the eval windows (group pastis):
python -m olmoearth_pretrain.evals.datasets.tessera_v2_export create_windows \
    --ds_path $DS_PATH --dataset pastis_rslearn
# 2. Fetch every 2019 acquisition (restrict to the new layers!):
rslearn dataset prepare     --root $DS_PATH --group pastis_tessera_v2 --workers 16 \
    --no-use-initial-job --retry-max-attempts 12 --retry-backoff-seconds 2 \
    --enabled-layers sentinel2_l2a_all,sentinel1_ascending_all,sentinel1_descending_all
rslearn dataset materialize ... (same flags, --retry-backoff-seconds 60)
# or fan out with scripts/tools/launch_year_aligned_prepare.sh (LAYER_SET=tessera_v2_fetch)
# 3. Student weights (HF geotessera/TESSERA-V-2.0-2B-L); already on weka at
#    /weka/dfive-default/helios/models/tessera_v2/ckpt/student_large.pt
python -m olmoearth_pretrain.evals.datasets.tessera_v2_export infer \
    --ds_path $DS_PATH --dataset pastis_rslearn \
    --checkpoint_path $CKPT --model_size large
# 4. Eval (identical probes/splits/metrics as AEF):
python -m olmoearth_pretrain.internal.embedding_eval_sweep --cluster=... --model=tessera_v2_precomputed
```

### v2 is the one baseline that must be quantized at eval time

**Read this before launching any v2 eval.** The AEF baseline already carries
its product's int8 loss in the stored values — the GSE fetcher dequantizes int8
COGs — so the sweeps pass it through unquantized, which scores it at the
precision it ships. **v2 does not**, because we bake it ourselves and
`infer_v2.py` defaults to float32 (`--int8` is opt-in). The shipped v2 product
*is* int8, via quantization-aware training, so an unquantized v2 arm is scored
**above its own release precision** — which is what the 2026-08-07 and
2026-08-11 africa/ethiopia sweeps did, flattering v2 on exactly the cell where
it beats us hardest.

Two constants in `full_eval_sweep.py` handle this, and both sweeps read them:
`QUANTIZE_AT_EVAL_MODALITIES` (holds `tessera_v2` alone) and
`QUANTIZE_SCHEME_BY_MODALITY` (maps it to `TESSERA_PER_VECTOR`). **Do not**
re-add a blanket `quantize_embeddings=False` for precomputed arms; if a future
product is downloaded rather than self-baked, leave it out of both.

**Each product is quantized under its own scheme, which matters more than the
flag.** Our default quantizer is AlphaEarth's published fixed-scale power scheme
(`POWER=2.0`, `SCALE=127.5`), which clips at |x| > ~0.992 and therefore fits
AEF's unit-L2 64-d vectors. The v2 student ends in a **non-affine LayerNorm**
(`tessera_v2_model.py`) at per-coordinate std ~1, so scoring it under the power
scheme would clip ~a third of its coordinates and cost it ~5 points of
round-trip cosine that its real product never pays. Tessera's own scheme is
linear with one float32 scale **per pixel** — verified against the shipped
client, `geotessera/store.py:207-216`,
`emb_int8.astype(np.float32) * scales[np.newaxis, :, :]`, published as
`grid_{lon}_{lat}.npy` + `_scales.npy` — so it is clip-free by construction.
`QuantizationScheme.TESSERA_PER_VECTOR` reproduces it, and
`TestTesseraQuantization::test_matches_geotessera_decoder` asserts our
dequantization is bit-identical to theirs (skipped if `geotessera` is not
installed). Because the per-vector scales are needed to reconstruct, that scheme
returns round-tripped **float32** rather than int8 codes, and the callback's
later dequantize step skips it.

Only their decoder ships in the client, so our encoder is the natural inverse
(`scale = max|x| / 127`); any per-vector scale is clip-free, so the choice moves
the step size by a few percent, not the conclusion.

**Still outstanding, and charged to us:** our own register bottleneck ends in the
same LayerNorm geometry (`flexi_vit.py:1917`), so the power scheme clips ~32% of
*our* coordinates on every embedding-eval number in every dashboard — cosine
0.95 against AEF's 0.9999. Fix on our side with `CENTER_L2` before the round-trip
(what `EmbeddingNormalization` recommends) or per-vector quantization; note the
bottleneck's `unit_norm` option does **not** fix it, because
`unit_norm_scale` defaults to `sqrt(register_dim)`, not 1.
`quantization_clip_stats` in `embedding_diagnostics.py` measures the real
fraction; nothing has logged it yet.

`--retry-backoff-seconds` is deliberately small for prepare: rslearn sleeps
`backoff * (attempt+1) * random(1..2)`, so 60 parks a worker for 60-120s on its
first (routine) 403 and collapses effective parallelism to ~1 worker. Keep 60
for materialize, where a retry covers a real mid-download failure.

The `tessera_v2` modality/layer plumbing (constants, datatypes, model.yaml,
registry, `tessera_v2_precomputed` baseline) is in place. NOTE: the `*_all`
fetch layers exist in
the shared dataset config, so any prepare/materialize of OTHER groups must
keep using `--enabled-layers` (or the defaults per layer type) to avoid
fetching a year of scenes for the eval windows.

Alternative considered and rejected as primary path: building d-pixels straight
from PASTIS-R's own arrays (it ships 43 S2 dates + 65 asc/70 desc S1 dates on
the exact patch grids, matching band sets). Zero download, ~1 day total — but
PASTIS reflectances come from a different processing chain (Theia), there are
no SCL cloud masks, and S1 units differ from their int16-scaled RTC, so the
hardcoded v2 normalization stats would be off-distribution. Keep as a fallback
/ sanity check only. Open question either way: v2 ships a single set of
normalization stats without stating whether they match MPC or AWS sources
(v1.1 shipped separate checkpoints per source); pre-2021 their own global runs
can only have used MPC, so MPC is the safer bet.

## Other datasets: africa_crop_mask_year_aligned, ethiopia_crops_year_aligned

Why these two: the published Tessera products are global for 2024 only,
reaching back to 2017 for the US/EU, so neither dataset can carry a reportable
Tessera number from a download. Running v2 ourselves is the only way to get a
Tessera column on non-US, non-2024 data, and it lands at 100% coverage by
construction.

Two things make this simpler than PASTIS. The `*_year_aligned` copies were
already re-anchored to `(Jan 1 Y, Jan 1 Y+1)`
(`scripts/tools/reanchor_year_aligned_dataset.py`), so **the eval window's own
range is the product year** — `create_windows` reads it per window instead of
taking a `--year`, which is what a multi-year dataset needs. And the windows
are 32×32 rather than 128×128: 2,556 + 2,530 windows ≈ **5.2M pixels total**,
13% of PASTIS, so student inference is minutes.

Runbook (weka-side; `$NAME` is `africa_crop_mask_year_aligned` or
`ethiopia_crops_year_aligned`):

```
STAGE=/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/$NAME
EVAL=/weka/dfive-default/olmoearth/eval_datasets/$NAME
EXPORT="python -m olmoearth_pretrain.evals.datasets.tessera_v2_export"
CKPT=/weka/dfive-default/helios/models/tessera_v2/ckpt/student_large.pt

# 1. Write the standalone fetch config (the *_all layers stay OUT of the
#    dataset's own config.json; prepare/materialize take them via --config).
$EXPORT write_fetch_config --ds_path $STAGE

# 2. Fetch group, one window per eval window, on that window's calendar year.
$EXPORT create_windows --ds_path $STAGE --dataset $NAME
#    -> logs the year histogram; sanity-check it against the label years.

# 3. Fetch a year of scenes for THAT GROUP ONLY.
LAUNCH=1 LAYER_SET=tessera_v2_fetch GROUP=${NAME%_year_aligned}_tessera_v2 \
    ONLY=$NAME HOSTS=<host>,<host> scripts/tools/launch_year_aligned_prepare.sh
LAUNCH=1 COMMAND=materialize LAYER_SET=tessera_v2_fetch \
    GROUP=${NAME%_year_aligned}_tessera_v2 ONLY=$NAME HOSTS=... \
    scripts/tools/launch_year_aligned_prepare.sh

# 4. Inference: read scenes from staging, write the layer + manifest into the
#    ingested copy model.yaml actually points at (same windows, same grids).
$EXPORT infer --ds_path $STAGE --eval_ds_path $EVAL --dataset $NAME \
    --checkpoint_path $CKPT --model_size large

# 5. Wire the layer up, then re-stamp provenance and commit.
python scripts/tools/wire_embedding_modalities.py \
    --datasets $NAME --products tessera_v2 --required
python scripts/tools/backfill_eval_registry_provenance.py

# 6. Eval.
python -m olmoearth_pretrain.internal.embedding_eval_sweep \
    --cluster=... --model=tessera_v2_precomputed
```

Notes, in rough order of how likely they are to bite:

- **The student is `large`**, at
  `/weka/dfive-default/helios/models/tessera_v2/ckpt/student_large.pt`. That is
  what PASTIS was run with (`product_version: v2-large` in
  `/weka/dfive-default/rslearn-eai/datasets/pastis_rslearn/embedding_materializer_manifest_tessera_v2.json`
  — note the manifest lives on the rslearn-eai source, not the ingested copy),
  and every dataset must match it or the three are not one Tessera column.
  `--model_size` defaults to `medium`, so pass it explicitly.
- **S2 `max_matches` is 300 here, 150 for PASTIS.** The layers sort `datetime`
  ascending, so hitting the cap keeps the *earliest* N scenes and drops the end
  of the year — a seasonal loss, not random attrition. Measured 2026-08-06
  after step 3: at 150, **4.94%** of ethiopia windows were truncated; at 300 it
  is **0.32%** (ethiopia) / **0.35%** (africa), with S1 at 100 truncating
  0.23–0.35% of africa windows. That residual was accepted. The cap only binds
  in MGRS overlap zones (median ~72 S2 scenes), so the extra download is a few
  percent. Ceiling for any future raise: the student bins valid observations at
  256 (`tessera_v2_infer.BIN_EDGES`), so scenes past roughly 500–600 cannot
  reach the model at all. **PASTIS was built at 150 and may carry the full
  4.94%-scale truncation — measure its fetch group before putting the three
  datasets in one table.**
- **Re-preparing after a cap change needs `--force`** and re-queries the whole
  group (rslearn cannot target just the capped windows). Note the config is
  read once at process start, so a job already running will not pick up an
  edited fetch config. The `duration` in the prepare summary is summed across
  workers, not wall clock — divide by `--workers`.
- **A window whose S1 layer has prepared items but none materialized fails by
  default**, because that normally means materialize is incomplete. Africa hit
  this on the 3 windows whose ascending scenes 404 (`2553 written, 3 failed`).
  Left as-is: 0.12% of windows, and `--required` resolves the same set for
  every model, so the comparison is unaffected. `infer
  --allow_unmaterialized_s1` embeds them with that layer treated as absent —
  use it only for scenes confirmed unfetchable at the source, and note it is
  recorded in the manifest's `cli_args`. S2 stays strict either way.
- **Zero ascending S1 is normal in Ethiopia** — all 2530 windows, against 23–58
  descending. Not a query bug: africa, fetched identically, gets ascending fine
  (median 30), so the absence is in MPC's `sentinel-1-rtc` archive. Tessera
  pulls the same collection, so their product is descending-only there too, and
  `build_dpixel_inputs` handles a zero-item layer. Zero *S2* over a window is
  never normal.
- **`--required` in step 5 changes the window set for every eval on that
  dataset**, so previously recorded OlmoEarth/AEF numbers on it are no longer
  measured on the same windows. It is the right flag *only* if inference wrote
  every window (the manifest's `num_windows_failed` is 0); drop it and re-run
  the wiring later otherwise.
- The fetch group is disposable: delete `$STAGE/windows/<fetch group>` once
  step 4 succeeds. It is only ~15 GB, but ~3M small files.
- Unlike PASTIS, these datasets' `config.json` never learns about the `*_all`
  layers — they are handed to rslearn as a separate `--config`, and the
  inference reads the rasters through `window.get_raster_dir(...)` rather than
  the dataset config. So a later `prepare`/`materialize` that forgets
  `--enabled-layers` cannot accidentally fetch a year of scenes here, and an
  ingest that overwrites `config.json` cannot silently drop the layers. On
  `pastis_rslearn` (built first, layers inline) both hazards are still live.

## Estimates

**Wall clock** (footprint scope, ~40M px):
- Engineering (ROI script, env for their pipeline incl. Rust build, output
  join): **2–3 days**.
- Download + preprocess: their v1-era figure is ~10 h per full tile-year on a
  128-core node; at 8% of the area but with per-scene overheads, expect
  **~1–2 days wall clock** (network-bound, parallelizable across ROI chunks).
- GPU inference: students **minutes** (medium ≈ 3.3 GFLOPs/px → ~0.13 EFLOPs
  total ≈ a few minutes on one H100 at realistic MFU; paper's own scaling law
  gives ~80 H100-seconds). Teacher ≈ 283 GFLOPs/px → ~11 EFLOPs ≈ **~4–8 h**
  on one H100.
- Join + eval launch: **~1 day**.
- Total: **≈ 1 calendar week**, mostly preprocessing + plumbing.

**Storage** (footprint scope): d-pixel npys ≈ 60 GB S2 (uint16, ~73 frames) +
~3 GB masks + ~22 GB S1 (int16, ~135 frames) ≈ **~85–100 GB**, plus ~2–4× that
transient during download (per-band GeoTIFFs) → budget **~0.5 TB scratch**.
Outputs: 40M px × 128 × f32 = **20 GB per student size** (5 GB int8+scales);
teacher 1024-d = 163 GB fp32 → use int8 (~41 GB) or store only patch pixels.
(Full-tile scope for reference: ≥1 TB per tile-year working storage per their
README, ~1–1.8 TB d-pixels total, 62–247 GB embeddings — not worth it.)

## FLOPs comparison (per 10 m pixel embedded, single pass, S1+S2)

OlmoEarth ws16/ps1: N = (16/1)² spatial tokens × 12 monthly timesteps ×
{1 S2, 1 S1} bandgroup tokens = 3,072 (S2) / 6,144 (S1+S2) tokens of d=768
through 12 layers, full self-attention; 64 such windows per 128×128 patch.
Tessera v2: ~160–190 tokens of d≤640 through 4 layers, per pixel.

| Model | GFLOPs / pixel | vs OlmoEarth S1+S2 |
|---|---|---|
| Tessera v2 nano | ~0.14 | 0.015× |
| Tessera v2 small | ~1.1 | 0.12× |
| Tessera v2 medium | ~3.3 | 0.35× |
| Tessera v2 large | ~6.9 | 0.73× |
| **OlmoEarth base ws16/ps1 S2-only** | **3.4** | 0.36× |
| **OlmoEarth base ws16/ps1 S1+S2** | **9.5** | 1× |
| Tessera v2 2B teacher | ~283 | ~30× |

(Tessera per-pixel = T_total × per-token cost of ONE branch — each token passes
through half the model; medium = 4 layers × (4d² + 2·d·FFN) MACs ≈ 18.8
MFLOPs/token × ~176 tokens. OlmoEarth per-pixel = per-window TFLOPs / 256 px;
window cost = 6,144 × (170 weight + 226 attention) MFLOPs/token ≈ 2.44 TFLOPs.
Attention is 57% of the OlmoEarth S1+S2 cost because N=6,144 is large, vs <5%
for Tessera at T≤256.)

**The intuition "Tessera runs 16×16 separate forward passes so it must cost
more" turns out backwards for the students**: per pixel they see ~15× more
timesteps than our 12 monthly mosaics, but through a 4-layer, ≤640-wide model
(≈19 MFLOPs/token for medium) instead of a 12-layer 768-wide one
(≈396 MFLOPs/token incl. attention at N=6144). Net: medium ≈ ⅓ of our S1+S2
per-pixel cost; even large is ~0.7×. Only the 2B teacher is meaningfully more
expensive (~30×). Caveats: (i) these are architectural FLOPs — our harness
re-embeds per LP-LR job (×8) and per modality variant (×2), which is harness
overhead, not model cost; (ii) OlmoEarth amortizes better at coarser output —
at ps=8 (semantic-level tasks) our cost/pixel drops ~64× on the linear term,
whereas Tessera's per-pixel cost is fixed; (iii) Tessera's true end-to-end cost
is dominated by preprocessing (their v1 figure: ~10 h/tile on 128 cores),
which FLOPs don't capture.

Totals for embedding all 39.9M PASTIS pixels once: OlmoEarth S1+S2 ≈ 0.38
EFLOPs, S2-only ≈ 0.14; Tessera medium ≈ 0.13, large ≈ 0.27, teacher ≈ 11.3.
