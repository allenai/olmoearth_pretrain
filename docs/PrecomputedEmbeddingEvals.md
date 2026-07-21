# Precomputed Embedding Evals (AlphaEarth, Tessera) — Plan & Status

Branch: `gabi/precomputed-embedding-evals`

## Goal

One command that evaluates OlmoEarth checkpoints, AlphaEarth (AEF/GSE), and
Tessera (v1.1 precomputed now, v2 forward-pass when released) on the same
datasets with a structural guarantee that everything downstream of the
embeddings — datasets, splits, probes, KNN, seeds, metrics — is identical:

```bash
python -m olmoearth_pretrain.internal.full_eval_sweep --model=aef ...
python -m olmoearth_pretrain.internal.full_eval_sweep --model=tessera_precomputed ...
python -m olmoearth_pretrain.internal.full_eval_sweep --checkpoint_path=<olmoearth ckpt> ...
```

## Design invariants

1. **Precomputed embedding products are data modalities, not models.**
   AlphaEarth = `Modality.GSE` (64 bands, A00–A63), Tessera =
   `Modality.TESSERA` (128 bands, T000–T127); both 10 m, annual,
   non-temporal. They are baked into eval dataset stores offline by the
   *embedding materializer* and read at eval time by a parameter-free
   "model" (`PrecomputedEmbedding`) through the standard `EvalWrapper`
   contract, so `run_knn` / `train_and_eval_probe` cannot tell them apart
   from a forward-pass model.
2. **OlmoEarth stays a live forward-pass model.** Embedding-production
   choices (patch size, window size, pooling, center-token, modality set)
   remain sweepable config so we can experiment with the best way to
   produce embeddings. Selection happens on val only; results tables must
   name the readout config used. For product parity with AEF/Tessera, run
   OlmoEarth with `--patch_size 1` (per-pixel 10 m embeddings; consider a
   lower embedding_batch_size — token counts grow 16x vs patch size 4) and
   `--quantize_embeddings` (int8 round-trip using AEF's exact
   sqrt-companding scheme; AEF/Tessera are already int8-at-source, so they
   must NOT be quantized a second time — their sweep args correctly set no
   quantize flags). Both flags mark the run name (`_ps1`, `_qt`).
3. **The temporal convention is dataset-scoped, not task-scoped.** rslearn
   eval datasets carry `start_time`/`end_time` in their ingested
   `model.yaml`; the materializer picks the annual product layer from the
   window time range (or an explicit `--year`). Everyone looks at the same
   year by construction. (This replaced the earlier idea of a `label_year`
   field on `DownstreamTaskConfig`.)
4. **Embedding products are consumed exactly as stored** — no imagery
   normalization is applied to them (loaders skip it; sweeps set
   `NO_NORM`).

## Status

### Done on this branch

- **`PrecomputedEmbedding` baseline + wrapper** —
  `olmoearth_pretrain/evals/models/precomputed/precomputed.py`
  (parameter-free `nn.Module` reading a named modality off the sample;
  `patch_size=1` so segmentation probes see the native 10 m pixel grid) and
  `PrecomputedEmbeddingEvalWrapper` in `evals/eval_wrapper.py`. Registered as
  `BaselineModelName.AEF` ("aef") and `BaselineModelName.TESSERA_PRECOMPUTED`
  ("tessera_precomputed") with launch scripts in `evals/models/precomputed/`.
- **`Modality.TESSERA`** in `data/constants.py`, sample fields in
  `datatypes.py` (3 classes), identity norm-config entries
  (`data/norm_configs/{computed,predefined}.json`), mock-dataloader branch.
  Also `EMBEDDING_PRODUCT_MODALITIES` constant.
- **Sweep integration** (`internal/full_eval_sweep.py`):
  `--model=aef|tessera_precomputed` works end-to-end (dry-run validated).
  Per-model args override capable tasks to `input_modalities=[<modality>]`
  + `NO_NORM`; tasks whose dataset lacks the modality keep imagery inputs
  and are skipped at runtime by the existing modality check. Tasks that
  differ only by input imagery are deduped (they collapse to identical
  embedding evals). GSE added to the seven `pretrain_subset*` dataset
  configs' `supported_modalities` (the osm_sampling h5 store already
  contains GSE data) → 12 AEF-capable tasks today (6 probe targets ×
  random/geographic splits).
- **`PretrainSubsetDataset` input-presence filter** (default on): drops
  samples missing an input modality *after* split assignment, so split
  membership stays aligned with model-probe runs (ports the manual filter
  from `scripts/tools/20260528_gse_linear_probe.py`, which is the numeric
  precedent to reproduce).
- **rslearn loader embedding-awareness** (`evals/datasets/rslearn_dataset.py`):
  `gse`/`tessera` in `allowed_modalities`; normalization skipped for
  embedding products.
- **`EvalMetric.BALANCED_ACCURACY`** (mean per-class recall, the AlphaEarth
  paper's protocol metric, arXiv:2507.22291) computed for all single-label
  classification tasks; selectable as primary metric.
- Unit tests: `tests/unit/eval/test_precomputed_embedding.py` (+ balanced
  accuracy test in `test_metrics.py`). Full unit suite green.

### Done (code-complete; needs a real-data run to validate the fetchers)

- **Embedding materializer** — `olmoearth_pretrain/evals/embedding_materializer/`:
  `EmbeddingFetcher` ABC returning float32 (C, H, W) on the window's grid
  (or None for coverage gaps); `AEFFetcher` wraps rslearn's
  `GoogleSatelliteEmbeddingV1` (AWS Open Data COGs, warped reads, int8→
  float dequantization to [-1,1], nodata -1.0, index CSV cached under
  `~/.cache/olmoearth_pretrain/aef_index`); `TesseraFetcher` wraps
  `geotessera` (lazy-imported — NOT in the venv yet, `pip install
  geotessera`), NaN nodata; shared tested mosaic/warp helper.
  `RslearnWindowProvider` writes per-window raster layers named after the
  modality (band dir is a sha256 hash — rslearn's own behavior for >64-char
  band lists; always resolve via `window.get_raster_dir`) and marks layers
  completed. Idempotent CLI:
  `python -m olmoearth_pretrain.evals.embedding_materializer
  --dataset_path <p> --products aef,tessera [--year YYYY] [--overwrite]
  [--workers N]`. Year policy: explicit `--year`, else window time-range
  midpoint; windows with no time range are recorded separately from
  coverage gaps in the per-product provenance manifest
  (`<dataset>/embedding_materializer_manifest_<product>.json`).
  14 unit tests (`tests/unit/eval/test_embedding_materializer.py`).

  **Verify on first real run** (written against docs, not a live service):
  geotessera's `registry.load_blocks_for_region(bounds=..., year=...)` and
  `fetch_embeddings(tiles)` yielding
  `(year, lon, lat, (H,W,128) float32, crs, transform)`; `GeoTessera()`
  kwargs for pinning dataset version/variant (pass via `client_kwargs`,
  update the recorded `product_version`, default "v1.1"). AEF years are
  2018–2024 on the AWS bucket; out-of-range years surface as coverage gaps.
  Also note: for rslearn `ModelDataset` to *read* the new layers, each eval
  dataset's on-WEKA `config.json`/`model.yaml` needs a `gse`/`tessera`
  layer entry — dataset-side change at onboarding time (remaining work #2).

### Done (code-complete; data materialization pending on cluster)

- **PASTIS join** — `pastis_processor.py --embedding_products=aef,tessera
  [--embedding_year 2019]` fetches each product on the patch's native 10 m
  UTM grid (footprint from `metadata.geojson`, snapped via
  `patch_grid_from_geometry`; UTM zone per patch centroid — PASTIS spans
  zones 30/31/32) and stores `gse_images/` / `tessera_images/` per split
  alongside `s2_images/`, quadrant-split identically. Processing is
  deterministic, so re-running into the existing output dir keeps sample
  indices aligned with prior splits. `pastis_dataset.py` loads them as
  `(H, W, 1, C)` un-normalized, only loads requested modalities from disk,
  and raises a clear error pointing at `--embedding_products` when splits
  lack them. `pastis`/`pastis128` configs list gse+tessera, so the AEF
  sweep now includes the PASTIS tasks (they fail with that clear error
  until the WEKA splits are reprocessed — expected).
  Default year 2019 = PASTIS label year (French LPIS 2019).
- **Launch-time guard** — `full_eval_sweep.py` exits at launch with the
  exact materializer command when a precomputed-embedding baseline has
  zero capable tasks, and logs the capable task list otherwise.

### Remaining work

1. **Add embeddings to the PASTIS splits** — preferred: the additive
   embeddings-only mode, which needs ONLY `metadata.geojson` (no 29 GB raw
   download), verifies per-sample month sequences against the existing
   `months.pt` before writing (aborts on any count/order drift), and writes
   only the new `<modality>_images` dirs — existing imagery/labels stay
   byte-identical so all prior PASTIS numbers remain comparable:

   ```bash
   # 64x64 splits (pastis)
   python -m olmoearth_pretrain.evals.datasets.pastis_processor \
     --data_dir <dir with metadata.geojson> --output_dir $PASTIS_DIR \
     --embeddings_only --embedding_products aef,tessera

   # 128x128 splits (pastis128): add --orig_size, point at $PASTIS_DIR_ORIG
   ```

   Resumable (re-runs skip written patches; `--overwrite_embeddings` to
   redo). AEF ≈ 2.5 GB, Tessera ≈ 5 GB per split set. The full-reprocess
   path (`--embedding_products` without `--embeddings_only`) needs the raw
   PASTIS-R download and rewrites everything — only use it into a fresh
   directory.
2. **AEF supplemental datasets** — already ingested in
   `evals/studio_ingest/registry.json` (africa_crop_mask, canada_crops_*,
   descals, ethiopia_crops, …; sourced from rslearn_projects'
   conversions). Run the materializer over them, add `gse`/`tessera` to
   their configs' `supported_modalities`, add `DownstreamTaskConfig`
   entries in `internal/all_evals.py` with
   `primary_metric=BALANCED_ACCURACY` (+ consider bootstrap CIs for parity
   with Google's published numbers; `n_bootstrap` already exists on the
   callback).
3. **Validation on cluster** — reproduce the GSE probe script's numbers
   through `--model=aef` on the pretrain_subset tasks (same splits/probe →
   numbers should match `scripts/tools/20260528_gse_linear_probe.py`);
   sanity-check probe `height_width` semantics per task (the script
   overrode `height_width` to the native GSE grid — watch for the same
   issue in the callback path).
4. **Tessera data + v2** — install/pin `geotessera`, materialize Tessera
   v1.1 for the target datasets (coverage: global 2024, US+EU 2017–2025 —
   gaps must be logged, not zero-filled). Tessera v2: neither weights nor
   embeddings published yet (arXiv:2607.03949); when weights drop, update
   `evals/models/tessera/tessera_model.py` (forward-pass path, same
   command); if they publish v2 embeddings, add a `TESSERA_V2` modality +
   fetcher (~day of work given the above).
5. **OlmoEarth annualized-readout sweep options** — `window_size` is now a
   `DownstreamTaskConfig` field for the pastis datasets: it tiles every
   sample + labels into window_size x window_size windows at load time
   (e.g. 16 -> sixteen 16x16 windows per 64x64 sample), and the probe
   geometry follows automatically. Set per task via CLI override, e.g.
   `--trainer.callbacks.downstream_evaluator.tasks.pastis_sentinel2.window_size=16`
   (combine with `--patch_size 1` for the ws16/ps1 readout). Note it changes
   the task definition, so compare models at the SAME window_size. Still
   deferred: window_size for other datasets and centered-crop
   (`use_center_token` already exists; rslearn ingest already fixes window
   sizes per dataset).
6. **Docs** — extend `docs/Evaluation.md` / `docs/Adding-Eval-Datasets.md`:
   a new eval task is embedding-eligible iff its samples have real
   geometry + a resolvable year; onboarding = ingest as usual → run
   materializer → add modality to its config's `supported_modalities` →
   task config entry.

## Extensibility contract

- **New eval task**: ingest as usual (rslearn windows are the common case →
  zero new code), run the materializer once per product, list the modality
  in the dataset config, add a `DownstreamTaskConfig`. No probe / wrapper /
  metric / sweep code changes.
- **New embedding product**: one `ModalitySpec` + sample fields + norm-config
  entries, one fetcher, one `BaselineModelName` + launch script. Everything
  else is shared.

## Fairness fine print (record in any results table)

- AEF/Tessera are annual products; OlmoEarth sees the actual (monthly) time
  series of the same year. Inherent, not fixable.
- AEF/Tessera are native per-pixel 10 m; OlmoEarth embeddings are per-token
  at `patch_size` granularity. Probes interpolate — same property as
  published AEF-vs-Tessera comparisons.
- OlmoEarth's readout is val-selected; AEF/Tessera get their one published
  product. State the selected readout config next to results.
- Tessera v1.1 could also be run forward-pass (weights are public); the
  precomputed product is primary (it's what users consume), forward-pass is
  a sanity check.
