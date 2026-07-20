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
   name the readout config used.
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

### In flight (uncommitted output may exist under `evals/embedding_materializer/`)

- **Embedding materializer** — `olmoearth_pretrain/evals/embedding_materializer/`:
  `EmbeddingFetcher` protocol with `AEFFetcher` (wraps rslearn's
  `GoogleSatelliteEmbeddingV1`: AWS Open Data COGs, streamed reads, int8→
  float dequantization, returns [-1,1] float32) and `TesseraFetcher`
  (wraps `geotessera`, lazy-imported — not in the venv yet);
  `RslearnWindowProvider` writes a fetched array as a per-window raster
  layer named after the modality; idempotent CLI
  (`python -m olmoearth_pretrain.evals.embedding_materializer
  --dataset_path <p> --products aef,tessera [--year YYYY] [--overwrite]`);
  provenance manifest JSON per (dataset, product) recording product
  version, year policy, and coverage gaps. Was being built by an agent
  when this doc was written — review its files and tests before
  committing.

### Remaining work

1. **PASTIS join** — extend `evals/datasets/pastis_processor.py` +
   `pastis_dataset.py` to store/load `gse_images` / `tessera_images`
   alongside `s2_images`/`s1_images`, fetched by patch geometry from
   PASTIS `metadata.geojson`, **year 2019** (labels are the 2019 French
   LPIS crop season; AEF ≈ 2.5 GB, Tessera ≈ 5 GB for all 2,433 patches).
   Then add `gse`/`tessera` to the PASTIS dataset config's
   `supported_modalities`.
2. **AEF supplemental datasets** — already ingested in
   `evals/studio_ingest/registry.json` (africa_crop_mask, canada_crops_*,
   descals, ethiopia_crops, …; sourced from rslearn_projects'
   conversions). Run the materializer over them, add `gse`/`tessera` to
   their configs' `supported_modalities`, add `DownstreamTaskConfig`
   entries in `internal/all_evals.py` with
   `primary_metric=BALANCED_ACCURACY` (+ consider bootstrap CIs for parity
   with Google's published numbers; `n_bootstrap` already exists on the
   callback).
3. **Launch-time guard** — in `full_eval_sweep.py` main: if
   `--model=aef|tessera_precomputed` and `_modality_capable_tasks(...)` is
   empty (or a requested task's store lacks the layer), fail fast printing
   the exact materializer command. (Data-level errors currently surface as
   an informative `ValueError` from `PrecomputedEmbedding.forward`.)
4. **Validation on cluster** — reproduce the GSE probe script's numbers
   through `--model=aef` on the pretrain_subset tasks (same splits/probe →
   numbers should match `scripts/tools/20260528_gse_linear_probe.py`);
   sanity-check probe `height_width` semantics per task (the script
   overrode `height_width` to the native GSE grid — watch for the same
   issue in the callback path).
5. **Tessera data + v2** — install/pin `geotessera`, materialize Tessera
   v1.1 for the target datasets (coverage: global 2024, US+EU 2017–2025 —
   gaps must be logged, not zero-filled). Tessera v2: neither weights nor
   embeddings published yet (arXiv:2607.03949); when weights drop, update
   `evals/models/tessera/tessera_model.py` (forward-pass path, same
   command); if they publish v2 embeddings, add a `TESSERA_V2` modality +
   fetcher (~day of work given the above).
6. **OlmoEarth annualized-readout sweep options** — optional fixed
   `window_size` / centered-crop as sweepable task options (deferred;
   `use_center_token` already exists, rslearn ingest already fixes window
   sizes per dataset).
7. **Docs** — extend `docs/Evaluation.md` / `docs/Adding-Eval-Datasets.md`:
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
