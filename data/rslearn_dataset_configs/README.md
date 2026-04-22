These are rslearn dataset configuration files for OlmoEarth Pretrain.

## Canonical single-config flow: `config_corpus.json`

`config_corpus.json` is the single canonical rslearn config for building an OlmoEarth
Pretrain dataset from a corpus of `(sample_id, lon, lat, start_time[, end_time])`
rows. All layers (Sentinel-1, Sentinel-2, SRTM, WorldCover, WorldCereal, OpenStreetMap,
CDL, WRI canopy height map, etc.) live side-by-side in one file.

### Recommended setup

`from_corpus` with `--config-path` **copies** that file to `ds_path/config.json` by
default so each dataset root is self-contained (branch moves, checkout changes, or
deleted repo paths do not break an in-flight run). To symlink instead:

`--config-mode symlink` (same filesystem only).

```bash
python -m olmoearth_pretrain.dataset_creation.create_windows.from_corpus \
    --ds_path /path/to/ds \
    --fname corpus.csv \
    --config-path data/rslearn_dataset_configs/config_corpus.json
```

### Standard rslearn commands

```bash
rslearn dataset prepare --root /path/to/ds
rslearn dataset ingest --root /path/to/ds
rslearn dataset materialize --root /path/to/ds
```

### Disable layers during testing

`rslearn` supports `--disabled-layers layer1,layer2,...` on `prepare`, `ingest`, and
`materialize`. You do not need separate config files just to run smaller smoke tests.

Examples:

```bash
rslearn dataset prepare --root /path/to/ds \
    --disabled-layers landsat

rslearn dataset ingest --root /path/to/ds \
    --disabled-layers landsat,sentinel2_l2a_mo01,sentinel2_l2a_mo02

rslearn dataset materialize --root /path/to/ds \
    --disabled-layers landsat,sentinel2_l2a_mo01,sentinel2_l2a_mo02
```

### Adding modalities to an existing dataset

Windows are created once; materialization is per-layer and idempotent (`completed`
markers). To add a modality:

1. Update the **dataset’s** `config.json` (or re-copy from an updated canonical file).
2. Rerun `prepare`, `ingest`, and `materialize`.

### Tiny end-to-end smoke run

Use ~30 rows and the full config copy; if one family is too slow or flaky, rerun with
`--disabled-layers ...` instead of editing the frozen `config.json`.

## Per-modality configs (legacy)

The `config_<modality>.json` files were the previous one-config-per-modality layout.
They are preserved for pipelines that still invoke rslearn per modality, but new
corpus-driven datasets should use `config_corpus.json`.
