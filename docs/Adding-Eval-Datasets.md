# Adding Evaluation Datasets (Internal)

> **This guide is for AI2 researchers with access to Weka and internal rslearn datasets.**
> It covers the end-to-end process of adding a new rslearn dataset as a linear probe/knn
> evaluation task in OlmoEarth pretraining.

---

## Overview

Adding a new eval dataset requires two things:

1. **Ingest** — copy the rslearn dataset to Weka, compute splits and band stats, register it
2. **Wire up** — add a `DownstreamTaskConfig` to a training script so it runs as a loop eval

---

## Prerequisites

- Access to Weka (`/weka/dfive-default/`)
- An rslearn dataset with a prepared `config.json` and a `model.yaml` training config

---

## Step 1: Ingest the Dataset

### What you need

| Thing | Where to find it |
|-------|-----------------|
| **Source path** | Path to the rslearn dataset (Weka or GCS) |
| **Config path** | Directory containing `model.yaml` for this task |
| **Dataset name** | A short unique identifier, e.g. `nandi_crop_map` |

### Run the ingest command

```bash
OLMOEARTH_INGEST_WORKERS=16 \
  NAME=my_task \
  SOURCE=/weka/dfive-default/rslearn-eai/datasets/my_task \
  CONFIG=/weka/dfive-default/henryh/helios/olmoearth_projects/olmoearth_run_data/my_task

OLMOEARTH_INGEST_WORKERS=16 nohup python -m olmoearth_pretrain.evals.studio_ingest.cli ingest \
  --name "$NAME" \
  --source "$SOURCE" \
  --olmoearth-run-config-path "$CONFIG" \
  --register \
  --overwrite \
  > "${NAME}_ingest.out" 2>&1 &
```

Tail the log to monitor progress:
```bash
tail -f "${NAME}_ingest.out"
```

### What ingest does

1. **Copies** the dataset to `/weka/dfive-default/olmoearth/eval_datasets/{name}/`
2. **Copies** `model.yaml` into the dataset folder as the canonical config
3. **Detects splits** — if the dataset has `train`/`val`/`test` tags it uses them; otherwise it auto-splits:
   - `train` + `test` → splits test into `val` + `test`
   - `train` + `val` → splits val into `val` + `test`
   - No splits → random 80/10/10
4. **Computes band stats** from up to 50k train samples for normalization
5. **Registers** the dataset in the eval registry

### Flags

| Flag | Required | Description |
|------|----------|-------------|
| `--name` | ✓ | Unique dataset identifier |
| `--source` | ✓ | Path to rslearn dataset (Weka or GCS) |
| `--olmoearth-run-config-path` | ✓ | Directory containing `model.yaml` |
| `--register` | ✓ | Write entry to eval registry |
| `--overwrite` | | Re-ingest if already exists |
| `--source-groups` | | Comma-separated list of rslearn groups to include |
| `--untar-source` | | Use if source is a `.tar.gz` archive on GCS |

### Verify

```bash
python -m olmoearth_pretrain.evals.studio_ingest.cli list
python -m olmoearth_pretrain.evals.studio_ingest.cli info --name my_task
```

### Common issues

**`pydantic ValidationError: layers.output.format: Extra inputs are not permitted`**
The dataset config has a deprecated `output` layer format. Ingest strips this automatically — if you see this error, make sure you're on the latest code.

**`KeyError: Unknown rslearn layer name: 'xyz'`**
The `model.yaml` references a layer that isn't in `RSLEARN_TO_OLMOEARTH` in `olmoearth_pretrain/evals/constants.py`. Either add the mapping or the layer will be skipped with a warning.

**`ValueError: could not get all the needed bands from window X layer label group 0`**
The `model.yaml` target layer name doesn't match the rasterized layer in the dataset. Check that `layers: ["label_raster"]` (not `layers: ["label"]`) is set in `model.yaml` for the segmentation target.

**All band stat batches skipped → `Stats for sentinel2_l2a B02 are None`**
Usually caused by the same `label` vs `label_raster` mismatch above — every sample fails to load. Fix the `model.yaml` and re-run.

---

## Step 2: Add to a Training Script

Once ingested, add a `DownstreamTaskConfig` to the eval tasks dict in your training script (e.g. `scripts/base_loop_evals/script.py`):

```python
"my_task": DownstreamTaskConfig(
    dataset="my_task",                          # must match --name from ingest
    embedding_batch_size=32,
    probe_batch_size=8,                         # ⬅ tune: reduce if OOM
    num_workers=8,
    pooling_type=PoolingType.MEAN,
    norm_stats_from_pretrained=True,
    norm_method=NormMethod.NORM_NO_CLIP_2_STD,
    probe_lr=0.01,                              # ⬅ tune: try 0.001, 0.01, 0.1
    eval_interval=LOOP_EVAL_INTERVAL,
    input_modalities=[                          # ⬅ set: modalities available in dataset
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.LANDSAT.name,
    ],
    epochs=100,                                 # ⬅ tune: 50 for fast tasks, 100+ for hard ones
    eval_mode=EvalMode.LINEAR_PROBE,
),
```

### Fields you must set

| Field | What to set |
|-------|-------------|
| `dataset` | The `--name` you used during ingest |
| `input_modalities` | The modalities actually present in the dataset (check `config.json`) |
| `probe_lr` | Start with `0.01`; tune with a quick sweep if needed |
| `epochs` | 50 for simpler tasks, 100 for harder segmentation |
| `probe_batch_size` | Reduce to 4 if OOM; larger is faster |

### Fields that rarely need changing

`embedding_batch_size`, `num_workers`, `pooling_type`, `norm_stats_from_pretrained`, `norm_method`, `eval_interval`, `eval_mode` — keep as shown unless you have a specific reason.

---

## What a PR adding a new dataset needs

A PR adding a new eval dataset should include:

1. **The ingest has been run** and the dataset is registered (verify with `cli list`)
2. **`DownstreamTaskConfig`** added to the relevant training script(s)
3. **A local eval test** confirming the dataset loads and the eval runs end-to-end (even just step 0)
4. **The dataset name and paths** documented in the PR description so teammates can reproduce

---

## Storage layout

```
/weka/dfive-default/olmoearth/eval_datasets/
└── {name}/
    ├── config.json          # rslearn dataset config (patched to remove deprecated fields)
    ├── model.yaml           # canonical model/task config for this eval
    ├── windows/             # rslearn windows with split tags written
    └── .rslearn_dataset_index/   # cached window index

Registry: /weka/dfive-default/olmoearth/eval_datasets/registry/registry.json
```

---

## Example: nandi_crop_map

```bash
OLMOEARTH_INGEST_WORKERS=16 \
  NAME=nandi_crop_map \
  SOURCE=/weka/dfive-default/rslearn-eai/datasets/nandi \
  CONFIG=/weka/dfive-default/henryh/helios/olmoearth_projects/olmoearth_run_data/nandi

OLMOEARTH_INGEST_WORKERS=16 nohup python -m olmoearth_pretrain.evals.studio_ingest.cli ingest \
  --name "$NAME" --source "$SOURCE" --olmoearth-run-config-path "$CONFIG" \
  --register --overwrite > "${NAME}_ingest.out" 2>&1 &
```

Then in `scripts/base_loop_evals/script.py`:

```python
"nandi_crop_map": DownstreamTaskConfig(
    dataset="nandi_crop_map",
    embedding_batch_size=32,
    probe_batch_size=8,
    num_workers=8,
    pooling_type=PoolingType.MEAN,
    norm_stats_from_pretrained=True,
    norm_method=NormMethod.NORM_NO_CLIP_2_STD,
    probe_lr=0.01,
    eval_interval=LOOP_EVAL_INTERVAL,
    input_modalities=[Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name, Modality.LANDSAT.name],
    epochs=100,
    eval_mode=EvalMode.LINEAR_PROBE,
),
```
