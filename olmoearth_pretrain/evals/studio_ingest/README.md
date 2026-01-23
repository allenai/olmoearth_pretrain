# Studio Dataset Ingestion (Internal)

> ⚠️ **INTERNAL USE ONLY** - This module is for AI2 internal use to ingest
> datasets from the Studio platform into OlmoEarth evaluations.

## For External Users

If we publish these evals on HuggingFace, external users can download
and use them by setting an environment variable:

```bash
# Set this to your local directory containing downloaded rslearn datasets
export OLMOEARTH_EVAL_DATASETS=/path/to/downloaded/datasets

# Then use the registry loader as normal
```


Goals:
1. Mindless to add a new eval to olmoearth pretrain
2. Possible to discover existing evals not in codebase

3. Possible to configure different splits of the dataset and partitioning percentages
4. Fast to load
5. These evals don't have a properly split test set and so we would need to re-split the data for that and save that



Notes
6. Takes a lot fo workers and a really long time to scan for bigger datasets so to use those as in loop evals we want to cache index

## Overview

This module provides tooling to:

1. **Validate** rslearn datasets from Studio/GCS
2. **Copy** data to Weka storage
3. **Compute** per-band normalization statistics
4. **Register** datasets in the eval registry for unified loading

## Requirements

Requires the `ingest` dependency group:

```bash
uv sync --group ingest
```

## Steps I have done so far
1. Found the appropriate path for the tolbi dataset in some gcs bucket (there needs to be a canonical locaiton studio can write datasets to if needed)
2. I want to run `uv run --group ingest python -m olmoearth_pretrain.evals.studio_ingest.cli ingest  --name tolbi_crops --source gs://rslearn-eai/datasets/tolbi ` and infer all the neccessary information for the task
3. Register an eval config and the location of the dataset into whatever the registry is
4? Actually run the eval with it
## Quick Start

```bash
# Ingest a dataset from Studio
uv run --group ingest python -m olmoearth_pretrain.evals.studio_ingest.cli ingest \
    --name lfmc \
    --display-name "Live Fuel Moisture Content" \
    --source gs://bucket/path/to/rslearn/dataset \
    --task-type regression \
    --modalities sentinel2_l2a sentinel1 \
    --temporal-range 2022-09-01 2023-09-01 \
    --property-name lfmc_value

# List registered datasets
uv run --group ingest python -m olmoearth_pretrain.evals.studio_ingest.cli list

# Validate a dataset without ingesting
uv run --group ingest python -m olmoearth_pretrain.evals.studio_ingest.cli validate \
    --source gs://bucket/path/to/rslearn/dataset

# Show details for a registered dataset
uv run --group ingest python -m olmoearth_pretrain.evals.studio_ingest.cli info --name lfmc
```

## Registry Location

- **Registry JSON**: `weka://dfive-default/olmoearth/eval_datasets/registry.json`
- **Dataset Data**: `weka://dfive-default/olmoearth/eval_datasets/{name}/`

Each dataset directory contains:

```
{name}/
├── metadata.json       # Full dataset configuration
├── norm_stats.json     # Per-band normalization statistics
├── train/              # Training split (rslearn format)
├── val/                # Validation split
└── test/               # Test split
```

## Ingestion Workflow

1. **Validate**: Check rslearn dataset structure, verify modalities exist, check splits gather information for the config

Next we need to see what do we need to quickly read this when it is copied over and have a dataset class to be used dependent on the configuration that does not involve having to load all the windows and that bs with the rslearn dataset and potentially is a different format

Goal for today is tolbi and fire through both of these steps
2. **Register**: Add entry to registry.json, create metadata.json in dataset dir
3. **Copy**: Copy data from GCS to Weka, preserving rslearn structure (For now we will just stream from gcs)
4. Then we need to make sure that given the eval info we can load and run that as an eval


5. Make sure that we can discover new datasets

`python -m olmoearth_pretrain.internal.full_eval_sweep     --cluster=local     --checkpoint_path=/weka/dfive-default/helios/checkpoints/joer/tiny_lr0.0002_wd0.02/step360000     --module_path=scripts/official/tiny.py     --trainer.callbacks.downstream_evaluator.tasks_to_run="[tolbi_crops]"     --trainer.callbacks.downstream_evaluator.eval_on_startup=True     --trainer.callbacks.downstream_evaluator.cancel_after_first_eval=True     --trainer.callbacks.wandb.enabled=False --defaults_only `
