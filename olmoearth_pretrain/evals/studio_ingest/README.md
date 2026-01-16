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


2. **Register**: Add entry to registry.json, create metadata.json in dataset dir
2. **Copy**: Copy data from GCS to Weka, preserving rslearn structure
3. Make sure that we can discover new datasets
## Normalization Statistics

Band statistics are computed using `band_stats.py` (moved from `scripts/tools/`).

Per-band statistics:

- **mean**: Mean pixel value
- **std**: Standard deviation
- **min/max**: Absolute min/max values

To compute stats for a dataset:

```bash
uv run --group ingest python -m olmoearth_pretrain.evals.studio_ingest.band_stats \
    --ds_path gs://bucket/dataset \
    --input_layers sentinel2 sentinel1 \
    --output_json /path/to/norm_stats.json
```

TODO: Add p1/p99 percentile computation for robust normalization.

## Using Ingested Datasets

After ingestion, datasets are available via the unified loader:

```python
from olmoearth_pretrain.evals.datasets import RegistryDataset

# Load by name
dataset = RegistryDataset("lfmc", split="val")

# With cross-validation
dataset = RegistryDataset("lfmc", split="train", fold=0, num_folds=5)
```

## Module Structure

```
studio_ingest/
├── __init__.py      # Public exports
├── README.md        # This file
├── schema.py        # Dataclasses for registry entries
├── registry.py      # Read/write registry JSON
├── band_stats.py    # Compute band normalization statistics (moved from scripts/tools/)
├── validate.py      # Validate rslearn dataset structure
├── ingest.py        # Main ingestion logic
└── cli.py           # CLI entry point
```

## Target Datasets

Initial datasets to ingest from Studio:

- [ ] lfmc - Live Fuel Moisture Content
- [ ] forest_loss_driver - Forest Loss Driver Classification
- [ ] mangrove_subset - Mangrove Mapping
- [ ] tolbi - TOLBI Classification
- [ ] mozambique - Mozambique Land Cover
- [ ] ecosystem - Ecosystem Classification

## Troubleshooting

### "Registry not found"

The registry is created on first ingest. If it doesn't exist, run any ingest
command and it will be initialized.

### "Permission denied on Weka"

Ensure you have write access to `weka://dfive-default/olmoearth/eval_datasets/`.
Contact the OlmoEarth team for access.

### "Modality not found in dataset"

The rslearn dataset must have layers matching the specified modalities.
Check the dataset structure with `validate` before ingesting.
