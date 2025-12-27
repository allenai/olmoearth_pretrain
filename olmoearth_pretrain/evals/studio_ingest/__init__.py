"""Studio Dataset Ingestion Module (Internal Use Only).

This module provides tooling to ingest datasets from the Studio platform
into the OlmoEarth evaluation system. It handles:

1. Validation of rslearn dataset structure
2. Copying data from GCS to Weka
3. Computing normalization statistics (via band_stats.py)
4. Registering datasets in the eval registry

⚠️  INTERNAL USE ONLY - This is for AI2 internal workflows.

For external users: Set OLMOEARTH_EVAL_DATASETS env var to point to your
local directory containing downloaded rslearn datasets.

Usage:
    uv run --group ingest python -m olmoearth_pretrain.evals.studio_ingest.cli ingest ...

    # Compute band stats separately:
    uv run --group ingest python -m olmoearth_pretrain.evals.studio_ingest.band_stats ...

See README.md in this directory for full documentation.
"""

from olmoearth_pretrain.evals.studio_ingest.schema import (
    BandStats,
    EvalDatasetEntry,
    ModalityStats,
)

__all__ = [
    "EvalDatasetEntry",
    "ModalityStats",
    "BandStats",
]
