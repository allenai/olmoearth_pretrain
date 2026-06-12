"""Path constants and helpers for Studio ingest."""

from __future__ import annotations

import os

# Environment variable for external eval dataset roots.
EVAL_DATASETS_ENV_VAR = "OLMOEARTH_EVAL_DATASETS"

# Default base URI used when loading eval datasets through UPath.
DEFAULT_WEKA_BASE_PATH = "weka://dfive-default/olmoearth/eval_datasets"

# Local Weka mount used by ingestion copy commands.
WEKA_EVAL_DATASETS_BASE_PATH = "/weka/dfive-default/olmoearth/eval_datasets"


def get_eval_datasets_base_path() -> str:
    """Get the base path for eval dataset loading."""
    return os.environ.get(EVAL_DATASETS_ENV_VAR, DEFAULT_WEKA_BASE_PATH)
