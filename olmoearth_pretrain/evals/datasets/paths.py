"""Dataset paths configured via environment variables."""

import os

from upath import UPath

# Only available to internal users
DEFAULT_PATHS = {
    "GEOBENCH_DIR": "/weka/dfive-default/presto-geobench/dataset/geobench",
    "BREIZHCROPS_DIR": "/weka/dfive-default/skylight/presto_eval_sets/breizhcrops",
    "MADOS_DIR": "/weka/dfive-default/presto_eval_sets/mados",
    "FLOODS_DIR": "/weka/dfive-default/presto_eval_sets/floods",
    "PASTIS_DIR": "/weka/dfive-default/presto_eval_sets/pastis_r",
    "PASTIS_DIR_ORIG": "/weka/dfive-default/presto_eval_sets/pastis_r_origsize",
    "PASTIS_DIR_PARTITION": "/weka/dfive-default/presto_eval_sets/pastis",
}

__all__ = [
    "BREIZHCROPS_DIR",
    "DEFAULT_PATHS",
    "FLOODS_DIR",
    "GEOBENCH_DIR",
    "MADOS_DIR",
    "PASTIS_DIR",
    "PASTIS_DIR_ORIG",
    "PASTIS_DIR_PARTITION",
    "get_path",
]


def get_path(name: str) -> UPath:
    """Resolve an eval dataset path from the current environment."""
    if name not in DEFAULT_PATHS:
        raise KeyError(f"Unknown eval dataset path key: {name}")
    return UPath(os.getenv(name, DEFAULT_PATHS[name]))


# Backward-compatible constants for callers that import paths directly. The
# dataset factory calls get_path so environment changes are picked up at runtime.
GEOBENCH_DIR = get_path("GEOBENCH_DIR")
BREIZHCROPS_DIR = get_path("BREIZHCROPS_DIR")
MADOS_DIR = get_path("MADOS_DIR")
FLOODS_DIR = get_path("FLOODS_DIR")
PASTIS_DIR = get_path("PASTIS_DIR")
PASTIS_DIR_ORIG = get_path("PASTIS_DIR_ORIG")
PASTIS_DIR_PARTITION = get_path("PASTIS_DIR_PARTITION")
