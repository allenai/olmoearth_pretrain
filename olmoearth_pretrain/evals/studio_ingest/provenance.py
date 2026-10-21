"""Config provenance for eval datasets.

Eval datasets split their state across two places: the rslearn data on Weka
(windows, split tags, config.json) and the task config (model.yaml) that
defines how the data is read. Historically both were read from the Weka
dataset folder, where the config copies could silently drift from their
git-tracked sources between ingests.

This module keeps the two halves honest:

- model.yaml is read from the copy installed with this package (under
  ``dataset_configs/<name>/``) when the registry entry records a
  ``packaged_config``, so the config a run uses is pinned by the commit being
  run rather than by whatever a past ingest left on Weka.
- config.json must stay physically in the dataset folder (rslearn reads it
  from the dataset root), so instead of relocating it we pin its sha256 in
  the registry at ingest time and verify it at eval time.
- Both hashes are logged to the active W&B run so any historical result can
  be tied back to the exact configs it ran with.
"""

from __future__ import annotations

import hashlib
import logging
from importlib.resources import files
from pathlib import Path
from typing import Any

from upath import UPath

logger = logging.getLogger(__name__)

# Per-dataset model.yaml configs, installed with the package (see
# [tool.setuptools.package-data] in pyproject.toml).
DATASET_CONFIGS_DIR = Path(
    str(files("olmoearth_pretrain.evals.studio_ingest") / "dataset_configs")
)


def packaged_config_name(config_dir: str) -> str | None:
    """Return the packaged config name for *config_dir*, or None if not packaged.

    Used at ingest time: when the config dir is one of the packaged
    ``dataset_configs/<name>/`` dirs, the registry entry records ``<name>`` so
    eval jobs read the packaged model.yaml instead of the Weka copy.
    """
    config_path = Path(config_dir).resolve()
    if config_path.parent != DATASET_CONFIGS_DIR.resolve():
        return None
    return config_path.name


def packaged_model_yaml_path(packaged_config: str) -> str:
    """Return the packaged model.yaml for *packaged_config*, failing loudly.

    Raises:
        FileNotFoundError: If the packaged config is missing. We deliberately
            do not fall back to the Weka copy here — a silent fallback would
            reintroduce exactly the stale-config drift this path exists to
            prevent.
    """
    path = DATASET_CONFIGS_DIR / packaged_config / "model.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Registry entry points at packaged config {packaged_config}, but "
            f"{path} does not exist. Was the config moved or deleted without "
            f"updating the registry?"
        )
    return str(path)


def verify_config_json_hash(
    name: str, weka_path: str, expected_sha256: str | None
) -> str | None:
    """Verify the dataset folder's config.json against the registry hash.

    Returns the actual sha256 (for provenance logging), or None if the file
    is missing. Entries with no recorded hash only warn — that keeps the
    check safe to land before the backfill has stamped existing entries.

    Raises:
        ValueError: If a recorded hash exists and does not match — the
            dataset folder has drifted from what was ingested/registered.
    """
    config_json = UPath(weka_path) / "config.json"
    if not config_json.exists():
        logger.warning(
            "Dataset '%s' has no config.json at %s; skipping hash verification.",
            name,
            weka_path,
        )
        return None

    actual = hashlib.sha256(config_json.read_bytes()).hexdigest()
    if expected_sha256 is None:
        logger.warning(
            "Dataset '%s' has no config_json_sha256 recorded in the registry; "
            "cannot verify config.json integrity. Re-ingest or run "
            "scripts/tools/backfill_eval_registry_provenance.py to stamp it.",
            name,
        )
    elif actual != expected_sha256:
        raise ValueError(
            f"config.json for dataset '{name}' at {config_json} does not match "
            f"the registry (expected sha256 {expected_sha256}, got {actual}). "
            f"The dataset folder has drifted from what was registered — "
            f"re-ingest with --overwrite (and commit the registry update), or "
            f"restore the original config.json."
        )
    return actual


def log_eval_dataset_provenance_to_wandb(name: str, provenance: dict[str, Any]) -> None:
    """Record dataset config provenance on the active W&B run, if any.

    Stored under ``eval_dataset_provenance/<name>`` in the run config so a
    historical result can always be tied back to the exact configs it used.
    The key is flat (not a nested dict) because ``config.update`` merges
    shallowly — nesting would let each dataset clobber the previous one's
    provenance in multi-task runs. No-op when wandb is not installed or no
    run is active.
    """
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    try:
        wandb.run.config.update(
            {f"eval_dataset_provenance/{name}": provenance}, allow_val_change=True
        )
    except Exception:
        logger.warning(
            "Failed to log eval dataset provenance for '%s' to wandb",
            name,
            exc_info=True,
        )
