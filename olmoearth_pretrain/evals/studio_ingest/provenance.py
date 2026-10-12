"""Config provenance for eval datasets.

Eval datasets split their state across two places: the rslearn data on Weka
(windows, split tags, config.json) and the task config (model.yaml) that
defines how the data is read. Historically both were read from the Weka
dataset folder, where the config copies could silently drift from their
git-tracked sources between ingests.

This module keeps the two halves honest:

- model.yaml is read from the git checkout when the registry entry records a
  ``config_repo_dir``, so the config a run uses is pinned by the commit being
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
import os
from pathlib import Path
from typing import Any

from upath import UPath

logger = logging.getLogger(__name__)

# Env var override for the repo root, for run modes where walking up from
# __file__ does not land in a checkout (e.g. a non-editable install).
REPO_ROOT_ENV_VAR = "OLMOEARTH_REPO_ROOT"

# Directory (repo-relative) where eval dataset configs are expected to live.
RSLEARN_DATASET_CONFIGS_DIR = "data/rslearn_dataset_configs"


def find_repo_root() -> Path | None:
    """Locate the repo checkout root.

    Resolution order:
    1. ``OLMOEARTH_REPO_ROOT`` env var, if set.
    2. Walk up from this file looking for ``pyproject.toml`` — works for
       editable installs and for Beaker jobs, which clone the repo and
       ``uv sync`` it.

    Returns None when no checkout can be found (e.g. wheel install).
    """
    env_root = os.environ.get(REPO_ROOT_ENV_VAR)
    if env_root:
        root = Path(env_root)
        if not root.is_dir():
            raise FileNotFoundError(
                f"{REPO_ROOT_ENV_VAR}={env_root} does not point to a directory"
            )
        return root

    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def sha256_of_file(path: str | Path | UPath) -> str:
    """Compute the sha256 hex digest of a file (works on Weka/GCS via UPath)."""
    upath = UPath(path)
    digest = hashlib.sha256()
    with upath.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative_config_dir(config_dir: str) -> str | None:
    """Return *config_dir* relative to the repo root, or None if outside it.

    Used at ingest time to record where the model.yaml came from: when the
    config dir lives inside the checkout (``data/rslearn_dataset_configs/...``)
    the registry entry stores the repo-relative path so eval jobs read the
    git-tracked file directly instead of the Weka copy.
    """
    repo_root = find_repo_root()
    if repo_root is None:
        return None
    try:
        return str(Path(config_dir).resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return None


def resolve_repo_config_path(config_repo_dir: str, filename: str) -> str:
    """Resolve ``<repo_root>/<config_repo_dir>/<filename>``, failing loudly.

    Raises:
        FileNotFoundError: If no repo checkout can be located or the file is
            missing from it. We deliberately do not fall back to the Weka copy
            here — a silent fallback would reintroduce exactly the stale-config
            drift this path exists to prevent.
    """
    repo_root = find_repo_root()
    if repo_root is None:
        raise FileNotFoundError(
            f"Cannot resolve repo-tracked config {config_repo_dir}/{filename}: "
            f"no repo checkout found. Run from a checkout or set "
            f"{REPO_ROOT_ENV_VAR} to the repo root."
        )
    path = repo_root / config_repo_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Registry entry points at repo config {config_repo_dir}/{filename}, "
            f"but it does not exist under {repo_root}. Was the config moved or "
            f"deleted without updating the registry?"
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

    actual = sha256_of_file(config_json)
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
