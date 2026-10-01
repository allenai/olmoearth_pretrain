"""Backfill config provenance onto existing eval registry entries.

Run this once on a Weka-mounted machine, then commit the updated
olmoearth_pretrain/evals/studio_ingest/registry.json:

    python scripts/tools/backfill_eval_registry_provenance.py [--dry-run]

For every registry entry this:

1. Stamps ``config_json_sha256`` with the hash of the dataset folder's
   current config.json (trust-on-first-record — eval jobs then fail loudly
   if the file drifts afterwards).
2. Sets ``config_repo_dir`` when data/rslearn_dataset_configs/<name>/model.yaml
   exists in the repo AND is byte-identical to the Weka copy, so eval jobs
   read the git-tracked file. A repo config that differs from the Weka copy
   is reported but NOT linked — switching it would silently change eval
   behavior; reconcile the two and re-run (or re-ingest with --overwrite).

Entries whose dataset folders are missing (no Weka mount, deleted dataset)
are skipped with a warning and left unchanged.
"""

from __future__ import annotations

import argparse
import logging

from upath import UPath

from olmoearth_pretrain.evals.studio_ingest.provenance import (
    RSLEARN_DATASET_CONFIGS_DIR,
    find_repo_root,
    sha256_of_file,
)
from olmoearth_pretrain.evals.studio_ingest.registry import Registry

logger = logging.getLogger(__name__)


def main() -> int:
    """Stamp config provenance onto registry entries and save the registry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing registry.json",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Backfill a single dataset by name",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    repo_root = find_repo_root()
    if repo_root is None:
        raise RuntimeError("Could not locate the repo root; run from a checkout.")

    registry = Registry.load()
    changed = 0
    for entry in registry:
        if args.only and entry.name != args.only:
            continue

        config_json = UPath(entry.weka_path) / "config.json"
        if not config_json.exists():
            logger.warning(
                "%s: no config.json at %s — skipping (is Weka mounted?)",
                entry.name,
                entry.weka_path,
            )
            continue

        entry_changed = False

        sha = sha256_of_file(config_json)
        if entry.config_json_sha256 != sha:
            if entry.config_json_sha256 is not None:
                logger.warning(
                    "%s: recorded config_json_sha256 differs from disk; "
                    "restamping to current contents",
                    entry.name,
                )
            entry.config_json_sha256 = sha
            entry_changed = True
            logger.info("%s: config_json_sha256 = %s", entry.name, sha)

        if entry.config_repo_dir is None:
            repo_dir = f"{RSLEARN_DATASET_CONFIGS_DIR}/{entry.name}"
            repo_yaml = repo_root / repo_dir / "model.yaml"
            weka_yaml = UPath(entry.weka_path) / "model.yaml"
            if repo_yaml.exists() and weka_yaml.exists():
                if repo_yaml.read_bytes() == weka_yaml.open("rb").read():
                    entry.config_repo_dir = repo_dir
                    entry_changed = True
                    logger.info("%s: config_repo_dir = %s", entry.name, repo_dir)
                else:
                    logger.warning(
                        "%s: %s exists but differs from the Weka copy — NOT "
                        "linking. Reconcile the two (re-ingest with --overwrite "
                        "or update the repo config) and re-run.",
                        entry.name,
                        repo_yaml,
                    )

        changed += entry_changed

    if changed == 0:
        logger.info("Nothing to backfill.")
        return 0
    if args.dry_run:
        logger.info("[dry-run] %d entries would change; not writing.", changed)
        return 0

    registry.save()
    logger.info(
        "Updated %d entries. Commit the registry: "
        "olmoearth_pretrain/evals/studio_ingest/registry.json",
        changed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
