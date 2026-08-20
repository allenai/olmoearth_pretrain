"""Register the all-scenes S2 arm's registry entries by hand.

Each is a clone of its ``*_year_aligned`` parent -- same windows, labels, split
tags and gse/tessera_v2 layers, with S2 swapped from twelve monthly mosaics to a
stack of up to 36 individual acquisitions
(``olmoearth_pretrain/evals/datasets/allscenes_export.py``). Cloning is why this is
not an ingest, for the same two reasons the ccmos pilot documents:

- The data is placed by rsync + allscenes_export, not by a materialize, so there is
  nothing for an ingest's copy step to do.
- ``band_stats._get_bands_by_modality`` derives bands from ``ModalitySpec.band_order``
  (twelve S2 bands) rather than from the model.yaml's declared ``bands:``, so on
  these ten-band datasets ``compute_band_stats`` sees C_actual=10 vs C=12, skips
  every sample, and raises "Stats for sentinel2_l2a B02 are None". Those stats are
  inert here -- ``use_pretrain_norm`` is true, so the loader normalizes with the
  pretraining config and reads ``entry.norm_stats`` only on the
  ``norm_stats_from_pretrained=False`` branch -- so cloning the parent's is honest
  and costs nothing.

``split_stats`` is cloned rather than recounted, and that is only correct while the
window set is unchanged. It is: the clone keeps the parent's required gse/tessera_v2
inputs, and every window in both datasets has at least one S2 acquisition (verified
by the export's manifest, ``num_coverage_gaps``). If a future dataset shows coverage
gaps, recount instead of cloning.

Idempotent: re-running overwrites the entries in place.

Run from the repo root on a weka-mounted machine (it hashes the datasets'
config.json), after the export has run, then commit the registry.json diff::

    python scripts/tools/register_allscenes_entries.py --go
"""

import argparse
import hashlib
import logging
from pathlib import Path

from olmoearth_pretrain.evals.studio_ingest.registry import Registry

logger = logging.getLogger(__name__)

CONFIG_DIR = "data/rslearn_dataset_configs"

# parent entry -> the all-scenes clone to register from it.
DATASETS = {
    "ethiopia_crops_year_aligned": "ethiopia_crops_s2all36_year_aligned",
    "africa_crop_mask_year_aligned": "africa_crop_mask_s2all36_year_aligned",
}


def sha256_of_file(path: str) -> str:
    """Hex sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_entries(registry: Registry) -> list:
    """Clone each parent entry into its all-scenes entry.

    Args:
        registry: the loaded registry, which must already hold every parent.

    Returns:
        The new entries, unsaved.

    Raises:
        SystemExit: if a clone's config.json is missing on weka.
    """
    entries = []
    for parent_name, clone_name in DATASETS.items():
        parent = registry.get(parent_name)

        clone = parent.model_copy(deep=True)
        clone.name = clone_name
        clone.weka_path = parent.weka_path.replace(parent_name, clone_name)
        clone.source_path = parent.source_path.replace(parent_name, clone_name)
        clone.config_repo_dir = f"{CONFIG_DIR}/{clone_name}"
        # Its config.json declares sentinel2_l2a_all and drops the monthly S2
        # layers (allscenes_export.patch_config), so it hashes differently from the
        # parent's. verify_config_json_hash fails loudly if this goes stale.
        config_json = Path(clone.weka_path) / "config.json"
        if not config_json.exists():
            raise SystemExit(
                f"{config_json} not found -- run the rsync and "
                "allscenes_export build for this dataset first."
            )
        clone.config_json_sha256 = sha256_of_file(str(config_json))
        entries.append(clone)
    return entries


def main() -> int:
    """Entry point."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--go", action="store_true", help="Write registry.json (default: dry run)."
    )
    args = parser.parse_args()

    registry = Registry.load()
    entries = build_entries(registry)
    for entry in entries:
        logger.info(
            f"{entry.name}\n"
            f"    weka_path        {entry.weka_path}\n"
            f"    config_repo_dir  {entry.config_repo_dir}\n"
            f"    config_json_sha  {entry.config_json_sha256}\n"
            f"    split_stats      {entry.split_stats}\n"
            f"    modalities       {entry.modalities}"
        )
    if not args.go:
        logger.info("dry run -- pass --go to write registry.json")
        return 0
    for entry in entries:
        registry.add(entry, overwrite=True)
    registry.save()
    logger.info(f"wrote {len(entries)} entries; commit the registry.json diff")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
