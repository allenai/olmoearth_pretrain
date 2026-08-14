"""Register the ethiopia cloud-mosaic pilot's two registry entries by hand.

Both are clones of ``ethiopia_crops_year_aligned``, which is why this is not an
ingest:

- ``ethiopia_crops_ccmos_year_aligned`` -- the composited dataset
  (pixel_mosaic_export.py). Its data is already copied to EVAL_DATASETS_BASE_PATH
  by an earlier ingest attempt, and its model.yaml is in place; only the entry is
  missing. A full ``studio_ingest ingest`` cannot finish it anyway:
  ``band_stats._get_bands_by_modality`` uses ``ModalitySpec.band_order`` (twelve S2
  bands) rather than the model.yaml's declared ``bands:``, so on this ten-band
  dataset ``compute_band_stats`` sees C_actual=10 vs C=12, skips every sample, and
  raises "Stats for sentinel2_l2a B02 are None". Those stats are inert here --
  ``use_pretrain_norm`` is true, so the loader normalizes with the pretraining
  config and reads ``entry.norm_stats`` only on the
  ``norm_stats_from_pretrained=False`` branch -- so cloning the parent's is honest
  and costs nothing.

- ``ethiopia_crops_10band_year_aligned`` -- the CONTROL. Shares the parent's
  ``weka_path`` (only its model.yaml's band list differs), i.e. exactly the
  ``us_trees_tessera`` pattern, which registry.json already carries as a
  hand-cloned entry rather than a second ingest.

Idempotent: re-running overwrites both entries in place.

Run from the repo root on a weka-mounted machine (it hashes the datasets'
config.json), then commit the registry.json diff::

    python scripts/tools/register_pixel_mosaic_entries.py --go
"""

import argparse
import hashlib
import logging
from pathlib import Path

from olmoearth_pretrain.evals.studio_ingest.registry import Registry

logger = logging.getLogger(__name__)

PARENT = "ethiopia_crops_year_aligned"
CCMOS = "ethiopia_crops_ccmos_year_aligned"
CONTROL = "ethiopia_crops_10band_year_aligned"
CONFIG_DIR = "data/rslearn_dataset_configs"


def sha256_of_file(path: str) -> str:
    """Hex sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_entries(registry: Registry) -> list:
    """Clone the parent entry into the composite and control entries.

    Args:
        registry: the loaded registry, which must already hold PARENT.

    Returns:
        The two new entries, unsaved.

    Raises:
        SystemExit: if a dataset's config.json is missing on weka.
    """
    parent = registry.get(PARENT)

    ccmos = parent.model_copy(deep=True)
    ccmos.name = CCMOS
    ccmos.weka_path = parent.weka_path.replace(PARENT, CCMOS)
    ccmos.source_path = parent.source_path.replace(PARENT, CCMOS)
    ccmos.config_repo_dir = f"{CONFIG_DIR}/{CCMOS}"
    # Its config.json is the patched one (a single ten-band set at window
    # resolution, see pixel_mosaic_export.patch_config), so it hashes differently
    # from the parent's. verify_config_json_hash fails loudly if this is stale.
    config_json = Path(ccmos.weka_path) / "config.json"
    if not config_json.exists():
        raise SystemExit(
            f"{config_json} not found -- copy the composited dataset into "
            "eval_datasets first (the ingest's step 1 does this)."
        )
    ccmos.config_json_sha256 = sha256_of_file(str(config_json))

    control = parent.model_copy(deep=True)
    control.name = CONTROL
    control.config_repo_dir = f"{CONFIG_DIR}/{CONTROL}"
    # weka_path/source_path/config_json_sha256 stay the PARENT's on purpose: the
    # control reads the parent's own windows and mosaics, just at ten bands.

    return [ccmos, control]


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
