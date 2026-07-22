"""Export PASTIS-R as an rslearn dataset that mirrors the pretraining dataset.

The existing PASTIS eval splits (pastis_processor.py) consume the imagery
shipped with the PASTIS benchmark (10 S2 bands with 3 imputed,
benchmark-specific preprocessing). This script instead initializes an rslearn
dataset whose satellite inputs are materialized from the same sources and
conventions as the OlmoEarth pretraining dataset — 12 monthly 30-day
Planetary Computer mosaics per sensor (S2 L2A and S1), per
data/rslearn_dataset_configs/config_pastis_rslearn.json:

- one 128x128 window per PASTIS patch on the patch's native 10 m UTM grid
  (patch_grid_from_geometry), named by ID_PATCH, in the "pastis" group, with
  window.options["split"] assigned from the PASTIS folds (1-3 train / 4 val /
  5 test) so studio_ingest keeps the official splits;
- a "label" raster layer written from the PASTIS annotations, storing raw
  classes 0-18 plus the void label 19 (masked to invalid by
  SegmentationTask(nodata_value=19) in the dataset's model.yaml);
- "gse"/"tessera" raster layers converted from the embeddings previously
  fetched into the processed .pt splits by
  pastis_processor.py --embedding_products (no re-fetch): the stored 64x64
  quadrants are stitched back to 128x128 on the identical pixel grid, after
  verifying the split ordering against months.pt.

After running this script (on a machine with Weka access), materialize the
satellite layers and ingest:

    export DS_PATH=/weka/dfive-default/rslearn-eai/datasets/pastis_rslearn
    rslearn dataset prepare --root $DS_PATH --group pastis --workers 64 \
        --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60
    rslearn dataset materialize --root $DS_PATH --group pastis --workers 64 \
        --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60

    python -m olmoearth_pretrain.evals.studio_ingest.cli ingest \
        --name pastis_rslearn --source $DS_PATH \
        --olmoearth-run-config-path data/rslearn_dataset_configs/pastis_rslearn \
        --start-time 2018-09-01 --end-time 2019-09-01 \
        --register --overwrite
"""

import argparse
import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import torch
from rasterio.crs import CRS
from rslearn.dataset import Dataset as RslearnDataset
from rslearn.dataset import Window
from rslearn.utils.raster_array import RasterArray, RasterMetadata
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.datasets.pastis_processor import (
    patch_grid_from_geometry,
    replay_split_patches,
    verify_months_alignment,
)
from olmoearth_pretrain.evals.embedding_materializer.providers import (
    RslearnWindowProvider,
)

logger = logging.getLogger(__name__)

WINDOW_GROUP = "pastis"
PATCH_SIZE_PX = 128
QUADRANT_SIZE_PX = 64

# The PASTIS series starts September 2018; 12 x 30-day monthly mosaic layers
# (time_offset 0d..330d in config_pastis_rslearn.json) tile this window.
WINDOW_START_TIME = datetime(2018, 9, 1, tzinfo=UTC)
WINDOW_TIME_RANGE = (WINDOW_START_TIME, WINDOW_START_TIME + timedelta(days=360))

# rslearn window split tag values understood by studio_ingest's split scan.
FOLD_TO_SPLIT_TAG = {"train": "train", "valid": "val", "test": "test"}

# Nodata fill values the embeddings were fetched with (see
# embedding_materializer/fetchers.py: AEF dequantizes with -1.0 fill, Tessera
# uses NaN).
EMBEDDING_NODATA = {Modality.GSE.name: -1.0, Modality.TESSERA.name: float("nan")}


def stitch_quadrants(quadrants: torch.Tensor) -> torch.Tensor:
    """Reassemble (4, C, 64, 64) quadrants into a (C, 128, 128) raster.

    Inverse of pastis_processor.split_into_quadrants (order: top-left,
    bottom-left, top-right, bottom-right).
    """
    if quadrants.shape[0] != 4 or quadrants.shape[-2:] != (
        QUADRANT_SIZE_PX,
        QUADRANT_SIZE_PX,
    ):
        raise ValueError(f"Expected (4, C, 64, 64) quadrants, got {quadrants.shape}")
    full = torch.empty(
        (quadrants.shape[1], PATCH_SIZE_PX, PATCH_SIZE_PX), dtype=quadrants.dtype
    )
    half = QUADRANT_SIZE_PX
    full[:, :half, :half] = quadrants[0]
    full[:, half:, :half] = quadrants[1]
    full[:, :half, half:] = quadrants[2]
    full[:, half:, half:] = quadrants[3]
    return full


def create_window_with_label(
    dataset: RslearnDataset,
    pastis_dir: UPath,
    meta_crs: CRS,
    patch: dict[str, Any],
    split_tag: str,
) -> Window:
    """Create the rslearn window for one PASTIS patch and write its label layer."""
    bounds, projection = patch_grid_from_geometry(
        patch["geometry"], meta_crs, patch_size_px=PATCH_SIZE_PX
    )
    window = Window(
        storage=dataset.storage,
        group=WINDOW_GROUP,
        name=str(patch["patch_id"]),
        projection=projection,
        bounds=bounds,
        time_range=WINDOW_TIME_RANGE,
        options={"split": split_tag},
    )
    window.save()

    # First channel of TARGET_*.npy is the semantic map: classes 0-18 + void
    # 19, stored raw (the void label is masked via the task's nodata_value).
    target_path = pastis_dir / f"ANNOTATIONS/TARGET_{patch['patch_id']}.npy"
    labels = np.load(target_path)[0].astype(np.uint8)
    if labels.shape != (PATCH_SIZE_PX, PATCH_SIZE_PX):
        raise ValueError(
            f"patch {patch['patch_id']}: expected "
            f"({PATCH_SIZE_PX}, {PATCH_SIZE_PX}) labels, got {labels.shape}"
        )
    raster_dir = window.get_raster_dir("label", ["label"])
    GeotiffRasterFormat().encode_raster(
        raster_dir,
        window.projection,
        window.bounds,
        RasterArray(
            chw_array=labels[None],
            time_range=window.time_range,
            metadata=RasterMetadata(),
        ),
    )
    window.mark_layer_completed("label")
    return window


def write_embedding_layers(
    provider: RslearnWindowProvider,
    windows: dict[Any, Window],
    split_patches: dict[str, list[dict[str, Any]]],
    processed_splits_dir: UPath,
    embedding_modalities: list[str],
    workers: int,
) -> None:
    """Convert the processed splits' quadrant embeddings into window layers.

    The .pt files under <processed_splits_dir>/pastis_r_<split>/<modality>_images
    are 64x64 quadrants indexed 4*patch_pos + q in split order (see
    pastis_processor.process_embeddings_only); verify_months_alignment must
    have been run first so patch_pos is trusted.
    """
    for split, patches in split_patches.items():
        split_dir = processed_splits_dir / f"pastis_r_{split}"

        def handle_patch(patch_pos: int, patch: dict[str, Any]) -> None:
            window = windows[patch["patch_id"]]
            for modality_name in embedding_modalities:
                quadrants = torch.stack(
                    [
                        torch.load(
                            split_dir
                            / f"{modality_name}_images"
                            / f"{4 * patch_pos + q}.pt"
                        )
                        for q in range(4)
                    ]
                )
                provider.write_embedding(
                    window=window,
                    modality=Modality.get(modality_name),
                    array=stitch_quadrants(quadrants).numpy(),
                    nodata_value=EMBEDDING_NODATA[modality_name],
                )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for done, _ in enumerate(
                executor.map(handle_patch, range(len(patches)), patches), start=1
            ):
                if done % 200 == 0 or done == len(patches):
                    logger.info(
                        f"{split}: embeddings written for {done}/{len(patches)} patches"
                    )


def main() -> None:
    """Initialize the pastis_rslearn dataset: windows, labels, and embeddings."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pastis_dir",
        type=str,
        default="/weka/dfive-default/helios/evaluation/PASTIS-R",
        help="PASTIS-R benchmark directory (metadata.geojson + ANNOTATIONS)",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        required=True,
        help="Output rslearn dataset root",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="data/rslearn_dataset_configs/config_pastis_rslearn.json",
        help="rslearn dataset config to install as <ds_path>/config.json",
    )
    parser.add_argument(
        "--processed_splits_dir",
        type=str,
        default="/weka/dfive-default/presto_eval_sets/pastis_r",
        help="Processed PASTIS splits holding the fetched embedding .pt files",
    )
    parser.add_argument(
        "--embedding_modalities",
        type=str,
        default="gse,tessera",
        help="Comma-separated embedding modalities to convert ('' to skip)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Worker threads for window creation and embedding conversion",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    pastis_dir = UPath(args.pastis_dir)
    ds_path = UPath(args.ds_path)
    embedding_modalities = [m for m in args.embedding_modalities.split(",") if m]
    for modality_name in embedding_modalities:
        if modality_name not in EMBEDDING_NODATA:
            raise ValueError(
                f"Unsupported embedding modality '{modality_name}'; expected "
                f"one of {sorted(EMBEDDING_NODATA)}"
            )

    ds_path.mkdir(parents=True, exist_ok=True)
    with (
        open(args.dataset_config, "rb") as src,
        (ds_path / "config.json").open("wb") as dst,
    ):
        shutil.copyfileobj(src, dst)
    dataset = RslearnDataset(ds_path)

    with (pastis_dir / "metadata.geojson").open() as f:
        meta_data = json.load(f)
    crs_name = meta_data.get("crs", {}).get("properties", {}).get("name")
    meta_crs = (
        CRS.from_user_input(crs_name) if crs_name is not None else CRS.from_epsg(4326)
    )
    split_patches = replay_split_patches(meta_data)

    # Verify the processed splits share this metadata replay BEFORE writing
    # anything, so the embedding conversion below cannot mis-align.
    if embedding_modalities:
        verify_months_alignment(
            UPath(args.processed_splits_dir), split_patches, samples_per_patch=4
        )

    windows: dict[Any, Window] = {}
    for split, patches in split_patches.items():
        split_tag = FOLD_TO_SPLIT_TAG[split]

        def create(patch: dict[str, Any]) -> Window:
            return create_window_with_label(
                dataset, pastis_dir, meta_crs, patch, split_tag
            )

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for done, window in enumerate(executor.map(create, patches), start=1):
                windows[int(window.name)] = window
                if done % 200 == 0 or done == len(patches):
                    logger.info(f"{split}: created {done}/{len(patches)} windows")

    if embedding_modalities:
        write_embedding_layers(
            provider=RslearnWindowProvider(ds_path, groups=[WINDOW_GROUP]),
            windows=windows,
            split_patches=split_patches,
            processed_splits_dir=UPath(args.processed_splits_dir),
            embedding_modalities=embedding_modalities,
            workers=args.workers,
        )

    total = sum(len(p) for p in split_patches.values())
    logger.info(
        f"Done: {total} windows in group '{WINDOW_GROUP}' at {ds_path} "
        f"(splits: {[f'{s}={len(p)}' for s, p in split_patches.items()]})"
    )


if __name__ == "__main__":
    main()
