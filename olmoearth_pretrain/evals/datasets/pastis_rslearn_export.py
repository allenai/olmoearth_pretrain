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
After running this script (on a machine with Weka access), materialize the
satellite layers and ingest:

    export DS_PATH=/weka/dfive-default/rslearn-eai/datasets/pastis_rslearn
    rslearn dataset prepare --root $DS_PATH --group pastis --workers 64 \
        --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60
    rslearn dataset materialize --root $DS_PATH --group pastis --workers 64 \
        --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60

    python -m olmoearth_pretrain.evals.studio_ingest.cli ingest \
        --name pastis_rslearn --source $DS_PATH \
        --olmoearth-run-config-path \
            olmoearth_pretrain/evals/studio_ingest/dataset_configs/pastis_rslearn \
        --start-time 2018-09-01 --end-time 2019-09-01 \
        --register --overwrite

Precomputed embedding products are baked into the windows like any other
eval dataset's, then registered (each window's 2018-09 - 2019-09 time range
is centred on 2019, the PASTIS label year, so AEF reads its 2019 layer):

    python -m olmoearth_pretrain.evals.embedding_materializer \
        --dataset_path /weka/dfive-default/olmoearth/eval_datasets/pastis_rslearn \
        --products aef
    python scripts/tools/register_embedding_products.py \
        --datasets pastis_rslearn --products aef
"""

import argparse
import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform
from rslearn.dataset import Dataset as RslearnDataset
from rslearn.dataset import Window
from rslearn.utils.geometry import PixelBounds, Projection
from rslearn.utils.raster_array import RasterArray, RasterMetadata
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

logger = logging.getLogger(__name__)

# Repo-checkout location of the dataset config, resolved relative to this file
# so the script works from any working directory.
DEFAULT_DATASET_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "data/rslearn_dataset_configs/config_pastis_rslearn.json"
)

WINDOW_GROUP = "pastis"
PATCH_SIZE_PX = 128

# The PASTIS series starts September 2018; 12 x 30-day monthly mosaic layers
# (time_offset 0d..330d in config_pastis_rslearn.json) tile this window.
WINDOW_START_TIME = datetime(2018, 9, 1, tzinfo=UTC)
WINDOW_TIME_RANGE = (WINDOW_START_TIME, WINDOW_START_TIME + timedelta(days=360))

# rslearn window split tag values understood by studio_ingest's split scan.
FOLD_TO_SPLIT_TAG = {"train": "train", "valid": "val", "test": "test"}

# Calendar months spanned by the PASTIS time series; each sample keeps the
# first 12 months that have S2 acquisitions (as pastis_processor.py does).
PASTIS_ALL_MONTHS = [
    "201809",
    "201810",
    "201811",
    "201812",
    "201901",
    "201902",
    "201903",
    "201904",
    "201905",
    "201906",
    "201907",
    "201908",
    "201909",
    "201910",
]


def replay_months(dates: dict[str, int]) -> list[int]:
    """Replay pastis_processor's month selection from metadata dates alone.

    The per-sample month sequence depends only on which months have S2
    acquisitions (``dates-S2`` in metadata.geojson), not on the imagery, so
    it can be reproduced without loading any .npy files.
    """
    months_present = {str(date)[:6] for date in dates.values()}
    months = [int(month) for month in PASTIS_ALL_MONTHS if month in months_present]
    return months[:12]


def replay_split_patches(
    meta_data: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Replay pastis_processor's deterministic split ordering from metadata alone.

    Patches appear in metadata feature order within folds, folds 1-3
    concatenated for train (matching PASTISRProcessor.process()); patches
    without 12 months of S2 data are dropped, mirroring the processor's
    12-month filter.

    Returns:
        Dict split name ("train"/"valid"/"test") -> ordered list of dicts with
        "patch_id", "geometry", "fold", and "months" keys.
    """
    fold_patches: dict[str, list[dict[str, Any]]] = {
        f"fold_{i}": [] for i in range(1, 6)
    }
    for feature in meta_data["features"]:
        properties = feature["properties"]
        months = replay_months(properties["dates-S2"])
        if len(months) != 12:
            continue
        fold_patches[f"fold_{properties['Fold']}"].append(
            {
                "patch_id": properties["ID_PATCH"],
                "geometry": feature["geometry"],
                "fold": properties["Fold"],
                "months": months,
            }
        )
    return {
        "train": (
            fold_patches["fold_1"] + fold_patches["fold_2"] + fold_patches["fold_3"]
        ),
        "valid": fold_patches["fold_4"],
        "test": fold_patches["fold_5"],
    }


def patch_grid_from_geometry(
    geometry: dict[str, Any],
    source_crs: CRS,
    patch_size_px: int = PATCH_SIZE_PX,
    resolution: float = 10.0,
) -> tuple[PixelBounds, Projection]:
    """Compute a patch's native UTM pixel grid from its footprint geometry.

    PASTIS patches are cut from Sentinel-2 tiles, so their native grid is the
    10 m UTM grid of the tile. The footprint (in the metadata.geojson CRS) is
    reprojected to the centroid's UTM zone and the min corner is snapped to the
    10 m grid; the grid size is forced to ``patch_size_px`` so the window
    aligns pixel-for-pixel with the patch imagery and labels.

    Args:
        geometry: GeoJSON polygon geometry of the patch footprint.
        source_crs: CRS of the geometry coordinates.
        patch_size_px: Patch size in pixels (128 for PASTIS).
        resolution: Grid resolution in meters.

    Returns:
        Tuple of (pixel bounds, rslearn Projection) for the patch grid.
    """
    if geometry["type"] == "Polygon":
        exterior = geometry["coordinates"][0]
    elif geometry["type"] == "MultiPolygon":
        exterior = geometry["coordinates"][0][0]
    else:
        raise ValueError(f"Unsupported geometry type: {geometry['type']}")
    xs = [float(pt[0]) for pt in exterior]
    ys = [float(pt[1]) for pt in exterior]

    # UTM zone from the footprint centroid (PASTIS is in France: northern zones).
    lons, lats = warp_transform(
        source_crs,
        CRS.from_epsg(4326),
        [sum(xs) / len(xs)],
        [sum(ys) / len(ys)],
    )
    zone = int((lons[0] + 180) // 6) + 1
    utm_crs = CRS.from_epsg(32600 + zone)

    utm_xs, utm_ys = warp_transform(source_crs, utm_crs, xs, ys)
    x0 = round(min(utm_xs) / resolution)
    # rslearn pixel y grows southward (y_res is negative), so the min pixel row
    # corresponds to the max geographic y.
    y0 = round(-max(utm_ys) / resolution)
    bounds = (x0, y0, x0 + patch_size_px, y0 + patch_size_px)
    return bounds, Projection(utm_crs, resolution, -resolution)


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


def main() -> None:
    """Initialize the pastis_rslearn dataset: windows and labels."""
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
        default=str(DEFAULT_DATASET_CONFIG),
        help="rslearn dataset config to install as <ds_path>/config.json",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Worker threads for window creation",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    pastis_dir = UPath(args.pastis_dir)
    ds_path = UPath(args.ds_path)

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

    for split, patches in split_patches.items():
        split_tag = FOLD_TO_SPLIT_TAG[split]

        def create(patch: dict[str, Any]) -> Window:
            return create_window_with_label(
                dataset, pastis_dir, meta_crs, patch, split_tag
            )

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for done, _ in enumerate(executor.map(create, patches), start=1):
                if done % 200 == 0 or done == len(patches):
                    logger.info(f"{split}: created {done}/{len(patches)} windows")

    total = sum(len(p) for p in split_patches.values())
    logger.info(
        f"Done: {total} windows in group '{WINDOW_GROUP}' at {ds_path} "
        f"(splits: {[f'{s}={len(p)}' for s, p in split_patches.items()]})"
    )


if __name__ == "__main__":
    main()
