"""Export GeoPathfinder as an rslearn dataset that keeps the released imagery.

GeoPathfinder (https://huggingface.co/datasets/isaaccorley/GeoPathfinder) asks
whether two dots on a 256x256 Landsat 8/9 patch are connected by land (1) or
separated by water (0). Every patch yields one connect and one disconnect
sample, so it is written as two windows sharing the patch's imagery. The
imagery is the released raw/ GeoTIFF (Collection 2 Level-2 surface-reflectance
DNs, RGB+NIR) on its native 30 m grid, with no dots drawn in. Each window's
label raster is IGNORE_LABEL except at the two dot pixels, which carry the
sample's class; the eval loader reads them back with label_at_point_pair.

Each window's options carry the released patch-level split as "split" and the
basin-disjoint protocol (splits_basin.json) as "basin_split".

    python -m olmoearth_pretrain.evals.datasets.geopathfinder_rslearn_export \
        --src <extracted geopathfinder/> --basin_splits <splits_basin.json> \
        --ds_path <out rslearn root>
"""

import argparse
import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rslearn.dataset import Dataset as RslearnDataset
from rslearn.dataset import Window
from rslearn.utils.geometry import Projection
from rslearn.utils.raster_array import RasterArray, RasterMetadata
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

logger = logging.getLogger(__name__)

CONFIG_PATH = (
    Path(__file__).resolve().parents[3]
    / "data/rslearn_dataset_configs/config_geopathfinder.json"
)

GROUP = "geopathfinder"
PATCH_PX = 256
IGNORE_LABEL = 255
# raw/ band order (RGBN) mapped to Landsat 8/9 OLI band names.
RAW_BANDS = ["B4", "B3", "B2", "B5"]
LANDSAT_BANDS = ["B2", "B3", "B4", "B5"]
LABELS = {"connect": 1, "disconnect": 0}


def basin_split_lookup(path: Path) -> dict[str, str]:
    """sample_id -> split under the basin-disjoint protocol."""
    splits = json.loads(path.read_text())["splits"]
    return {sid: split for split, ids in splits.items() for sid in ids}


def dot_label(dots: tuple[tuple[int, int], tuple[int, int]], cls: int) -> np.ndarray:
    """A (1, H, W) label raster that is IGNORE_LABEL except at the two dots."""
    label = np.full((1, PATCH_PX, PATCH_PX), IGNORE_LABEL, dtype=np.uint8)
    for row, col in dots:
        if not (0 <= row < PATCH_PX and 0 <= col < PATCH_PX):
            raise ValueError(f"dot ({row}, {col}) outside the patch")
        label[0, row, col] = cls
    if (label != IGNORE_LABEL).sum() != 2:
        raise ValueError(f"dots {dots} do not mark two distinct pixels")
    return label


def write_raster(
    window: Window,
    layer: str,
    bands: list[str],
    chw: np.ndarray,
    time_range: tuple[datetime, datetime] | None,
) -> None:
    """Write one completed raster layer covering the whole window."""
    GeotiffRasterFormat().encode_raster(
        window.get_raster_dir(layer, bands),
        window.projection,
        window.bounds,
        RasterArray(
            chw_array=chw,
            time_range=time_range,
            metadata=RasterMetadata(nodata_value=None),
        ),
    )
    window.mark_layer_completed(layer)


def export_patch(
    dataset: RslearnDataset, src: Path, row: pd.Series, basin_split: str
) -> None:
    """Write the connect and disconnect windows for one patch."""
    with rasterio.open(src / f"raw/{row.sample_id}.tif") as f:
        image = f.read()
        tf, crs = f.transform, f.crs
    if image.shape != (len(RAW_BANDS), PATCH_PX, PATCH_PX):
        raise ValueError(f"{row.sample_id}: unexpected raw shape {image.shape}")
    if tf.b != 0 or tf.d != 0:
        raise ValueError(f"{row.sample_id}: rotated geotransform")
    image = image[[RAW_BANDS.index(b) for b in LANDSAT_BANDS]]

    acquired = datetime.fromisoformat(row.datetime)
    time_range = (acquired, acquired + timedelta(minutes=1))
    projection = Projection(CRS.from_wkt(crs.to_wkt()), tf.a, tf.e)
    # Patch origins sit half a pixel off the 30 m lattice, so windows snap to
    # it (a 15 m georeference shift; pixels and dots are unchanged).
    col0, row0 = round(tf.c / tf.a), round(tf.f / tf.e)
    bounds = (col0, row0, col0 + PATCH_PX, row0 + PATCH_PX)

    for name, cls in LABELS.items():
        dots = (
            (int(row[f"{name}_row0"]), int(row[f"{name}_col0"])),
            (int(row[f"{name}_row1"]), int(row[f"{name}_col1"])),
        )
        window = Window(
            storage=dataset.storage,
            group=GROUP,
            name=f"{row.sample_id}_{name}",
            projection=projection,
            bounds=bounds,
            time_range=time_range,
            options={
                "split": row.split,
                "basin_split": basin_split,
                "sample_id": row.sample_id,
                "region": row.region,
            },
        )
        window.save()
        write_raster(window, "landsat", LANDSAT_BANDS, image, time_range)
        write_raster(window, "label", ["label"], dot_label(dots, cls), None)


def main() -> None:
    """Run the export."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", required=True, help="Extracted geopathfinder/")
    parser.add_argument("--basin_splits", required=True, help="splits_basin.json")
    parser.add_argument("--ds_path", required=True, help="Output rslearn root")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    src, ds_path = Path(args.src), UPath(args.ds_path)
    if ds_path.exists():
        raise FileExistsError(f"{ds_path} exists; pick a new path")
    meta = pd.read_parquet(src / "metadata.parquet")
    basin = basin_split_lookup(Path(args.basin_splits))
    missing = set(meta.sample_id) - set(basin)
    if missing:
        raise ValueError(f"{len(missing)} patches have no basin split")

    ds_path.mkdir(parents=True)
    shutil.copyfile(CONFIG_PATH, ds_path / "config.json")
    dataset = RslearnDataset(ds_path)
    rows = [row for _, row in meta.iterrows()]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(lambda r: export_patch(dataset, src, r, basin[r.sample_id]), rows))
    logger.info(json.dumps(meta.split.value_counts().to_dict()))


if __name__ == "__main__":
    main()
