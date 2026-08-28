"""Process "Bark Beetle Disturbance, Pokljuka (Slovenia)" into open-set-segmentation patches.

Source: Zenodo 10.5281/zenodo.15260584 ("Bark beetle geospatial dataset from 2017 to
2021 (Pokljuka, Slovenia)"). A single 10 m raster "Change detection mask 2017-2021.tif":
spruce bark-beetle disturbance derived from Sentinel-2 NDVI + NBSI time series processed
with the CuSum change-detection algorithm (intersection of high-magnitude breakpoint maps
from both indices), overlaid with in-situ ground truth. Field-validated.

The raster is EPSG:32633 (UTM 33N), 3000x2500 px at 10 m, over the Pokljuka plateau
(~30 x 25 km). Values: 0 = no disturbance / background (99.24%, file nodata=0),
1 = bark-beetle disturbance (0.758%, 56,854 px). One foreground class only.

Recipe (label_type = dense_raster, single positive class -> POSITIVE-ONLY, spec 5):
- The source is already local UTM at 10 m, so we reuse its CRS/grid directly (no reproject).
- We tile the raster into a non-overlapping 64x64 grid (640 m tiles) and keep every tile
  that contains >=1 disturbance pixel (585 of 1794 grid cells). 585 < the 1000/class cap,
  so all are kept.
- Per pixel: disturbance (source value 1) -> class id 0; every other pixel -> 255 (ignore).
  Per spec 5 we do NOT fabricate a "no disturbance" negative class for a positive-only
  dataset; downstream assembly supplies negatives from other datasets.

Time range (pre/post scheme): the mask is a cumulative 2017-2021 disturbance product with no
per-pixel disturbance dates, so the disturbance occurred somewhere in that span. Under the
pre/post change scheme we bracket the whole span with two fixed six-month windows (each <=
183 days) and ``time_range`` = null: ``pre_time_range`` = summer 2016 and ``post_time_range``
= summer 2022; ``change_time`` stays null (the exact date is unknown). Summer windows avoid
snow-cover confusion. This replaces the previous "annual disturbance-presence anchored on
2021 (1-year window, change_time = null)" encoding and the change-timing rejection: with the
disturbance bracketed between a genuine before/after image pair the timing imprecision is no
longer a problem, so the dataset is now usable. See the summary.
"""

import argparse
import multiprocessing
import os
from datetime import UTC, datetime
from typing import Any

import numpy as np
import rasterio
import tqdm
from rasterio.crs import CRS
from rasterio.windows import Window
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest

SLUG = "bark_beetle_disturbance_pokljuka_slovenia"
NAME = "Bark Beetle Disturbance, Pokljuka (Slovenia)"
SRC_NAME = "Change detection mask 2017-2021.tif"

VAL_DISTURBANCE = 1  # source value for bark-beetle disturbance
CLASS_DISTURBANCE = 0  # output class id

T = 64  # output tile size (64 px * 10 m = 640 m)
LABELED_YEAR = (
    2021  # end of the 2017-2021 period (cumulative disturbance fully expressed)
)
# Cumulative 2017-2021 disturbance mask with no per-pixel dates: the disturbance occurred
# somewhere in that span. Under the pre/post scheme we bracket the whole span -- a "before"
# window in summer 2016 and an "after" window in summer 2022 -- so the mask of where
# disturbance occurred is aligned with a genuine before/after image pair (change_time stays
# null; the exact date is unknown). Summer windows avoid snow-cover confusion.
PRE_WINDOW = (datetime(2016, 6, 1, tzinfo=UTC), datetime(2016, 9, 1, tzinfo=UTC))
POST_WINDOW = (datetime(2022, 6, 1, tzinfo=UTC), datetime(2022, 9, 1, tzinfo=UTC))

# Source grid constants (EPSG:32633, res 10 m, origin from the file transform).
ORIGIN_X = 410000
ORIGIN_Y_TOP = 5145000
RES = 10
SRC_CRS = "EPSG:32633"


def _src_path() -> str:
    return os.path.join(str(io.raw_dir(SLUG)), SRC_NAME)


def _projection() -> Projection:
    return Projection(CRS.from_string(SRC_CRS), RES, -RES)


def _bounds(iy: int, jx: int) -> tuple[int, int, int, int]:
    """Integer pixel bounds in the output Projection (x_res=10, y_res=-10)."""
    col_min = ORIGIN_X // RES + jx * T
    row_min = -(ORIGIN_Y_TOP // RES) + iy * T
    return (col_min, row_min, col_min + T, row_min + T)


def scan() -> list[dict[str, Any]]:
    """Return one record per 64x64 grid tile containing >=1 disturbance pixel."""
    with rasterio.open(_src_path()) as ds:
        arr = ds.read(1)
    mask = arr == VAL_DISTURBANCE
    H, W = mask.shape
    nby, nbx = H // T, W // T
    recs: list[dict[str, Any]] = []
    for iy in range(nby):
        for jx in range(nbx):
            n = int(mask[iy * T : (iy + 1) * T, jx * T : (jx + 1) * T].sum())
            if n > 0:
                recs.append({"iy": iy, "jx": jx, "n_disturb": n})
    return recs


def _write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    iy, jx = rec["iy"], rec["jx"]
    with rasterio.open(_src_path()) as ds:
        src = ds.read(1, window=Window(jx * T, iy * T, T, T))
    out = np.full((T, T), io.CLASS_NODATA, dtype=np.uint8)
    out[src == VAL_DISTURBANCE] = CLASS_DISTURBANCE
    proj = _projection()
    bounds = _bounds(iy, jx)
    io.write_label_geotiff(SLUG, sample_id, out, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        None,
        change_time=None,
        source_id=f"{SRC_NAME}:tile_{iy}_{jx}",
        classes_present=[CLASS_DISTURBANCE],
        pre_time_range=PRE_WINDOW,
        post_time_range=POST_WINDOW,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    recs = scan()
    recs.sort(key=lambda r: (r["iy"], r["jx"]))
    for i, r in enumerate(recs):
        r["sample_id"] = f"{i:06d}"
    total_disturb = sum(r["n_disturb"] for r in recs)
    print(f"positive tiles: {len(recs)} (total disturbance px {total_disturb})")

    io.check_disk()

    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in recs]),
            total=len(recs),
            desc="write",
        ):
            pass

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Zenodo (10.5281/zenodo.15260584)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.org/10.5281/zenodo.15260584",
                "have_locally": False,
                "annotation_method": (
                    "Sentinel-2 NDVI + NBSI time-series change detection (CuSum algorithm; "
                    "intersection of high-magnitude breakpoint maps), overlaid with in-situ "
                    "ground-truth validation."
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {
                    "id": CLASS_DISTURBANCE,
                    "name": "bark beetle disturbance",
                    "description": (
                        "Spruce forest disturbance / die-off attributed to bark-beetle "
                        "(Ips typographus) outbreak on the Pokljuka plateau, detected from "
                        "2017-2021 Sentinel-2 vegetation-index breakpoints and field-validated."
                    ),
                }
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(recs),
            "class_counts": {"bark beetle disturbance": len(recs)},
            "labeled_year": LABELED_YEAR,
            "notes": (
                "Positive-only single-class dense_raster. 64x64 (640 m) tiles cropped "
                "directly from the source EPSG:32633 10 m mask (no reprojection); every grid "
                "tile containing >=1 disturbance pixel is kept (585). Per pixel: source "
                "value 1 -> class 0 (disturbance); all other pixels -> 255 (ignore) since no "
                "confident negative class is provided (spec 5 positive-only; negatives added "
                "downstream). Cumulative 2017-2021 disturbance with no recoverable per-pixel "
                "dates, so encoded as annual disturbance-presence classification anchored on "
                "2021 (change_time=null) rather than a dated change label."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(recs)
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
