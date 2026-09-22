"""Stage 1: region x ERA5-Land cell weights for the CY-Bench admin units.

For every (crop, country, admin unit) with a CY-Bench yield csv, rasterize the
admin polygon on a 0.01 deg sub-grid (10 x 10 per ERA5-Land cell), read the
WorldCereal crop area-fraction image (AFI, 0..100 %) resampled onto the same
sub-grid, and reduce to per-cell weights:

* ``w_area`` — fraction of the ERA5 cell covered by the polygon;
* ``w_crop`` — coverage-weighted mean crop fraction (what CY-Bench used to
  aggregate its own predictors). Falls back to ``w_area`` when the region has
  no crop pixels at all (``crop_fallback`` flag in the regions table).

Outputs two parquet files:

* ``weights.parquet``: ``region, crop, cc, adm_id, i, j, w_area, w_crop`` with
  ``(i, j)`` the canonical ERA5-Land grid indices (see ``common``).
* ``regions.parquet``: one row per region with centroid, cell count, label year
  span and the fallback flag.

Usage::

    python -m olmoearth_pretrain.dataset_creation.cybench.build_weights \
        --polygons-root /weka/.../cybench/raw/polygons/polygons \
        --afi-dir /path/to/AgML-CY-Bench/data_preparation/global_crop_AFIs_ESA_WC \
        --labels-root /weka/.../cybench/raw/cybench-data/cybench-data \
        --out-dir /weka/.../cybench/meta
"""

from __future__ import annotations

import argparse
import logging
import math
import multiprocessing
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.windows
import shapely
from rasterio.enums import Resampling
from rasterio.transform import from_origin

from .common import (
    CELL_DEG,
    CROP_TO_AFI,
    DEFAULT_LABELS_ROOT,
    DEFAULT_POLYGONS_ROOT,
    DEFAULT_ROOT,
    FINE_DEG,
    FINE_PER_CELL,
    N_LAT,
    N_LON,
    list_crop_countries,
    load_labels,
    region_key,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HALF = CELL_DEG / 2.0


def _cell_rows(miny: float, maxy: float) -> tuple[int, int]:
    """Inclusive canonical row range covering [miny, maxy]."""
    i_min = int(math.floor((90.0 + HALF - maxy) / CELL_DEG))
    i_max = int(math.floor((90.0 + HALF - miny) / CELL_DEG))
    return max(i_min, 0), min(i_max, N_LAT - 1)


def _cell_cols(minx: float, maxx: float) -> tuple[int, int]:
    """Inclusive (unwrapped) column range covering [minx, maxx] (lon in -180..180)."""
    j_min = int(math.floor((minx + HALF) / CELL_DEG))
    j_max = int(math.floor((maxx + HALF) / CELL_DEG))
    return j_min, j_max


def _block_mean(a: np.ndarray, k: int) -> np.ndarray:
    """Mean over k x k blocks of a 2-D array whose dims are multiples of k."""
    h, w = a.shape
    return a.reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def _part_weights(
    geom: shapely.Geometry, afi: rasterio.DatasetReader
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Weights for one polygon part that does not cross the antimeridian.

    Returns ``(i, j, w_area, w_crop)`` arrays over the cells with w_area > 0.
    """
    minx, miny, maxx, maxy = geom.bounds
    i_min, i_max = _cell_rows(miny, maxy)
    j_min, j_max = _cell_cols(minx, maxx)
    n_rows = i_max - i_min + 1
    n_cols = j_max - j_min + 1
    north = 90.0 + HALF - CELL_DEG * i_min
    west = CELL_DEG * j_min - HALF
    south = north - CELL_DEG * n_rows
    east = west + CELL_DEG * n_cols
    fine_shape = (n_rows * FINE_PER_CELL, n_cols * FINE_PER_CELL)
    transform = from_origin(west, north, FINE_DEG, FINE_DEG)

    cover = rasterio.features.rasterize(
        [(geom, 1)], out_shape=fine_shape, transform=transform, fill=0, dtype="uint8"
    )
    if cover.sum() == 0:
        # Polygon smaller than a 0.01 deg sub-pixel: take every touched sub-pixel.
        cover = rasterio.features.rasterize(
            [(geom, 1)],
            out_shape=fine_shape,
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=True,
        )
    cover_f = cover.astype(np.float32)

    # Crop area fraction resampled (mean) onto the fine grid; 0 outside the AFI
    # extent (AFI covers lat -56..75, which contains every CY-Bench region).
    win = rasterio.windows.from_bounds(west, south, east, north, afi.transform)
    afi_fine = afi.read(
        1,
        window=win,
        out_shape=fine_shape,
        resampling=Resampling.average,
        boundless=True,
        fill_value=0,
        out_dtype="float32",
    )
    afi_fine = np.clip(np.nan_to_num(afi_fine, nan=0.0), 0.0, 100.0) / 100.0

    w_area = _block_mean(cover_f, FINE_PER_CELL)
    w_crop = _block_mean(cover_f * afi_fine, FINE_PER_CELL)
    ii, jj = np.nonzero(w_area > 0)
    i = ii + i_min
    j = (jj + j_min) % N_LON
    return i, j, w_area[ii, jj], w_crop[ii, jj]


def region_weights(
    geom: shapely.Geometry, afi: rasterio.DatasetReader
) -> tuple[pd.DataFrame, bool]:
    """Per-cell weights for one admin polygon (handles antimeridian crossing)."""
    minx, _, maxx, _ = geom.bounds
    if maxx - minx > 180.0:
        parts = [
            geom.intersection(shapely.box(-180.0, -90.0, 0.0, 90.0)),
            geom.intersection(shapely.box(0.0, -90.0, 180.0, 90.0)),
        ]
        parts = [p for p in parts if not p.is_empty]
    else:
        parts = [geom]
    frames = []
    for p in parts:
        i, j, wa, wc = _part_weights(p, afi)
        frames.append(pd.DataFrame({"i": i, "j": j, "w_area": wa, "w_crop": wc}))
    df = pd.concat(frames, ignore_index=True)
    # A polygon split into parts (or a multipolygon) can touch the same cell
    # twice; sum the contributions.
    df = df.groupby(["i", "j"], as_index=False)[["w_area", "w_crop"]].sum()
    fallback = False
    # Fall back to plain area weights when the crop mask is (near-)empty for the
    # region; a handful of crop sub-pixels would otherwise dominate the mean.
    if df["w_crop"].sum() < 1e-3 * df["w_area"].sum():
        df["w_crop"] = df["w_area"]
        fallback = True
    return df, fallback


def _process_country(args: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Worker: weights + regions for every crop of one country."""
    cc, crops, polygons_root, afi_dir, labels_root = args
    shp = Path(polygons_root) / cc / f"{cc}.shp"
    gdf = gpd.read_file(shp)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(4326)
    gdf["adm_id"] = gdf["adm_id"].astype(str)
    gdf = gdf.drop_duplicates("adm_id")

    weight_frames, region_rows = [], []
    for crop in crops:
        labels = None
        if labels_root is not None:
            labels = load_labels(Path(labels_root), crop, cc)
            keep = set(labels["adm_id"])
            sub = gdf[gdf["adm_id"].isin(keep)]
            missing = keep - set(sub["adm_id"])
            if missing:
                logger.warning(
                    "%s/%s: %d labelled adm_ids have no polygon (e.g. %s)",
                    crop,
                    cc,
                    len(missing),
                    sorted(missing)[:3],
                )
        else:
            sub = gdf
        with rasterio.open(Path(afi_dir) / CROP_TO_AFI[crop]) as afi:
            for row in sub.itertuples(index=False):
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                df, fallback = region_weights(geom, afi)
                key = region_key(crop, cc, row.adm_id)
                df.insert(0, "adm_id", row.adm_id)
                df.insert(0, "cc", cc)
                df.insert(0, "crop", crop)
                df.insert(0, "region", key)
                weight_frames.append(df)
                c = geom.representative_point()
                yrs = (
                    labels.loc[labels["adm_id"] == row.adm_id, "year"]
                    if labels is not None
                    else pd.Series(dtype=int)
                )
                region_rows.append(
                    dict(
                        region=key,
                        crop=crop,
                        cc=cc,
                        adm_id=row.adm_id,
                        centroid_lat=float(c.y),
                        centroid_lon=float(c.x),
                        area_km2=float(
                            gpd.GeoSeries([geom], crs=4326)
                            .to_crs("+proj=cea")
                            .area.iloc[0]
                            / 1e6
                        ),
                        n_cells=int(len(df)),
                        crop_fallback=bool(fallback),
                        n_labels=int(len(yrs)),
                        min_year=int(yrs.min()) if len(yrs) else -1,
                        max_year=int(yrs.max()) if len(yrs) else -1,
                    )
                )
    w = pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
    r = pd.DataFrame(region_rows)
    logger.info("%s: %d regions, %d cells", cc, len(r), len(w))
    return w, r


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--polygons-root", default=str(DEFAULT_POLYGONS_ROOT))
    ap.add_argument(
        "--afi-dir", required=True, help="dir with the WorldCereal AFI tifs"
    )
    ap.add_argument(
        "--labels-root",
        default=str(DEFAULT_LABELS_ROOT),
        help="CY-Bench data root (<crop>/<CC>/yield_*.csv); 'none' to use all polygons",
    )
    ap.add_argument("--crops", nargs="+", default=sorted(CROP_TO_AFI))
    ap.add_argument(
        "--only-cc", nargs="*", default=None, help="restrict to these country codes"
    )
    ap.add_argument("--out-dir", default=str(DEFAULT_ROOT / "meta"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    labels_root = None if args.labels_root == "none" else Path(args.labels_root)
    if labels_root is not None:
        pairs = [
            (c, cc) for c, cc in list_crop_countries(labels_root) if c in args.crops
        ]
    else:
        pairs = [
            (c, p.name)
            for p in sorted(Path(args.polygons_root).iterdir())
            if p.is_dir()
            for c in args.crops
        ]
    by_cc: dict[str, list[str]] = {}
    for crop, cc in pairs:
        if args.only_cc and cc not in args.only_cc:
            continue
        if not (Path(args.polygons_root) / cc / f"{cc}.shp").exists():
            logger.warning("no polygons for %s, skipping", cc)
            continue
        by_cc.setdefault(cc, []).append(crop)
    jobs = [
        (
            cc,
            crops,
            args.polygons_root,
            args.afi_dir,
            str(labels_root) if labels_root else None,
        )
        for cc, crops in sorted(by_cc.items())
    ]
    logger.info("%d countries", len(jobs))

    if args.workers > 1:
        with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
            results = pool.map(_process_country, jobs)
    else:
        results = [_process_country(j) for j in jobs]

    weights = pd.concat([w for w, _ in results if len(w)], ignore_index=True)
    regions = pd.concat([r for _, r in results if len(r)], ignore_index=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights.to_parquet(out_dir / "weights.parquet", index=False)
    regions.to_parquet(out_dir / "regions.parquet", index=False)
    logger.info(
        "wrote %d weight rows for %d regions (%d crop-mask fallbacks) to %s",
        len(weights),
        len(regions),
        int(regions["crop_fallback"].sum()),
        out_dir,
    )
    cells = weights[["i", "j"]].drop_duplicates()
    logger.info(
        "distinct ERA5 cells: %d; distinct (lat,lon) 150x300 chunks: %d",
        len(cells),
        len((cells[["i", "j"]] // np.array([150, 300])).drop_duplicates()),
    )


if __name__ == "__main__":
    main()
