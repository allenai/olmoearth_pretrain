"""Stage 1: region x ERA5-Land cell weights for the CY-Bench admin units.

For every (crop, country, admin unit) with a CY-Bench yield csv:

1. rasterize the admin polygon on a 0.01 deg sub-grid (10 x 10 per ERA5 cell)
   -> ``w_area`` = fraction of each cell inside the polygon;
2. resample the WorldCereal crop area-fraction image onto the same sub-grid
   -> ``w_crop`` = coverage-weighted crop fraction (CY-Bench's own weighting);
3. fall back to ``w_area`` when the region has no crop pixels
   (``crop_fallback`` flag).

Outputs ``weights.parquet`` (one row per region x cell, canonical ``i, j``
indices) and ``regions.parquet`` (one row per region).

Usage::

    python -m olmoearth_pretrain.dataset_creation.cybench.build_weights \
        --polygons-root .../raw/polygons/polygons \
        --afi-dir <AgML-CY-Bench>/data_preparation/global_crop_AFIs_ESA_WC \
        --labels-root .../raw/cybench-data/cybench-data --out-dir .../meta
"""

from __future__ import annotations

import argparse
import logging
import math
import multiprocessing
from pathlib import Path
from typing import Any, NamedTuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.windows
import shapely
from affine import Affine
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

HALF_CELL = CELL_DEG / 2.0
CROP_FALLBACK_RATIO = 1e-3  # w_crop total below this fraction of w_area -> use area


class Footprint(NamedTuple):
    """The block of canonical cells covering a polygon, plus its fine sub-grid."""

    i_min: int
    j_min: int
    n_rows: int
    n_cols: int

    @classmethod
    def around(cls, bounds: tuple[float, float, float, float]) -> Footprint:
        """Smallest block of cells containing ``(minx, miny, maxx, maxy)``."""
        minx, miny, maxx, maxy = bounds
        i_min = max(int(math.floor((90.0 + HALF_CELL - maxy) / CELL_DEG)), 0)
        i_max = min(int(math.floor((90.0 + HALF_CELL - miny) / CELL_DEG)), N_LAT - 1)
        j_min = int(math.floor((minx + HALF_CELL) / CELL_DEG))
        j_max = int(math.floor((maxx + HALF_CELL) / CELL_DEG))
        return cls(i_min, j_min, i_max - i_min + 1, j_max - j_min + 1)

    @property
    def geo_bounds(self) -> tuple[float, float, float, float]:
        """(west, south, east, north) of the block in degrees."""
        north = 90.0 + HALF_CELL - CELL_DEG * self.i_min
        west = CELL_DEG * self.j_min - HALF_CELL
        return (
            west,
            north - CELL_DEG * self.n_rows,
            west + CELL_DEG * self.n_cols,
            north,
        )

    @property
    def fine_shape(self) -> tuple[int, int]:
        """(rows, cols) of the 0.01 deg sub-grid over the block."""
        return self.n_rows * FINE_PER_CELL, self.n_cols * FINE_PER_CELL

    @property
    def fine_transform(self) -> Affine:
        """Affine transform of the sub-grid (north-up, 0.01 deg pixels)."""
        west, _, _, north = self.geo_bounds
        return from_origin(west, north, FINE_DEG, FINE_DEG)


def _fine_mask(geom: shapely.Geometry, fp: Footprint) -> np.ndarray:
    """1.0 on sub-pixels inside ``geom``, 0.0 elsewhere."""
    kwargs = dict(
        out_shape=fp.fine_shape, transform=fp.fine_transform, fill=0, dtype="uint8"
    )
    inside = rasterio.features.rasterize([(geom, 1)], **kwargs)
    if inside.sum() == 0:  # polygon smaller than one sub-pixel
        inside = rasterio.features.rasterize([(geom, 1)], all_touched=True, **kwargs)
    return inside.astype(np.float32)


def crop_fraction(afi: rasterio.DatasetReader, fp: Footprint) -> np.ndarray:
    """Crop area fraction (0..1) of the AFI raster on the fine sub-grid."""
    west, south, east, north = fp.geo_bounds
    window = rasterio.windows.from_bounds(west, south, east, north, afi.transform)
    pct = afi.read(
        1,
        window=window,
        out_shape=fp.fine_shape,
        resampling=Resampling.average,
        boundless=True,
        fill_value=0,
        out_dtype="float32",
    )
    return np.clip(np.nan_to_num(pct, nan=0.0), 0.0, 100.0) / 100.0


def _block_mean(fine: np.ndarray) -> np.ndarray:
    """Mean over each cell's FINE_PER_CELL x FINE_PER_CELL sub-pixels."""
    h, w = fine.shape
    k = FINE_PER_CELL
    return fine.reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def part_weights(geom: shapely.Geometry, afi: rasterio.DatasetReader) -> pd.DataFrame:
    """Per-cell weights for a polygon that does not cross the antimeridian."""
    fp = Footprint.around(geom.bounds)
    inside = _fine_mask(geom, fp)
    w_area = _block_mean(inside)
    w_crop = _block_mean(inside * crop_fraction(afi, fp))
    rows, cols = np.nonzero(w_area > 0)
    return pd.DataFrame(
        {
            "i": rows + fp.i_min,
            "j": (cols + fp.j_min) % N_LON,
            "w_area": w_area[rows, cols],
            "w_crop": w_crop[rows, cols],
        }
    )


def region_weights(
    geom: shapely.Geometry, afi: rasterio.DatasetReader
) -> tuple[pd.DataFrame, bool]:
    """Per-cell weights for one admin polygon; returns (weights, used_area_fallback)."""
    minx, _, maxx, _ = geom.bounds
    if maxx - minx > 180.0:  # crosses the antimeridian: handle each side separately
        parts = [
            geom.intersection(shapely.box(-180.0, -90.0, 0.0, 90.0)),
            geom.intersection(shapely.box(0.0, -90.0, 180.0, 90.0)),
        ]
        parts = [p for p in parts if not p.is_empty]
    else:
        parts = [geom]
    df = pd.concat([part_weights(p, afi) for p in parts], ignore_index=True)
    df = df.groupby(["i", "j"], as_index=False)[["w_area", "w_crop"]].sum()
    fallback = df["w_crop"].sum() < CROP_FALLBACK_RATIO * df["w_area"].sum()
    if fallback:
        df["w_crop"] = df["w_area"]
    return df, bool(fallback)


def weights_for_country(job: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Worker: weights and region rows for every crop of one country."""
    cc, crops, polygons_root, afi_dir, labels_root = job
    polygons = _load_polygons(Path(polygons_root) / cc / f"{cc}.shp")
    weight_frames, region_rows = [], []
    for crop in crops:
        labels = load_labels(Path(labels_root), crop, cc) if labels_root else None
        regions = (
            polygons
            if labels is None
            else _labelled_polygons(polygons, labels, crop, cc)
        )
        with rasterio.open(Path(afi_dir) / CROP_TO_AFI[crop]) as afi:
            for row in regions.itertuples(index=False):
                if row.geometry is None or row.geometry.is_empty:
                    continue
                weights, fallback = region_weights(row.geometry, afi)
                key = region_key(crop, cc, row.adm_id)
                weight_frames.append(
                    weights.assign(region=key, crop=crop, cc=cc, adm_id=row.adm_id)
                )
                region_rows.append(
                    _region_row(key, crop, cc, row, weights, fallback, labels)
                )
    weights_df = (
        pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
    )
    regions_df = pd.DataFrame(region_rows)
    logger.info("%s: %d regions, %d cells", cc, len(regions_df), len(weights_df))
    return weights_df, regions_df


def _load_polygons(shp: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(shp)
    gdf = (gdf if gdf.crs is not None else gdf.set_crs(4326)).to_crs(4326)
    gdf["adm_id"] = gdf["adm_id"].astype(str)
    return gdf.drop_duplicates("adm_id")


def _labelled_polygons(
    polygons: gpd.GeoDataFrame, labels: pd.DataFrame, crop: str, cc: str
) -> gpd.GeoDataFrame:
    """Polygons that have at least one label; warn about labels without a polygon."""
    labelled = set(labels["adm_id"])
    sub = polygons[polygons["adm_id"].isin(labelled)]
    missing = labelled - set(sub["adm_id"])
    if missing:
        logger.warning(
            "%s/%s: %d labelled adm_ids have no polygon (e.g. %s)",
            crop,
            cc,
            len(missing),
            sorted(missing)[:3],
        )
    return sub


def _region_row(
    key: str,
    crop: str,
    cc: str,
    row: Any,
    weights: pd.DataFrame,
    fallback: bool,
    labels: pd.DataFrame | None,
) -> dict[str, Any]:
    point = row.geometry.representative_point()
    years = (
        labels.loc[labels["adm_id"] == row.adm_id, "year"]
        if labels is not None
        else pd.Series(dtype=int)
    )
    return dict(
        region=key,
        crop=crop,
        cc=cc,
        adm_id=row.adm_id,
        centroid_lat=float(point.y),
        centroid_lon=float(point.x),
        area_km2=float(
            gpd.GeoSeries([row.geometry], crs=4326).to_crs("+proj=cea").area.iloc[0]
            / 1e6
        ),
        n_cells=int(len(weights)),
        crop_fallback=fallback,
        n_labels=int(len(years)),
        min_year=int(years.min()) if len(years) else -1,
        max_year=int(years.max()) if len(years) else -1,
    )


def plan_jobs(args: argparse.Namespace) -> list[tuple]:
    """One job per country: (cc, crops, polygons_root, afi_dir, labels_root | None)."""
    labels_root = None if args.labels_root == "none" else args.labels_root
    if labels_root:
        pairs = [
            (c, cc)
            for c, cc in list_crop_countries(Path(labels_root))
            if c in args.crops
        ]
    else:
        pairs = [
            (c, p.name)
            for p in sorted(Path(args.polygons_root).iterdir())
            if p.is_dir()
            for c in args.crops
        ]
    crops_by_cc: dict[str, list[str]] = {}
    for crop, cc in pairs:
        if args.only_cc and cc not in args.only_cc:
            continue
        if not (Path(args.polygons_root) / cc / f"{cc}.shp").exists():
            logger.warning("no polygons for %s, skipping", cc)
            continue
        crops_by_cc.setdefault(cc, []).append(crop)
    return [
        (cc, crops, args.polygons_root, args.afi_dir, labels_root)
        for cc, crops in sorted(crops_by_cc.items())
    ]


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
        help="CY-Bench data root; 'none' = all polygons",
    )
    ap.add_argument("--crops", nargs="+", default=sorted(CROP_TO_AFI))
    ap.add_argument(
        "--only-cc", nargs="*", default=None, help="restrict to these country codes"
    )
    ap.add_argument("--out-dir", default=str(DEFAULT_ROOT / "meta"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    jobs = plan_jobs(args)
    logger.info("%d countries", len(jobs))
    if args.workers > 1:
        with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
            results = pool.map(weights_for_country, jobs)
    else:
        results = [weights_for_country(j) for j in jobs]

    weights = pd.concat([w for w, _ in results if len(w)], ignore_index=True)
    regions = pd.concat([r for _, r in results if len(r)], ignore_index=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights.to_parquet(out_dir / "weights.parquet", index=False)
    regions.to_parquet(out_dir / "regions.parquet", index=False)
    logger.info(
        "wrote %d weight rows for %d regions (%d crop-mask fallbacks) to %s",
        len(weights), len(regions), int(regions["crop_fallback"].sum()), out_dir,
    )  # fmt: skip


if __name__ == "__main__":
    main()
