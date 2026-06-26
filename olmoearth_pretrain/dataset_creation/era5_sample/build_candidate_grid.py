"""Build candidate grid for ERA5 climate-stratified sampling.

Creates an adaptive ~10 km global land grid, snaps each centroid to its
ERA5-Land 0.1° cell, deduplicates to one point per cell, then attaches
Köppen class, elevation band, and latitude band.

Output: candidates.parquet in the metadata directory.

Usage:
    python -m olmoearth_pretrain.dataset_creation.era5_sample.build_candidate_grid \
        --metadata-dir /weka/dfive-default/helios/dataset/era5enc_pretrain/metadata
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Geod
from shapely.geometry import Point, box
from shapely.prepared import prep

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

STEP_M = 10_000  # 10 km grid spacing
TILE_SIZE_DEG = 10.0  # Processing tile size

# ERA5-Land grid: 0.1° cells. Cell ID encodes row/col from (-90, -180).
ERA5_RESOLUTION_DEG = 0.1

# Elevation bands (meters)
ELEV_BAND_EDGES = [0, 250, 750, 1500, 3000, 9000]
ELEV_BAND_LABELS = ["<250", "250-750", "750-1500", "1500-3000", ">3000"]

# Latitude bands (degrees)
LAT_BAND_EDGES = [-90, -60, -30, 0, 30, 60, 90]
LAT_BAND_LABELS = ["[-90,-60)", "[-60,-30)", "[-30,0)", "[0,30)", "[30,60)", "[60,90]"]

# Köppen-Geiger class codes (integer values from Beck et al. 2023 GeoTIFF)
KOPPEN_CODES = {
    1: "Af",
    2: "Am",
    3: "Aw",
    4: "BWh",
    5: "BWk",
    6: "BSh",
    7: "BSk",
    8: "Csa",
    9: "Csb",
    10: "Csc",
    11: "Cwa",
    12: "Cwb",
    13: "Cwc",
    14: "Cfa",
    15: "Cfb",
    16: "Cfc",
    17: "Dsa",
    18: "Dsb",
    19: "Dsc",
    20: "Dsd",
    21: "Dwa",
    22: "Dwb",
    23: "Dwc",
    24: "Dwd",
    25: "Dfa",
    26: "Dfb",
    27: "Dfc",
    28: "Dfd",
    29: "ET",
    30: "EF",
}


def _era5_cell_id(lon: float, lat: float) -> int:
    """Compute a unique integer cell ID for the ERA5-Land 0.1° cell containing (lon, lat).

    Cell (0,0) is at the top-left corner of the global grid: (-180, 90).
    """
    col = int(math.floor((lon + 180.0) / ERA5_RESOLUTION_DEG))
    row = int(math.floor((90.0 - lat) / ERA5_RESOLUTION_DEG))
    col = max(0, min(col, 3599))
    row = max(0, min(row, 1799))
    return row * 3600 + col


def _era5_cell_center(cell_id: int) -> tuple[float, float]:
    """Return the (lon, lat) center of an ERA5-Land cell given its ID."""
    row = cell_id // 3600
    col = cell_id % 3600
    lon = -180.0 + (col + 0.5) * ERA5_RESOLUTION_DEG
    lat = 90.0 - (row + 0.5) * ERA5_RESOLUTION_DEG
    return lon, lat


def _adaptive_grid_centroids(
    minx: float, maxx: float, miny: float, maxy: float, step_m: int
) -> list[tuple[float, float]]:
    """Generate adaptive-spacing grid centroids (lon, lat) over a geographic tile.

    Grid spacing is constant in meters (so longitude step widens near equator,
    shrinks near poles) for roughly equal ground-area cells.
    """
    geod = Geod(ellps="WGS84")

    _POLE_EPS = 0.05
    miny = max(miny, -90.0 + _POLE_EPS)
    maxy = min(maxy, 90.0 - _POLE_EPS)

    if miny >= maxy:
        return []

    lat_grid: list[float] = [miny]
    while lat_grid[-1] < maxy:
        _, lat_next, _ = geod.fwd(minx, lat_grid[-1], 0, step_m)
        lat_next = min(lat_next, maxy)
        lat_grid.append(lat_next)
        if lat_next >= maxy:
            break

    centroids: list[tuple[float, float]] = []
    for i in range(len(lat_grid) - 1):
        lat_center = (lat_grid[i] + lat_grid[i + 1]) / 2.0

        lon_row: list[float] = [minx]
        while lon_row[-1] < maxx:
            lon_next, _, _ = geod.fwd(lon_row[-1], lat_center, 90, step_m)
            if lon_next <= lon_row[-1]:
                lon_row.append(maxx)
                break
            lon_next = min(lon_next, maxx)
            lon_row.append(lon_next)
            if lon_next >= maxx:
                break

        for j in range(len(lon_row) - 1):
            lon_center = (lon_row[j] + lon_row[j + 1]) / 2.0
            centroids.append((lon_center, lat_center))

    return centroids


def _sample_raster_at_points(
    raster_path: Path, lons: np.ndarray, lats: np.ndarray
) -> np.ndarray:
    """Sample a raster at given lon/lat coordinates. Returns 1D array of values."""
    with rasterio.open(raster_path) as src:
        # Convert lon/lat to pixel coordinates
        transform = src.transform
        nodata = src.nodata
        data = src.read(1)

        cols, rows = ~transform * (lons, lats)
        cols = np.round(cols).astype(int)
        rows = np.round(rows).astype(int)

        # Clip to valid range
        rows = np.clip(rows, 0, data.shape[0] - 1)
        cols = np.clip(cols, 0, data.shape[1] - 1)

        values = data[rows, cols]
        if nodata is not None:
            values = np.where(values == nodata, np.nan, values.astype(float))
        else:
            values = values.astype(float)

    return values


def _assign_elev_band(elevations: np.ndarray) -> np.ndarray:
    """Assign elevation band index (0-4) from elevation values."""
    return np.digitize(elevations, ELEV_BAND_EDGES[1:], right=False).astype(np.int8)


def _assign_lat_band(lats: np.ndarray) -> np.ndarray:
    """Assign latitude band index (0-5) from latitude values."""
    return np.digitize(lats, LAT_BAND_EDGES[1:-1], right=False).astype(np.int8)


def build_candidate_grid(
    metadata_dir: str | Path,
    step_m: int = STEP_M,
    tile_size_deg: float = TILE_SIZE_DEG,
) -> Path:
    """Build the candidate grid and write candidates.parquet.

    Args:
        metadata_dir: Directory containing downloaded source data (from fetch step)
            and where candidates.parquet will be written.
        step_m: Grid cell spacing in meters.
        tile_size_deg: Processing tile width/height in degrees.

    Returns:
        Path to the output parquet file.
    """
    metadata_dir = Path(metadata_dir)
    output_path = metadata_dir / "candidates.parquet"

    if output_path.exists():
        logger.info("candidates.parquet already exists at %s, skipping", output_path)
        return output_path

    # Load land polygons
    land_shp = metadata_dir / "ne_10m_land.shp"
    if not land_shp.exists():
        raise FileNotFoundError(
            f"Land shapefile not found at {land_shp}. "
            "Run fetch_stratification_sources.py first."
        )
    logger.info("Loading land polygons from %s", land_shp)
    land_gdf = gpd.read_file(land_shp)
    land_union = land_gdf.union_all()
    land_prepared = prep(land_union)

    # Generate global adaptive grid filtered to land
    logger.info("Generating global adaptive grid at %d m spacing...", step_m)
    lat_starts = np.arange(-90.0, 90.0, tile_size_deg)
    lon_starts = np.arange(-180.0, 180.0, tile_size_deg)

    all_centroids: list[tuple[float, float]] = []
    for row_idx, lat_start in enumerate(lat_starts):
        lat_end = min(lat_start + tile_size_deg, 90.0)
        row_count = 0

        for lon_start in lon_starts:
            lon_end = min(lon_start + tile_size_deg, 180.0)

            tile_bbox = box(lon_start, lat_start, lon_end, lat_end)
            if not land_prepared.intersects(tile_bbox):
                continue

            centroids = _adaptive_grid_centroids(
                lon_start, lon_end, lat_start, lat_end, step_m
            )

            # Filter to land
            land_centroids = [
                (lon, lat)
                for lon, lat in centroids
                if land_prepared.intersects(Point(lon, lat))
            ]
            all_centroids.extend(land_centroids)
            row_count += len(land_centroids)

        logger.info(
            "Lat band %+6.1f to %+6.1f  |  row %2d/%d  |  +%d points (total %d)",
            lat_start,
            lat_end,
            row_idx + 1,
            len(lat_starts),
            row_count,
            len(all_centroids),
        )

    logger.info("Total land centroids before dedup: %d", len(all_centroids))

    # Convert to arrays
    lons = np.array([c[0] for c in all_centroids])
    lats = np.array([c[1] for c in all_centroids])

    # Compute ERA5-Land cell IDs and deduplicate (one point per cell)
    cell_ids = np.array([_era5_cell_id(lon, lat) for lon, lat in all_centroids])
    _, unique_indices = np.unique(cell_ids, return_index=True)
    unique_indices = np.sort(unique_indices)

    lons = lons[unique_indices]
    lats = lats[unique_indices]
    cell_ids = cell_ids[unique_indices]
    logger.info("After ERA5-cell dedup: %d unique cells", len(lons))

    # Snap centroids to ERA5-Land cell centers for consistency
    snapped = np.array([_era5_cell_center(cid) for cid in cell_ids])
    lons = snapped[:, 0]
    lats = snapped[:, 1]

    # Sample Köppen class
    koppen_path = metadata_dir / "koppen_geiger_0p1.tif"
    if not koppen_path.exists():
        raise FileNotFoundError(
            f"Köppen GeoTIFF not found at {koppen_path}. "
            "Run fetch_stratification_sources.py first."
        )
    logger.info("Sampling Köppen classes...")
    koppen_values = _sample_raster_at_points(koppen_path, lons, lats)
    koppen_int = np.nan_to_num(koppen_values, nan=0).astype(np.int8)

    # Sample elevation
    dem_path = metadata_dir / "ETOPO_2022_v1_60s_N90W180_surface.tif"
    if not dem_path.exists():
        raise FileNotFoundError(
            f"DEM not found at {dem_path}. Run fetch_stratification_sources.py first."
        )
    logger.info("Sampling elevations...")
    elevations = _sample_raster_at_points(dem_path, lons, lats)
    elevations = np.nan_to_num(elevations, nan=0)

    # Assign bands
    elev_bands = _assign_elev_band(elevations)
    lat_bands = _assign_lat_band(lats)

    # Build dataframe
    df = pd.DataFrame(
        {
            "lon": lons,
            "lat": lats,
            "era5_cell_id": cell_ids,
            "koppen_code": koppen_int,
            "koppen_class": [KOPPEN_CODES.get(int(k), "unknown") for k in koppen_int],
            "elevation_m": elevations,
            "elev_band": elev_bands,
            "elev_band_label": [ELEV_BAND_LABELS[min(b, 4)] for b in elev_bands],
            "lat_band": lat_bands,
            "lat_band_label": [LAT_BAND_LABELS[min(b, 5)] for b in lat_bands],
        }
    )

    # Filter out ocean/invalid (koppen_code == 0 typically means ocean)
    n_before = len(df)
    df = df[df["koppen_code"] > 0].reset_index(drop=True)
    logger.info(
        "Filtered %d ocean/invalid points, %d candidates remain",
        n_before - len(df),
        len(df),
    )

    # Write output
    df.to_parquet(output_path, index=False)
    logger.info("Wrote %d candidates to %s", len(df), output_path)

    # Print summary statistics
    logger.info("--- Candidate Grid Summary ---")
    logger.info("  Total candidates: %d", len(df))
    logger.info("  Köppen classes present: %d / 30", df["koppen_code"].nunique())
    logger.info(
        "  Elevation band distribution:\n%s",
        df["elev_band_label"].value_counts().to_string(),
    )
    logger.info(
        "  Latitude band distribution:\n%s",
        df["lat_band_label"].value_counts().to_string(),
    )

    return output_path


def main() -> None:
    """Run the candidate grid build CLI."""
    parser = argparse.ArgumentParser(
        description="Build candidate grid for ERA5 climate-stratified sampling."
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        required=True,
        help="Directory with source data (from fetch step) and output location.",
    )
    parser.add_argument(
        "--step-m",
        type=int,
        default=STEP_M,
        help=f"Grid spacing in meters (default: {STEP_M}).",
    )
    args = parser.parse_args()

    build_candidate_grid(metadata_dir=args.metadata_dir, step_m=args.step_m)


if __name__ == "__main__":
    main()
