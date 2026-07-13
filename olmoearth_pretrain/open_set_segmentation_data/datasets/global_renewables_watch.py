"""Process Global Renewables Watch (GRW) into open-set-segmentation label patches.

Source: Microsoft / Planet / TNC "Global Renewables Watch", a quarterly global inventory
of solar PV installations (polygons) and wind turbines (points) detected from PlanetScope
imagery with deep learning + human QC, with per-feature construction dates. We use the
v1.0 2024-Q2 release GeoPackages published on the project's GitHub releases:

  https://github.com/microsoft/global-renewables-watch/releases/download/v1.0/solar_all_2024q2_v1.gpkg
  https://github.com/microsoft/global-renewables-watch/releases/download/v1.0/wind_all_2024q2_v1.gpkg

Two label kinds are combined into one classification dataset with a shared class scheme:
  0 = background/negative
  1 = solar_pv     (PV installation polygons, rasterized into a <=64x64 UTM tile)
  2 = wind_turbine (turbine points, encoded with the tunable DETECTION encoding:
                    a 1px positive square + 10px nodata buffer ring (21x21 ignore) +
                    background fill in a 32x32 context tile)
  255 = nodata / ignore (detection buffer rings)

Per-feature ``construction_year`` sets a ~1-year time range. Bounded to <=1000 tiles per
class (solar_pv, wind_turbine), stratified across construction years for temporal
diversity, plus a small set of background-only negative tiles for the detection class.

Run (idempotent):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_renewables_watch
"""

import argparse
import multiprocessing
import random
from collections import Counter
from typing import Any

import fiona
import numpy as np
import shapely
import tqdm
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered
from scipy.spatial import cKDTree

from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    balance_by_class,
    encode_detection_tile,
)

SLUG = "global_renewables_watch"
NAME = "Global Renewables Watch"
RELEASE = "https://github.com/microsoft/global-renewables-watch/releases/download/v1.0"
SOLAR_FILE = "solar_all_2024q2_v1.gpkg"
WIND_FILE = "wind_all_2024q2_v1.gpkg"

# Shared class scheme.
CID_BACKGROUND = 0
CID_SOLAR = 1
CID_WIND = 2
CLASSES = [
    {
        "id": CID_BACKGROUND,
        "name": "background",
        "description": "Negative / non-target land: pixels outside any detected PV "
        "installation or wind turbine.",
    },
    {
        "id": CID_SOLAR,
        "name": "solar_pv",
        "description": "Ground-mounted / large solar photovoltaic installation footprint "
        "(polygon rasterized at 10 m), from GRW PlanetScope deep-learning detections with "
        "human QC.",
    },
    {
        "id": CID_WIND,
        "name": "wind_turbine",
        "description": "Individual wind turbine location (point), from GRW PlanetScope "
        "deep-learning detections with human QC. Detection-encoded: a small positive "
        "square with a nodata buffer ring in a background context tile.",
    },
]

# Sampling / encoding parameters.
PER_CLASS = 1000
YEARS = list(range(2017, 2025))  # construction_year buckets present in the release.
PER_YEAR = PER_CLASS // len(YEARS)  # 125 -> 1000 total per class.
N_NEGATIVES = 300  # background-only detection negatives.

# Detection encoding for wind turbines (~1px positive at 10 m). Buffer >=10 px: turbine
# coordinates aren't pixel-exact, so a thick nodata ring (21x21 ignore for a 1px positive)
# avoids penalizing a few-pixel offset; a 32x32 tile still keeps ample background.
DET_TILE = 32
DET_POS_SIZE = 1
DET_BUFFER = 10

# Solar polygon tiling.
MAX_SOLAR_TILE = io.MAX_TILE  # 64

SRC_PROJ = Projection(
    CRS.from_epsg(3857), 1, 1
)  # source geometries are EPSG:3857 metres.
_TO_WGS84 = None  # lazily-built pyproj transformer (per process).


def _lonlat(x: float, y: float) -> tuple[float, float]:
    """EPSG:3857 (x, y) metres -> (lon, lat) degrees."""
    global _TO_WGS84
    if _TO_WGS84 is None:
        from pyproj import Transformer

        _TO_WGS84 = Transformer.from_crs(3857, 4326, always_xy=True)
    lon, lat = _TO_WGS84.transform(x, y)
    return lon, lat


# --------------------------------------------------------------------------------------
# Reading source features.
# --------------------------------------------------------------------------------------
def read_solar() -> list[dict[str, Any]]:
    """Read solar PV polygons into records (year, centroid lon/lat, geometry WKB)."""
    path = io.raw_dir(SLUG) / SOLAR_FILE
    recs: list[dict[str, Any]] = []
    with fiona.open(path.path) as src:
        for i, feat in enumerate(src):
            props = feat["properties"]
            year = props.get("construction_year")
            if year is None:
                continue
            geom = shapely.geometry.shape(feat["geometry"])
            if geom.is_empty:
                continue
            c = geom.centroid
            lon, lat = _lonlat(c.x, c.y)
            recs.append(
                {
                    "kind": "solar",
                    "year": int(year),
                    "lon": lon,
                    "lat": lat,
                    "geom_wkb": shapely.to_wkb(geom),
                    "source_id": f"solar/{i}",
                }
            )
    return recs


def read_wind() -> tuple[list[dict[str, Any]], np.ndarray]:
    """Read wind turbine points; return (records, Nx2 array of EPSG:3857 x,y)."""
    path = io.raw_dir(SLUG) / WIND_FILE
    recs: list[dict[str, Any]] = []
    xy: list[tuple[float, float]] = []
    with fiona.open(path.path) as src:
        for i, feat in enumerate(src):
            props = feat["properties"]
            year = props.get("construction_year")
            if year is None:
                continue
            x, y = feat["geometry"]["coordinates"]
            lon, lat = _lonlat(x, y)
            recs.append(
                {
                    "kind": "wind",
                    "year": int(year),
                    "lon": lon,
                    "lat": lat,
                    "x3857": x,
                    "y3857": y,
                    "source_id": f"wind/{i}",
                }
            )
            xy.append((x, y))
    return recs, np.array(xy, dtype=float)


# --------------------------------------------------------------------------------------
# Writers (run in worker processes).
# --------------------------------------------------------------------------------------
def _write_solar(rec: dict[str, Any]) -> str | None:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    geom = shapely.from_wkb(rec["geom_wkb"])
    pix = geom_to_pixels(geom, SRC_PROJ, proj)
    minx, miny, maxx, maxy = pix.bounds
    # Center a tile (sized to the polygon footprint, capped at 64) on the bbox center.
    cx = int(round((minx + maxx) / 2))
    cy = int(round((miny + maxy) / 2))
    w = min(MAX_SOLAR_TILE, max(1, int(np.ceil(maxx - minx))))
    h = min(MAX_SOLAR_TILE, max(1, int(np.ceil(maxy - miny))))
    bounds = io.centered_bounds(cx, cy, w, h)
    arr = rasterize_shapes(
        [(pix, CID_SOLAR)], bounds, fill=CID_BACKGROUND, dtype="uint8", all_touched=True
    )
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        source_id=rec["source_id"],
        classes_present=sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA}),
    )
    return "solar"


def _write_wind(rec: dict[str, Any]) -> str | None:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    _, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"], proj)
    bounds = io.centered_bounds(col, row, DET_TILE, DET_TILE)
    x_min, y_min, _, _ = bounds
    # Collect this turbine + any neighbor turbines that fall inside the tile.
    positives: list[tuple[int, int, int]] = []
    cands = rec.get("neighbors", [])
    cands = [(rec["lon"], rec["lat"])] + list(cands)
    for lon, lat in cands:
        _, c, r = io.lonlat_to_utm_pixel(lon, lat, proj)
        lc, lr = c - x_min, r - y_min
        if 0 <= lc < DET_TILE and 0 <= lr < DET_TILE:
            positives.append((lr, lc, CID_WIND))
    arr = encode_detection_tile(
        positives,
        tile_size=DET_TILE,
        positive_size=DET_POS_SIZE,
        buffer_size=DET_BUFFER,
        nodata=io.CLASS_NODATA,
        background=CID_BACKGROUND,
    )[np.newaxis]
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        source_id=rec["source_id"],
        classes_present=sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA}),
    )
    return "wind"


def _write_negative(rec: dict[str, Any]) -> str | None:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    _, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"], proj)
    bounds = io.centered_bounds(col, row, DET_TILE, DET_TILE)
    arr = encode_detection_tile(
        [], tile_size=DET_TILE, background=CID_BACKGROUND, nodata=io.CLASS_NODATA
    )[np.newaxis]
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        source_id=rec["source_id"],
        classes_present=[CID_BACKGROUND],
    )
    return "negative"


def _dispatch(rec: dict[str, Any]) -> str | None:
    if rec["kind"] == "solar":
        return _write_solar(rec)
    if rec["kind"] == "wind":
        return _write_wind(rec)
    return _write_negative(rec)


# --------------------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------------------
def _make_negatives(
    wind_recs: list[dict[str, Any]], tree: cKDTree, n: int, seed: int = 7
) -> list[dict[str, Any]]:
    """Sample background-only tile centers offset from turbines, guaranteed turbine-free.

    Offset a random turbine by 5-15 km, then reject the center if any turbine lies within
    1 km in EPSG:3857 (>> the tile half-diagonal even at high latitudes), so the 32x32
    context tile contains no turbine.
    """
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        base = wind_recs[rng.randrange(len(wind_recs))]
        ang = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(5000, 15000)  # metres in EPSG:3857
        x = base["x3857"] + dist * np.cos(ang)
        y = base["y3857"] + dist * np.sin(ang)
        if tree.query_ball_point([x, y], r=1000.0):
            continue
        lon, lat = _lonlat(x, y)
        if not (-60 <= lat <= 75):  # keep within plausible land latitudes
            continue
        out.append(
            {
                "kind": "negative",
                "year": base["year"],
                "lon": lon,
                "lat": lat,
                "source_id": f"negative/{len(out)}",
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    # NOTE: registry.json is intentionally NOT modified by this script (per task rules);
    # status is managed externally.

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Global Renewables Watch v1.0 (2024 Q2), Microsoft/Planet/TNC.\n"
            f"{RELEASE}/{SOLAR_FILE}\n{RELEASE}/{WIND_FILE}\n"
        )

    print("reading solar polygons ...")
    solar = read_solar()
    print(f"  {len(solar)} solar PV polygons")
    print("reading wind points ...")
    wind, wind_xy = read_wind()
    print(f"  {len(wind)} wind turbine points")

    io.check_disk()

    # Neighbor turbines for each wind record (candidates within 700 m in EPSG:3857).
    tree = cKDTree(wind_xy)
    # Stratify by construction year, then cap at PER_CLASS.
    solar_sel = balance_by_class(solar, "year", per_class=PER_YEAR, seed=42)[:PER_CLASS]
    wind_sel = balance_by_class(wind, "year", per_class=PER_YEAR, seed=42)[:PER_CLASS]

    for r in wind_sel:
        idxs = tree.query_ball_point([r["x3857"], r["y3857"]], r=700.0)
        r["neighbors"] = [
            (wind[i]["lon"], wind[i]["lat"])
            for i in idxs
            if wind[i]["source_id"] != r["source_id"]
        ]

    negatives = _make_negatives(wind, tree, N_NEGATIVES)
    print(
        f"selected {len(solar_sel)} solar, {len(wind_sel)} wind, "
        f"{len(negatives)} negatives"
    )

    selected = solar_sel + wind_sel + negatives
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"

    results: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _dispatch, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            results[res] += 1
    print("write results:", dict(results))

    io.check_disk()

    class_counts = {
        "solar_pv": len(solar_sel),
        "wind_turbine": len(wind_sel),
        "background_negative_tiles": len(negatives),
    }
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "GitHub (Microsoft/Planet/TNC)",
            "license": "MIT",
            "provenance": {
                "url": "https://github.com/microsoft/global-renewables-watch",
                "have_locally": False,
                "annotation_method": "deep learning (PlanetScope) + human QC",
                "release": "v1.0 (2024 Q2)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "detection_encoding": {
                "applies_to": "wind_turbine",
                "tile_size": DET_TILE,
                "positive_size": DET_POS_SIZE,
                "buffer_size": DET_BUFFER,
            },
            "num_samples": len(selected),
            "class_counts": class_counts,
            "notes": (
                "Combined dataset: solar_pv (class 1) from rasterized PV polygons in "
                "variable <=64x64 UTM tiles; wind_turbine (class 2) from turbine points "
                "using the tunable detection encoding (1px positive, 10px nodata buffer = "
                "21x21 ignore, background) in 32x32 tiles, with neighboring turbines in-tile also marked "
                "positive; plus background-only negative tiles. Time range is a 1-year "
                "window anchored on each feature's construction_year (2017-2024; 2017 is "
                "the earliest-baseline bucket). All classes global. Derived product "
                "(DL+QC)."
            ),
        },
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
