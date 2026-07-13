"""Process GOODD (Global Georeferenced Dams) into open-set-segmentation detection tiles.

Source: Mulligan, M., van Soesbergen, A. & Saenz, L. "GOODD, a global dataset of more
than 38,000 georeferenced dams." Scientific Data 7, 31 (2020). Distributed by Global Dam
Watch (https://www.globaldamwatch.org/goodd), license CC0. Downloaded as a zip of two
ESRI shapefiles:
  - GOOD2_dams.shp      -> 38,667 dam-wall POINTS (EPSG:4326), digitized by manual
                           photointerpretation from Landsat/SPOT imagery.
  - GOOD2_catchments.shp -> upstream drainage catchment POLYGONS, one per dam.

We build a single-class, positive-only **object-detection** dataset of dam walls
(label_type "points" that mark presence; spec section 4). The catchment polygons are
DROPPED: they delineate the full upstream hydrological drainage basin of each dam (often
thousands of km2), not a feature observable/segmentable at the dam location from S2/S1/
Landsat at 10-30 m, and they are not a per-pixel land-cover class. See the summary.

Encoding (tunable detection, spec section 4): each dam point becomes a 1 px positive at
the dam location, ringed by a 10 px nodata (255) buffer to absorb the coordinate
imprecision of manual Landsat/SPOT digitizing, with background (0) filling the rest of a
32x32 (320 m) context tile. All other GOODD dams falling inside a tile are also marked as
positives. Per spec section 4, we additionally emit background-only NEGATIVE tiles away
from any dam so the class has spatially-meaningful negatives.

Class scheme (id 0 = background; 255 = nodata/ignore = detection buffer rings):
  0 background
  1 dam

Time range: dams are persistent structures (undated in the source). Per spec section 5
(static labels) each sample gets a 1-year window at a representative Sentinel-era year,
spread pseudo-randomly across 2016-2022 for temporal diversity.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.goodd_global_georeferenced_dams
"""

import argparse
import multiprocessing
import random
from collections import Counter
from typing import Any

import fiona
import numpy as np
import tqdm
from rslearn.utils.mp import star_imap_unordered
from scipy.spatial import cKDTree

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import encode_detection_tile

SLUG = "goodd_global_georeferenced_dams"
NAME = "GOODD (Global Georeferenced Dams)"
DOWNLOAD_URL = "https://www.globaldamwatch.org/goodd"
DAMS_SHP = "Data/GOOD2_dams.shp"

CID_BACKGROUND = 0
CID_DAM = 1
CLASSES = [
    {
        "id": 0,
        "name": "background",
        "description": "Negative / non-dam land: pixels outside any mapped dam wall.",
    },
    {
        "id": 1,
        "name": "dam",
        "description": "Dam wall location from GOODD, digitized by manual photointerpretation "
        "of Landsat/SPOT imagery. Marks a barrier/dam wall on a watercourse (all dam types; "
        "the source records only a point, no dam-type attribute).",
    },
]

# Sampling / encoding parameters.
PER_CLASS = 1000  # positive dam tiles (spec section 5, single class)
N_NEGATIVES = 500  # background-only tiles
YEARS = list(range(2016, 2023))

DET_TILE = 32
DET_POS_SIZE = 1
DET_BUFFER = 10

NEIGHBOR_RADIUS_M = 500.0  # 3857 prefilter radius for in-tile neighbor dams
NEG_MIN_DIST_M = 1000.0  # min distance a negative tile center keeps from any dam
NEG_OFFSET_MIN_M = 3000.0
NEG_OFFSET_MAX_M = 20000.0

_TO_3857 = None
_TO_4326 = None


def _to_3857(lon: float, lat: float) -> tuple[float, float]:
    global _TO_3857
    if _TO_3857 is None:
        from pyproj import Transformer

        _TO_3857 = Transformer.from_crs(4326, 3857, always_xy=True)
    return _TO_3857.transform(lon, lat)


def _to_4326(x: float, y: float) -> tuple[float, float]:
    global _TO_4326
    if _TO_4326 is None:
        from pyproj import Transformer

        _TO_4326 = Transformer.from_crs(3857, 4326, always_xy=True)
    return _TO_4326.transform(x, y)


def dams_path() -> str:
    return str(io.raw_dir(SLUG) / DAMS_SHP)


def ensure_extracted() -> None:
    """Extract GOODD_data.zip into raw_dir if the dam shapefile is not present."""
    import zipfile

    raw = io.raw_dir(SLUG)
    if (raw / DAMS_SHP).exists():
        return
    zip_path = raw / "GOODD_data.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} missing; download GOODD_data.zip from {DOWNLOAD_URL}"
        )
    with zipfile.ZipFile(str(zip_path)) as z:
        z.extractall(str(raw))


def read_dams() -> list[dict[str, Any]]:
    """Read GOODD dam points into records with lon/lat + source id."""
    recs: list[dict[str, Any]] = []
    with fiona.open(dams_path()) as src:
        for i, feat in enumerate(src):
            if feat["geometry"] is None:
                continue
            lon, lat = feat["geometry"]["coordinates"][:2]
            dam_id = feat["properties"].get("DAM_ID")
            recs.append(
                {
                    "lon": float(lon),
                    "lat": float(lat),
                    "source_id": f"DAM_ID/{int(dam_id)}"
                    if dam_id is not None
                    else f"row/{i}",
                }
            )
    return recs


# --------------------------------------------------------------------------------------
# Writers (worker processes).
# --------------------------------------------------------------------------------------
def _write_positive(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    _, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"], proj)
    bounds = io.centered_bounds(col, row, DET_TILE, DET_TILE)
    x_min, y_min, _, _ = bounds
    positives: list[tuple[int, int, int]] = []
    cands = [(rec["lon"], rec["lat"])] + rec.get("neighbors", [])
    for lon, lat in cands:
        _, c, r = io.lonlat_to_utm_pixel(lon, lat, proj)
        lc, lr = c - x_min, r - y_min
        if 0 <= lc < DET_TILE and 0 <= lr < DET_TILE:
            positives.append((lr, lc, CID_DAM))
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
    return "positive"


def _write_negative(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    _, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"], proj)
    bounds = io.centered_bounds(col, row, DET_TILE, DET_TILE)
    arr = encode_detection_tile(
        [],
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
        classes_present=[CID_BACKGROUND],
    )
    return "negative"


def _dispatch(rec: dict[str, Any]) -> str:
    if rec["kind"] == "negative":
        return _write_negative(rec)
    return _write_positive(rec)


# --------------------------------------------------------------------------------------
# Negatives.
# --------------------------------------------------------------------------------------
def make_negatives(
    tree: cKDTree, dams: list[dict[str, Any]], n: int, seed: int = 7
) -> list[dict[str, Any]]:
    """Background-only tile centers offset from dams, guaranteed dam-free."""
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    attempts = 0
    while len(out) < n and attempts < n * 100:
        attempts += 1
        base = dams[rng.randrange(len(dams))]
        ang = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(NEG_OFFSET_MIN_M, NEG_OFFSET_MAX_M)
        bx, by = _to_3857(base["lon"], base["lat"])
        x, y = bx + dist * np.cos(ang), by + dist * np.sin(ang)
        if tree.query_ball_point([x, y], r=NEG_MIN_DIST_M):
            continue
        lon, lat = _to_4326(x, y)
        if not (-58 <= lat <= 74):
            continue
        out.append(
            {
                "kind": "negative",
                "lon": float(lon),
                "lat": float(lat),
                "source_id": f"negative/{len(out)}",
            }
        )
    return out


# --------------------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "GOODD (Global Georeferenced Dams). Mulligan, van Soesbergen & Saenz, "
            "Sci Data 7, 31 (2020). Global Dam Watch. License CC0.\n"
            f"{DOWNLOAD_URL}\n"
            "GOODD_data.zip -> Data/GOOD2_dams.shp (38,667 dam points, EPSG:4326) + "
            "Data/GOOD2_catchments.shp (upstream drainage catchments; DROPPED for this "
            "dataset - not observable/segmentable at the dam location at 10-30 m).\n"
        )

    ensure_extracted()
    print("reading dam points ...", flush=True)
    dams = read_dams()
    print(f"  {len(dams)} dam points", flush=True)

    io.check_disk()

    # Global KDTree over ALL dams (EPSG:3857) for negatives + in-tile neighbor marking.
    dams_xy = np.array([_to_3857(d["lon"], d["lat"]) for d in dams], dtype=float)
    tree = cKDTree(dams_xy)

    # Select positive tile centers.
    rng = random.Random(42)
    idxs = list(range(len(dams)))
    rng.shuffle(idxs)
    selected = [dict(dams[i]) for i in idxs[:PER_CLASS]]

    # Mark neighboring dams that fall inside each positive tile.
    for r in selected:
        x, y = _to_3857(r["lon"], r["lat"])
        near = tree.query_ball_point([x, y], r=NEIGHBOR_RADIUS_M)
        r["neighbors"] = [
            (dams[i]["lon"], dams[i]["lat"])
            for i in near
            if dams[i]["source_id"] != r["source_id"]
        ][:200]

    negatives = make_negatives(tree, dams, N_NEGATIVES)
    print(
        f"selected {len(selected)} positive tiles + {len(negatives)} negatives",
        flush=True,
    )

    for r in selected:
        r["kind"] = "positive"
    yrng = random.Random(123)
    all_recs = selected + negatives
    for r in all_recs:
        r["year"] = YEARS[yrng.randrange(len(YEARS))]
    for i, r in enumerate(all_recs):
        r["sample_id"] = f"{i:06d}"

    results: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _dispatch, [dict(rec=r) for r in all_recs]),
            total=len(all_recs),
        ):
            results[res] += 1
    print("write results:", dict(results), flush=True)

    io.check_disk()

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Global Dam Watch / Scientific Data (Mulligan et al. 2020)",
            "license": "CC0",
            "provenance": {
                "url": DOWNLOAD_URL,
                "paper": "https://doi.org/10.1038/s41597-020-0362-5",
                "have_locally": False,
                "annotation_method": "manual photointerpretation of Landsat/SPOT imagery",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "detection_encoding": {
                "applies_to": "dam points (single foreground class)",
                "tile_size": DET_TILE,
                "positive_size": DET_POS_SIZE,
                "buffer_size": DET_BUFFER,
            },
            "num_samples": len(all_recs),
            "class_counts": {
                "dam_positive_tiles": len(selected),
                "background_negative_tiles": len(negatives),
            },
            "notes": (
                "Positive-only dam-point object detection. 1 px positive at each dam wall + "
                "10 px nodata buffer ring (absorbs Landsat/SPOT digitizing imprecision), "
                "background fill in a 32x32 (320 m) context tile; all GOODD dams inside a "
                "tile are marked positive. 500 background-only negative tiles emitted away "
                "from any dam (>=1 km). 1000 of 38,667 dams sampled as tile centers "
                "(spec section 5 per-class cap). Catchment polygons (GOOD2_catchments) "
                "dropped: upstream drainage basins, not observable at the dam location. "
                "Persistent features -> 1-year window at a representative Sentinel-era year "
                "(2016-2022)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(all_recs)
    )
    print("done:", len(all_recs), "samples", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
