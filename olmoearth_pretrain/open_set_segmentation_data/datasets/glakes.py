"""Process the GLAKES global lake polygon product into binary lake-water label tiles.

Source: GLAKES, "Mapping global lake dynamics reveals the emerging roles of small
lakes" (Pi et al., Nat. Commun. 2022), Zenodo record 7016548 (CC-BY-4.0). The
``GLAKES.rar`` archive contains a global lake-polygon product covering ~3.4M lakes
> 0.03 km2, split into 7 per-continent ESRI shapefiles (Asia ``as``, Africa ``af``,
North America ``na1``/``na2``, South America ``sa``, Europe ``eu``, Oceania ``oc``),
all in EPSG:4326. Each polygon is a lake footprint (max water extent over 1984-2019)
with attributes Lake_id, Area_bound (km2), Continent, Lat/Lon (centroid), and several
flags. Polygons are validated/derived from a U-Net water-segmentation pipeline.

Task: **binary lake-water vs background segmentation** (label_type: polygons). We
rasterize lake polygons into 64x64 UTM 10 m tiles:

    0 = background (not GLAKES lake water)
    1 = lake water (inside a GLAKES lake polygon)

The source is a huge global vector (3.4M polygons), so we do BOUNDED, geographically
stratified sampling (round-robin over 1-degree cells across all continents), capped at
25,000 tiles total. Two kinds of tile:

  * positive : centered on a geographically-stratified sample of lake centroids. All
               GLAKES lake polygons intersecting the 640 m tile are rasterized to
               class 1 (so multiple nearby lakes and shorelines are captured); the
               rest is background 0. Most lakes are small (median ~0.085 km2, ~87%
               fit within a tile), so positive tiles typically contain both classes.
  * negative : background-only tiles, produced by offsetting a stratified sample of
               lake anchors by a random ~3-9 km vector so no lake falls in the tile
               (verified per-tile against the polygons); all background 0. Rare cases
               where the offset still clips a lake are simply rasterized correctly
               (counted as a lake tile).

Per-tile intersecting polygons are read directly from the shapefiles with a pyogrio
bbox spatial filter (using the .sbn spatial index), so no giant in-memory STRtree is
needed and the write phase parallelizes cleanly over a multiprocessing Pool.

Time range: lakes are quasi-static; GLAKES covers 1984-2019 and the manifest window is
2016-2021. We assign each tile a 1-year window with the start year uniformly sampled in
2016-2020 (Sentinel era), matching how pretraining pairs labels with imagery.

Run: ``python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.glakes``
Idempotent: existing ``locations/{id}.tif`` are skipped.
"""

import argparse
import math
import multiprocessing
import os
import random
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import tqdm
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io

SLUG = "glakes"
NAME = "GLAKES"
RAW = str(io.raw_dir(SLUG))
SHP_DIR = os.path.join(RAW, "extract", "GLAKES", "GLAKES")
CONTINENT_FILES = [
    "GLAKES_af",
    "GLAKES_as",
    "GLAKES_eu",
    "GLAKES_na1",
    "GLAKES_na2",
    "GLAKES_oc",
    "GLAKES_sa",
]

TILE = 64
POS_TARGET = 20000
NEG_TARGET = 5000
CELL = 1.0  # geographic stratification cell size (degrees)
YEARS = [2016, 2017, 2018, 2019, 2020]

CLASSES = [
    (
        "background",
        "Not a GLAKES lake: land or any surface outside a mapped lake polygon.",
    ),
    (
        "lake water",
        "Inside a GLAKES lake polygon: lake/reservoir water at maximum extent over "
        "1984-2019, from the GLAKES global lake product (>0.03 km2 lakes).",
    ),
]
BG, LAKE = 0, 1


def shp_path(fbase: str) -> str:
    return os.path.join(SHP_DIR, fbase + ".shp")


# --------------------------------------------------------------------------- scan


def load_centroids(fbase: str) -> dict[str, np.ndarray]:
    """Read only Lon/Lat/Area_bound (no geometry) for one continent."""
    import pyogrio

    df = pyogrio.read_dataframe(
        shp_path(fbase), columns=["Lon", "Lat", "Area_bound"], read_geometry=False
    )
    return {
        "lon": df["Lon"].to_numpy(dtype="float64"),
        "lat": df["Lat"].to_numpy(dtype="float64"),
        "area": df["Area_bound"].to_numpy(dtype="float64"),
    }


def stratified_indices(
    lons: np.ndarray, lats: np.ndarray, n: int, seed: int
) -> list[int]:
    """Round-robin over 1-degree lon/lat cells for geographic spread; up to n indices."""
    cells: dict[tuple, list] = defaultdict(list)
    for i in range(len(lons)):
        cells[
            (int(math.floor(lons[i] / CELL)), int(math.floor(lats[i] / CELL)))
        ].append(i)
    rng = random.Random(seed)
    order = list(cells.values())
    for lst in order:
        rng.shuffle(lst)
    rng.shuffle(order)
    out: list[int] = []
    i = 0
    while len(out) < n and order:
        lst = order[i % len(order)]
        if lst:
            out.append(lst.pop())
        i += 1
        if i % len(order) == 0:
            order = [l for l in order if l]
    return out[:n]


# --------------------------------------------------------------------------- write


def _read_lakes_bbox(fbase: str, bbox: tuple[float, float, float, float]) -> list[Any]:
    """Read lake geometries whose envelope intersects bbox (lon/lat)."""
    import pyogrio

    gdf = pyogrio.read_dataframe(shp_path(fbase), bbox=bbox, columns=[])
    return [g for g in gdf.geometry.values if g is not None and not g.is_empty]


def _write_one(rec: dict[str, Any]) -> str | None:
    from rslearn.const import WGS84_PROJECTION
    from shapely.geometry import box

    from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
        geom_to_pixels,
        rasterize_shapes,
    )

    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return None

    lon, lat = rec["lon"], rec["lat"]
    proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)

    # lon/lat query box covering the tile (with generous margin), for the spatial filter.
    mlat = (TILE * io.RESOLUTION) / 111320.0  # ~full tile size in deg lat (2x safety)
    mlon = mlat / max(math.cos(math.radians(lat)), 0.1)
    qbox = (lon - mlon, lat - mlat, lon + mlon, lat + mlat)
    clip_ll = box(*qbox)

    geoms = _read_lakes_bbox(rec["file"], qbox)
    shapes = []
    for g in geoms:
        # Clip to the query box first so huge lakes don't blow up reprojection.
        try:
            gc = g.intersection(clip_ll)
        except Exception:
            gc = g
        if gc.is_empty:
            continue
        px = geom_to_pixels(gc, WGS84_PROJECTION, proj)
        if not px.is_empty:
            shapes.append((px, LAKE))

    if shapes:
        label = rasterize_shapes(
            shapes, bounds, fill=BG, dtype="uint8", all_touched=True
        )[0]
    else:
        label = np.full((TILE, TILE), BG, dtype=np.uint8)

    present = sorted(int(v) for v in np.unique(label))
    io.write_label_geotiff(SLUG, sample_id, label, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        source_id=rec["source_id"],
        classes_present=present,
    )
    return "lake" if LAKE in present else "background"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    io.check_disk()

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "GLAKES global lake polygons, Pi et al., Nat. Commun. 2022. "
            "Zenodo record 7016548 (CC-BY-4.0). "
            "https://zenodo.org/records/7016548\n"
            "File: GLAKES.rar (1.8 GB) -> extract/GLAKES/GLAKES/GLAKES_{af,as,eu,na1,"
            "na2,oc,sa}.shp (7 per-continent lake-polygon shapefiles, EPSG:4326).\n"
        )

    # ---- Phase A: cheap attribute-only scan + geographic stratified selection.
    per_file: dict[str, dict[str, np.ndarray]] = {}
    for fbase in CONTINENT_FILES:
        per_file[fbase] = load_centroids(fbase)
        print(f"loaded centroids {fbase}: {len(per_file[fbase]['lon'])}")

    # Global concatenation with an index back to (file, local_idx).
    all_lon = np.concatenate([per_file[f]["lon"] for f in CONTINENT_FILES])
    all_lat = np.concatenate([per_file[f]["lat"] for f in CONTINENT_FILES])
    file_of = []
    local_of = []
    for f in CONTINENT_FILES:
        n = len(per_file[f]["lon"])
        file_of.append(np.full(n, CONTINENT_FILES.index(f), dtype="int32"))
        local_of.append(np.arange(n, dtype="int64"))
    file_of = np.concatenate(file_of)
    local_of = np.concatenate(local_of)
    print(f"total lakes: {len(all_lon)}")

    io.check_disk()

    pos_idx = stratified_indices(all_lon, all_lat, POS_TARGET, seed=1)
    neg_idx = stratified_indices(all_lon, all_lat, NEG_TARGET, seed=2)
    print(f"selected positives={len(pos_idx)} negative-anchors={len(neg_idx)}")

    rng = random.Random(7)
    records: list[dict[str, Any]] = []
    sid = 0

    for gi in pos_idx:
        fbase = CONTINENT_FILES[int(file_of[gi])]
        records.append(
            {
                "sample_id": f"{sid:06d}",
                "kind": "pos",
                "lon": float(all_lon[gi]),
                "lat": float(all_lat[gi]),
                "file": fbase,
                "year": rng.choice(YEARS),
                "source_id": f"{fbase}:{int(local_of[gi])}",
            }
        )
        sid += 1

    for gi in neg_idx:
        fbase = CONTINENT_FILES[int(file_of[gi])]
        lat0 = float(all_lat[gi])
        # random offset ~3-9 km so the tile lands off the anchor lake.
        dist_deg = rng.uniform(0.03, 0.08)
        ang = rng.uniform(0, 2 * math.pi)
        dlat = dist_deg * math.sin(ang)
        dlon = dist_deg * math.cos(ang) / max(math.cos(math.radians(lat0)), 0.1)
        records.append(
            {
                "sample_id": f"{sid:06d}",
                "kind": "neg",
                "lon": float(all_lon[gi]) + dlon,
                "lat": max(min(lat0 + dlat, 83.0), -83.0),
                "file": fbase,
                "year": rng.choice(YEARS),
                "source_id": f"{fbase}:{int(local_of[gi])}:neg",
            }
        )
        sid += 1

    print(f"total records to write: {len(records)}")
    io.check_disk()

    counts: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in records]),
            total=len(records),
            desc="write tiles",
        ):
            if res is not None:
                counts[res] += 1

    n_written = len(list(io.locations_dir(SLUG).glob("*.tif")))
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Zenodo / Nat Commun (GLAKES)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://zenodo.org/records/7016548",
                "have_locally": False,
                "annotation_method": "derived + validated (U-Net water segmentation), "
                "global lake polygon product",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_written,
            "tile_counts": {
                "lake_tiles": counts.get("lake", 0),
                "background_tiles": counts.get("background", 0),
            },
            "notes": (
                "Binary lake-water vs background segmentation from the GLAKES global "
                "lake polygon product (~3.4M lakes >0.03 km2). 64x64 uint8 tiles in "
                "local UTM at 10 m; classes: 0 background, 1 lake water (255 nodata, "
                "unused). Bounded geographically-stratified sampling (round-robin over "
                "1-degree cells across all 7 continents): ~20k positive tiles centered "
                "on lake centroids (all GLAKES polygons intersecting a tile rasterized "
                "to class 1, all_touched=True) + ~5k background-only negative tiles "
                "offset ~3-9 km from lake anchors. Capped at 25,000 tiles total. Lake "
                "polygons are max water extent over 1984-2019; time range = 1-year "
                "window with start year uniform in 2016-2020. Caveat: negatives are "
                "labeled background even if they contain unmapped small ponds/rivers or "
                "(rarely, near coasts) ocean, since GLAKES maps lakes only."
            ),
        },
    )
    print("tile counts:", dict(counts))
    print("total tif on disk:", n_written)
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
