"""Process the GMIE global irrigation / central-pivot dataset into label patches.

Source: GMIE (Global Maximum Irrigation Extent) + GCPIS (Global Central Pivot
Irrigation System), Tian et al. (ESSD), Harvard Dataverse doi:10.7910/DVN/HKBAQQ
(license CC0). Two products:

* ``GMIE-100_*.tif`` (67 tiles): single-band irrigation-proportion raster in EPSG:4326
  at ~100 m. Pixel value = irrigation proportion in [0, 1]; background = -99. Produced
  from dry months over 2017-2019 (regularly irrigated regions) / 2010-2019 (occasionally
  irrigated). This is a derived-product *map*.
* ``GCPIS.shp`` (179,942 polygons): footprints of detected centre-pivot irrigation
  systems (the visually-distinctive circles), EPSG:4326.

This is a **global derived-product raster** so we do BOUNDED-TILE sampling (<=1000 tiles
per class, no attempt at global coverage), preferring spatially-homogeneous windows.
Every label patch is a 64x64 uint8 tile in local UTM at 10 m (GMIE reprojected with
nearest resampling), with a unified 3-class segmentation:

    0 = central pivot irrigation system   (GCPIS polygon overlay; wins where present)
    1 = irrigated cropland                 (GMIE proportion >= IRR_THRESH)
    2 = non-irrigated                      (GMIE proportion in [0, NON_THRESH], observed)
  255 = nodata/ignore                      (GMIE background, or ambiguous mid proportion)

Sampling per class:
  * central pivot   : centre a tile on each of a geographically-stratified set of GCPIS
                      polygons (guaranteed class 0 present).
  * irrigated       : homogeneous GMIE windows (coarse ~640 m cell fully >= IRR_THRESH).
  * non-irrigated   : homogeneous GMIE windows (coarse ~640 m cell fully in [0, NON_THRESH]).
GCPIS polygons intersecting any selected tile are overlaid on all tiles, so an irrigated
or non-irrigated window that contains a pivot still gets class 0 there.

Run: ``python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.gmie_central_pivot_irrigation``
Idempotent: existing ``locations/{id}.tif`` are skipped.
"""

import argparse
import glob
import multiprocessing
import os
import random
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import tqdm
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io

SLUG = "gmie_central_pivot_irrigation"
NAME = "GMIE Central Pivot Irrigation"
RAW = str(io.raw_dir(SLUG))
GCPIS_SHP = os.path.join(RAW, "GCPIS_extract", "GCPIS.shp")

PER_CLASS = 1000
TILE = 64
# GMIE proportion thresholds.
IRR_THRESH = 0.5  # >= this => irrigated cropland
NON_THRESH = 0.05  # <= this (and observed) => non-irrigated
BACKGROUND = -99.0
# Coarse block factor for homogeneity scan: 7 GMIE px (~700 m) ~ one 640 m tile.
COARSE = 7
# Cap candidates gathered per GMIE tile per class (before global stratified sampling).
CAND_PER_TILE = 400
# geographic stratification cell size (degrees).
CELL = 1.0

CLASSES = [
    (
        "central pivot irrigation system",
        "Footprint of a machine-detected centre-pivot irrigation system (the "
        "characteristic irrigated circle), from the GCPIS product.",
    ),
    (
        "irrigated cropland",
        "Land with high assessed irrigation proportion (GMIE irrigation proportion "
        ">= 0.5), i.e. regularly/heavily irrigated agricultural land.",
    ),
    (
        "non-irrigated",
        "Assessed land with ~zero irrigation proportion (GMIE proportion <= 0.05): "
        "observed but not irrigated (rainfed / natural).",
    ),
]
CID = {name: i for i, (name, _d) in enumerate(CLASSES)}


# --------------------------------------------------------------------------- scan


def gmie_tiles() -> list[str]:
    return sorted(glob.glob(os.path.join(RAW, "GMIE-100_*.tif")))


def _scan_one(path: str) -> list[dict[str, Any]]:
    """Read one GMIE tile, find homogeneous irrigated / non-irrigated coarse cells."""
    import rasterio

    ds = rasterio.open(path)
    a = ds.read(1).astype("float32")
    h, w = a.shape
    hh, ww = (h // COARSE) * COARSE, (w // COARSE) * COARSE
    b = a[:hh, :ww].reshape(hh // COARSE, COARSE, ww // COARSE, COARSE)
    cmin = b.min(axis=(1, 3))
    cmax = b.max(axis=(1, 3))
    irr = cmin >= IRR_THRESH
    non = (cmin >= -0.001) & (cmax <= NON_THRESH)
    tf = ds.transform
    rng = random.Random(hash(os.path.basename(path)) & 0xFFFFFFFF)
    out: list[dict[str, Any]] = []
    for cls, mask in (("irrigated cropland", irr), ("non-irrigated", non)):
        rows, cols = np.nonzero(mask)
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        idx = idx[:CAND_PER_TILE]
        for i in idx:
            # centre full-res pixel of the coarse cell.
            px = int(cols[i]) * COARSE + COARSE // 2
            py = int(rows[i]) * COARSE + COARSE // 2
            lon, lat = tf * (px + 0.5, py + 0.5)
            out.append(
                {
                    "lon": float(lon),
                    "lat": float(lat),
                    "cls": cls,
                    "source_id": f"{os.path.basename(path)}:{px},{py}",
                }
            )
    return out


def scan_gmie(workers: int) -> list[dict[str, Any]]:
    tiles = gmie_tiles()
    recs: list[dict[str, Any]] = []
    with multiprocessing.Pool(min(workers, 48)) as p:
        for r in tqdm.tqdm(
            star_imap_unordered(p, _scan_one, [dict(path=t) for t in tiles]),
            total=len(tiles),
            desc="scan GMIE",
        ):
            recs.extend(r)
    return recs


def load_pivots() -> tuple[list[Any], list[dict[str, Any]]]:
    """Load GCPIS polygons; return (geoms, centroid records)."""
    import fiona
    from shapely.geometry import shape

    geoms: list[Any] = []
    recs: list[dict[str, Any]] = []
    with fiona.open(GCPIS_SHP) as c:
        for i, f in enumerate(c):
            g = shape(f["geometry"])
            if g.is_empty:
                continue
            ct = g.centroid
            geoms.append(g)
            recs.append(
                {
                    "lon": float(ct.x),
                    "lat": float(ct.y),
                    "cls": "central pivot irrigation system",
                    "source_id": f"GCPIS:{i}",
                    "pivot_idx": len(geoms) - 1,
                }
            )
    return geoms, recs


def stratified(
    records: list[dict[str, Any]], n: int, seed: int
) -> list[dict[str, Any]]:
    """Round-robin over 1-degree cells for geographic spread; up to n records."""
    cells: dict[tuple, list] = defaultdict(list)
    rng = random.Random(seed)
    for r in records:
        cells[(int(r["lon"] // CELL), int(r["lat"] // CELL))].append(r)
    order = list(cells.values())
    for lst in order:
        rng.shuffle(lst)
    rng.shuffle(order)
    out: list[dict[str, Any]] = []
    i = 0
    while len(out) < n and any(order):
        lst = order[i % len(order)]
        if lst:
            out.append(lst.pop())
        i += 1
        if i % len(order) == 0:
            order = [l for l in order if l]
            if not order:
                break
    return out[:n]


# --------------------------------------------------------------------------- write

_GMIE_INDEX: list[tuple[str, float, float, float, float]] = []


def build_gmie_index() -> None:
    import rasterio

    _GMIE_INDEX.clear()
    for path in gmie_tiles():
        with rasterio.open(path) as ds:
            b = ds.bounds
            _GMIE_INDEX.append((path, b.left, b.bottom, b.right, b.top))


def covering_gmie(lon: float, lat: float) -> str | None:
    for path, west, south, east, north in _GMIE_INDEX:
        if west <= lon <= east and south <= lat <= north:
            return path
    return None


def _write_one(rec: dict[str, Any]) -> str | None:
    import rasterio
    from affine import Affine
    from rasterio.warp import Resampling, reproject
    from rslearn.const import WGS84_PROJECTION

    from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
        geom_to_pixels,
        rasterize_shapes,
    )

    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return None

    proj, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"])
    bounds = io.centered_bounds(col, row, TILE, TILE)

    # Base classes from GMIE (nearest reproject into the UTM 10 m tile grid).
    label = np.full((TILE, TILE), io.CLASS_NODATA, dtype=np.uint8)
    src_path = rec.get("gmie_path")
    if src_path:
        dst = np.full((TILE, TILE), BACKGROUND, dtype="float32")
        dst_transform = Affine(10, 0, bounds[0] * 10, 0, -10, bounds[1] * -10)
        with rasterio.open(src_path) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=proj.crs.to_string(),
                resampling=Resampling.nearest,
                src_nodata=BACKGROUND,
                dst_nodata=BACKGROUND,
            )
        label[(dst >= -0.001) & (dst <= NON_THRESH)] = CID["non-irrigated"]
        label[dst >= IRR_THRESH] = CID["irrigated cropland"]

    # Overlay any GCPIS pivot polygons intersecting this tile (class 0 wins).
    pivots = rec.get("pivots") or []
    if pivots:
        shapes = []
        for g in pivots:
            px = geom_to_pixels(g, WGS84_PROJECTION, proj)
            if not px.is_empty:
                shapes.append((px, 1))
        if shapes:
            mask = rasterize_shapes(
                shapes, bounds, fill=0, dtype="uint8", all_touched=True
            )[0]
            label[mask == 1] = CID["central pivot irrigation system"]

    present = sorted(int(v) for v in np.unique(label) if v != io.CLASS_NODATA)
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
    return rec["cls"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    io.check_disk()

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "GMIE / GCPIS, Tian et al. (ESSD), Harvard Dataverse "
            "doi:10.7910/DVN/HKBAQQ (CC0).\n"
            "Files: GMIE-100_*.tif (67 irrigation-proportion tiles, EPSG:4326 ~100 m), "
            "GCPIS.shp (central-pivot polygons). Downloaded via Dataverse file access API.\n"
        )

    # Scan GMIE for homogeneous irrigated / non-irrigated candidates.
    gmie_recs = scan_gmie(args.workers)
    by_cls: dict[str, list] = defaultdict(list)
    for r in gmie_recs:
        by_cls[r["cls"]].append(r)
    print(
        f"GMIE candidates: irrigated={len(by_cls['irrigated cropland'])} "
        f"non-irrigated={len(by_cls['non-irrigated'])}"
    )

    # Load GCPIS pivots.
    print("loading GCPIS polygons ...")
    pivot_geoms, pivot_recs = load_pivots()
    print(f"loaded {len(pivot_geoms)} pivot polygons")

    io.check_disk()

    # Select up to PER_CLASS per class, geographically stratified.
    sel_cpis = stratified(pivot_recs, PER_CLASS, seed=1)
    sel_irr = stratified(by_cls["irrigated cropland"], PER_CLASS, seed=2)
    sel_non = stratified(by_cls["non-irrigated"], PER_CLASS, seed=3)
    selected = sel_cpis + sel_irr + sel_non
    print(
        f"selected: cpis={len(sel_cpis)} irrigated={len(sel_irr)} "
        f"non-irrigated={len(sel_non)} total={len(selected)}"
    )

    # Index for GMIE coverage + spatial index of pivots for overlay on every tile.
    build_gmie_index()
    from shapely import STRtree
    from shapely.geometry import box

    tree = STRtree(pivot_geoms)

    rng = random.Random(7)
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
        r["year"] = rng.choice([2017, 2018, 2019])
        r["gmie_path"] = covering_gmie(r["lon"], r["lat"])
        # pivots intersecting a ~tile-sized lon/lat box around the centre.
        m = 0.01
        qbox = box(r["lon"] - m, r["lat"] - m, r["lon"] + m, r["lat"] + m)
        idxs = tree.query(qbox)
        pv = [pivot_geoms[j] for j in idxs if pivot_geoms[j].intersects(qbox)]
        r["pivots"] = pv

    io.check_disk()

    counts: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for cls in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write tiles",
        ):
            if cls is not None:
                counts[cls] += 1

    # Class balance among selected (primary class per sample).
    sel_counts = Counter(r["cls"] for r in selected)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "ESSD / Harvard Dataverse",
            "license": "CC0-1.0",
            "provenance": {
                "url": "https://doi.org/10.7910/DVN/HKBAQQ",
                "have_locally": False,
                "annotation_method": "derived-product (GMIE irrigation map + GCPIS "
                "centre-pivot detection) with manual VHR validation",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts_primary": {
                name: sel_counts.get(name, 0) for name, _ in CLASSES
            },
            "notes": (
                "Global derived-product map -> bounded-tile sampling (<=1000 tiles/class). "
                "64x64 uint8 tiles in local UTM at 10 m; GMIE (EPSG:4326 ~100 m) reprojected "
                "with nearest resampling. Classes: 0 central pivot (GCPIS polygons, overlaid "
                "on all tiles), 1 irrigated cropland (GMIE proportion >=0.5), 2 non-irrigated "
                "(GMIE proportion <=0.05, observed); 255 nodata (background or ambiguous "
                "0.05-0.5 proportion). Irrigated/non-irrigated windows chosen homogeneous "
                "(all pixels in a ~640 m coarse cell pass the threshold). Tiles are "
                "multi-class where classes co-occur; class_counts_primary counts the class a "
                "tile was sampled for. Time range: 1-year window uniformly in 2017-2019 "
                "(GMIE production period)."
            ),
        },
    )
    print("class_counts_primary:", dict(sel_counts))
    print("written this run:", dict(counts))
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
