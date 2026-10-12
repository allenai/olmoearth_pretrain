"""Process GRanD (Global Reservoir and Dam Database) into unified reservoir+dam label tiles.

Source: the manifest dataset is GRanD v1.3 (Lehner et al. 2011; v1.3 = 7,320 large dams
plus reservoir polygons, expert-curated). GRanD as a standalone product has been
*discontinued* and fully integrated into the **Global Dam Watch (GDW) consensus database
v1.0** (Nature Scientific Data 2024, doi:10.1038/s41597-024-03752-9), which is the current,
freely downloadable realization (figshare 25988293, CC-BY-4.0). We therefore download GDW
v1.0 (barriers point layer + reservoirs polygon layer, EPSG:4326) and **scope to GRanD
provenance only** so this dataset stays distinct from the separately-cataloged GOODD
dataset (GOODD contributes ~24k of GDW's barriers; GRanD contributes ~7.4k).

GRanD subset (ORIG_SRC == 'GRanD' / GRAND_ID > 0):
  * 7,424 barrier (dam) points
  * 7,378 reservoir polygons (all have a matching GRanD barrier; 46 barriers are dam-only)

This is a MIXED-MODALITY dataset (reservoir polygons = segmentation, dam points =
detection). Per spec section 5 we combine both into ONE unified class scheme:

    0 = background   (land / other, inside a detection/segmentation tile)
    1 = reservoir    (inside a GRanD reservoir polygon; visible water body at 10-30 m)
    2 = dam          (barrier point; detection positive, ringed by a nodata buffer)
    255 = nodata/ignore (dam detection buffer ring)

Encoding: for each GRanD barrier we build a 64x64 (640 m) local-UTM tile at 10 m CENTERED
ON THE DAM POINT (the dam sits at the reservoir margin/outlet, so a dam-centered window
captures reservoir water + surrounding land + the dam structure). All GRanD reservoir
polygons intersecting the tile are rasterized to class 1 (all_touched, invalid rings
repaired via make_valid); then every GRanD dam point in the tile is stamped as a class-2
positive (1 px) ringed by a 10 px nodata (255) buffer -- dam coordinates are not pixel
exact, so the ring avoids penalizing a few-pixel offset (spec section 4). Background (land)
fills the rest and supplies spatially-meaningful negatives for the dam class, so no separate
negative tiles are fabricated. The 46 dam-only barriers yield background+dam tiles.

Time range: dams/reservoirs are persistent structures visible throughout the Sentinel era.
Most GRanD dams predate 2016 (YEAR_DAM median unknown/-99). Per spec section 5 (static
labels) we assign each tile a 1-year window with start year uniformly sampled in 2016-2020,
matching how pretraining pairs labels with imagery. (A handful of dams built >= 2016 exist
but yearly precision makes a change-label ill-posed; treated as persistent -- noted.)

Total: ~7,424 tiles, well under the 25k per-dataset cap, so ALL GRanD records are used (no
subsampling needed).

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.grand_global_reservoir_and_dam_database
Idempotent: existing locations/{id}.tif are skipped.
"""

import argparse
import math
import multiprocessing
import os
import random
from collections import Counter
from typing import Any

import numpy as np
import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)

SLUG = "grand_global_reservoir_and_dam_database"
NAME = "GRanD (Global Reservoir and Dam Database)"

FIGSHARE_URL = "https://ndownloader.figshare.com/files/47913754"  # GDW_v1_0_shp.zip
RAW = io.raw_dir(SLUG)
SHP_DIR = os.path.join(str(RAW), "extract", "GDW_v1_0_shp")
BARRIERS = os.path.join(SHP_DIR, "GDW_barriers_v1_0.shp")
RESERVOIRS = os.path.join(SHP_DIR, "GDW_reservoirs_v1_0.shp")

TILE = io.MAX_TILE  # 64 px @ 10 m = 640 m
BUFFER = 10  # nodata ring (px) around each dam positive (spec >= 10)
YEARS = [2016, 2017, 2018, 2019, 2020]

BG, RES, DAM = 0, 1, 2
CLASSES = [
    (
        BG,
        "background",
        "Land or any surface outside a GRanD reservoir polygon, within the tile.",
    ),
    (
        RES,
        "reservoir",
        "Inside a GRanD reservoir polygon: the artificial water body impounded by the dam "
        "(max reservoir extent, expert-curated). Visible at 10-30 m.",
    ),
    (
        DAM,
        "dam",
        "GRanD barrier/dam point (detection positive, 1 px), ringed by a 10 px nodata buffer. "
        "Marks the dam structure at the reservoir outlet.",
    ),
]

# Reservoir polygons + dam points for the GRanD subset are held module-global (loaded once
# per worker) so tile writes can spatial-query them without re-reading the shapefile.
_RES_TREE = None
_RES_GEOMS = None
_DAM_TREE = None
_DAM_XY = None


def _load_grand_subset():
    """Return (res_geoms[list], res_lonlat[Nx2], dam_lonlat[Mx2]) for the GRanD subset."""
    import pyogrio

    b = pyogrio.read_dataframe(BARRIERS, columns=["ORIG_SRC", "GRAND_ID"])
    bg = b[(b["ORIG_SRC"] == "GRanD") | (b["GRAND_ID"] > 0)]
    dam_xy = np.array([[g.x, g.y] for g in bg.geometry.values], dtype="float64")

    r = pyogrio.read_dataframe(RESERVOIRS, columns=["GRAND_ID"])
    rg = r[r["GRAND_ID"] > 0]
    res_geoms = [shapely.make_valid(g) for g in rg.geometry.values]
    res_xy = np.array(
        [[g.centroid.x, g.centroid.y] for g in rg.geometry.values], dtype="float64"
    )
    return res_geoms, res_xy, dam_xy


def _ensure_worker_state():
    """Lazily build the reservoir/dam STRtrees inside each worker process."""
    global _RES_TREE, _RES_GEOMS, _DAM_TREE, _DAM_XY
    if _RES_TREE is not None:
        return
    res_geoms, _res_xy, dam_xy = _load_grand_subset()
    _RES_GEOMS = res_geoms
    _RES_TREE = shapely.STRtree(res_geoms)
    _DAM_XY = dam_xy
    _DAM_TREE = shapely.STRtree([shapely.Point(x, y) for x, y in dam_xy])


def _write_one(rec: dict[str, Any]) -> str | None:
    from shapely.geometry import box

    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return rec.get("prev_key")

    _ensure_worker_state()
    lon, lat = rec["lon"], rec["lat"]
    proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)

    # lon/lat query box covering the tile (+margin) for spatial filtering.
    mlat = (TILE * io.RESOLUTION) / 111320.0
    mlon = mlat / max(math.cos(math.radians(lat)), 0.1)
    qbox = box(lon - mlon, lat - mlat, lon + mlon, lat + mlat)

    # --- reservoirs -> class 1 ---
    shapes = []
    for idx in _RES_TREE.query(qbox):
        g = _RES_GEOMS[int(idx)]
        try:
            gc = g.intersection(qbox)
        except Exception:
            gc = g
        if gc.is_empty:
            continue
        px = geom_to_pixels(gc, WGS84_PROJECTION, proj)
        if not px.is_empty:
            shapes.append((px, RES))
    if shapes:
        label = rasterize_shapes(
            shapes, bounds, fill=BG, dtype="uint8", all_touched=True
        )[0]
    else:
        label = np.full((TILE, TILE), BG, dtype=np.uint8)

    # --- dams -> class 2 detection (buffer ring then positive), overlaid on reservoirs ---
    positives = []
    for idx in _DAM_TREE.query(qbox):
        dlon, dlat = _DAM_XY[int(idx)]
        _, dc, dr = io.lonlat_to_utm_pixel(float(dlon), float(dlat), proj)
        pr, pc = dr - bounds[1], dc - bounds[0]
        if 0 <= pr < TILE and 0 <= pc < TILE:
            positives.append((pr, pc))
    for pr, pc in positives:  # buffer first
        r0, r1 = max(0, pr - BUFFER), min(TILE, pr + BUFFER + 1)
        c0, c1 = max(0, pc - BUFFER), min(TILE, pc + BUFFER + 1)
        label[r0:r1, c0:c1] = io.CLASS_NODATA
    for pr, pc in positives:  # positive on top
        label[pr, pc] = DAM

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
    return "|".join(str(c) for c in present)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    # 1. Download + extract GDW v1.0 shapefiles (idempotent).
    RAW.mkdir(parents=True, exist_ok=True)
    zip_path = RAW / "GDW_v1_0_shp.zip"
    download.download_http(FIGSHARE_URL, zip_path)
    if not os.path.exists(BARRIERS):
        import zipfile

        with zipfile.ZipFile(zip_path.path) as z:
            z.extractall((RAW / "extract").path)
    with (RAW / "SOURCE.txt").open("w") as f:
        f.write(
            "GRanD (Global Reservoir and Dam Database) v1.3, now integrated into the\n"
            "Global Dam Watch (GDW) consensus database v1.0 (the current, downloadable form).\n"
            "GDW v1.0: Nature Scientific Data 2024 doi:10.1038/s41597-024-03752-9,\n"
            "figshare 25988293 (CC-BY-4.0). File: GDW_v1_0_shp.zip ->\n"
            "extract/GDW_v1_0_shp/GDW_{barriers,reservoirs}_v1_0.shp (EPSG:4326).\n"
            "Scoped to GRanD provenance (ORIG_SRC=='GRanD' / GRAND_ID>0): 7424 dam points,\n"
            "7378 reservoir polygons. GOODD-sourced barriers are excluded (separate dataset).\n"
        )

    # 2. Load GRanD subset in the parent to build the per-anchor record list.
    res_geoms, _res_xy, dam_xy = _load_grand_subset()
    print(
        f"GRanD subset: {len(dam_xy)} dam points, {len(res_geoms)} reservoir polygons",
        flush=True,
    )

    # 3. One tile per GRanD dam point (anchored on the dam). All records fit under the cap.
    rng = random.Random(7)
    records: list[dict[str, Any]] = []
    for i in range(len(dam_xy)):
        lon, lat = float(dam_xy[i][0]), float(dam_xy[i][1])
        records.append(
            {
                "sample_id": f"{i:06d}",
                "lon": lon,
                "lat": lat,
                "year": rng.choice(YEARS),
                "source_id": f"GDW/GRanD:dam:{i}",
            }
        )
    print(f"records to write: {len(records)}", flush=True)
    io.check_disk()

    # 4. Write tiles in parallel.
    class_counts: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for present in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in records]),
            total=len(records),
            desc="write tiles",
        ):
            for c in (present or "").split("|"):
                if c != "":
                    class_counts[int(c)] += 1

    n_written = len(list(io.locations_dir(SLUG).glob("*.tif")))
    year_counts = dict(sorted(Counter(r["year"] for r in records).items()))
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Global Dam Watch (GDW) v1.0 / GRanD v1.3 (Global Dam Watch / SEDAC)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.org/10.6084/m9.figshare.25988293",
                "have_locally": False,
                "annotation_method": "manual / expert-curated (GRanD subset of the GDW "
                "consensus database)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": cid, "name": name, "description": desc}
                for cid, name, desc in CLASSES
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_written,
            "class_tile_counts": {
                "background": class_counts.get(BG, 0),
                "reservoir": class_counts.get(RES, 0),
                "dam": class_counts.get(DAM, 0),
            },
            "year_tile_counts": year_counts,
            "tile_size": TILE,
            "detection_encoding": {
                "positive_size": 1,
                "buffer_px": BUFFER,
                "class": DAM,
            },
            "notes": (
                "GRanD reservoirs+dams unified into one scheme: 0 background, 1 reservoir "
                "(polygon segmentation), 2 dam (point detection), 255 nodata (dam buffer "
                "ring). Source = GDW v1.0 (figshare 25988293, CC-BY-4.0), the current form "
                "of the discontinued standalone GRanD; scoped to GRanD provenance "
                "(ORIG_SRC=='GRanD'/GRAND_ID>0) to stay distinct from the separately-"
                "cataloged GOODD dataset. 64x64 UTM 10 m tiles, one per GRanD dam point, "
                "centered on the dam; reservoir polygons rasterized all_touched (invalid "
                "rings repaired via make_valid), dam stamped 1 px + 10 px nodata buffer. "
                "In-tile land supplies dam negatives; no fabricated negative tiles. "
                "Persistent structures -> 1-year window, start year uniform 2016-2020; the "
                "few dams built >=2016 are treated as persistent (yearly change ill-posed). "
                "All ~7424 GRanD records used (under 25k cap)."
            ),
        },
    )
    print(f"class tile counts: {dict(class_counts)}  years: {year_counts}")
    print(f"tif on disk: {n_written}")

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=n_written
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
