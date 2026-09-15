"""Oil Slicks & Look-Alikes (E. Mediterranean SAR) -> open-set-segmentation tiles.

Source: PANGAEA https://doi.org/10.1594/PANGAEA.980773 (Yang & Singha 2025), CC-BY-4.0;
ESSD data descriptor https://doi.org/10.5194/essd-17-6807-2025. Manually interpreted
Sentinel-1 SAR patches over the Eastern Mediterranean Sea in 2019:
  * OIL set: 1365 patches with 3225 oil-slick objects (subsets ``ow`` oil/water,
    ``oc`` oil/coast), each object a manually drawn bounding box.
  * NO-OIL set: 2290 look-alike patches (``nw`` no_oil/water, ``nc`` no_oil/coast) --
    oceanic/atmospheric phenomena that mimic oil in SAR but are NOT oil; whole-patch
    scenes with no localized box.

Georeferencing: the PANGAEA tab-delimited data matrix (raw/.../pangaea_data.txt) carries,
per row, the patch corner lon/lat (ul/ur/br/bl), the oil-object bbox corner lon/lat
(ul/ur/br/bl), pixel positions, the S1 acquisition start/end time, and the Sentinel-1
.SAFE granule id. So every oil footprint and every patch is fully georeferenced -- the
JPG SAR patches / PASCAL-VOC XML boxes are NOT needed (labels are placed directly on a
UTM 10 m grid). No raster download required.

Encoding (label_type = bounding boxes -> detection, spec section 4). Binary class map
matching the manifest's two classes (spec section 5, multi-target -> one class map):
  * 0 = oil_slick
  * 1 = look_alike_no_oil  (the "look-alike/no-oil" negative/confuser class: both the
        dedicated look-alike scenes AND non-oil sea surrounding a slick)
  * 255 = nodata/ignore (buffer ring around imprecise oil-box edges)
For each selected oil object we write a 64x64 UTM 10 m tile centered on the object's geo
centroid: the object's geo quadrilateral (plus any sibling oil objects of the same patch
that fall in the tile) is rasterized as oil (0), dilated by a 10 px nodata ring (255) to
absorb bounding-box imprecision, and the rest of the tile is look_alike_no_oil (1). For
each selected no-oil patch we write a 64x64 tile centered on the patch centroid filled
entirely with look_alike_no_oil (1) -- a spatially-meaningful hard-negative confuser tile
(spec section 5 detection exception).

Time range: each sample uses its own S1 acquisition window [start_time, end_time] (~1-2
min, well under the ~1 hour specific-image budget, spec section 5) -- an oil slick /
look-alike is visible only in the matching S1 acquisition. All labels are 2019 (post-2016).

Counts: up to 1000 oil-slick tiles + 1000 look-alike tiles (spec section 5; well under the
25k cap). class 1 additionally appears as background in every oil tile.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.oil_slicks_look_alikes_e_mediterranean_sar
"""

import argparse
import multiprocessing
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.mp import star_imap_unordered
from scipy.ndimage import binary_dilation

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)

SLUG = "oil_slicks_look_alikes_e_mediterranean_sar"
NAME = "Oil Slicks & Look-Alikes (E. Mediterranean SAR)"
DATA_TABLE = io.raw_dir(SLUG) / "pangaea_data.txt"

OIL_ID = 0
NOOIL_ID = 1
CLASS_NAMES = {OIL_ID: "oil_slick", NOOIL_ID: "look_alike_no_oil"}

TILE = 64
BUFFER = 10  # nodata ring around oil boxes (spec: buffer >= 10 px; box edges imprecise)
PER_CLASS_OIL = 1000
N_NOOIL = 1000
SEED = 42

# Column indices in the PANGAEA data matrix (see raw/.../pangaea_data.txt header).
C_SET, C_PATCH, C_START, C_END = 0, 4, 5, 6
C_PATCH_CORNERS = list(
    range(10, 18)
)  # ul_lon,ul_lat,ur_lon,ur_lat,br_lon,br_lat,bl_lon,bl_lat
C_OBJ_CORNERS = list(range(18, 26))  # obj_ul_lon,lat, ur, br, bl


def _corners(row: list[str], cols: list[int]) -> list[tuple[float, float]]:
    """Return [(lon,lat) x4] from 8 consecutive lon/lat columns."""
    vals = [float(row[c]) for c in cols]
    return [
        (vals[0], vals[1]),
        (vals[2], vals[3]),
        (vals[4], vals[5]),
        (vals[6], vals[7]),
    ]


def _centroid(corners: list[tuple[float, float]]) -> tuple[float, float]:
    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]
    return sum(lons) / len(lons), sum(lats) / len(lats)


def parse_table() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse the data matrix into (oil_object_records, nooil_patch_records).

    Each oil record carries the geo quads of ALL oil objects sharing its patch so a tile
    can mark neighbouring slicks. No-oil records carry the patch quad/centroid.
    """
    with open(DATA_TABLE) as f:
        lines = f.readlines()
    hdr_idx = next(i for i, ln in enumerate(lines) if ln.startswith("Image set"))
    rows = [ln.rstrip("\n").split("\t") for ln in lines[hdr_idx + 1 :] if ln.strip()]

    oil_by_patch: dict[str, list[list[tuple[float, float]]]] = defaultdict(list)
    oil_rows: list[dict[str, Any]] = []
    nooil: list[dict[str, Any]] = []
    for r in rows:
        if len(r) <= C_OBJ_CORNERS[-1]:
            continue
        subset, patch = r[C_SET], r[C_PATCH]
        start, end = r[C_START], r[C_END]
        obj_empty = r[C_OBJ_CORNERS[0]].strip() == ""
        if obj_empty:  # no-oil / look-alike patch (one row, no box)
            pc = _corners(r, C_PATCH_CORNERS)
            lon, lat = _centroid(pc)
            nooil.append(
                {
                    "subset": subset,
                    "patch": patch,
                    "start": start,
                    "end": end,
                    "lon": lon,
                    "lat": lat,
                }
            )
        else:  # oil object
            quad = _corners(r, C_OBJ_CORNERS)
            oil_by_patch[patch].append(quad)
            lon, lat = _centroid(quad)
            oil_rows.append(
                {
                    "subset": subset,
                    "patch": patch,
                    "start": start,
                    "end": end,
                    "lon": lon,
                    "lat": lat,
                    "quad": quad,
                }
            )
    # Attach sibling quads (all oil objects of the same patch) to each oil record.
    for rec in oil_rows:
        rec["oil_quads"] = oil_by_patch[rec["patch"]]
    return oil_rows, nooil


def _time_range(start: str, end: str) -> tuple[datetime, datetime]:
    t0 = datetime.fromisoformat(start).replace(tzinfo=UTC)
    t1 = datetime.fromisoformat(end).replace(tzinfo=UTC)
    if t1 <= t0:
        t1 = t0 + timedelta(minutes=1)
    return t0, t1


def _write_oil(rec: dict[str, Any]) -> str:
    sid = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sid}.tif").exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    _, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"], proj)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    shapes = []
    for quad in rec["oil_quads"]:
        poly_px = geom_to_pixels(shapely.Polygon(quad), WGS84_PROJECTION, proj)
        if not poly_px.is_empty:
            shapes.append((poly_px, 1))
    oil_mask = rasterize_shapes(
        shapes, bounds, fill=0, dtype="uint8", all_touched=True
    )[0].astype(bool)
    if not oil_mask.any():
        oil_mask[TILE // 2, TILE // 2] = (
            True  # tiny/edge box: guarantee a positive pixel
        )
    ring = binary_dilation(oil_mask, iterations=BUFFER) & ~oil_mask
    arr = np.full((TILE, TILE), NOOIL_ID, dtype=np.uint8)
    arr[ring] = io.CLASS_NODATA
    arr[oil_mask] = OIL_ID
    io.write_label_geotiff(SLUG, sid, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sid,
        proj,
        bounds,
        _time_range(rec["start"], rec["end"]),
        source_id=f"{rec['subset']}/{rec['patch']}",
        classes_present=sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA}),
    )
    return "oil"


def _write_nooil(rec: dict[str, Any]) -> str:
    sid = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sid}.tif").exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    _, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"], proj)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    arr = np.full(
        (TILE, TILE), NOOIL_ID, dtype=np.uint8
    )  # whole tile = look-alike/no-oil
    io.write_label_geotiff(SLUG, sid, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sid,
        proj,
        bounds,
        _time_range(rec["start"], rec["end"]),
        source_id=f"{rec['subset']}/{rec['patch']}",
        classes_present=[NOOIL_ID],
    )
    return "nooil"


def _dispatch(rec: dict[str, Any]) -> str:
    return _write_oil(rec) if rec["kind"] == "oil" else _write_nooil(rec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    oil_rows, nooil = parse_table()
    print(
        f"parsed {len(oil_rows)} oil objects across "
        f"{len({r['patch'] for r in oil_rows})} oil patches; "
        f"{len(nooil)} no-oil/look-alike patches",
        flush=True,
    )

    rng = random.Random(SEED)
    rng.shuffle(oil_rows)
    rng.shuffle(nooil)
    sel_oil = oil_rows[:PER_CLASS_OIL]
    sel_nooil = nooil[:N_NOOIL]
    for r in sel_oil:
        r["kind"] = "oil"
    for r in sel_nooil:
        r["kind"] = "nooil"
    all_recs = sel_oil + sel_nooil
    for i, r in enumerate(all_recs):
        r["sample_id"] = f"{i:06d}"
    print(
        f"selected {len(sel_oil)} oil tiles + {len(sel_nooil)} look-alike tiles",
        flush=True,
    )

    io.check_disk()
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
            "task_type": "classification",  # detection encoded as per-pixel classes
            "source": "PANGAEA / ESSD",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.org/10.1594/PANGAEA.980773",
                "have_locally": False,
                "annotation_method": "manual (two interpreters); Sentinel-1 SAR, 2019",
            },
            "sensors_relevant": ["sentinel1", "sentinel2"],
            "classes": [
                {
                    "id": OIL_ID,
                    "name": "oil_slick",
                    "description": "Manually interpreted oil slick in Sentinel-1 SAR "
                    "(dark, low-backscatter surface film). Rasterized from the annotated "
                    "geo-referenced bounding box footprint.",
                },
                {
                    "id": NOOIL_ID,
                    "name": "look_alike_no_oil",
                    "description": "Look-alike / no-oil sea surface: oceanic or atmospheric "
                    "phenomena (low wind, biogenic films, rain cells, etc.) that resemble oil "
                    "in SAR but are not oil, plus ordinary non-oil sea surrounding a slick. "
                    "The manifest's 'look-alike/no-oil' confuser class.",
                },
            ],
            "nodata_value": io.CLASS_NODATA,
            "detection_encoding": {
                "tile_size": TILE,
                "buffer_size": BUFFER,
                "note": "Oil geo-bbox footprint rasterized as class 0, dilated 10 px "
                "nodata ring, rest class 1; no-oil patches filled entirely class 1.",
            },
            "num_samples": len(all_recs),
            "class_tile_counts": {
                "oil_slick_tiles": len(sel_oil),
                "look_alike_no_oil_tiles": len(sel_nooil),
            },
            "available": {
                "oil_objects": len(oil_rows),
                "oil_patches": len({r["patch"] for r in oil_rows}),
                "look_alike_patches": len(nooil),
            },
            "notes": (
                "PANGAEA E. Mediterranean Sentinel-1 oil-slick / look-alike dataset "
                "(Yang & Singha 2025). Labels taken from the PANGAEA tab-delimited data "
                "matrix (patch + object corner lon/lat, S1 acquisition times, .SAFE ids); "
                "JPG patches / XML boxes not needed. label_type='bounding boxes' -> "
                "detection encoding: 64x64 UTM 10 m tile per oil object centered on its geo "
                "centroid, oil geo-bbox footprint (+ sibling oil boxes of the patch) as "
                "class 0, 10 px nodata (255) ring around it, rest class 1 (look-alike/no-oil "
                "sea). No-oil/look-alike patches -> 64x64 tiles filled entirely class 1 "
                "(hard-negative confusers; whole-patch phenomena, no localized box). Time "
                "range = each sample's own S1 acquisition window [start,end] (~1-2 min, "
                "specific-image; all 2019, post-2016). Up to 1000 oil tiles + 1000 "
                "look-alike tiles (spec section 5). class 1 also fills the background of "
                "every oil tile."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(all_recs)
    )
    print(f"done: {len(all_recs)} samples", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
