"""Process OpenSentinelMap (CVPR EarthVision 2022) into open-set-segmentation tiles.

Source: OpenSentinelMap, Johnson/Treible/Crispell (Vision Systems Inc.), CVPRW 2022.
137k global ~1.9 km spatial cells, each with a per-pixel OSM-derived land-use label plus
multi-year Sentinel-2 imagery. We use ONLY the labels (imagery is ~445 GB and not needed
here -- pretraining supplies its own S2). Labels ship as a 425 MB tarball of PNG masks:

    osm_label_images_v10/{MGRS_TILE}/{cell_id}.png     # 192x192x3 uint8

Each PNG is 192x192 px at 10 m/px (= 1920 m, the ~3.7 km^2 cell) in the cell's MGRS UTM
zone. The three PNG channels are three OSM label "channels" from osm_categories.json:
    ch0 OSM_land_use        : 0..11 (wooded, agricultural, residential, industrial,
                              commercial, recreation, airport, quarry, military,
                              desert_sand, mountain_rock, other_natural)
    ch1 OSM_water_and_roads : 12 water, 13 road
    ch2 buildings           : 14 building
with 254 ("none", explicitly no label) and 255 ("unlabeled", outside OSM coverage) as
non-classes in every channel.

Georeferencing (spec 8.2): the PNGs carry no CRS, but spatial_cell_info.csv gives each
cell's WGS84 bounds + MGRS tile. The cells are axis-aligned 1920 m squares in the MGRS UTM
zone (verified: at 66 N the WGS84 bbox envelope is exactly a 1920 m box rotated by the
meridian-convergence angle). We recover each cell's UTM box by transforming its center to
the MGRS UTM zone and laying a 192 px box (+/-960 m) around it, snapped to the nearest
pixel. No resampling -- the label is already native UTM 10 m.

Recipe (spec 4, dense_raster): flatten the 3 channels to ONE single-band 15-class map by
OSM precedence compositing (highest-precedence label at each pixel wins; e.g. building 100
> road 97 > water 96 > residential 50 > wooded 2). 192 = 3*64, so each cell splits cleanly
into nine 64x64 patches. Keep a patch if >= 5% of its pixels are labeled (OSM coverage is
sparse; a higher threshold would drop thin road/building patches). Tiles-per-class balanced
(rare classes first), up to 1000 tiles/class, 25k cap. Time range = 1-year window on 2019
(mid-point of the 2017-2020 imagery span; OSM land-use is ~static). Labels are OSM reference
polygons (annotation_method OSM-derived), a preferred reference source.

Reproduce (after staging raw/opensentinelmap/{osm_categories.json,spatial_cell_info.csv}
and untarring osm_label_images.tgz there):
    python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.opensentinelmap
"""

import argparse
import csv
import json
import multiprocessing
import os
from collections import Counter
from functools import lru_cache
from typing import Any

import numpy as np
import tqdm
from PIL import Image
from pyproj import Transformer
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest, sampling

SLUG = "opensentinelmap"
NAME = "OpenSentinelMap"
URL = "https://visionsystemsinc.github.io/open-sentinel-map/"

RAW = io.raw_dir(SLUG)
LABELS_ROOT = RAW / "osm_label_images_v10"
CSV_PATH = RAW / "spatial_cell_info.csv"
CATS_PATH = RAW / "osm_categories.json"

CELL_PX = 192
TILE = 64  # 192 = 3 * 64 -> 3x3 grid of patches per cell
RES = 10.0
MIN_LABELED_FRAC = 0.05  # keep a patch only if >=5% of pixels carry a class
PER_CLASS = 1000
YEAR = 2019  # 1-year label window (imagery spans 2017-2020; OSM ~static)
SEED = 42

# Output class scheme: unified 15-class land-use/land-cover from the 3 OSM channels.
# (id, name, description). id == the pixel value stored in the PNG channels.
CLASSES: list[tuple[int, str, str]] = [
    (0, "wooded", "OSM natural=wood / landuse=forest: forested / wooded land."),
    (
        1,
        "agricultural",
        "OSM landuse farmland/meadow/orchard/vineyard/farmyard/allotments: cultivated land.",
    ),
    (
        2,
        "residential",
        "OSM landuse=residential plus dilated residential buildings/living streets.",
    ),
    (
        3,
        "industrial",
        "OSM landuse=industrial and industrial/warehouse/factory buildings, works, "
        "water/wastewater plants, piers, prisons.",
    ),
    (
        4,
        "commercial",
        "OSM landuse commercial/retail and parking/commercial amenities.",
    ),
    (
        5,
        "recreation",
        "OSM leisure: golf courses, sports pitches, parks and other recreation grounds.",
    ),
    (6, "airport", "OSM aeroway:* and military airfields: airport / aerodrome land."),
    (7, "quarry", "OSM landuse=quarry: surface mineral extraction sites."),
    (8, "military", "OSM landuse=military / military:*: military installations."),
    (9, "desert_sand", "OSM natural desert/sand/beach: sandy / desert surfaces."),
    (
        10,
        "mountain_rock",
        "OSM natural mountain_range/bare_rock/scree: exposed rock / mountainous terrain.",
    ),
    (
        11,
        "other_natural",
        "OSM natural scrub/wetland/grassland and other unbuilt natural cover.",
    ),
    (
        12,
        "water",
        "OSM natural=water / water:* / waterway:*: rivers, lakes, reservoirs.",
    ),
    (13, "road", "OSM highway:* road network (dilated to be visible at 10 m)."),
    (14, "building", "OSM building:* / man_made towers/tanks (built structures)."),
]
NUM_CLASSES = len(CLASSES)
CLASS_NAMES = {c[0]: c[1] for c in CLASSES}


def _build_precedence_lut() -> np.ndarray:
    """LUT[value] = OSM precedence for class values 0..14, else -1 (254/255/other)."""
    with CATS_PATH.open() as f:
        cats = json.load(f)
    lut = np.full(256, -1.0, dtype=np.float64)
    for ch in cats["channels"]:
        for _name, info in ch["labels"].items():
            v = info["value"]
            if v <= 14:
                lut[v] = float(info["precedence"])
    return lut


_PREC_LUT = _build_precedence_lut()


def composite(im: np.ndarray) -> np.ndarray:
    """Flatten a 192x192x3 OSM label PNG to a single-band uint8 class map.

    At each pixel, the label with the highest OSM precedence across the 3 channels wins;
    pixels with no class in any channel (all 254/255) become nodata (255).
    """
    prec = _PREC_LUT[im]  # H x W x 3
    winner = prec.argmax(axis=2)  # channel index of max precedence
    out = np.take_along_axis(im, winner[..., None], axis=2)[..., 0].astype(np.uint8)
    out[prec.max(axis=2) < 0] = io.CLASS_NODATA
    return out


def epsg_from_mgrs(mgrs: str) -> int:
    zone = int(mgrs[:2])
    north = (
        mgrs[2].upper() >= "N"
    )  # MGRS latitude band N and above -> northern hemisphere
    return (32600 if north else 32700) + zone


@lru_cache(maxsize=64)
def _transformer(epsg: int) -> Transformer:
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)


def cell_topleft_pixel(rec: dict[str, Any]) -> tuple[int, int, int]:
    """Return (epsg, col0, row0): the top-left pixel of the full 192px cell in its UTM zone."""
    epsg = epsg_from_mgrs(rec["mgrs"])
    clon = (rec["min_lon"] + rec["max_lon"]) / 2.0
    clat = (rec["min_lat"] + rec["max_lat"]) / 2.0
    cx, cy = _transformer(epsg).transform(clon, clat)
    half = CELL_PX * RES / 2.0
    col0 = int(round((cx - half) / RES))
    row0 = int(
        round(-(cy + half) / RES)
    )  # pixel row = -northing/res (north-up, y_res<0)
    return epsg, col0, row0


def _png_path(mgrs: str, cell_id: str) -> str:
    return os.path.join(LABELS_ROOT.path, mgrs, f"{cell_id}.png")


def _scan_chunk(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Composite each cell PNG and emit candidate 64x64 sub-tiles passing the labeled-frac
    filter, each tagged with its class set + georeferencing (no arrays kept).
    """
    min_labeled = int(MIN_LABELED_FRAC * TILE * TILE)
    out: list[dict[str, Any]] = []
    for rec in cells:
        path = _png_path(rec["mgrs"], rec["cell_id"])
        try:
            im = np.asarray(Image.open(path))
        except (FileNotFoundError, OSError):
            continue
        if im.shape != (CELL_PX, CELL_PX, 3):
            continue
        comp = composite(im)
        epsg, col0, row0 = cell_topleft_pixel(rec)
        for ti in range(CELL_PX // TILE):
            for tj in range(CELL_PX // TILE):
                sub = comp[ti * TILE : (ti + 1) * TILE, tj * TILE : (tj + 1) * TILE]
                labeled = int((sub != io.CLASS_NODATA).sum())
                if labeled < min_labeled:
                    continue
                present = sorted(int(v) for v in np.unique(sub) if v != io.CLASS_NODATA)
                if not present:
                    continue
                out.append(
                    {
                        "cell_id": rec["cell_id"],
                        "mgrs": rec["mgrs"],
                        "epsg": epsg,
                        "bounds": (
                            col0 + tj * TILE,
                            row0 + ti * TILE,
                            col0 + tj * TILE + TILE,
                            row0 + ti * TILE + TILE,
                        ),
                        "ti": ti,
                        "tj": tj,
                        "classes_present": present,
                    }
                )
    return out


def _write_one(rec: dict[str, Any]) -> tuple[str, list[int]]:
    sid = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sid}.tif").exists():
        return sid, rec["classes_present"]
    im = np.asarray(Image.open(_png_path(rec["mgrs"], rec["cell_id"])))
    comp = composite(im)
    ti, tj = rec["ti"], rec["tj"]
    sub = comp[ti * TILE : (ti + 1) * TILE, tj * TILE : (tj + 1) * TILE]
    proj = Projection(CRS.from_epsg(rec["epsg"]), RES, -RES)
    bounds = tuple(rec["bounds"])
    io.write_label_geotiff(SLUG, sid, sub, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sid,
        proj,
        bounds,
        io.year_range(YEAR),
        source_id=f"{rec['mgrs']}/{rec['cell_id']}_r{ti}c{tj}",
        classes_present=rec["classes_present"],
    )
    return sid, rec["classes_present"]


def load_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    with CSV_PATH.open() as f:
        for r in csv.DictReader(f):
            cells.append(
                {
                    "cell_id": r["cell_id"],
                    "mgrs": r["MGRS_tile"],
                    "min_lon": float(r["min_lon"]),
                    "max_lon": float(r["max_lon"]),
                    "min_lat": float(r["min_lat"]),
                    "max_lat": float(r["max_lat"]),
                }
            )
    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--limit-cells", type=int, default=0, help="debug: only scan the first N cells"
    )
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    cells = load_cells()
    # Keep only cells whose label PNG actually exists on disk.
    cells = [c for c in cells if os.path.exists(_png_path(c["mgrs"], c["cell_id"]))]
    if args.limit_cells:
        cells = cells[: args.limit_cells]
    print(f"cells with labels: {len(cells)}")

    # ---- Phase 1: scan all cells -> candidate 64x64 patches --------------------------
    chunks = [cells[i : i + 200] for i in range(0, len(cells), 200)]
    cands: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as pool:
        for recs in tqdm.tqdm(
            star_imap_unordered(pool, _scan_chunk, [dict(cells=c) for c in chunks]),
            total=len(chunks),
            desc="scan",
        ):
            cands.extend(recs)
    # Sort into a deterministic order: the parallel scan returns chunks out of order, and
    # the seeded selection below depends on candidate order, so sorting makes the whole
    # pipeline reproducible/idempotent across runs.
    cands.sort(key=lambda r: (r["mgrs"], r["cell_id"], r["ti"], r["tj"]))
    print(f"candidate patches (>= {MIN_LABELED_FRAC:.0%} labeled): {len(cands)}")
    avail = Counter()
    for r in cands:
        for cid in r["classes_present"]:
            avail[cid] += 1
    print("candidate patches per class:")
    for cid, name, _d in CLASSES:
        print(f"  {cid:>2} {name:14} {avail.get(cid, 0)}")

    # ---- Phase 2: tiles-per-class balanced selection ---------------------------------
    selected = sampling.select_tiles_per_class(
        cands,
        classes_key="classes_present",
        per_class=PER_CLASS,
        total_cap=sampling.MAX_SAMPLES_PER_DATASET,
        seed=SEED,
    )
    selected.sort(key=lambda r: (r["mgrs"], r["cell_id"], r["ti"], r["tj"]))
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} patches (<= {PER_CLASS}/class, 25k cap)")

    # ---- Phase 3: write patches in parallel ------------------------------------------
    tile_counts = Counter()
    with multiprocessing.Pool(args.workers) as pool:
        done = 0
        for _sid, present in tqdm.tqdm(
            star_imap_unordered(pool, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write",
        ):
            for cid in present:
                tile_counts[cid] += 1
            done += 1
            if done % 4000 == 0:
                io.check_disk()

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "OpenSentinelMap (Vision Systems Inc., CVPRW 2022)",
            "license": "open (CC BY 4.0 per dataset site)",
            "provenance": {
                "url": URL,
                "paper": "Johnson, Treible, Crispell. OpenSentinelMap. CVPRW 2022.",
                "have_locally": False,
                "annotation_method": "OSM-derived per-pixel land-use labels (v10) over Sentinel-2 cells",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": cid, "name": name, "description": desc}
                for cid, name, desc in CLASSES
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_tile_counts": {
                CLASS_NAMES[cid]: int(tile_counts.get(cid, 0))
                for cid in range(NUM_CLASSES)
            },
            "notes": (
                "Labels-only use of OpenSentinelMap (Sentinel-2 imagery not downloaded). Each "
                "192x192 OSM label PNG (3 channels: OSM_land_use / water_and_roads / buildings) "
                "is precedence-composited into one 15-class uint8 map and split into nine 64x64 "
                "patches. Georeferenced by reconstructing each cell's UTM box from its WGS84 "
                "center + MGRS zone (native 10 m UTM; no resampling). Patches kept with >=5% "
                "labeled pixels (OSM coverage is sparse). Tiles-per-class balanced, <=1000 "
                "tiles/class, 25k cap. Time range = 1-year window on 2019 (imagery spans "
                "2017-2020; OSM land-use ~static). Precedence compositing: building>road>water/"
                "industrial>commercial>recreation>airport>quarry>military>residential>desert>"
                "other_natural>agricultural>wooded>mountain_rock."
            ),
        },
    )
    print("tile counts per class:")
    for cid, name, _d in CLASSES:
        print(f"  {cid:>2} {name:14} {tile_counts.get(cid, 0)}")

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
