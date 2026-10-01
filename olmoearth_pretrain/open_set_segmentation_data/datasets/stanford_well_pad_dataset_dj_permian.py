"""Process the Stanford Well-Pad Dataset (DJ & Permian) into open-set-segmentation tiles.

Source (external, GitHub): https://github.com/stanfordmlgroup/well-pad-denver-permian
Paper: "Deep learning for detecting and characterizing oil and gas well pads in
satellite imagery" (Stanford ML Group). Expert- and crowd-curated bounding-box
annotations of oil/gas **well pads** (and, in a separate file, individual **storage
tanks**) over the Permian and Denver-Julesburg (DJ) basins.

We download only the LABEL tables (label-only extraction, no imagery pulled):
  data/training/datasets/well-pad_dataset.csv     (88,044 image rows, 12,490 well-pad boxes)
  data/training/datasets/storage-tank_dataset.csv (5,435 image rows, 10,470 tank boxes)
Each CSV row is one Google-Earth-basemap image chip (well pad: 640x640, EPSG:3857) with:
  centroid_lat/lon, extent_image (WKT POLYGON, the chip's lon/lat extent),
  annotations_latlon (list of {"bbox": WKT POLYGON} well-pad/tank boxes in EPSG:4326),
  split, basin, source. Rows with an empty annotation list are true negatives (the chip
  contains no object of that type). Annotations cover the WHOLE chip, so within a chip
  every well pad is labeled and background pixels are true negatives.

Decisions (spec sections 2-5):
  * label_type polygons/boxes -> POLYGON rasterization recipe (spec section 4). Each
    well-pad box is rasterized as class 1 (well_pad) into the chip's own UTM 10 m tile;
    outside boxes = background (0). Well pads are 30-200 m (median ~89 m, i.e. ~9 px at
    10 m) -> clearly observable at 10 m.
  * STORAGE TANKS ARE DROPPED. Individual tanks in this dataset are ~4-6 m across
    (median ~4.7 m, well under one 10 m pixel), not observable at 10 m from
    S2/S1/Landsat. We keep a single foreground class (well_pad); note in summary.
  * Tile = the chip's UTM pixel extent (~20-23 px square, <= 64 cap). Because all well
    pads in a chip are annotated, background within the tile is a true negative -> we can
    emit both positive tiles (>=1 well pad) and background-only NEGATIVE tiles (detection
    exception, spec section 5).
  * Time range: well pads are persistent structures and the Google-basemap chips are
    undated mosaics; manifest range is 2016-2022. We assign a static representative
    1-year window (2021) to every sample (spec section 5, static labels; post-2016).
    change_time = null.
  * Sampling: single foreground class -> up to PER_CLASS (1000) positive well-pad tiles
    + N_NEGATIVES (1000) background tiles (well under the 25k cap), matching the
    turbine/vessel detection precedent. All source splits used (splits pretraining-agnostic).

Classes: 0 background, 1 well_pad.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.stanford_well_pad_dataset_dj_permian
"""

import argparse
import ast
import math
import multiprocessing
import random
import re
from collections import Counter
from typing import Any

import numpy as np
import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import (
    download,
    io,
    manifest,
    rasterize,
)

SLUG = "stanford_well_pad_dataset_dj_permian"
NAME = "Stanford Well-Pad Dataset (DJ & Permian)"
URL = "https://github.com/stanfordmlgroup/well-pad-denver-permian"
RAW_BASE = (
    "https://raw.githubusercontent.com/stanfordmlgroup/"
    "well-pad-denver-permian/main/data/training/datasets"
)
WELL_PAD_CSV = "well-pad_dataset.csv"
STORAGE_TANK_CSV = "storage-tank_dataset.csv"

BACKGROUND_ID = 0
WELL_PAD_ID = 1
CLASS_NAMES = {BACKGROUND_ID: "background", WELL_PAD_ID: "well_pad"}

PER_CLASS = 1000  # positive well-pad tiles (single foreground class, spec section 5)
N_NEGATIVES = 1000  # background-only tiles from well-pad-free chips
STATIC_YEAR = 2021  # representative 1-year window (persistent structures; post-2016)
MAX_TILE = io.MAX_TILE
SEED = 42

_FLOAT_RE = re.compile(r"-?\d+\.\d+")


def _wkt_coords(wkt: str) -> list[tuple[float, float]]:
    """Parse a simple WKT POLYGON ring into a list of (lon, lat) tuples."""
    nums = [float(n) for n in _FLOAT_RE.findall(wkt)]
    return list(zip(nums[0::2], nums[1::2]))


def _load_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (positive_records, negative_records) parsed from the well-pad CSV.

    A positive record has >=1 well-pad box; a negative record has none. We prefer
    negatives inside the Permian/DJ basins (hard, in-context negatives) over 'none'/'other'.
    """
    import pandas as pd

    csv_path = io.raw_dir(SLUG) / WELL_PAD_CSV
    df = pd.read_csv(str(csv_path))
    positives: list[dict[str, Any]] = []
    neg_basin: list[dict[str, Any]] = []
    neg_other: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        try:
            anns = ast.literal_eval(row["annotations_latlon"])
        except (ValueError, SyntaxError):
            anns = []
        ext = _wkt_coords(row["extent_image"])
        if len(ext) < 4:
            continue
        rec: dict[str, Any] = {
            "clon": float(row["centroid_lon"]),
            "clat": float(row["centroid_lat"]),
            "ext": ext,
            "src": f"well-pad/{row['basin']}/{row['split']}/img{row['image_id']}",
            "basin": str(row["basin"]),
        }
        if anns:
            boxes = []
            for a in anns:
                ring = _wkt_coords(a["bbox"])
                if len(ring) >= 4:
                    boxes.append(ring)
            if not boxes:
                continue
            rec["kind"] = "pos"
            rec["boxes"] = boxes
            positives.append(rec)
        else:
            rec["kind"] = "neg"
            (neg_basin if rec["basin"] in ("permian", "denver") else neg_other).append(
                rec
            )
    negatives = neg_basin + neg_other  # basin negatives first
    return positives, negatives


def _tile_bounds(proj, ext_lonlat: list[tuple[float, float]]):
    """Integer UTM pixel bounds for a chip's extent polygon (clamped to MAX_TILE)."""
    g = rasterize.geom_to_pixels(shapely.Polygon(ext_lonlat), WGS84_PROJECTION, proj)
    minx, miny, maxx, maxy = g.bounds
    x0, y0 = int(math.floor(minx)), int(math.floor(miny))
    x1, y1 = int(math.ceil(maxx)), int(math.ceil(maxy))
    w, h = x1 - x0, y1 - y0
    if w > MAX_TILE:  # center-crop (chips are ~20-23 px so this is a safety net)
        x0 += (w - MAX_TILE) // 2
        x1 = x0 + MAX_TILE
    if h > MAX_TILE:
        y0 += (h - MAX_TILE) // 2
        y1 = y0 + MAX_TILE
    return (x0, y0, x1, y1)


def _write_sample(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["clon"], rec["clat"])
    bounds = _tile_bounds(proj, rec["ext"])
    shapes: list[tuple[Any, int]] = []
    for ring in rec.get("boxes", []):
        gp = rasterize.geom_to_pixels(shapely.Polygon(ring), WGS84_PROJECTION, proj)
        if not gp.is_empty:
            shapes.append((gp, WELL_PAD_ID))
    if shapes:
        arr = rasterize.rasterize_shapes(
            shapes, bounds, fill=BACKGROUND_ID, dtype="uint8", all_touched=True
        )
    else:
        w, h = bounds[2] - bounds[0], bounds[3] - bounds[1]
        arr = np.full((1, h, w), BACKGROUND_ID, dtype=np.uint8)
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(STATIC_YEAR),
        change_time=None,
        source_id=rec["src"],
        classes_present=sorted(set(np.unique(arr).tolist())),
    )
    return rec["kind"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    for fname in (WELL_PAD_CSV, STORAGE_TANK_CSV):
        download.download_http(f"{RAW_BASE}/{fname}", raw / fname)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Stanford Well-Pad Dataset (DJ & Permian)\n"
            f"{URL}\n"
            "Label-only download: data/training/datasets/{well-pad,storage-tank}_dataset.csv "
            "(expert/crowd-curated bounding boxes; annotations_latlon in EPSG:4326).\n"
            "Only well-pad boxes are used (storage tanks ~4-6 m are sub-pixel at 10 m and "
            "are dropped). Imagery is supplied by pretraining, not downloaded here.\n"
        )

    io.check_disk()

    positives, negatives = _load_records()
    print(
        f"parsed {len(positives)} well-pad chips (positives), "
        f"{len(negatives)} well-pad-free chips (negatives)",
        flush=True,
    )

    rng = random.Random(SEED)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    sel_pos = positives[:PER_CLASS]
    sel_neg = negatives[:N_NEGATIVES]
    all_recs = sel_pos + sel_neg
    for i, r in enumerate(all_recs):
        r["sample_id"] = f"{i:06d}"
    n_boxes = sum(len(r["boxes"]) for r in sel_pos)
    print(
        f"selected {len(sel_pos)} positive tiles ({n_boxes} well-pad boxes) + "
        f"{len(sel_neg)} negative tiles = {len(all_recs)} samples",
        flush=True,
    )

    io.check_disk()
    results: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_sample, [dict(rec=r) for r in all_recs]),
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
            "task_type": "classification",  # detection/polygon encoded as per-pixel classes
            "source": "Stanford ML Group (GitHub / Nature Communications)",
            "license": "check repo (public GitHub research release)",
            "provenance": {
                "url": URL,
                "have_locally": False,
                "annotation_method": "manual/expert- and crowd-curated bounding boxes",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {
                    "id": BACKGROUND_ID,
                    "name": "background",
                    "description": "Land within the annotated chip that contains no oil/gas "
                    "well pad (true negative: chips are fully annotated).",
                },
                {
                    "id": WELL_PAD_ID,
                    "name": "well_pad",
                    "description": "Oil/gas well pad: cleared/graded pad hosting wellheads, "
                    "tanks and access roads (typ. 30-200 m). Bounding-box footprint rasterized "
                    "at 10 m over the Permian and Denver-Julesburg basins.",
                },
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(all_recs),
            "class_tile_counts": {
                "well_pad_positive_tiles": len(sel_pos),
                "background_negative_tiles": len(sel_neg),
                "well_pad_boxes_in_positives": n_boxes,
            },
            "available": {
                "positive_chips": len(positives),
                "negative_chips": len(negatives),
            },
            "static_year": STATIC_YEAR,
            "notes": (
                "Well pads as polygon/box footprints rasterized to class 1 in each chip's "
                "own UTM 10 m tile (~20-23 px square = the chip extent); background = 0. "
                "Chips are fully annotated so background is a true negative; positive tiles "
                "(>=1 well pad) + background-only negative tiles are both emitted (detection "
                "exception, spec section 5). Storage-tank class DROPPED: individual tanks "
                "(~4-6 m) are sub-pixel at 10 m. Time range = static representative 1-year "
                "window (2021; well pads persistent, basemap chips undated, manifest range "
                "2016-2022); change_time=null. All source splits used. Single foreground "
                "class -> up to 1000 positive + 1000 negative tiles."
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
