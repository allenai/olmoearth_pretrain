"""Process OlmoEarth vessel attributes (type) into detection tiles by vessel TYPE.

Source: local rslearn dataset (have_locally=true, not copied)
``/weka/dfive-default/rslearn-eai/datasets/sentinel2_vessel_attribute/dataset_v1/20250205``.
This is the existing OlmoEarth vessel-attribute eval. Unlike the plain vessel-DETECTION
eval (``olmoearth_sentinel_2_vessels``), here each window is a small **per-vessel crop**
(128x128 px, local UTM @ 10 m) centered on ONE AIS-matched vessel, with a ~2-hour
Sentinel-2 acquisition ``time_range``. The target attribute processed here is vessel
**type** (length/width/course/speed regression attributes are excluded per the manifest).

Label layer ``info`` (vector GeoJSON, ``layers/info/data.geojson``): a single ``Point``
feature at the vessel, in the window's projection (pixel) coordinates. Its
``properties.type`` is the vessel category, present ONLY for the 9 valid categories
(``rslp.vessel_attribute.ship_types.VESSEL_CATEGORIES`` / ``train.SHIP_TYPE_CATEGORIES``):
cargo, tanker, passenger, service, tug, pleasure, fishing, enforcement, sar. Windows whose
vessel type is ``unknown``/``other`` have NO ``type`` property (matching the eval, which
marks those examples invalid) and are skipped.

Encoding (label_type points -> detection by TYPE, spec section 4). Each positive tile is a
32x32 (DET_TILE) context tile centered on the vessel, in the window's own UTM projection
(source already local UTM @ 10 m -> georeferencing is exact):
  * the vessel is a 1x1 positive whose class id is its vessel TYPE (1..9; background=0),
    ringed by a 10 px nodata (255) buffer (vessel centroids are not pixel-exact), all other
    pixels background (0 = open water / non-vessel).

Negatives: this dataset has NO vessel-free windows (every window is centered on a vessel),
and neighboring vessels within a crop are UNLABELED (only the central target vessel is
annotated), so background-only "negative" tiles sampled from window corners would be
unreliable (they could contain real, unlabeled vessels). We therefore do NOT emit separate
negative tiles. The background class (0) is still abundantly and spatially-meaningfully
represented WITHIN every positive tile (the open water surrounding each vessel, outside the
21x21 ignore ring). Downstream assembly supplements cross-dataset negatives (spec section 5).

Classes (background=0, then eval order): 0 background, 1 cargo, 2 tanker, 3 passenger,
4 service, 5 tug, 6 pleasure, 7 fishing, 8 enforcement, 9 sar.

Sampling: 9 type classes, up to PER_CLASS=1000 positive tiles per type, class-balanced
(``balance_by_class`` by type id), prioritizing rare types (tug/sar/enforcement). Total well
under the 25k cap. All splits used (pretraining-agnostic, spec section 5).

Time range: each sample uses its window's own ~2-hour S2 acquisition ``time_range``
(specific-image label, spec section 5; well under the ~1 hour-scale budget / <= 1 yr).

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_vessel_attributes_type
"""

import argparse
import json
import math
import multiprocessing
import os
from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np
import tqdm
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    balance_by_class,
    encode_detection_tile,
)

SLUG = "olmoearth_vessel_attributes_type"
NAME = "OlmoEarth vessel attributes (type)"
SOURCE = (
    "/weka/dfive-default/rslearn-eai/datasets/sentinel2_vessel_attribute/"
    "dataset_v1/20250205"
)
WINDOWS_ROOT = os.path.join(SOURCE, "windows")

# Eval vessel-type order (rslp.vessel_attribute.train.SHIP_TYPE_CATEGORIES); ids offset by
# +1 so background can take id 0 in the detection encoding.
SHIP_TYPE_CATEGORIES = [
    "cargo",
    "tanker",
    "passenger",
    "service",
    "tug",
    "pleasure",
    "fishing",
    "enforcement",
    "sar",
]
BACKGROUND_ID = 0
TYPE_TO_ID = {name: i + 1 for i, name in enumerate(SHIP_TYPE_CATEGORIES)}
ID_TO_NAME = {0: "background", **{i + 1: n for i, n in enumerate(SHIP_TYPE_CATEGORIES)}}

TYPE_DESCRIPTIONS = {
    "cargo": "Cargo vessel (AIS ship-type 70-79).",
    "tanker": "Tanker (AIS ship-type 80-89).",
    "passenger": "Passenger vessel / ferry (AIS ship-type 60-69).",
    "service": "Service vessel: pilot, port tender, dredger, anti-pollution, etc.",
    "tug": "Tug / towing vessel (AIS ship-type 52, 31-32).",
    "pleasure": "Pleasure craft / sailing vessel (AIS ship-type 36-37).",
    "fishing": "Fishing vessel (AIS ship-type 30).",
    "enforcement": "Law-enforcement vessel (AIS ship-type 55).",
    "sar": "Search-and-rescue vessel (AIS ship-type 51).",
}

# Detection encoding parameters (spec section 4).
DET_TILE = 32
DET_POS_SIZE = 1
DET_BUFFER = 10

PER_CLASS = 1000  # positive tiles per vessel type (9 types -> up to 9k, << 25k cap)


def _list_windows() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for g in sorted(os.listdir(WINDOWS_ROOT)):
        gd = os.path.join(WINDOWS_ROOT, g)
        if not os.path.isdir(gd):
            continue
        for name in os.listdir(gd):
            out.append((g, name))
    return out


def _scan_window(group: str, name: str) -> dict[str, Any] | None:
    """Return a per-vessel record (crs/bounds/time/center/type_id) or None to skip."""
    wdir = os.path.join(WINDOWS_ROOT, group, name)
    try:
        md = json.load(open(os.path.join(wdir, "metadata.json")))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    tr = md.get("time_range")
    if not tr or tr[0] is None:
        return None
    proj = md["projection"]
    if abs(proj.get("x_resolution", 0)) != io.RESOLUTION:
        return None
    try:
        lab = json.load(open(os.path.join(wdir, "layers", "info", "data.geojson")))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    feats = lab.get("features", [])
    if not feats:
        return None
    f = feats[0]
    geom = f.get("geometry") or {}
    if geom.get("type") != "Point":
        return None
    vtype = f.get("properties", {}).get("type")
    if vtype not in TYPE_TO_ID:
        return None  # unknown / other -> not a valid type label (matches the eval)
    x, y = geom["coordinates"][:2]
    return {
        "crs": proj["crs"],
        "tr": tr,
        "src": f"{group}/{name}",
        "center": (float(x), float(y)),
        "type_id": TYPE_TO_ID[vtype],
        "vtype": vtype,
    }


def _projection(crs: str) -> Projection:
    return Projection(CRS.from_string(crs), io.RESOLUTION, -io.RESOLUTION)


def _time_range(tr: list[str]) -> tuple[datetime, datetime]:
    return (datetime.fromisoformat(tr[0]), datetime.fromisoformat(tr[1]))


def _write_positive(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return "skip"
    proj = _projection(rec["crs"])
    cx, cy = rec["center"]
    px, py = int(math.floor(cx)), int(math.floor(cy))
    bounds = io.centered_bounds(px, py, DET_TILE, DET_TILE)
    x_min, y_min = bounds[0], bounds[1]
    lc = px - x_min
    lr = py - y_min
    positives = [(lr, lc, rec["type_id"])]
    arr = encode_detection_tile(
        positives,
        tile_size=DET_TILE,
        positive_size=DET_POS_SIZE,
        buffer_size=DET_BUFFER,
        nodata=io.CLASS_NODATA,
        background=BACKGROUND_ID,
    )
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        _time_range(rec["tr"]),
        source_id=rec["src"],
        classes_present=sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA}),
    )
    return "pos"


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
            "local rslearn dataset (have_locally=true, not copied):\n"
            f"{SOURCE}\n"
            "OlmoEarth vessel-ATTRIBUTE eval. Each window is a 128x128 per-vessel S2 crop "
            "in a local UTM projection @ 10 m with a ~2-hour acquisition time_range, "
            "centered on ONE AIS-matched vessel.\n"
            "layer 'info' (vector GeoJSON): one Point per window at the vessel; "
            "properties.type is the vessel category, present only for the 9 valid "
            "categories (cargo/tanker/passenger/service/tug/pleasure/fishing/enforcement/"
            "sar); unknown/other have no type property and are skipped.\n"
            "Target attribute = vessel TYPE (length/width/course/speed excluded per "
            "manifest).\n"
        )

    windows = _list_windows()
    print(f"scanning {len(windows)} windows", flush=True)
    records: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for rec in tqdm.tqdm(
            star_imap_unordered(
                p, _scan_window, [dict(group=g, name=n) for g, n in windows]
            ),
            total=len(windows),
        ):
            if rec is not None:
                records.append(rec)

    avail = Counter(r["vtype"] for r in records)
    print(f"typed vessels: {len(records)}", flush=True)
    for name in SHIP_TYPE_CATEGORIES:
        print(f"  {name}: {avail.get(name, 0)}", flush=True)

    io.check_disk()

    # Class-balanced selection: up to PER_CLASS positive tiles per vessel type.
    selected = balance_by_class(records, key="type_id", per_class=PER_CLASS)
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    sel_counts = Counter(r["vtype"] for r in selected)
    print(
        f"selected {len(selected)} positive tiles ({dict(sorted(sel_counts.items()))})",
        flush=True,
    )

    results: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_positive, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            results[res] += 1
    print("write results:", dict(results), flush=True)

    io.check_disk()

    classes_meta = [
        {
            "id": 0,
            "name": "background",
            "description": "Open water / non-vessel ocean surface within the tile.",
        }
    ]
    for i, name in enumerate(SHIP_TYPE_CATEGORIES):
        classes_meta.append(
            {"id": i + 1, "name": name, "description": TYPE_DESCRIPTIONS[name]}
        )

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",  # detection-by-type encoded as per-pixel classes
            "source": "olmoearth",
            "license": "internal",
            "provenance": {
                "url": SOURCE,
                "have_locally": True,
                "annotation_method": "AIS-matched vessel attributes",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes_meta,
            "nodata_value": io.CLASS_NODATA,
            "detection_encoding": {
                "tile_size": DET_TILE,
                "positive_size": DET_POS_SIZE,
                "buffer_size": DET_BUFFER,
            },
            "num_samples": len(selected),
            "class_tile_counts": {
                ID_TO_NAME[cid]: n
                for cid, n in sorted(Counter(r["type_id"] for r in selected).items())
            },
            "available_per_type": {
                name: avail.get(name, 0) for name in SHIP_TYPE_CATEGORIES
            },
            "notes": (
                "Local OlmoEarth vessel-ATTRIBUTE rslearn dataset; target = vessel TYPE. "
                "Each source window is a 128x128 per-vessel S2 crop (local UTM @ 10 m) "
                "centered on ONE AIS-matched vessel; layer 'info' has one Point with "
                "properties.type in the 9 eval categories (unknown/other omitted -> "
                "skipped). Detection-by-type encoding: 32x32 UTM 10 m tile centered on the "
                "vessel, 1 px positive whose class id is the vessel TYPE (1..9) + 10 px "
                "nodata (255) buffer ring, rest background (id 0). Written in the window's "
                "own UTM projection (no reprojection). NO separate negative tiles: every "
                "window is vessel-centered and neighboring vessels are unlabeled, so corner "
                "'background' tiles would be unreliable; background (0) is still abundant "
                "within each positive tile. Class-balanced up to 1000 tiles/type. All splits "
                "used. Time range = each window's own ~2-hour S2 acquisition window "
                "(specific-image, spec section 5). Length/width/course/speed attributes "
                "excluded (regression, out of scope per manifest)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print(f"done: {len(selected)} samples", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
