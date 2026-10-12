"""Process OlmoEarth vessel attributes (type) into presence-only POINTS by vessel TYPE.

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

Task type / encoding (presence-only POINTS): each source crop is centered on ONE
AIS-matched vessel; neighboring vessels in the crop are UNLABELED, so the crop's background
is NOT a genuine negative -- it is effectively one labeled typed point per crop. We emit
each labeled vessel as one presence POINT (at the labeled vessel's own lon/lat, converted
from its pixel coordinate in the window's UTM projection) carrying its multi-class vessel
TYPE into a dataset-wide ``points.geojson``. Cross-dataset negatives are supplied by
assembly. The earlier per-detection GeoTIFF tile encoding (1 px positive + nodata buffer +
background fill) is dropped.

Classes (renumbered 0..8, eval order): 0 cargo, 1 tanker, 2 passenger, 3 service, 4 tug,
5 pleasure, 6 fishing, 7 enforcement, 8 sar.

Sampling: 9 type classes, up to PER_CLASS=1000 presence points per type, class-balanced
(``balance_by_class`` by type id). Total well under the 25k cap. All splits used.

Time range: each sample uses its window's own ~2-hour S2 acquisition ``time_range``
(specific-image label, spec section 5).

Run (idempotent; reuses cached raw):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_vessel_attributes_type
"""

import argparse
import json
import multiprocessing
import os
from collections import Counter
from typing import Any

import shapely
import tqdm
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "olmoearth_vessel_attributes_type"
NAME = "OlmoEarth vessel attributes (type)"
SOURCE = (
    "/weka/dfive-default/rslearn-eai/datasets/sentinel2_vessel_attribute/"
    "dataset_v1/20250205"
)
WINDOWS_ROOT = os.path.join(SOURCE, "windows")

# Eval vessel-type order (rslp.vessel_attribute.train.SHIP_TYPE_CATEGORIES); ids 0..8.
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
TYPE_TO_ID = {name: i for i, name in enumerate(SHIP_TYPE_CATEGORIES)}
ID_TO_NAME = {i: n for i, n in enumerate(SHIP_TYPE_CATEGORIES)}

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

PER_CLASS = 1000  # presence points per vessel type (9 types -> up to 9k, << 25k cap)


def _list_windows() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for g in sorted(os.listdir(WINDOWS_ROOT)):
        gd = os.path.join(WINDOWS_ROOT, g)
        if not os.path.isdir(gd):
            continue
        for name in os.listdir(gd):
            out.append((g, name))
    return out


def _projection(crs: str) -> Projection:
    return Projection(CRS.from_string(crs), io.RESOLUTION, -io.RESOLUTION)


def _scan_window(group: str, name: str) -> dict[str, Any] | None:
    """Return a presence-point record (lon/lat/time/type) for the labeled vessel, or None."""
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
    # The vessel Point is in the window's UTM pixel coordinates; convert to WGS84 lon/lat.
    stg = STGeometry(_projection(proj["crs"]), shapely.Point(float(x), float(y)), None)
    ll = stg.to_projection(WGS84_PROJECTION)
    return {
        "lon": float(ll.shp.x),
        "lat": float(ll.shp.y),
        "tr": tr,
        "src": f"{group}/{name}",
        "label": TYPE_TO_ID[vtype],
        "vtype": vtype,
    }


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
            "manifest). Emitted as presence-only points (one typed point per crop).\n"
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

    # Class-balanced selection: up to PER_CLASS presence points per vessel type.
    selected = balance_by_class(records, "label", per_class=PER_CLASS)
    sel_counts = Counter(r["vtype"] for r in selected)
    print(
        f"selected {len(selected)} presence points ({dict(sorted(sel_counts.items()))})",
        flush=True,
    )

    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": r["label"],
                "time_range": [r["tr"][0], r["tr"][1]],  # window's own ~2-hour S2 window
                "change_time": None,
                "source_id": r["src"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    io.check_disk()

    classes_meta = [
        {"id": i, "name": name, "description": TYPE_DESCRIPTIONS[name]}
        for i, name in enumerate(SHIP_TYPE_CATEGORIES)
    ]

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "olmoearth",
            "license": "internal",
            "provenance": {
                "url": SOURCE,
                "have_locally": True,
                "annotation_method": "AIS-matched vessel attributes",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes_meta,
            "num_samples": len(selected),
            "class_counts": {
                ID_TO_NAME[cid]: n
                for cid, n in sorted(Counter(r["label"] for r in selected).items())
            },
            "available_per_type": {
                name: avail.get(name, 0) for name in SHIP_TYPE_CATEGORIES
            },
            "notes": (
                "Local OlmoEarth vessel-ATTRIBUTE rslearn dataset; target = vessel TYPE. "
                "Emitted as presence-only POINTS (converted from the earlier per-detection "
                "GeoTIFF tile encoding; negatives now come from assembly). Each source window "
                "is a 128x128 per-vessel S2 crop (local UTM @ 10 m) centered on ONE "
                "AIS-matched vessel; layer 'info' has one Point with properties.type in the 9 "
                "eval categories (unknown/other omitted -> skipped). Because neighboring "
                "vessels in a crop are unlabeled, the crop background is not a genuine "
                "negative, so each crop yields exactly one typed presence point at the labeled "
                "vessel's lon/lat (converted from its pixel coordinate in the window's UTM "
                "projection). Vessel-type classes only, ids 0..8. Class-balanced up to 1000 "
                "points/type. All splits used. Time range = each window's own ~2-hour S2 "
                "acquisition window (specific-image, spec section 5). Length/width/course/"
                "speed attributes excluded (regression, out of scope per manifest)."
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
