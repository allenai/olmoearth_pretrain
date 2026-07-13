"""Process OlmoEarth landslide (Sen12Landslides) into open-set-segmentation label patches.

Source: local rslearn dataset (have_locally=true), NOT copied. Path:
`/weka/dfive-default/piperw/rslearn_projects/data/landslide/sen12landslides/all_positives`.
Upstream is Sen12Landslides: binary landslide-scar segmentation from Sentinel-1 / Sentinel-2
/ SRTM pre/post-event acquisitions, manual annotation, global.

Layout (rslearn on-disk): windows/<group>/<name>/{metadata.json, layers/label_raster/label/
geotiff.tif}. We use ONLY the `sen12_landslides` group (the Sen12Landslides subset). The
other groups in this project (glc, icimod, fwn_mtli, osm_ski, osm_ski_resorts_trial) are
separate landslide inventories and are NOT part of this manifest entry.

Each location has a paired `positive` window (time range spanning the event; contains the
landslide scar) and a `negative` window (same location, one year earlier; all no_landslide).
We use only the **positive** windows: they carry the actual landslide scars and already
contain abundant no_landslide background, so with 2 classes and tiles-per-class balancing
they saturate both classes. We do NOT add the negative windows (they would be near-duplicate
all-background tiles at the same locations; the assembly step supplies negatives from other
datasets, spec 5).

Each source window is already 64x64 at 10 m in a local UTM CRS, so no reprojection/resampling
is needed -- we read the label raster, remap, and re-emit.

Class scheme (dense per-pixel CLASSIFICATION, matching the manifest's 2 classes):
    id 0 = no_landslide  (source label 0, observed)
    id 1 = landslide     (source label 1; manually annotated landslide scar)
    255  = nodata/ignore (source label 2 = no_data: a 30 m buffer ring around each
                          landslide polygon, left ambiguous on purpose)

Time range: landslide is a change/event label. `change_time` = the event date
(options.event_date); `time_range` = a 1-year window centered on it (spec 5). All Sen12
events are 2016-2023 (Sentinel era); any pre-2016 window is defensively filtered.

Sampling: tiles-per-class balanced (spec 5), <= 1000 tiles/class. Every positive tile
contains both classes, so this yields ~1000 tiles.

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_landslide_sen12landslides
"""

import argparse
import multiprocessing
import os
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import rasterio
import tqdm
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest, sampling

SLUG = "olmoearth_landslide_sen12landslides"
NAME = "OlmoEarth landslide (Sen12Landslides)"

SRC_ROOT = (
    "/weka/dfive-default/piperw/rslearn_projects/data/landslide/sen12landslides/"
    "all_positives"
)
GROUP = "sen12_landslides"
WINDOWS_DIR = os.path.join(SRC_ROOT, "windows", GROUP)

PER_CLASS = 1000
MIN_YEAR = 2016  # Sentinel era; reject/filter labels entirely before this.

NO_LANDSLIDE, LANDSLIDE = 0, 1
SRC_NODATA = 2  # source label value for the no_data buffer ring
CLASSES = [
    (
        "no_landslide",
        "No landslide at this pixel for the mapped event, among observed pixels "
        "(Sen12Landslides source label 0).",
    ),
    (
        "landslide",
        "Manually annotated landslide scar for the pre/post event (Sen12Landslides source "
        "label 1), from Sentinel-1/Sentinel-2/SRTM pre- and post-event imagery.",
    ),
]


def _list_positive_windows() -> list[str]:
    """Names of the positive windows in the sen12_landslides group."""
    return sorted(n for n in os.listdir(WINDOWS_DIR) if "_positive_" in n)


def _read_metadata(name: str) -> dict[str, Any] | None:
    import json

    mp = os.path.join(WINDOWS_DIR, name, "metadata.json")
    if not os.path.exists(mp):
        return None
    with open(mp) as f:
        return json.load(f)


def _event_datetime(options: dict[str, Any]) -> datetime | None:
    """Parse the event date (change_time). Fall back to mid-year of event_year."""
    ed = options.get("event_date")
    if ed and str(ed).lower() != "nan":
        try:
            d = datetime.fromisoformat(str(ed))
            return d if d.tzinfo else d.replace(tzinfo=UTC)
        except ValueError:
            pass
    yr = options.get("event_year")
    if yr:
        return datetime(int(yr), 7, 1, tzinfo=UTC)
    return None


def _remap_label(arr: np.ndarray) -> np.ndarray:
    """Source uint8 (0/1/2) -> label uint8 (0 no_landslide, 1 landslide, 255 nodata)."""
    out = arr.astype(np.uint8).copy()
    out[arr == SRC_NODATA] = io.CLASS_NODATA
    # Any unexpected value -> nodata (defensive).
    out[(arr != NO_LANDSLIDE) & (arr != LANDSLIDE) & (arr != SRC_NODATA)] = (
        io.CLASS_NODATA
    )
    return out


def _scan_window(name: str) -> dict[str, Any] | None:
    """Return a candidate record for one positive window, or None to skip."""
    md = _read_metadata(name)
    if md is None:
        return None
    options = md.get("options", {})
    change_time = _event_datetime(options)
    if change_time is None or change_time.year < MIN_YEAR:
        return None
    tif = os.path.join(WINDOWS_DIR, name, "layers/label_raster/label/geotiff.tif")
    if not os.path.exists(tif):
        return None
    with rasterio.open(tif) as ds:
        arr = ds.read(1)
    out = _remap_label(arr)
    present = sorted(
        int(c) for c in (NO_LANDSLIDE, LANDSLIDE) if int((out == c).sum()) > 0
    )
    if not present:
        return None  # all-nodata tile carries no signal
    proj = md["projection"]
    return {
        "name": name,
        "crs": proj["crs"],
        "bounds": [int(v) for v in md["bounds"]],
        "change_time": change_time.isoformat(),
        "classes_present": present,
    }


def _write_window(rec: dict[str, Any]) -> None:
    """Read the source raster, remap, and write one label tile + sidecar JSON."""
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    tif = os.path.join(
        WINDOWS_DIR, rec["name"], "layers/label_raster/label/geotiff.tif"
    )
    with rasterio.open(tif) as ds:
        arr = ds.read(1)
    out = _remap_label(arr)
    proj = Projection(CRS.from_string(rec["crs"]), io.RESOLUTION, -io.RESOLUTION)
    bounds = tuple(rec["bounds"])
    change_time = datetime.fromisoformat(rec["change_time"])
    tr = (change_time - timedelta(days=182), change_time + timedelta(days=183))
    io.write_label_geotiff(SLUG, sample_id, out, proj, bounds, nodata=io.CLASS_NODATA)
    present = sorted(int(v) for v in np.unique(out) if v != io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        tr,
        change_time=change_time,
        source_id=f"{GROUP}/{rec['name']}",
        classes_present=present,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    names = _list_positive_windows()
    print(f"{len(names)} positive windows in group {GROUP}")

    print("Scanning windows (metadata + label raster)...")
    with multiprocessing.Pool(args.workers) as p:
        recs: list[dict[str, Any]] = []
        for rec in tqdm.tqdm(
            star_imap_unordered(p, _scan_window, [dict(name=n) for n in names]),
            total=len(names),
        ):
            if rec is not None:
                recs.append(rec)
    print(f"  {len(recs)} candidate tiles (>= {MIN_YEAR}, non-empty)")

    selected = sampling.select_tiles_per_class(
        recs, classes_key="classes_present", per_class=PER_CLASS
    )
    selected.sort(key=lambda r: r["name"])  # stable, idempotent ids
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(
        f"  selected {len(selected)} tiles (tiles-per-class balanced, <= {PER_CLASS}/class)"
    )

    io.check_disk()
    print(f"Writing {len(selected)} label tiles...")
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_window, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            pass

    tile_class_counts = {name: 0 for name, _ in CLASSES}
    for r in selected:
        for c in r["classes_present"]:
            tile_class_counts[CLASSES[c][0]] += 1
    print("tiles containing each class:", tile_class_counts)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "olmoearth",
            "license": "internal",
            "provenance": {
                "url": SRC_ROOT,
                "have_locally": True,
                "annotation_method": "manual annotation (Sen12Landslides)",
                "group": GROUP,
                "note": (
                    "Local rslearn dataset; positive windows of the sen12_landslides group. "
                    "Label read from layers/label_raster/label/geotiff.tif."
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "tile_class_counts": tile_class_counts,
            "notes": (
                "Sen12Landslides binary landslide-scar masks. Source windows are already "
                "64x64 @ 10 m in local UTM, read directly and remapped (0 no_landslide, "
                "1 landslide, source 2=no_data buffer -> 255 nodata/ignore). Only the "
                "positive windows of the sen12_landslides group are used (74847 available); "
                "negative windows (same locations, prior year, all background) are omitted. "
                "Landslide is an event label: change_time = options.event_date, time_range = "
                "1-year window centered on it. Tiles-per-class balanced, <=1000/class; every "
                "positive tile contains both classes so the two classes saturate together."
            ),
        },
    )

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
