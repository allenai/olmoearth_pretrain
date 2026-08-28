"""Process OlmoEarth WorldCover land cover into open-set-segmentation label patches.

Source: local rslearn eval at rslearn-eai/datasets/worldcover. Each window is a ~53x53 UTM
tile (10 m/pixel, local UTM, north-up) whose ``label_raster`` layer carries a single-band
land-cover label. Only a central ~10x10 pixel block (a 100 m reference plot) is labeled;
the surrounding pixels are the source's ``no_data`` class (pixel value 0). The 12 real
classes follow the reference legend used to validate ESA WorldCover
(bare / burnt / crops / fallow-shifting-cultivation / grassland / lichen-and-moss / shrub /
snow-and-ice / tree / urban / water / wetland). This is a dense multi-class raster
(``label_type: dense_raster``): the 10x10 plot is frequently multi-class, so we write one
single-band GeoTIFF per plot (spec §2), tiles-per-class balanced to <=1000 tiles/class.

Class remap: the source pixel value ``v`` (0..12) maps to output id ``v-1`` for v>=1
(bare=0 .. wetland=11); the source ``no_data`` class (0) becomes 255 (nodata/ignore).

Provenance note: the manifest labels this "ESA WorldCover" (a derived product), but the
label legend here (with burnt / fallow-shifting-cultivation) is the crowd-sourced Geo-Wiki
reference legend used for WorldCover validation, i.e. reference plots rather than the raw
map. Either way the ``label_raster`` layer is treated as ground truth. Each source window's
own ~1-year time range (2016) is used verbatim; land cover is near-static so the small
label-vs-image year offset is immaterial (noted in the summary).
"""

import argparse
import json
import multiprocessing
import os
from collections import Counter
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    balance_tiles_by_class,
)

SLUG = "olmoearth_worldcover"
SOURCE = "/weka/dfive-default/rslearn-eai/datasets/worldcover"
PER_CLASS = 1000

# Source label_raster class_names (index = source pixel value); index 0 = no_data.
SOURCE_CLASS_NAMES = [
    "no_data",
    "bare",
    "burnt",
    "crops",
    "fallow/shifting cultivation",
    "grassland",
    "Lichen and moss",
    "shrub",
    "snow and ice",
    "tree",
    "urban/built-up",
    "water",
    "wetland (herbaceous)",
]

# Output classes (id 0..11) = source value - 1. Descriptions from the reference legend.
CLASSES = [
    (
        "bare",
        "Bare / sparsely vegetated ground: exposed soil, sand, rock with little or no vegetation.",
    ),
    ("burnt", "Recently burnt land: fire-scarred surfaces showing charred vegetation."),
    (
        "crops",
        "Cultivated cropland: actively managed annual/perennial agricultural fields.",
    ),
    (
        "fallow/shifting cultivation",
        "Fallow fields or shifting (swidden) cultivation land temporarily out of active crop production.",
    ),
    (
        "grassland",
        "Herbaceous grass-dominated vegetation (natural or managed grassland).",
    ),
    (
        "lichen and moss",
        "Lichen- and moss-dominated cover, typical of tundra / high-latitude or high-altitude ground.",
    ),
    (
        "shrub",
        "Shrub-dominated vegetation (woody plants generally shorter than trees).",
    ),
    (
        "snow and ice",
        "Permanent or seasonal snow and ice cover (glaciers, snowfields).",
    ),
    ("tree", "Tree-dominated cover: forest and woodland."),
    (
        "urban/built-up",
        "Human-made impervious surfaces: buildings, roads, and other built infrastructure.",
    ),
    ("water", "Open water bodies: lakes, rivers, reservoirs, ocean."),
    (
        "wetland (herbaceous)",
        "Herbaceous wetland: seasonally or permanently flooded ground with herbaceous vegetation.",
    ),
]
N_CLASSES = len(CLASSES)  # 12


def _remap(arr: np.ndarray) -> np.ndarray:
    """Remap source pixel values (0=no_data, 1..12=classes) to output ids.

    0 -> 255 (nodata); v in 1..12 -> v-1. Any unexpected value -> 255.
    """
    out = np.full(arr.shape, io.CLASS_NODATA, dtype=np.uint8)
    valid = (arr >= 1) & (arr <= N_CLASSES)
    out[valid] = (arr[valid] - 1).astype(np.uint8)
    return out


def _scan_one(path: str) -> dict[str, Any] | None:
    """Read one window: crop the labeled block, remap, return a record (array included)."""
    try:
        with open(os.path.join(path, "metadata.json")) as f:
            md = json.load(f)
    except FileNotFoundError:
        return None
    tif = os.path.join(path, "layers", "label_raster", "label", "geotiff.tif")
    if not os.path.exists(tif):
        return None
    with rasterio.open(tif) as ds:
        a = ds.read(1)
    remapped = _remap(a)
    ys, xs = np.where(remapped != io.CLASS_NODATA)
    if len(ys) == 0:
        return None  # fully no_data plot; nothing to label
    r0, r1 = int(ys.min()), int(ys.max())
    c0, c1 = int(xs.min()), int(xs.max())
    # Cap crop to MAX_TILE just in case (labeled block is ~10x10 in practice).
    if (r1 - r0 + 1) > io.MAX_TILE:
        r1 = r0 + io.MAX_TILE - 1
    if (c1 - c0 + 1) > io.MAX_TILE:
        c1 = c0 + io.MAX_TILE - 1
    patch = remapped[r0 : r1 + 1, c0 : c1 + 1]
    win = md["bounds"]  # [x_min, y_min, x_max, y_max] in pixel coords; row 0 = y_min
    bounds = (win[0] + c0, win[1] + r0, win[0] + c1 + 1, win[1] + r1 + 1)
    classes_present = sorted(int(v) for v in np.unique(patch) if v != io.CLASS_NODATA)
    return {
        "patch": patch,
        "crs": md["projection"]["crs"],
        "bounds": bounds,
        "time_range": md.get("time_range"),
        "classes_present": classes_present,
        "source_id": f"{md.get('_group', os.path.basename(os.path.dirname(path)))}/{os.path.basename(path)}",
    }


def scan_records(workers: int) -> list[dict[str, Any]]:
    jobs = []
    windows_root = os.path.join(SOURCE, "windows")
    for group in sorted(os.listdir(windows_root)):
        gd = os.path.join(windows_root, group)
        if os.path.isdir(gd):
            for name in os.listdir(gd):
                jobs.append(os.path.join(gd, name))
    with multiprocessing.Pool(workers) as p:
        recs = [r for r in p.imap_unordered(_scan_one, jobs, chunksize=128) if r]
    return recs


def _write_one(args: tuple[str, dict[str, Any]]) -> None:
    sample_id, r = args
    d = io.locations_dir(SLUG)
    if (d / f"{sample_id}.tif").exists() and (d / f"{sample_id}.json").exists():
        return
    proj = Projection(CRS.from_string(r["crs"]), io.RESOLUTION, -io.RESOLUTION)
    tr = r["time_range"]
    from datetime import datetime

    time_range = (
        (datetime.fromisoformat(tr[0]), datetime.fromisoformat(tr[1])) if tr else None
    )
    io.write_label_geotiff(
        SLUG, sample_id, r["patch"], proj, r["bounds"], nodata=io.CLASS_NODATA
    )
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        r["bounds"],
        time_range,
        source_id=r["source_id"],
        classes_present=r["classes_present"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(f"local rslearn dataset: {SOURCE}\n")

    recs = scan_records(args.workers)
    print(f"scanned {len(recs)} labeled plots")
    avail = Counter()
    for r in recs:
        for c in r["classes_present"]:
            avail[c] += 1
    print("tiles available per class id:", dict(sorted(avail.items())))

    selected = balance_tiles_by_class(
        recs, classes_key="classes_present", per_class=PER_CLASS
    )
    print(
        f"selected {len(selected)} tiles (tiles-per-class balanced, <= {PER_CLASS}/class)"
    )

    io.locations_dir(SLUG).mkdir(parents=True, exist_ok=True)
    jobs = [(f"{i:06d}", r) for i, r in enumerate(selected)]
    with multiprocessing.Pool(args.workers) as p:
        for _ in p.imap_unordered(_write_one, jobs, chunksize=64):
            pass

    sel_counts = Counter()
    for r in selected:
        for c in r["classes_present"]:
            sel_counts[c] += 1
    print("selected tiles per class id:", dict(sorted(sel_counts.items())))

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "OlmoEarth WorldCover",
            "task_type": "classification",
            "source": "olmoearth",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": SOURCE,
                "have_locally": True,
                "annotation_method": "derived-product / reference (ESA WorldCover legend)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_tile_counts": {
                CLASSES[c][0]: sel_counts.get(c, 0) for c in range(N_CLASSES)
            },
            "notes": (
                "Dense multi-class raster: one single-band uint8 GeoTIFF per ~10x10 labeled "
                "plot (the source's central labeled block; surrounding no_data pixels dropped "
                "as ignore). Source pixel value v (1..12) -> id v-1; source no_data (0) -> 255. "
                "Tiles-per-class balanced to <=1000 tiles/class; a tile counts toward every "
                "class it contains. All source splits (train/val/test) used. Time range is each "
                "source window's own ~1-year (2016) range."
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
