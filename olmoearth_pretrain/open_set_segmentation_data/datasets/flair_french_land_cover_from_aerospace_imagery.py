"""Process FLAIR (French Land cover from Aerospace ImageRy) into open-set-segmentation patches.

Source: IGN France, distributed on Hugging Face as ``IGNF/FLAIR-1-2`` (Etalab Open
Licence 2.0). The release ships per-département zip archives under
``data/{train-val,flair#1-test,flair#2-test}/D0XX_YYYY.zip``. Each archive contains
``labels/Z*/MSK_*.tif`` land-cover masks (512x512, 0.2 m, RGF93/Lambert-93 = EPSG:2154,
uint8, 19 classes) alongside the aerial imagery (not needed here) and Sentinel-2 series.
The département folder name carries the acquisition YEAR (``D041_2021`` -> 2021), which
sets each patch's 1-year time range (all patches are 2018-2021, i.e. Sentinel era).

VHR handling (task spec §4): each 0.2 m mask patch (102.4 m footprint) is reprojected to
a local UTM grid at 10 m with **mode** resampling (categorical majority; never bilinear),
yielding one ~11x11 tile per patch. Class ids are remapped to the FLAIR 13-class baseline
nomenclature: the 12 main classes (source 1..12 -> output 0..11) plus a merged **other**
(output 12) folding source 13..19 (swimming pool, snow, clear cut, mixed, ligneous,
greenhouse, other) -- these fine/rare classes are largely unresolvable as distinct
categories at 10 m, so IGN's own baseline groups them, and we follow suit. Source value 0
(should not occur) -> nodata 255.

Masks are read directly out of the remote zips over HTTP (only the small MSK members are
pulled, not the multi-GB imagery) via ``huggingface_hub.HfFileSystem`` + ``zipfile``; the
full 121 GB of archives is never downloaded. Scanned tile records are cached to
``raw/{slug}/scan_cache.pkl`` so re-runs skip the remote reading.

Sampling: one tile per patch; tiles-per-class balanced to <=1000 tiles per class,
rarest-class-first, capped at 25,000 total (task spec §5). All three source splits
(train/val + both official test sets) are used.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.flair_french_land_cover_from_aerospace_imagery
"""

import argparse
import itertools
import math
import multiprocessing
import pickle
import random
import zipfile
from collections import defaultdict
from io import BytesIO
from typing import Any

import numpy as np
import rasterio
import tqdm
from affine import Affine
from huggingface_hub import HfFileSystem, list_repo_files
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject
from rslearn.utils.geometry import Projection
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest

SLUG = "flair_french_land_cover_from_aerospace_imagery"
NAME = "FLAIR (French Land cover from Aerospace ImageRy)"
REPO_ID = "IGNF/FLAIR-1-2"
SRC_EPSG = 2154  # RGF93 v1 / Lambert-93 (masks store it as an unlabeled LOCAL_CS)
TARGET_RES = 10.0
PER_CLASS = 1000

# Output class list (output id -> name, description). Source ids 1..12 map to 0..11;
# source 13..19 fold into "other" (id 12), matching IGN's 13-class baseline nomenclature.
CLASSES = [
    ("building", "Buildings (any roofed built structure), FLAIR source class 1."),
    (
        "pervious surface",
        "Pervious man-made surface (gravel, natural-material paths/soil-based surfaces), FLAIR source class 2.",
    ),
    (
        "impervious surface",
        "Impervious man-made surface (asphalt, concrete roads/parking), FLAIR source class 3.",
    ),
    ("bare soil", "Bare soil without vegetation, FLAIR source class 4."),
    ("water", "Water bodies (rivers, lakes, ponds, sea), FLAIR source class 5."),
    ("coniferous", "Coniferous trees/forest, FLAIR source class 6."),
    ("deciduous", "Deciduous trees/forest, FLAIR source class 7."),
    (
        "brushwood",
        "Brushwood / shrubland / low woody vegetation, FLAIR source class 8.",
    ),
    ("vineyard", "Vineyards, FLAIR source class 9."),
    (
        "herbaceous vegetation",
        "Herbaceous vegetation (grass, lawns, natural grassland), FLAIR source class 10.",
    ),
    ("agricultural land", "Agricultural land / crops, FLAIR source class 11."),
    ("plowed land", "Plowed / bare cultivated land, FLAIR source class 12."),
    (
        "other",
        "Merged rare/fine FLAIR classes (13 swimming pool, 14 snow, 15 clear cut, 16 mixed, "
        "17 ligneous, 18 greenhouse, 19 other) grouped per IGN's 13-class baseline; these are "
        "largely unresolvable as distinct categories at 10 m.",
    ),
]
NUM_CLASSES = len(CLASSES)  # 13

# Source value -> output id lookup (index by source uint8 value).
_LUT = np.full(256, io.CLASS_NODATA, dtype=np.uint8)
for _s in range(1, 13):
    _LUT[_s] = _s - 1  # source 1..12 -> 0..11
for _s in range(13, 20):
    _LUT[_s] = 12  # source 13..19 -> "other"


def _list_zip_tasks() -> list[dict[str, Any]]:
    """Return one task dict per département zip: {url, domain, year, split}."""
    files = list_repo_files(REPO_ID, repo_type="dataset")
    tasks = []
    for f in files:
        if not f.endswith(".zip"):
            continue
        # e.g. data/train-val/D041_2021.zip  or  data/flair#1-test/D012_2019.zip
        parts = f.split("/")
        split = parts[-2]
        base = parts[-1][: -len(".zip")]  # D041_2021
        domain, _, year = base.partition("_")
        tasks.append(
            {
                "url": f"datasets/{REPO_ID}/{f}",
                "domain": domain,
                "year": int(year),
                "split": split,
            }
        )
    return tasks


def _reproject_mask(arr: np.ndarray, src_t: Affine, W: int, H: int) -> tuple:
    """Reproject a 0.2 m Lambert-93 mask to local UTM 10 m (mode). Returns
    (out_uint8, utm_crs_str, (col0,row0,col1,row1)) or None if degenerate.
    """
    src_crs = CRS.from_epsg(SRC_EPSG)
    cx = src_t.c + src_t.a * W / 2.0
    cy = src_t.f + src_t.e * H / 2.0
    lon, lat = Transformer.from_crs(SRC_EPSG, 4326, always_xy=True).transform(cx, cy)
    utm_crs = get_utm_ups_projection(lon, lat, TARGET_RES, -TARGET_RES).crs
    to_utm = Transformer.from_crs(SRC_EPSG, utm_crs.to_epsg(), always_xy=True)
    xs = [src_t.c, src_t.c + src_t.a * W]
    ys = [src_t.f, src_t.f + src_t.e * H]
    pts = [to_utm.transform(X, Y) for X, Y in itertools.product(xs, ys)]
    cols = [p[0] / TARGET_RES for p in pts]
    rows = [p[1] / -TARGET_RES for p in pts]
    col0, col1 = math.floor(min(cols)), math.ceil(max(cols))
    row0, row1 = math.floor(min(rows)), math.ceil(max(rows))
    dw, dh = col1 - col0, row1 - row0
    if dw <= 0 or dh <= 0 or dw > io.MAX_TILE or dh > io.MAX_TILE:
        return None
    dst_t = Affine(TARGET_RES, 0, col0 * TARGET_RES, 0, -TARGET_RES, row0 * -TARGET_RES)
    dst = np.zeros((dh, dw), dtype=np.uint8)
    reproject(
        arr,
        dst,
        src_transform=src_t,
        src_crs=src_crs,
        dst_transform=dst_t,
        dst_crs=utm_crs,
        resampling=Resampling.mode,
    )
    out = _LUT[dst]
    return out, utm_crs.to_string(), (col0, row0, col1, row1)


def _scan_zip(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Read every MSK_*.tif in one remote zip, reproject to a 10 m tile, return records."""
    fs = HfFileSystem()
    try:
        z = zipfile.ZipFile(fs.open(task["url"], "rb"))
    except Exception as e:  # noqa: BLE001
        print(f"WARN open failed {task['url']}: {e}")
        return []
    members = [n for n in z.namelist() if n.rpartition("/")[2].startswith("MSK")]
    recs = []
    for m in members:
        try:
            data = z.read(m)
            with rasterio.open(BytesIO(data)) as ds:
                arr = ds.read(1)
                src_t = ds.transform
                W, H = ds.width, ds.height
        except Exception as e:  # noqa: BLE001
            print(f"WARN read failed {task['url']}::{m}: {e}")
            continue
        res = _reproject_mask(arr, src_t, W, H)
        if res is None:
            continue
        out, crs_str, bounds = res
        present = sorted(int(v) for v in np.unique(out) if v != io.CLASS_NODATA)
        if not present:
            continue
        patch_id = m.rpartition("/")[2][: -len(".tif")]  # MSK_027013
        recs.append(
            {
                "array": out,
                "crs": crs_str,
                "bounds": bounds,
                "year": task["year"],
                "classes_present": present,
                "source_id": f"{task['split']}/{task['domain']}_{task['year']}/{patch_id}",
            }
        )
    return recs


def _scan_all(workers: int) -> list[dict[str, Any]]:
    cache = io.raw_dir(SLUG) / "scan_cache.pkl"
    if cache.exists():
        print(f"loading cached scan from {cache}")
        with cache.open("rb") as f:
            return pickle.load(f)
    tasks = _list_zip_tasks()
    print(f"scanning {len(tasks)} département zips (mask-only, remote)")
    all_recs: list[dict[str, Any]] = []
    with multiprocessing.Pool(workers) as p:
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan_zip, [dict(task=t) for t in tasks]),
            total=len(tasks),
        ):
            all_recs.extend(recs)
    print(f"scanned {len(all_recs)} non-empty patches")
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    tmp = io.raw_dir(SLUG) / "scan_cache.pkl.tmp"
    with tmp.open("wb") as f:
        pickle.dump(all_recs, f)
    tmp.rename(cache)
    return all_recs


def _select(records: list[dict[str, Any]], seed: int = 42) -> list[dict[str, Any]]:
    """Tiles-per-class balanced selection: <=PER_CLASS tiles per class, rarest-first,
    total capped at MAX_SAMPLES_PER_DATASET. A tile counts toward every class it contains.
    """
    from olmoearth_pretrain.open_set_segmentation_data.sampling import (
        MAX_SAMPLES_PER_DATASET,
    )

    freq: dict[int, int] = defaultdict(int)
    for r in records:
        for c in r["classes_present"]:
            freq[c] += 1
    rng = random.Random(seed)
    order = list(records)
    rng.shuffle(order)
    # Process patches whose rarest present class is globally rarest first.
    order.sort(key=lambda r: min(freq[c] for c in r["classes_present"]))
    counts: dict[int, int] = defaultdict(int)
    selected = []
    for r in order:
        if len(selected) >= MAX_SAMPLES_PER_DATASET:
            break
        if any(counts[c] < PER_CLASS for c in r["classes_present"]):
            selected.append(r)
            for c in r["classes_present"]:
                counts[c] += 1
    return selected


def _write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    proj = Projection(CRS.from_string(rec["crs"]), TARGET_RES, -TARGET_RES)
    bounds = tuple(rec["bounds"])
    io.write_label_geotiff(
        SLUG, sample_id, rec["array"], proj, bounds, nodata=io.CLASS_NODATA
    )
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        source_id=rec["source_id"],
        classes_present=rec["classes_present"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--write-workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            f"Source: Hugging Face dataset {REPO_ID} (IGN France, Etalab Open Licence 2.0).\n"
            "Not downloaded in full (121 GB of zips). Only the small MSK_*.tif land-cover\n"
            "masks are streamed out of each per-département zip via HfFileSystem+zipfile\n"
            "(HTTP range reads). See scan_cache.pkl for the scanned tile records.\n"
            "Masks: 512x512, 0.2 m, EPSG:2154 (stored as unlabeled LOCAL_CS Lambert-93),\n"
            "uint8, source classes 1..19.\n"
        )

    records = _scan_all(args.workers)
    selected = _select(records)
    selected.sort(key=lambda r: r["source_id"])
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} tiles (of {len(records)} patches)")

    with multiprocessing.Pool(args.write_workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            pass

    tile_counts = {i: 0 for i in range(NUM_CLASSES)}
    for r in selected:
        for c in r["classes_present"]:
            tile_counts[c] += 1

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "IGN France / Hugging Face (IGNF/FLAIR-1-2)",
            "license": "Etalab Open Licence 2.0",
            "provenance": {
                "url": "https://huggingface.co/datasets/IGNF/FLAIR-1-2",
                "have_locally": False,
                "annotation_method": "manual photointerpretation of 0.2 m aerial imagery (IGN)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_tile_counts": {
                CLASSES[i][0]: tile_counts[i] for i in range(NUM_CLASSES)
            },
            "notes": (
                "VHR 0.2 m FLAIR land-cover masks reprojected from EPSG:2154 to local UTM "
                "at 10 m with MODE resampling and tiled to one ~11x11 patch each. Source "
                "classes 1..12 -> output 0..11; rare/fine source classes 13..19 (swimming "
                "pool, snow, clear cut, mixed, ligneous, greenhouse, other) merged into "
                "'other' (12) per IGN's 13-class baseline (unresolvable as distinct classes "
                "at 10 m). Time range = 1-year window for the patch's acquisition year "
                "(département folder suffix; 2018-2021). All three source splits used. "
                "Tiles-per-class balanced to <=1000/class, rarest-first, <=25k total."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("class tile counts:")
    for i in range(NUM_CLASSES):
        print(f"  {i:>2} {CLASSES[i][0]:24} {tile_counts[i]}")
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
