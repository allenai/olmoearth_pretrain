"""Process OlmoEarth solar farm into open-set-segmentation label patches.

Source: local rslearn dataset (have_locally=true, not copied)
``/weka/dfive-default/rslearn-eai/datasets/solar_farm/dataset_v1/20250605``. This is the
existing OlmoEarth eval / Satlas solar-farm segmentation dataset: 3561 manually annotated
windows (group ``default``; splits train=3115, val=446) spread over 58 UTM zones, each a
variable-size (~180-490 px) crop already in a **local UTM projection at 10 m/pixel**.

Two relevant layers per window:
  * ``label_raster`` band ``label`` -- single-band uint8 PNG, 0 = background,
    1 = solar_farm (photovoltaic footprint, manually annotated).
  * ``mask`` band ``mask`` -- single-band uint8 PNG, 255 = valid/annotated region,
    0 = outside the annotated footprint (window borders). Covers ~96-100% of each window.

Encoding (dense_raster): because the source is already local UTM @ 10 m, NO reprojection
or resampling is needed -- we read each window's label + mask PNG directly (exact, fast)
and tile the window into <=64x64 patches on the native pixel grid. Output tiles are
single-band uint8: 0 = background, 1 = solar_farm, 255 = nodata (pixels where mask==0, i.e.
unannotated). (The shared ``rslearn_read.read_label_raster`` targets GeoTIFF band dirs;
these layers are ``single_image`` PNG, so we read the PNG directly -- same array rslearn's
own decoder would return, since it also loads the PNG with PIL.)

Sampling: tiles-per-class balanced (spec section 5) over the two classes. solar_farm is the
rare class, so every solar tile is prioritized (up to 1000); background-only tiles are then
added until the background class also reaches ~1000 tiles. A tile counts toward every class
present in it. Time range = the window's own ~180-day acquisition range (<=1 year); solar
farms are persistent, so this is a valid annual-style label.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_solar_farm
"""

import argparse
import json
import multiprocessing
import os
import random
from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np
import tqdm
from PIL import Image
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io

SLUG = "olmoearth_solar_farm"
NAME = "OlmoEarth solar farm"
SOURCE = "/weka/dfive-default/rslearn-eai/datasets/solar_farm/dataset_v1/20250605"
WINDOWS_ROOT = os.path.join(SOURCE, "windows", "default")

TILE = io.MAX_TILE  # 64 px @ 10 m = 640 m
BACKGROUND_ID = 0
SOLAR_ID = 1
CLASS_NAMES = {BACKGROUND_ID: "background", SOLAR_ID: "solar_farm"}

# Minimum valid (mask==255) pixels for a tile to be worth keeping (~16x16 of real data).
MIN_VALID_PX = 256
# A tile counts as a solar_farm tile only if it has at least this many solar pixels, so
# a handful of stray annotation-edge pixels do not create noise-level positives.
MIN_SOLAR_PX = 10
PER_CLASS = 1000  # tiles-per-class target (spec section 5)
SEED = 42


def _load_window_png(band_dir: str) -> np.ndarray:
    """Read a single_image PNG band (label or mask) as a 2-D uint8 array."""
    arr = np.array(Image.open(os.path.join(band_dir, "image.png")))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def _scan_window(name: str) -> list[dict[str, Any]]:
    """Enumerate <=64x64 tiles of one window; return lightweight tile records.

    Records carry classes_present and pixel bounds only (not the array) so scanning stays
    cheap; the write phase re-reads the window once and slices out the selected tiles.
    """
    wdir = os.path.join(WINDOWS_ROOT, name)
    try:
        md = json.load(open(os.path.join(wdir, "metadata.json")))
    except FileNotFoundError:
        return []
    x0, y0, x1, y1 = md["bounds"]
    crs = md["projection"]["crs"]
    tr = md["time_range"]
    label = _load_window_png(os.path.join(wdir, "layers", "label_raster", "label"))
    mask = _load_window_png(os.path.join(wdir, "layers", "mask", "mask"))
    h, w = label.shape
    # Guard against any bounds/array mismatch.
    if (y1 - y0, x1 - x0) != (h, w):
        h = min(h, y1 - y0)
        w = min(w, x1 - x0)

    records: list[dict[str, Any]] = []
    for r0 in range(0, h, TILE):
        th = min(TILE, h - r0)
        for c0 in range(0, w, TILE):
            tw = min(TILE, w - c0)
            lab = label[r0 : r0 + th, c0 : c0 + tw]
            m = mask[r0 : r0 + th, c0 : c0 + tw]
            valid = m == 255
            n_valid = int(valid.sum())
            if n_valid < MIN_VALID_PX:
                continue
            solar = valid & (lab >= 1)
            n_solar = int(solar.sum())
            n_bg = int((valid & (lab == 0)).sum())
            present: list[int] = []
            if n_bg >= 1:
                present.append(BACKGROUND_ID)
            if n_solar >= MIN_SOLAR_PX:
                present.append(SOLAR_ID)
            if not present:
                continue
            records.append(
                {
                    "window": name,
                    "crs": crs,
                    "r0": r0,
                    "c0": c0,
                    "bounds": [x0 + c0, y0 + r0, x0 + c0 + tw, y0 + r0 + th],
                    "time_range": tr,
                    "classes_present": present,
                    "has_solar": SOLAR_ID in present,
                }
            )
    return records


def _write_window_tiles(name: str, tiles: list[dict[str, Any]]) -> list[list[int]]:
    """Read one window once and write all its selected tiles. Idempotent."""
    wdir = os.path.join(WINDOWS_ROOT, name)
    label = _load_window_png(os.path.join(wdir, "layers", "label_raster", "label"))
    mask = _load_window_png(os.path.join(wdir, "layers", "mask", "mask"))
    proj = Projection(CRS.from_string(tiles[0]["crs"]), io.RESOLUTION, -io.RESOLUTION)
    out_present: list[list[int]] = []
    for t in tiles:
        sample_id = t["sample_id"]
        out_present.append(t["classes_present"])
        if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
            continue
        r0, c0 = t["r0"], t["c0"]
        x0, y0, x1, y1 = t["bounds"]
        th, tw = y1 - y0, x1 - x0
        lab = label[r0 : r0 + th, c0 : c0 + tw].astype(np.uint8)
        m = mask[r0 : r0 + th, c0 : c0 + tw]
        out = np.where(m == 255, lab, io.CLASS_NODATA).astype(np.uint8)
        tr = t["time_range"]
        time_range = (datetime.fromisoformat(tr[0]), datetime.fromisoformat(tr[1]))
        io.write_label_geotiff(
            SLUG, sample_id, out, proj, tuple(t["bounds"]), nodata=io.CLASS_NODATA
        )
        io.write_sample_json(
            SLUG,
            sample_id,
            proj,
            tuple(t["bounds"]),
            time_range,
            source_id=f"{name}:{r0}:{c0}",
            classes_present=t["classes_present"],
        )
    return out_present


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "local rslearn dataset (have_locally=true, not copied):\n"
            f"{SOURCE}\n"
            "group 'default' = manually annotated Satlas/OlmoEarth solar-farm windows.\n"
            "layer label_raster/label (uint8 PNG): 0=background, 1=solar_farm.\n"
            "layer mask/mask (uint8 PNG): 255=valid annotated region, 0=nodata.\n"
        )

    names = sorted(os.listdir(WINDOWS_ROOT))
    print(f"scanning {len(names)} windows (tile @ native 10 m UTM)", flush=True)
    all_records: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan_window, [dict(name=n) for n in names]),
            total=len(names),
        ):
            all_records.extend(recs)
    print(f"non-empty tiles: {len(all_records)}", flush=True)

    solar_tiles = [r for r in all_records if r["has_solar"]]
    bg_only_tiles = [r for r in all_records if not r["has_solar"]]
    print(
        f"  solar tiles: {len(solar_tiles)}  background-only tiles: {len(bg_only_tiles)}"
    )

    rng = random.Random(SEED)
    rng.shuffle(solar_tiles)
    rng.shuffle(bg_only_tiles)

    # Prioritize the rare class: take up to PER_CLASS solar tiles (each also carries
    # background), then top up with background-only tiles until background ~= PER_CLASS.
    selected_solar = solar_tiles[:PER_CLASS]
    bg_from_solar = sum(
        1 for t in selected_solar if BACKGROUND_ID in t["classes_present"]
    )
    need_bg = max(0, PER_CLASS - bg_from_solar)
    selected_bg = bg_only_tiles[:need_bg]
    selected = selected_solar + selected_bg
    rng.shuffle(selected)
    print(
        f"selected {len(selected)} tiles "
        f"(solar={len(selected_solar)}, background-only added={len(selected_bg)})"
    )

    # Deterministic, stable sample ids.
    selected.sort(key=lambda r: (r["window"], r["r0"], r["c0"]))
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"

    # Group selected tiles by window so each window is read once in the write phase.
    by_window: dict[str, list[dict[str, Any]]] = {}
    for r in selected:
        by_window.setdefault(r["window"], []).append(r)

    io.check_disk()
    class_counts = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for present_lists in tqdm.tqdm(
            star_imap_unordered(
                p,
                _write_window_tiles,
                [dict(name=n, tiles=t) for n, t in by_window.items()],
            ),
            total=len(by_window),
        ):
            for present in present_lists:
                for cid in present:
                    class_counts[cid] += 1

    total = len(selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "olmoearth",
            "license": "ODbL/internal",
            "provenance": {
                "url": SOURCE,
                "have_locally": True,
                "annotation_method": "manual annotation (Satlas)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {
                    "id": BACKGROUND_ID,
                    "name": "background",
                    "description": "Non-solar-farm surface (any other land cover) within the annotated region.",
                },
                {
                    "id": SOLAR_ID,
                    "name": "solar_farm",
                    "description": "Ground-mounted photovoltaic solar-farm footprint (panel arrays), manually annotated in Sentinel-2 (Satlas).",
                },
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": total,
            "class_tile_counts": {
                CLASS_NAMES[c]: class_counts[c] for c in (BACKGROUND_ID, SOLAR_ID)
            },
            "tile_size": TILE,
            "notes": (
                "Local OlmoEarth/Satlas solar-farm rslearn dataset (3561 windows, splits "
                "train+val, both used). Source already local UTM @ 10 m, so no resampling: "
                "each window's label_raster/label + mask/mask PNGs are read directly and "
                "tiled into <=64x64 patches on the native grid. Output uint8: 0=background, "
                "1=solar_farm, 255=nodata where mask==0 (unannotated window borders). "
                "Tiles-per-class balanced: all solar tiles prioritized (rare class), "
                "background-only tiles added to balance; a tile counts toward every class "
                "present. A tile is a solar tile only if it has >=10 solar pixels, and any "
                "tile needs >=256 valid (masked-in) pixels. Time range = the window's own "
                "~180-day acquisition range (<=1 yr); solar farms are persistent."
            ),
        },
    )
    print(
        f"class tile counts: {{'background': {class_counts[BACKGROUND_ID]}, "
        f"'solar_farm': {class_counts[SOLAR_ID]}}}"
    )
    print(f"done: {total} tiles")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
