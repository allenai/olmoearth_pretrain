"""Process CaBuAr (California Burned Areas) into open-set-segmentation label patches.

Source: CaBuAr (Rege Cambrin, Colomba, Garza 2023; IEEE GRSM,
doi:10.1109/MGRS.2023.3292467), HuggingFace `DarthReca/california_burned_areas`,
https://huggingface.co/datasets/DarthReca/california_burned_areas . Sentinel-2 pre/post-
fire acquisitions over California wildfires with **binary burned-area masks** derived from
CAL FIRE (California Dept. of Forestry and Fire Protection) fire perimeters, mapped onto
the imagery.

We use the pre-patched file `raw/patched/512x512.hdf5`: 534 patches of 512x512 px at
**20 m/pixel** (Sentinel-2 20 m grid), each keyed `{uuid}_{patch}` and holding `post_fire`
(12-band uint16), optional `pre_fire`, and a binary `mask` (uint16 {0,1}; 1 = burned). Only
patches containing >=1 burned pixel are present in this file. Per-patch georeferencing (CRS
+ x/y pixel-center coordinate arrays + post-fire acquisition timestamp) comes from the
companion `metadata.parquet` (keyed on uuid+patch, `post==True` rows).

Class scheme (dense per-pixel CLASSIFICATION, matching the manifest's 2 classes):
    id 0 = unburned   (mask == 0, observed)
    id 1 = burned     (mask == 1)
    255  = nodata/ignore  (all-12-band-zero fill at S2 tile edges, ~14% of patches)

Processing (label_type = dense_raster): each 512x512 20 m patch is already in local UTM.
We cut it into 32x32 (20 m) blocks and **upsample each block 2x with nearest resampling**
to a 64x64 tile at 10 m (categorical label -> nearest, never bilinear), georeferenced from
the block's UTM coordinates. Sampling is **tiles-per-class balanced** (spec 5): a tile
counts toward every class present in it (>= MIN_CLASS_PX px), rarer class (burned) filled
first, up to PER_CLASS tiles/class.

Time range: the burn is a change/event label. `change_time` is set to the post-fire
Sentinel-2 acquisition timestamp (a post-event date - the fire occurred shortly before it,
between the pre- and post-fire acquisitions). We emit two independent six-month windows via
`io.pre_post_time_ranges(change_time, pre_offset_days=90)`: `post_time_range` starts at
`change_time` and runs ~6 months (<=183 days) forward, and `pre_time_range` ends 90 days
before `change_time` (a guard offset, since the fire precedes the acquisition) and spans
~6 months (<=183 days) backward from there, keeping the pre window entirely before the
fire. `time_range` is null. Pretraining pairs a "before" stack with an "after" stack and
probes on their difference (spec 5).

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.cabuar_california_burned_areas
"""

import argparse
import multiprocessing
import random
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import tqdm
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest

try:  # BZip2/other filters may be needed for some HDF5s; harmless if unavailable.
    import hdf5plugin  # noqa: F401
except Exception:  # pragma: no cover
    pass

import h5py  # noqa: E402  (import after optional hdf5plugin)

SLUG = "cabuar_california_burned_areas"
NAME = "CaBuAr (California Burned Areas)"

RAW = io.raw_dir(SLUG)
HDF5_PATH = RAW / "512x512.hdf5"
META_PATH = RAW / "metadata.parquet"

SRC_RES = 20  # source resolution (m)
SRC_BLOCK = 32  # source-pixel block edge (32 * 20 m = 640 m)
UPSAMPLE = 2  # 20 m -> 10 m
TILE = SRC_BLOCK * UPSAMPLE  # 64 output px at 10 m
GRID = 512 // SRC_BLOCK  # 16 blocks per axis

PER_CLASS = 1000
MIN_CLASS_PX = 32  # a tile counts toward a class only with >= this many output px
MAX_NODATA_FRAC = 0.5  # skip tiles that are more than half nodata

UNBURNED, BURNED = 0, 1
CLASSES = [
    (
        "unburned",
        "No wildfire burn at this pixel over the mapped fire event (Sentinel-2 pixel not "
        "inside the CAL FIRE burned-area perimeter), among observed pixels.",
    ),
    (
        "burned",
        "Wildfire burned area: pixel inside the CAL FIRE (California Dept. of Forestry and "
        "Fire Protection) fire perimeter for the event, mapped onto the post-fire "
        "Sentinel-2 acquisition.",
    ),
]


def _load_georef() -> dict[str, tuple[int, float, float, str]]:
    """Key -> (epsg, x0, y0, post_timestamp) from metadata.parquet (post==True rows).

    x0/y0 are the pixel-center coordinates of the patch's top-left (col 0, row 0) in the
    patch CRS; the grid is regular at 20 m (x ascending, y descending).
    """
    df = pd.read_parquet(META_PATH.path)
    df["key"] = df.uuid + "_" + df.patch.astype(str)
    post = df[df.post == True]  # noqa: E712
    out: dict[str, tuple[int, float, float, str]] = {}
    for key, x, y, epsg, ts in zip(
        post.key.values,
        post.x.values,
        post.y.values,
        post.crs.values,
        post.timestamp.values,
    ):
        if key in out:
            continue
        out[key] = (int(epsg), float(x[0]), float(y[0]), str(ts))
    return out


def _label_array(key: str) -> np.ndarray:
    """Read a patch's 512x512 uint8 label (0 unburned / 1 burned / 255 nodata).

    nodata = pixels where the post-fire image is all-zero (S2 tile-edge fill).
    """
    with h5py.File(HDF5_PATH.path, "r") as f:
        g = f[key]
        mask = np.asarray(g["mask"][..., 0]).astype(np.uint8)
        nodata = (np.asarray(g["post_fire"][...]) == 0).all(axis=2)
    mask[mask > 1] = UNBURNED  # defensive; source is {0,1}
    mask[nodata] = io.CLASS_NODATA
    return mask


def _block(label: np.ndarray, ti: int, tj: int) -> np.ndarray:
    return label[
        ti * SRC_BLOCK : (ti + 1) * SRC_BLOCK, tj * SRC_BLOCK : (tj + 1) * SRC_BLOCK
    ]


def _scan_patch(key: str) -> list[dict[str, Any]]:
    """One candidate record per non-mostly-nodata 64x64 tile of a patch."""
    label = _label_array(key)
    total = SRC_BLOCK * SRC_BLOCK
    recs: list[dict[str, Any]] = []
    for ti in range(GRID):
        for tj in range(GRID):
            b = _block(label, ti, tj)
            nod = int((b == io.CLASS_NODATA).sum())
            if nod > MAX_NODATA_FRAC * total:
                continue
            # counts are on the (2x upsampled) output grid -> source count * 4.
            present = [
                c
                for c in (UNBURNED, BURNED)
                if int((b == c).sum()) * (UPSAMPLE * UPSAMPLE) >= MIN_CLASS_PX
            ]
            if not present:
                continue
            recs.append({"key": key, "ti": ti, "tj": tj, "count_classes": present})
    return recs


def _select_tiles_per_class(all_recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tiles-per-class balanced selection (spec 5). Rarest class filled first."""
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in all_recs:
        for c in rec["count_classes"]:
            by_class[c].append(rec)
    order = sorted(by_class, key=lambda c: len(by_class[c]))  # rarest first
    rng = random.Random(42)
    selected_keys: set = set()
    selected: list[dict[str, Any]] = []
    counts: dict[int, int] = defaultdict(int)
    for c in order:
        tiles = by_class[c][:]
        rng.shuffle(tiles)
        for rec in tiles:
            if counts[c] >= PER_CLASS:
                break
            k = (rec["key"], rec["ti"], rec["tj"])
            if k in selected_keys:
                continue
            selected_keys.add(k)
            selected.append(rec)
            for cc in rec["count_classes"]:
                counts[cc] += 1
    return selected


def _event_time(
    ts: str,
) -> tuple[
    datetime,
    tuple[datetime, datetime],
    tuple[datetime, datetime],
    tuple[datetime, datetime],
]:
    """(change_time, outer time_range, pre_range, post_range) from a post-fire ISO ts.

    change_time is a post-fire acquisition date, so the pre window is pushed earlier
    (pre_offset_days=90) to end before the fire itself.
    """
    d = datetime.fromisoformat(ts)
    if d.tzinfo is None:
        d = d.replace(tzinfo=UTC)
    pre_range, post_range = io.pre_post_time_ranges(d, pre_offset_days=90)
    return d, (pre_range[0], post_range[1]), pre_range, post_range


def _tile_bounds(x0: float, y0: float, ti: int, tj: int) -> tuple[int, int, int, int]:
    """Pixel bounds of tile (ti, tj) under the patch CRS at 10 m."""
    c0, r0 = tj * SRC_BLOCK, ti * SRC_BLOCK
    x_left = x0 + c0 * SRC_RES - SRC_RES / 2.0  # west edge of source pixel c0
    y_top = y0 - r0 * SRC_RES + SRC_RES / 2.0  # north edge of source pixel r0
    col_min = int(round(x_left / io.RESOLUTION))
    row_min = int(round(-y_top / io.RESOLUTION))
    return (col_min, row_min, col_min + TILE, row_min + TILE)


def _write_patch(
    key: str, georef: tuple[int, float, float, str], tiles: list[dict[str, Any]]
) -> None:
    """Write all selected tiles of one patch."""
    epsg, x0, y0, ts = georef
    proj = Projection(CRS.from_epsg(epsg), io.RESOLUTION, -io.RESOLUTION)
    change_time, tr, pre_range, post_range = _event_time(ts)
    label = None
    for t in tiles:
        sample_id = t["sample_id"]
        if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
            continue
        if label is None:
            label = _label_array(key)
        ti, tj = t["ti"], t["tj"]
        b = _block(label, ti, tj)
        out = np.repeat(
            np.repeat(b, UPSAMPLE, axis=0), UPSAMPLE, axis=1
        )  # 64x64 nearest
        bounds = _tile_bounds(x0, y0, ti, tj)
        io.write_label_geotiff(
            SLUG, sample_id, out, proj, bounds, nodata=io.CLASS_NODATA
        )
        present = sorted(int(v) for v in np.unique(out) if v != io.CLASS_NODATA)
        io.write_sample_json(
            SLUG,
            sample_id,
            proj,
            bounds,
            tr,
            change_time=change_time,
            source_id=f"{key}_r{ti}_c{tj}",
            classes_present=present,
            pre_time_range=pre_range,
            post_time_range=post_range,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    assert HDF5_PATH.exists(), f"missing {HDF5_PATH}; download raw first"
    georef = _load_georef()
    with h5py.File(HDF5_PATH.path, "r") as f:
        keys = list(f.keys())
    keys = [k for k in keys if k in georef]
    print(f"{len(keys)} fire patches (512x512 @ 20 m)")

    print("Scanning patches into 64x64 tiles...")
    with multiprocessing.Pool(args.workers) as p:
        all_recs: list[dict[str, Any]] = []
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan_patch, [dict(key=k) for k in keys]),
            total=len(keys),
        ):
            all_recs.extend(recs)
    print(f"  {len(all_recs)} candidate tiles")

    selected = _select_tiles_per_class(all_recs)
    selected.sort(key=lambda r: (r["key"], r["ti"], r["tj"]))  # stable, idempotent ids
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(
        f"  selected {len(selected)} tiles (tiles-per-class balanced, <= {PER_CLASS}/class)"
    )

    by_patch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in selected:
        by_patch[r["key"]].append(r)

    io.check_disk()
    print(f"Writing tiles for {len(by_patch)} patches...")
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(
                p,
                _write_patch,
                [dict(key=k, georef=georef[k], tiles=ts) for k, ts in by_patch.items()],
            ),
            total=len(by_patch),
        ):
            pass

    tile_class_counts = {name: 0 for name, _ in CLASSES}
    for r in selected:
        for c in r["count_classes"]:
            tile_class_counts[CLASSES[c][0]] += 1
    print("tiles containing each class:", tile_class_counts)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "CaBuAr (HuggingFace DarthReca/california_burned_areas)",
            "license": "CDLA-Permissive-2.0",
            "provenance": {
                "url": "https://huggingface.co/datasets/DarthReca/california_burned_areas",
                "have_locally": False,
                "annotation_method": "derived (CAL FIRE fire perimeters mapped onto S2)",
                "citation": "Rege Cambrin, Colomba, Garza 2023, IEEE GRSM, doi:10.1109/MGRS.2023.3292467",
                "file": "raw/patched/512x512.hdf5 (534 burned patches, 512x512 @ 20 m)",
            },
            "sensors_relevant": ["sentinel2"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "tile_class_counts": tile_class_counts,
            "notes": (
                "CaBuAr binary burned-area masks (CAL FIRE perimeters) over California "
                "wildfires. Source patches are 512x512 @ 20 m in local UTM; each is cut into "
                "32x32 (20 m) blocks and upsampled 2x with nearest resampling to 64x64 tiles "
                "at 10 m. Classes: 0 unburned, 1 burned, 255 nodata (all-band-zero S2 "
                "tile-edge fill). Tiles-per-class balanced (<=1000/class), burned filled "
                "first. Burn is an event label: change_time = post-fire S2 acquisition "
                "timestamp, time_range = 1-year window centered on it. Only patches with "
                ">=1 burned pixel exist in the source file, so every patch supplies burned "
                "context; pure-unburned tiles come from unburned blocks within these patches."
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
