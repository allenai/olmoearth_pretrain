"""Process Sen1Floods11 into open-set-segmentation label patches.

Source: Sen1Floods11 (Bonafilia et al. 2020, CVPRW; Cloud to Street /
Google), https://github.com/cloudtostreet/Sen1Floods11 with data on the public
GCS bucket ``gs://sen1floods11/``. The dataset provides georeferenced 512x512
~10 m chips over 11 global flood events, each with a coincident Sentinel-1
(and Sentinel-2) acquisition. We use the **hand-labeled** subset
(``data/flood_events/HandLabeled/``, 446 chips), which is the high-quality
manually-annotated flood-extent product.

Two source label rasters per chip (both EPSG:4326, 512x512, ~10 m):
  * ``LabelHand`` (int16): hand-annotated surface-water extent at the flood
    acquisition. -1 = no data / not analyzed, 0 = not water, 1 = water.
  * ``JRCWaterHand`` (uint8): JRC Global Surface Water permanent-water mask
    co-registered to the chip. 0 = not permanent water, 1 = permanent water.

We fuse them into the manifest's 3-class scheme (dense per-pixel
CLASSIFICATION):
    id 0 = flood water      (LabelHand water AND not JRC permanent water)
    id 1 = permanent water  (JRC permanent water, where observed)
    id 2 = non-water        (LabelHand not-water)
    255  = nodata/ignore    (LabelHand == -1)
Permanent water takes priority over flood/non-water; nodata pixels
(LabelHand == -1) stay nodata even if JRC marks them permanent.

Processing (label_type = dense_raster): each 512x512 EPSG:4326 chip is
reprojected once to its local UTM zone at 10 m (nearest resampling -
categorical) and cut into 64x64 tiles. Sampling is **tiles-per-class
balanced** (spec 5): a tile counts toward every class present in it (>=
MIN_CLASS_PX pixels); rare classes (flood/permanent water) are filled first up
to PER_CLASS tiles. Non-water co-occurs in almost every tile so it needs no
dedicated pass.

Time range: each event has a Sentinel-1 acquisition date (from
``Sen1Floods11_Metadata.geojson``). The flood mask is an **event** label, so we
set ``change_time`` to the acquisition date and keep it as the reference used to
build two adjacent windows via ``io.pre_post_time_ranges``: ``pre_time_range`` (the
~6 months, <=183 days, immediately before change_time) and ``post_time_range`` (the
~6 months, <=183 days, immediately after); ``time_range`` is null (spec 5, change
labels). Pretraining pairs a "before" image stack with an "after" stack and probes
on their difference.

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.sen1floods11
"""

import argparse
import multiprocessing
import random
import subprocess
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import rasterio
import shapely
import tqdm
from rasterio.warp import Resampling, reproject
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import STGeometry
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds

from olmoearth_pretrain.open_set_segmentation_data import io

SLUG = "sen1floods11"
NAME = "Sen1Floods11"

GCS_HANDLABELED = "gs://sen1floods11/v1.1/data/flood_events/HandLabeled"
GCS_METADATA = "gs://sen1floods11/v1.1/Sen1Floods11_Metadata.geojson"

TILE = 64
PER_CLASS = 1000
MIN_CLASS_PX = 32  # a tile counts toward a class only with >= this many px of it
MAX_NODATA_FRAC = 0.5  # skip tiles that are more than half nodata

# Manifest class order -> id.
CLASSES = [
    (
        "flood water",
        "Surface water present at the flood-event Sentinel-1 acquisition that is NOT "
        "permanent water (i.e. hand-labeled water extent minus JRC permanent water) - "
        "the flood inundation.",
    ),
    (
        "permanent water",
        "Permanent open water (rivers, lakes, reservoirs) per the JRC Global Surface Water "
        "mask co-registered to the chip, restricted to observed (non-nodata) pixels.",
    ),
    (
        "non-water",
        "Land / non-water: hand-labeled as not water at the flood acquisition.",
    ),
]
FLOOD, PERM, NONWATER = 0, 1, 2

# LabelHand location prefix -> the metadata event (location name in the metadata
# geojson). The hand-labeled chips prefix the Cambodia (Mekong river) event as
# "Mekong". Colombia (metadata ID 12) has no hand labels. Dates are the S1
# acquisition (YYYY/MM/DD) from Sen1Floods11_Metadata.geojson.
EVENT_DATE = {
    "Bolivia": "2018/02/15",
    "Ghana": "2018/09/18",
    "India": "2016/08/12",
    "Mekong": "2018/08/05",  # Cambodia event, Mekong river
    "Nigeria": "2018/09/21",
    "Pakistan": "2017/06/28",
    "Paraguay": "2018/10/31",
    "Somalia": "2018/05/07",
    "Spain": "2019/09/17",
    "Sri-Lanka": "2017/05/30",
    "USA": "2019/05/22",
}


def raw_root():
    return io.raw_dir(SLUG)


def label_path(name: str):
    return raw_root() / "LabelHand" / f"{name}_LabelHand.tif"


def jrc_path(name: str):
    return raw_root() / "JRCWaterHand" / f"{name}_JRCWaterHand.tif"


def download_raw() -> list[str]:
    """Download the hand-labeled LabelHand + JRCWaterHand rasters + metadata (idempotent).

    Returns the list of chip base names (e.g. "Bolivia_103757").
    """
    root = raw_root()
    (root / "LabelHand").mkdir(parents=True, exist_ok=True)
    (root / "JRCWaterHand").mkdir(parents=True, exist_ok=True)
    io.check_disk()
    for sub in ("LabelHand", "JRCWaterHand"):
        dst = root / sub
        # gsutil -m rsync is idempotent and fast for the ~446 tiny files.
        subprocess.run(
            ["gsutil", "-q", "-m", "rsync", f"{GCS_HANDLABELED}/{sub}/", str(dst)],
            check=True,
        )
    meta_dst = root / "Sen1Floods11_Metadata.geojson"
    if not meta_dst.exists():
        subprocess.run(["gsutil", "-q", "cp", GCS_METADATA, str(meta_dst)], check=True)

    names = sorted(
        p.name[: -len("_LabelHand.tif")]
        for p in (root / "LabelHand").iterdir()
        if p.name.endswith("_LabelHand.tif")
    )
    return names


def _combined_label(name: str) -> tuple[np.ndarray, Any]:
    """Return (uint8 3-class array, rasterio src transform+crs handle) for a chip.

    Reads LabelHand + JRCWaterHand (EPSG:4326, 512x512) and fuses to the 3-class
    scheme with 255 nodata. Returns the array and the LabelHand rasterio profile
    fields needed for reprojection.
    """
    with rasterio.open(str(label_path(name))) as d:
        lab = d.read(1)
        src_transform = d.transform
        src_crs = d.crs
        src_bounds = d.bounds
    with rasterio.open(str(jrc_path(name))) as d:
        jrc = d.read(1)

    valid = lab != -1
    out = np.full(lab.shape, io.CLASS_NODATA, dtype=np.uint8)
    out[valid & (lab == 0)] = NONWATER
    out[valid & (lab == 1)] = FLOOD
    out[valid & (jrc == 1)] = PERM  # permanent water wins over flood/non-water
    return out, (src_transform, src_crs, src_bounds)


def _reproject_chip(name: str):
    """Reproject a chip's 3-class label to local UTM 10 m; return (arr, proj, col0, row0).

    ``arr`` is (H, W) uint8 with 255 nodata, sized to whole 64-px tiles. Pixel
    (col0+j, row0+i) is the top-left of the array under ``proj``.
    """
    combined, (src_transform, src_crs, src_bounds) = _combined_label(name)

    cx = (src_bounds.left + src_bounds.right) / 2.0
    cy = (src_bounds.bottom + src_bounds.top) / 2.0
    proj = io.utm_projection_for_lonlat(cx, cy)

    # Project the chip footprint into UTM pixel coordinates to size the output.
    box = shapely.box(
        src_bounds.left, src_bounds.bottom, src_bounds.right, src_bounds.top
    )
    utm_box = STGeometry(WGS84_PROJECTION, box, None).to_projection(proj).shp
    minx, miny, maxx, maxy = utm_box.bounds
    pad = 2
    col0 = int(np.floor(minx)) - pad
    row0 = int(np.floor(miny)) - pad
    w = int(np.ceil(maxx)) + pad - col0
    h = int(np.ceil(maxy)) + pad - row0
    # Round up to whole tiles so the grid cuts evenly.
    w = ((w + TILE - 1) // TILE) * TILE
    h = ((h + TILE - 1) // TILE) * TILE
    bounds = (col0, row0, col0 + w, row0 + h)
    dst_transform = get_transform_from_projection_and_bounds(proj, bounds)

    dst = np.full((h, w), io.CLASS_NODATA, dtype=np.uint8)
    reproject(
        source=combined,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=proj.crs,
        resampling=Resampling.nearest,
        src_nodata=io.CLASS_NODATA,
        dst_nodata=io.CLASS_NODATA,
    )
    return dst, proj, col0, row0


def _tile_counts(arr: np.ndarray, ti: int, tj: int) -> dict[int, int]:
    """Per-class pixel counts for tile (ti, tj) of a reprojected chip array."""
    sub = arr[ti * TILE : (ti + 1) * TILE, tj * TILE : (tj + 1) * TILE]
    u, c = np.unique(sub, return_counts=True)
    return {int(k): int(v) for k, v in zip(u, c)}


def _scan_chip(name: str) -> list[dict[str, Any]]:
    """Return one candidate record per non-mostly-nodata 64x64 tile of a chip."""
    arr, _proj, _col0, _row0 = _reproject_chip(name)
    nty, ntx = arr.shape[0] // TILE, arr.shape[1] // TILE
    recs: list[dict[str, Any]] = []
    total_px = TILE * TILE
    for ti in range(nty):
        for tj in range(ntx):
            counts = _tile_counts(arr, ti, tj)
            nodata = counts.get(io.CLASS_NODATA, 0)
            if nodata > MAX_NODATA_FRAC * total_px:
                continue
            count_classes = [
                c for c in (FLOOD, PERM, NONWATER) if counts.get(c, 0) >= MIN_CLASS_PX
            ]
            if not count_classes:
                continue
            recs.append(
                {
                    "name": name,
                    "ti": ti,
                    "tj": tj,
                    "count_classes": count_classes,
                }
            )
    return recs


def _select_tiles_per_class(all_recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tiles-per-class balanced selection (spec 5). Rare classes filled first."""
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in all_recs:
        for c in rec["count_classes"]:
            by_class[c].append(rec)
    # rarest first; tie-break on class id so the order is deterministic even though
    # by_class was built from a nondeterministically-ordered (star_imap_unordered) scan.
    order = sorted(by_class, key=lambda c: (len(by_class[c]), c))
    rng = random.Random(42)
    selected_keys: set = set()
    selected: list[dict[str, Any]] = []
    counts: dict[int, int] = defaultdict(int)
    for c in order:
        # Sort by the tile's stable identity before the seeded shuffle so the selected
        # subset is reproducible regardless of the scan's completion order.
        tiles = sorted(by_class[c], key=lambda r: (r["name"], r["ti"], r["tj"]))
        rng.shuffle(tiles)
        for rec in tiles:
            if counts[c] >= PER_CLASS:
                break
            key = (rec["name"], rec["ti"], rec["tj"])
            if key in selected_keys:
                continue
            selected_keys.add(key)
            selected.append(rec)
            for cc in rec["count_classes"]:
                counts[cc] += 1
    return selected


def event_time(name: str):
    """(change_time, outer time_range, pre_range, post_range) for a chip."""
    loc = name.split("_")[0]
    d = datetime.strptime(EVENT_DATE[loc], "%Y/%m/%d").replace(tzinfo=UTC)
    pre_range, post_range = io.pre_post_time_ranges(d)
    return d, (pre_range[0], post_range[1]), pre_range, post_range


def _write_chip(name: str, tiles: list[dict[str, Any]]) -> None:
    """Reproject one chip and write all its selected tiles."""
    arr, proj, col0, row0 = _reproject_chip(name)
    change_time, tr, pre_range, post_range = event_time(name)
    for t in tiles:
        sample_id = t["sample_id"]
        if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
            continue
        ti, tj = t["ti"], t["tj"]
        sub = arr[ti * TILE : (ti + 1) * TILE, tj * TILE : (tj + 1) * TILE].copy()
        x_min = col0 + tj * TILE
        y_min = row0 + ti * TILE
        bounds = (x_min, y_min, x_min + TILE, y_min + TILE)
        io.write_label_geotiff(
            SLUG, sample_id, sub, proj, bounds, nodata=io.CLASS_NODATA
        )
        present = sorted(int(x) for x in np.unique(sub) if x != io.CLASS_NODATA)
        io.write_sample_json(
            SLUG,
            sample_id,
            proj,
            bounds,
            tr,
            change_time=change_time,
            source_id=f"{name}_r{ti}_c{tj}",
            classes_present=present,
            pre_time_range=pre_range,
            post_time_range=post_range,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    from olmoearth_pretrain.open_set_segmentation_data import manifest

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    print("Downloading Sen1Floods11 hand-labeled rasters...")
    names = download_raw()
    print(f"  {len(names)} hand-labeled chips")
    io.check_disk()

    print("Scanning chips into 64x64 tiles...")
    with multiprocessing.Pool(args.workers) as p:
        all_recs: list[dict[str, Any]] = []
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan_chip, [dict(name=n) for n in names]),
            total=len(names),
        ):
            all_recs.extend(recs)
    print(f"  {len(all_recs)} candidate tiles")

    selected = _select_tiles_per_class(all_recs)
    # Stable sample-id ordering (by chip then tile) for idempotent reruns.
    selected.sort(key=lambda r: (r["name"], r["ti"], r["tj"]))
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(
        f"  selected {len(selected)} tiles (tiles-per-class balanced, <= {PER_CLASS}/class)"
    )

    # Group selected tiles by chip for the write phase.
    by_chip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in selected:
        by_chip[r["name"]].append(r)

    io.check_disk()
    print(f"Writing tiles for {len(by_chip)} chips...")
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(
                p,
                _write_chip,
                [dict(name=n, tiles=ts) for n, ts in by_chip.items()],
            ),
            total=len(by_chip),
        ):
            pass

    # Class tile-occurrence counts (a tile counts toward every class present).
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
            "source": "Cloud to Street / Google (GitHub cloudtostreet/Sen1Floods11)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://github.com/cloudtostreet/Sen1Floods11",
                "gcs_bucket": "gs://sen1floods11/v1.1",
                "have_locally": False,
                "annotation_method": "manual (hand-labeled flood-extent subset)",
                "citation": "Bonafilia et al. 2020, CVPRW (Sen1Floods11)",
                "subset": "HandLabeled (446 chips, 11 events)",
            },
            "sensors_relevant": ["sentinel1", "sentinel2"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "tile_class_counts": tile_class_counts,
            "notes": (
                "Hand-labeled subset of Sen1Floods11. Each 512x512 EPSG:4326 ~10 m chip "
                "reprojected to local UTM at 10 m (nearest, categorical) and cut into 64x64 "
                "tiles. 3-class fusion of LabelHand (surface-water extent: -1 nodata / 0 land "
                "/ 1 water) with JRCWaterHand (JRC permanent water 0/1): flood water = water & "
                "not permanent, permanent water = JRC permanent (observed), non-water = land. "
                "Tiles-per-class balanced (<=1000/class); non-water co-occurs widely. Flood is "
                "an event label: change_time set to the Sentinel-1 acquisition date, time_range "
                "a 1-year window centered on it. Colombia event (metadata ID 12) has no hand "
                "labels and is absent; Cambodia event chips are prefixed 'Mekong'."
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
