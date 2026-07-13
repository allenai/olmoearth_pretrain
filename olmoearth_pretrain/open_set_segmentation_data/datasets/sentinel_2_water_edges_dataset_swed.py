"""Process the Sentinel-2 Water Edges Dataset (SWED) into open-set-segmentation patches.

Source: SWED, UK Hydrographic Office (https://openmldata.ukho.gov.uk/). Globally
distributed Sentinel-2 L2A scenes with a **binary water / non-water** segmentation mask,
photointerpreted (expert-checked) on the S2 image. Distributed as one ~19 GB zip on a
public S3 bucket (``ukho-openmldata.s3.eu-west-2.amazonaws.com/SWED.zip``) containing
image+label pairs. **Only the label files are needed** (pretraining supplies its own
imagery), so we selectively extract just the labels via HTTP range requests (remotezip):

  * ``SWED/train/labels/{PRODUCT}_chip_{r}_{c}.npy`` -- 28,224 chips, int16 {0,1},
    256x256, one per 256x256 chip of a 42x42 grid cut from the S2 granule top-left.
    These carry **no embedded georeferencing**, but the filename gives the full S2
    product id (hence the MGRS tile) and the within-granule chip index (r,c). The S2
    granule origin + UTM CRS is deterministic per MGRS tile (looked up once from the
    Planetary Computer STAC and hardcoded in ``TILE_GEO``), so each chip's exact UTM
    10 m footprint is recoverable: chip (r,c) -> granule[r*256:(r+1)*256, c*256:...].
    16 distinct scenes (one product per MGRS tile), all 2017-2020 (post-2016).
  * ``SWED/test/labels/{PRODUCT}_label_{a}_{b}.tif`` -- 98 GeoTIFFs, uint16 {0,1}, in
    EPSG:4326 at ~10 m. These ARE georeferenced; we reproject each to its local UTM at
    10 m (nearest, categorical) before tiling.

label_type = dense_raster, binary CLASSIFICATION. Class ids follow the source label
values directly: **0 = non-water, 1 = water** (ids start at 0 per spec 2). All source
splits are used (spec 5). Each source scene is tiled into 64x64 UTM patches; a tile
counts toward a class only if it holds >= MIN_CLASS_PX px of it; tiles > MAX_NODATA_FRAC
nodata (test-set reprojection edges) are dropped. Selection is **tiles-per-class
balanced** (spec 5) via ``sampling.select_tiles_per_class`` (<= 1000 tiles/class, 25k
cap; the rarer class is filled first).

Time range: the water/non-water mask is a per-image STATE observed in one Sentinel-2
acquisition -- water extent varies with tide (the dataset ships a per-test-scene tidal
csv), so it is NOT a diffuse yearly label. Per spec 5 (specific-image labels) we set
``time_range`` to a ~1-hour window at the S2 acquisition datetime (parsed from the
product id) and leave ``change_time = null``. This mirrors the worldfloods_v2 decision.

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.sentinel_2_water_edges_dataset_swed
"""

import argparse
import multiprocessing
import os
import re
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest, sampling

SLUG = "sentinel_2_water_edges_dataset_swed"
NAME = "Sentinel-2 Water Edges Dataset (SWED)"
ZIP_URL = "https://ukho-openmldata.s3.eu-west-2.amazonaws.com/SWED.zip"
SCRATCH = "swed_scratch"  # local staging for extracted labels (not weka)

TILE = 64
CHIP = 256  # native SWED chip size (px)
PER_CLASS = 1000
MIN_CLASS_PX = 64  # a tile counts toward a class only with >= this many px of it
MAX_NODATA_FRAC = (
    0.5  # drop tiles that are more than half nodata (test reprojection edges)
)

# id -> (name, description). Follows the source binary encoding: 0 = non-water, 1 = water.
CLASSES = [
    (
        "non-water",
        "Land / non-water surface (SWED binary label value 0): everything in the scene not "
        "photointerpreted as water.",
    ),
    (
        "water",
        "Open water surface -- sea, coastal and inland water (SWED binary label value 1), "
        "expert-checked photointerpretation on the Sentinel-2 image. Water extent is "
        "image-specific (varies with tidal state).",
    ),
]
NONWATER, WATER = 0, 1

# S2 granule UTM origin (EPSG, ULX, ULY) per MGRS tile, from Planetary Computer STAC
# (proj:transform of the 10 m bands). Deterministic per tile; chips tile from the ULX/ULY.
TILE_GEO: dict[str, tuple[int, float, float]] = {
    "16RBU": (32616, 199980.0, 3400020.0),
    "17MNT": (32717, 499980.0, 9800020.0),
    "18TWL": (32618, 499980.0, 4600020.0),
    "19KCQ": (32719, 300000.0, 7500040.0),
    "20PLS": (32620, 300000.0, 1200000.0),
    "24MXV": (32724, 600000.0, 9500020.0),
    "28PCA": (32628, 300000.0, 1600020.0),
    "29NKH": (32629, 199980.0, 800040.0),
    "30STF": (32630, 199980.0, 4100040.0),
    "30UYC": (32630, 699960.0, 5800020.0),
    "31UFV": (32631, 600000.0, 6000000.0),
    "32TQR": (32632, 699960.0, 5100000.0),
    "34RCU": (32634, 300000.0, 3400020.0),
    "39PVL": (32639, 399960.0, 1100040.0),
    "51HYE": (32751, 699960.0, 6500020.0),
    "58KDB": (32758, 399960.0, 7700020.0),
}

_DT_RE = re.compile(r"MSIL2A_(\d{8}T\d{6})_")
_TILE_RE = re.compile(r"_T(\w{5})_")
_CHIP_RE = re.compile(r"_chip_(\d+)_(\d+)\.npy$")


def parse_dt(fname: str) -> datetime:
    """Sentinel-2 acquisition datetime (UTC) from the product id in the filename."""
    m = _DT_RE.search(fname)
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=UTC)


def parse_tile(fname: str) -> str:
    return _TILE_RE.search(fname).group(1)


# ---------------------------------------------------------------------------
# Download / selective extraction of label files only.
# ---------------------------------------------------------------------------


def _list_label_members() -> tuple[list[str], list[str]]:
    """Return (train_label_npy, test_label_tif) member names in the zip (cached)."""
    import json

    cache = os.path.join(SCRATCH, "_label_members.json")
    if os.path.exists(cache):
        with open(cache) as f:
            d = json.load(f)
        return d["train"], d["test"]
    from remotezip import RemoteZip

    with RemoteZip(ZIP_URL) as z:
        names = z.namelist()
    train = sorted(
        n for n in names if n.startswith("SWED/train/labels/") and n.endswith(".npy")
    )
    test = sorted(
        n for n in names if n.startswith("SWED/test/labels/") and n.endswith(".tif")
    )
    os.makedirs(SCRATCH, exist_ok=True)
    with open(cache, "w") as f:
        json.dump({"train": train, "test": test}, f)
    return train, test


def _extract_shard(members: list[str]) -> int:
    """Extract a shard of members from the zip to SCRATCH (idempotent, skip existing)."""
    from remotezip import RemoteZip

    todo = [m for m in members if not os.path.exists(os.path.join(SCRATCH, m))]
    if not todo:
        return 0
    with RemoteZip(ZIP_URL) as z:
        for m in todo:
            z.extract(m, SCRATCH)
    return len(todo)


def extract_labels(workers: int) -> tuple[list[str], list[str]]:
    """Selectively extract all label files to SCRATCH via parallel range requests."""
    train, test = _list_label_members()
    all_members = train + test
    missing = [m for m in all_members if not os.path.exists(os.path.join(SCRATCH, m))]
    print(
        f"  {len(all_members)} label files ({len(train)} train npy, {len(test)} test tif); "
        f"{len(missing)} missing -> extracting"
    )
    if missing:
        # Shard so each worker opens RemoteZip once and pulls a contiguous batch.
        n = max(1, len(missing) // (workers * 4) + 1)
        shards = [missing[i : i + n] for i in range(0, len(missing), n)]
        with multiprocessing.Pool(workers) as p:
            done = 0
            for c in p.imap_unordered(_extract_shard, shards):
                done += c
        print(f"  extracted {done} label files")
    return train, test


# ---------------------------------------------------------------------------
# Georeferencing.
# ---------------------------------------------------------------------------


def _train_geo(fname: str) -> tuple[Projection, int, int]:
    """Return (Projection, col0, row0): the top-left rslearn pixel of a train chip.

    chip (r,c) covers granule pixels [r*256:(r+1)*256, c*256:(c+1)*256] from the tile UL.
    """
    tile = parse_tile(fname)
    epsg, ulx, uly = TILE_GEO[tile]
    m = _CHIP_RE.search(fname)
    r, c = int(m.group(1)), int(m.group(2))
    proj = Projection(CRS.from_epsg(epsg), io.RESOLUTION, -io.RESOLUTION)
    col0 = int(round(ulx / io.RESOLUTION)) + c * CHIP  # geo_x / 10
    row0 = int(round(uly / (-io.RESOLUTION))) + r * CHIP  # geo_y / -10 (top row)
    return proj, col0, row0


def _load_train_label(path: str) -> np.ndarray:
    """Load a train label npy as a (256,256) uint8 array with values in {0,1}."""
    a = np.load(path)
    if a.ndim == 3:
        a = a[0]
    return a.astype(np.uint8)


def _reproject_test_label(path: str) -> tuple[np.ndarray, Projection, int, int]:
    """Reproject a test EPSG:4326 label tif to local UTM 10 m (nearest).

    Returns (uint8 array with 255 nodata outside source, Projection, col0, row0).
    """
    with rasterio.open(path) as d:
        src = d.read(1)
        src_crs = d.crs
        src_transform = d.transform
        left, bottom, right, top = d.bounds
    lon = (left + right) / 2.0
    lat = (bottom + top) / 2.0
    proj = io.utm_projection_for_lonlat(lon, lat)  # Projection(utm_crs, 10, -10)
    dst_crs = proj.crs
    # UTM extent of the tile from its 4 corners, snapped to the 10 m grid.
    from rasterio.warp import transform as warp_transform

    xs, ys = warp_transform(
        src_crs,
        dst_crs,
        [left, right, right, left],
        [bottom, bottom, top, top],
    )
    x_min = np.floor(min(xs) / io.RESOLUTION) * io.RESOLUTION
    x_max = np.ceil(max(xs) / io.RESOLUTION) * io.RESOLUTION
    y_min = np.floor(min(ys) / io.RESOLUTION) * io.RESOLUTION
    y_max = np.ceil(max(ys) / io.RESOLUTION) * io.RESOLUTION
    width = int(round((x_max - x_min) / io.RESOLUTION))
    height = int(round((y_max - y_min) / io.RESOLUTION))
    dst_transform = Affine(io.RESOLUTION, 0, x_min, 0, -io.RESOLUTION, y_max)
    dst = np.full((height, width), io.CLASS_NODATA, dtype=np.uint8)
    reproject(
        source=src.astype(np.uint8),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=None,
        dst_nodata=io.CLASS_NODATA,
        resampling=Resampling.nearest,
    )
    col0 = int(round(x_min / io.RESOLUTION))
    row0 = int(round(y_max / (-io.RESOLUTION)))
    return dst, proj, col0, row0


def _load_label(rec: dict[str, Any]) -> tuple[np.ndarray, Projection, int, int]:
    """Dispatch: return (label array, projection, col0, row0) for a scene record."""
    if rec["split"] == "train":
        arr = _load_train_label(rec["path"])
        proj, col0, row0 = _train_geo(os.path.basename(rec["path"]))
        return arr, proj, col0, row0
    return _reproject_test_label(rec["path"])


# ---------------------------------------------------------------------------
# Scan / write.
# ---------------------------------------------------------------------------


def _classes_in(sub: np.ndarray) -> list[int]:
    u, c = np.unique(sub, return_counts=True)
    counts = {int(k): int(v) for k, v in zip(u, c)}
    return [cid for cid in (NONWATER, WATER) if counts.get(cid, 0) >= MIN_CLASS_PX]


def _scan_scene(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one candidate record per usable 64x64 tile of a scene."""
    arr, _proj, _c0, _r0 = _load_label(rec)
    nty, ntx = arr.shape[0] // TILE, arr.shape[1] // TILE
    total = TILE * TILE
    out: list[dict[str, Any]] = []
    for si in range(nty):
        for sj in range(ntx):
            sub = arr[si * TILE : (si + 1) * TILE, sj * TILE : (sj + 1) * TILE]
            nod = int((sub == io.CLASS_NODATA).sum())
            if nod > MAX_NODATA_FRAC * total:
                continue
            present = _classes_in(sub)
            if not present:
                continue
            out.append(
                {
                    "split": rec["split"],
                    "path": rec["path"],
                    "si": si,
                    "sj": sj,
                    "classes_present": present,
                }
            )
    return out


def _write_scene(path: str, split: str, tiles: list[dict[str, Any]]) -> None:
    """Load a scene once and write all its selected 64x64 tiles + sidecars."""
    rec = {"split": split, "path": path}
    arr, proj, col0, row0 = _load_label(rec)
    dt = parse_dt(os.path.basename(path))
    tr = (dt, dt + timedelta(hours=1))  # per-image state (spec 5)
    for t in tiles:
        sid = t["sample_id"]
        if (io.locations_dir(SLUG) / f"{sid}.tif").exists():
            continue
        si, sj = t["si"], t["sj"]
        sub = arr[si * TILE : (si + 1) * TILE, sj * TILE : (sj + 1) * TILE].copy()
        x_min = col0 + sj * TILE
        y_min = row0 + si * TILE
        bounds = (x_min, y_min, x_min + TILE, y_min + TILE)
        io.write_label_geotiff(SLUG, sid, sub, proj, bounds, nodata=io.CLASS_NODATA)
        present = sorted(int(x) for x in np.unique(sub) if x != io.CLASS_NODATA)
        io.write_sample_json(
            SLUG,
            sid,
            proj,
            bounds,
            tr,
            change_time=None,
            source_id=f"{split}/{os.path.basename(path)}_t{si}_{sj}",
            classes_present=present,
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
        f.write(
            f"{NAME}\nPortal: https://openmldata.ukho.gov.uk/\nArchive: {ZIP_URL}\n"
        )
        f.write(
            "Only label files used (selective range extraction); staged at "
            f"{SCRATCH}.\nLicense: Geospatial Commission Data Exploration licence.\n"
        )

    print("Extracting SWED label files (selective, labels only)...")
    train, test = extract_labels(args.workers)
    io.check_disk()

    scenes = [{"split": "train", "path": os.path.join(SCRATCH, m)} for m in train] + [
        {"split": "test", "path": os.path.join(SCRATCH, m)} for m in test
    ]
    # Sanity: every train tile must be in our geo lookup.
    unknown = {parse_tile(os.path.basename(m)) for m in train} - set(TILE_GEO)
    if unknown:
        raise RuntimeError(f"train tiles without geo lookup: {unknown}")

    print(f"Scanning {len(scenes)} scenes into {TILE}x{TILE} tiles...")
    all_recs: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for recs in star_imap_unordered(p, _scan_scene, [dict(rec=s) for s in scenes]):
            all_recs.extend(recs)
    print(f"  {len(all_recs)} candidate tiles")

    selected = sampling.select_tiles_per_class(
        all_recs, classes_key="classes_present", per_class=PER_CLASS
    )
    selected.sort(key=lambda r: (r["split"], r["path"], r["si"], r["sj"]))
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(
        f"  selected {len(selected)} tiles (tiles-per-class balanced, <= {PER_CLASS}/class)"
    )

    by_scene: dict[str, list[dict[str, Any]]] = {}
    for r in selected:
        by_scene.setdefault(r["path"], []).append(r)
    split_by_path = {s["path"]: s["split"] for s in scenes}

    io.check_disk()
    print(f"Writing tiles for {len(by_scene)} scenes...")
    write_args = [
        dict(path=path, split=split_by_path[path], tiles=ts)
        for path, ts in by_scene.items()
    ]
    with multiprocessing.Pool(args.workers) as p:
        for _ in star_imap_unordered(p, _write_scene, write_args):
            pass

    tile_class_counts = {name: 0 for name, _ in CLASSES}
    split_counts = {"train": 0, "test": 0}
    for r in selected:
        split_counts[r["split"]] += 1
        for c in r["classes_present"]:
            tile_class_counts[CLASSES[c][0]] += 1
    print(
        "tiles containing each class:", tile_class_counts, "| by split:", split_counts
    )

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "UK Hydrographic Office (SWED)",
            "license": "Geospatial Commission Data Exploration licence",
            "provenance": {
                "url": "https://openmldata.ukho.gov.uk/",
                "have_locally": False,
                "annotation_method": "photointerpretation (expert-checked)",
                "archive": ZIP_URL,
                "splits_used": ["train", "test"],
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": nm, "description": desc}
                for i, (nm, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "tile_class_counts": tile_class_counts,
            "split_counts": split_counts,
            "notes": (
                "SWED binary water/non-water masks. Class ids follow the source encoding "
                "(0=non-water, 1=water). Train labels (.npy, 28224 chips over 16 S2 scenes) "
                "carry no embedded georeferencing; each chip's exact UTM 10 m footprint is "
                "reconstructed from the filename's MGRS tile (deterministic S2 granule origin "
                "+ UTM CRS, TILE_GEO) and within-granule chip index. Test labels (.tif, 98, "
                "EPSG:4326 ~10 m) are reprojected to local UTM 10 m (nearest). Both splits "
                "used; each scene tiled into 64x64 patches, tiles-per-class balanced "
                "(<=1000/class, rarer class filled first). Water extent is image-specific "
                "(tidal), so time_range is a ~1-hour window at the S2 acquisition datetime "
                "and change_time is null (specific-image label, spec 5)."
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
