"""Process AI4Boundaries (EC JRC) field-boundary masks into open-set-segmentation patches.

Source: European Commission, Joint Research Centre — "AI4Boundaries", an AI-ready dataset
to map agricultural field boundaries from Sentinel-2 and aerial photography (d'Andrimont
et al., ESSD 2023). Public JRC Open Data Catalogue (CC BY 4.0 / EC reuse notice), no
credential required. Labels are derived from openly-released GSAA parcel declarations for
2019 across 7 EU regions (Austria, Catalonia/ES, France, Luxembourg, Netherlands, Slovenia,
Sweden).

We use ONLY the **10 m Sentinel-2 label masks** (`sentinel2/masks.zip` on the JRC FTP), not
the imagery time series (.nc) and not the 1 m aerial orthophotos — pretraining supplies its
own imagery. Each mask is a 256x256, 10 m, EPSG:3035 (LAEA Europe), 4-band float32 GeoTIFF:
  band 1 = field-extent mask (1 = field, 0 = non-field/background)
  band 2 = field-boundary mask (1 = boundary pixel)
  band 3 = distance-to-boundary (unused)
  band 4 = field enumeration / instance id (unused)

Class scheme (dense 3-class segmentation, all resolvable at 10 m):
  0 = background / non-field
  1 = field interior  (extent==1 AND boundary==0)
  2 = field boundary  (boundary==1)   [priority over interior]

Field extent and boundaries ARE the signal the dataset was designed to expose at 10 m S2,
so this is a valid segmentation target. Caveat (from the source paper): GSAA parcels can be
missing, so "background" (0) mixes true non-field with un-declared fields — the labels are
meant for a masked-learning approach (learn the extent/boundary of *included* fields).

Processing (task spec §4 dense_raster, VHR-style reprojection §4):
  * Derive the 3-class array in EPSG:3035, reproject to a local UTM zone at 10 m with
    **nearest** resampling (preserves the 1-px boundary lines; never bilinear).
  * Tile the reprojected ~256x256 array into non-overlapping 64x64 windows (spec cap 64).
    Keep only windows containing at least one field pixel (class 1 or 2); slivers (<32 px on
    an axis) are dropped.
  * Tiles-per-class balanced selection: <=1000 tiles per class, rarest-class-first, capped at
    25,000 total (spec §5). All three source splits (train/val/test) are used.

Time range: the S2 composites are the 2019 growing season (Mar–Aug 2019); each patch gets a
1-year 2019 window (post-2016, Sentinel era). Not a change dataset (change_time=null).

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.ai4boundaries
"""

import argparse
import math
import multiprocessing
import pickle
from itertools import product
from typing import Any

import numpy as np
import rasterio
import tqdm
from affine import Affine
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject
from rslearn.utils.geometry import Projection
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest, sampling

SLUG = "ai4boundaries"
NAME = "AI4Boundaries"
SRC_EPSG = 3035  # ETRS89-LAEA Europe (source masks)
TARGET_RES = 10.0
TILE = 64
MIN_TILE = 32  # drop reprojection slivers smaller than this on either axis
PER_CLASS = 1000
YEAR = 2019

CLASSES = [
    (
        "background",
        "Non-field / background: land not delineated as an agricultural parcel in the 2019 "
        "GSAA declarations (includes genuine non-field land and any un-declared/missing fields; "
        "AI4Boundaries is a masked-learning benchmark, so background is not a pure negative).",
    ),
    (
        "field interior",
        "Interior of an agricultural field parcel (GSAA extent mask == 1, away from the parcel "
        "boundary).",
    ),
    (
        "field boundary",
        "Field-boundary pixel separating/enclosing agricultural parcels (GSAA-derived boundary "
        "mask == 1); the core signal AI4Boundaries was built to detect at 10 m.",
    ),
]
NUM_CLASSES = len(CLASSES)


def _mask_paths() -> list[dict[str, Any]]:
    """One task per mask tif: {path, split, file_id}. Splits are the extracted subfolders."""
    root = io.raw_dir(SLUG) / "masks"
    tasks: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        d = root / split
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.name.endswith(".tif"):
                # e.g. NL_4121_S2_10m_256.tif -> file_id NL_4121
                file_id = "_".join(p.name.split("_")[:2])
                tasks.append({"path": str(p), "split": split, "file_id": file_id})
    return tasks


def _load_class_array(path: str) -> tuple[np.ndarray, Affine, int, int]:
    """Read a mask and build the 3-class uint8 array in the source (EPSG:3035) grid."""
    with rasterio.open(path) as ds:
        extent = ds.read(1)
        boundary = ds.read(2)
        src_t = ds.transform
        W, H = ds.width, ds.height
    cls = np.zeros((H, W), dtype=np.uint8)
    cls[extent > 0.5] = 1
    cls[boundary > 0.5] = 2  # boundary wins over interior
    return cls, src_t, W, H


def _reproject_to_utm(
    cls: np.ndarray, src_t: Affine, W: int, H: int
) -> tuple[np.ndarray, str, int, int]:
    """Reproject the 3-class array from EPSG:3035 to a local UTM grid at 10 m (nearest).

    Returns (dst_uint8, utm_crs_str, col0, row0) where (col0,row0) is the dst grid's
    top-left pixel index under the UTM Projection(crs, 10, -10).
    """
    src_crs = CRS.from_epsg(SRC_EPSG)
    cx = src_t.c + src_t.a * W / 2.0
    cy = src_t.f + src_t.e * H / 2.0
    lon, lat = Transformer.from_crs(SRC_EPSG, 4326, always_xy=True).transform(cx, cy)
    utm = get_utm_ups_projection(lon, lat, TARGET_RES, -TARGET_RES).crs
    to_utm = Transformer.from_crs(SRC_EPSG, utm.to_epsg(), always_xy=True)
    xs = [src_t.c, src_t.c + src_t.a * W]
    ys = [src_t.f, src_t.f + src_t.e * H]
    pts = [to_utm.transform(X, Y) for X, Y in product(xs, ys)]
    cols = [p[0] / TARGET_RES for p in pts]
    rows = [p[1] / -TARGET_RES for p in pts]
    col0, col1 = math.floor(min(cols)), math.ceil(max(cols))
    row0, row1 = math.floor(min(rows)), math.ceil(max(rows))
    dw, dh = col1 - col0, row1 - row0
    dst_t = Affine(TARGET_RES, 0, col0 * TARGET_RES, 0, -TARGET_RES, row0 * -TARGET_RES)
    dst = np.zeros((dh, dw), dtype=np.uint8)
    reproject(
        cls,
        dst,
        src_transform=src_t,
        src_crs=src_crs,
        dst_transform=dst_t,
        dst_crs=utm,
        resampling=Resampling.nearest,
    )
    return dst, utm.to_string(), col0, row0


def _tile_windows(dh: int, dw: int) -> list[tuple[int, int, int, int]]:
    """Non-overlapping (r0, c0, h, w) windows of <=64 px; drop <MIN_TILE slivers."""
    out = []
    for r0 in range(0, dh, TILE):
        h = min(TILE, dh - r0)
        if h < MIN_TILE:
            continue
        for c0 in range(0, dw, TILE):
            w = min(TILE, dw - c0)
            if w < MIN_TILE:
                continue
            out.append((r0, c0, h, w))
    return out


def _scan_one(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Reproject a mask and emit one lightweight record per field-containing tile."""
    try:
        cls, src_t, W, H = _load_class_array(task["path"])
        dst, crs_str, col0, row0 = _reproject_to_utm(cls, src_t, W, H)
    except Exception as e:  # noqa: BLE001
        print(f"WARN scan failed {task['path']}: {e}")
        return []
    dh, dw = dst.shape
    recs = []
    for r0, c0, h, w in _tile_windows(dh, dw):
        sub = dst[r0 : r0 + h, c0 : c0 + w]
        present = sorted(int(v) for v in np.unique(sub))
        if not (1 in present or 2 in present):
            continue  # require at least one field pixel
        recs.append(
            {
                "path": task["path"],
                "split": task["split"],
                "file_id": task["file_id"],
                "crs": crs_str,
                "col0": col0,
                "row0": row0,
                "r0": r0,
                "c0": c0,
                "h": h,
                "w": w,
                "classes_present": present,
                "source_id": f"{task['split']}/{task['file_id']}/r{r0}_c{c0}",
            }
        )
    return recs


def _scan_all(workers: int) -> list[dict[str, Any]]:
    cache = io.raw_dir(SLUG) / "scan_cache.pkl"
    if cache.exists():
        print(f"loading cached scan from {cache}")
        with cache.open("rb") as f:
            return pickle.load(f)
    tasks = _mask_paths()
    print(f"scanning {len(tasks)} masks (mp, reproject+tile)")
    all_recs: list[dict[str, Any]] = []
    with multiprocessing.Pool(workers) as p:
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan_one, [dict(task=t) for t in tasks]),
            total=len(tasks),
        ):
            all_recs.extend(recs)
    print(f"scanned {len(all_recs)} field-containing candidate tiles")
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    tmp = io.raw_dir(SLUG) / "scan_cache.pkl.tmp"
    with tmp.open("wb") as f:
        pickle.dump(all_recs, f)
    tmp.rename(cache)
    return all_recs


def _write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    cls, src_t, W, H = _load_class_array(rec["path"])
    dst, crs_str, col0, row0 = _reproject_to_utm(cls, src_t, W, H)
    r0, c0, h, w = rec["r0"], rec["c0"], rec["h"], rec["w"]
    sub = dst[r0 : r0 + h, c0 : c0 + w]
    bounds = (col0 + c0, row0 + r0, col0 + c0 + w, row0 + r0 + h)
    proj = Projection(CRS.from_string(crs_str), TARGET_RES, -TARGET_RES)
    io.write_label_geotiff(SLUG, sample_id, sub, proj, bounds)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(YEAR),
        source_id=rec["source_id"],
        classes_present=sorted(int(v) for v in np.unique(sub)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--write-workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Source: EC JRC AI4Boundaries, JRC Open Data Catalogue (CC BY 4.0 / EC reuse).\n"
            "URL: https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/AI4BOUNDARIES/\n"
            "Downloaded only sentinel2/masks.zip (832 MB, 7598 label masks) + the split CSV\n"
            "ai4boundaries_ftp_urls_sentinel2_split.csv. NOT downloaded: S2 imagery (.nc) or\n"
            "the 1 m aerial orthophotos/masks (pretraining supplies imagery).\n"
            "Masks: 256x256, 10 m, EPSG:3035, 4 float32 bands (extent, boundary, distance,\n"
            "enumeration). We use bands 1-2 to build a 3-class label (0 background, 1 field\n"
            "interior, 2 field boundary). See scan_cache.pkl for scanned tile records.\n"
        )

    records = _scan_all(args.workers)
    selected = sampling.select_tiles_per_class(
        records,
        classes_key="classes_present",
        per_class=PER_CLASS,
        total_cap=sampling.MAX_SAMPLES_PER_DATASET,
    )
    selected.sort(key=lambda r: r["source_id"])
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} tiles (of {len(records)} candidates)")

    with multiprocessing.Pool(args.write_workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            pass

    tile_counts = {i: 0 for i in range(NUM_CLASSES)}
    split_counts: dict[str, int] = {}
    for r in selected:
        for c in r["classes_present"]:
            tile_counts[c] += 1
        split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "EC JRC (Joint Research Centre)",
            "license": "CC BY 4.0 / EC reuse notice",
            "provenance": {
                "url": "https://data.jrc.ec.europa.eu/dataset/0e79ce5d-e4c8-4721-8773-59a4acf2c9c9",
                "have_locally": False,
                "annotation_method": "GSAA (Geospatial Aid Application) parcel declarations, 2019, rasterized",
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
            "split_counts": split_counts,
            "notes": (
                "10 m Sentinel-2 field-boundary masks (bands 1-2: extent, boundary) from EC "
                "JRC AI4Boundaries (7 EU regions, 2019 GSAA). 3-class dense segmentation: "
                "0 background/non-field, 1 field interior, 2 field boundary (boundary wins "
                "over interior). Source masks (256x256, EPSG:3035, 10 m) reprojected to local "
                "UTM at 10 m with NEAREST resampling (preserves 1-px boundaries) and tiled "
                "into <=64x64 windows containing >=1 field pixel. Time range = 1-year 2019 "
                "window (S2 composites Mar-Aug 2019). All three source splits used. "
                "Tiles-per-class balanced to <=1000/class, rarest-first, <=25k total. Caveat: "
                "GSAA parcels can be missing, so background (0) mixes true non-field with "
                "un-declared fields (masked-learning benchmark)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("class tile counts:")
    for i in range(NUM_CLASSES):
        print(f"  {i} {CLASSES[i][0]:16} {tile_counts[i]}")
    print("split counts:", split_counts)
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
