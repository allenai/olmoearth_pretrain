"""Process the NCCM (Northeast China Crop Map) into open-set-segmentation label patches.

Source: You et al. (2021), "The 10-m crop type maps in Northeast China during 2017-2019"
(Scientific Data, https://doi.org/10.1038/s41597-021-00827-9; data DOI
10.5281/zenodo.8175171; also distributed as the TorchGeo ``NCCM`` dataset). It is an annual
10 m crop-type MAP for Northeast China (2017, 2018, 2019) produced by hierarchical
random-forest classification of interpolated/smoothed 10-day Sentinel-2 time series. It is a
derived-product map validated against ground-truth reference samples, so per the spec
(§4 dense_raster; §5 large derived-product) we do tiles-per-class balanced sampling and
prefer spatially-homogeneous / high-confidence windows.

Raw files (cached on weka, ~800 MB each, NOT re-downloaded here): ``CDL2017_clip.tif``,
``CDL2018_clip1.tif``, ``CDL2019_clip.tif`` under ``raw/nccm_northeast_china_crop_map/``.
Each is a single-band uint8 GeoTIFF in EPSG:4326 at ~8.98e-5 deg/pixel (~10 m), covering
lon 115.5-135.0 E, lat 38.7-53.5 N (216985 x 164926 px). Nodata = 15.

Native class codes (from the NCCM product / TorchGeo legend):
    0 = paddy rice, 1 = maize, 2 = soybean, 3 = others crops and lands, 15 = nodata.
We remap to a compact uint8 id space aligned with the manifest class order
(maize, soybean, rice, other):
    code 1 (maize)   -> id 0
    code 2 (soybean) -> id 1
    code 0 (rice)    -> id 2
    code 3 (other)   -> id 3
    code 15          -> nodata (255)

Sampling: the source raster is scanned in parallel over SUPER x SUPER native super-windows;
each is subdivided into BLOCK x BLOCK (~64 px, ~one 640 m UTM tile footprint) native blocks.
A block is a candidate only if it is well-observed (>= VALID_FRAC_MIN of pixels are not
nodata) and a crop/other class covers >= PRESENT_FRAC of the valid pixels (high-confidence /
homogeneous preference). Candidate class ids present in each block feed tiles-per-class
balanced selection (rarest class first) up to 1000 tiles/class, capped at the 25,000-sample
per-dataset limit (well under it here with 4 classes). Each selected block is reprojected
from EPSG:4326 to a local UTM projection at 10 m with NEAREST resampling (categorical) into
a 64x64 uint8 tile. Time range = the 1-year window of the block's map year.

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.nccm_northeast_china_crop_map
"""

import argparse
import multiprocessing
import random
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import rasterio
import tqdm
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds

from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    select_tiles_per_class,
)

SLUG = "nccm_northeast_china_crop_map"

RAW_FILES: dict[int, str] = {
    2017: "CDL2017_clip.tif",
    2018: "CDL2018_clip1.tif",
    2019: "CDL2019_clip.tif",
}
YEARS = [2017, 2018, 2019]

TILE = 64  # output UTM tile (10 m) side
BLOCK = 64  # native-block side scanned for composition (~one tile footprint)
SUPER = 4096  # native super-window read per scan task
VALID_FRAC_MIN = 0.7  # block must be >= 70% observed (not nodata) to be a candidate
PRESENT_FRAC = 0.25  # a class counts as present if >= 25% of the valid pixels
PER_CLASS = 1000  # tiles-per-class target
KEEP_PER_CLASS_PER_TASK = 16  # per-scan-task cap per class (bounds candidate memory)

NODATA_CODE = 15  # native fill value in the source rasters

# Native NCCM code -> compact uint8 id (manifest order: maize, soybean, rice, other).
CODE_TO_ID: dict[int, int] = {1: 0, 2: 1, 0: 2, 3: 3}
CLASS_NAMES: dict[int, str] = {0: "maize", 1: "soybean", 2: "rice", 3: "other"}
CLASS_DESC: dict[int, str] = {
    0: "Maize (corn) cropland (native NCCM code 1).",
    1: "Soybean cropland (native NCCM code 2).",
    2: "Paddy rice cropland (native NCCM code 0).",
    3: (
        "Others crops and lands: the residual class of the NCCM product covering all "
        "non-(maize/soybean/rice) crops and non-cropland land cover (native NCCM code 3)."
    ),
}


def raw_path(year: int):
    return io.raw_dir(SLUG) / RAW_FILES[year]


def _build_lut() -> np.ndarray:
    """256-entry LUT: raw NCCM code -> compact id, everything else -> CLASS_NODATA."""
    lut = np.full(256, io.CLASS_NODATA, dtype=np.uint8)
    for code, cid in CODE_TO_ID.items():
        lut[code] = cid
    return lut


def _scan_super(year: int, row_off: int, col_off: int) -> list[dict[str, Any]]:
    """Scan a native super-window; return candidate BLOCKxBLOCK-block records.

    Each record has the block-center lon/lat, year, the compact class ids present
    (>= PRESENT_FRAC of the valid pixels), and a source id. Well-observed
    (>= VALID_FRAC_MIN valid) blocks only. A per-class cap bounds the returned count.
    """
    path = str(raw_path(year))
    with rasterio.open(path) as ds:
        w = min(SUPER, ds.width - col_off)
        h = min(SUPER, ds.height - row_off)
        if w <= 0 or h <= 0:
            return []
        win = Window(col_off, row_off, w, h)
        arr = ds.read(1, window=win)
        tf = ds.window_transform(win)

    nby, nbx = h // BLOCK, w // BLOCK
    if nby == 0 or nbx == 0:
        return []
    a = arr[: nby * BLOCK, : nbx * BLOCK]
    # (nby, nbx, BLOCK*BLOCK)
    blocks = (
        a.reshape(nby, BLOCK, nbx, BLOCK)
        .transpose(0, 2, 1, 3)
        .reshape(nby * nbx, BLOCK * BLOCK)
    )
    nvalid = (blocks != NODATA_CODE).sum(axis=1).astype(np.float64)
    denom = float(BLOCK * BLOCK)
    min_valid = VALID_FRAC_MIN * denom
    # Per-class present counts (only the 4 real classes).
    code_counts = {code: (blocks == code).sum(axis=1) for code in CODE_TO_ID}

    cand = np.nonzero(nvalid >= min_valid)[0]
    rng = random.Random((year * 1_000_003 + row_off * 131 + col_off) & 0xFFFFFFFF)
    order = list(cand)
    rng.shuffle(order)

    kept_per_class: dict[int, int] = defaultdict(int)
    recs: list[dict[str, Any]] = []
    for idx in order:
        nv = nvalid[idx]
        present = [
            CODE_TO_ID[code]
            for code in CODE_TO_ID
            if code_counts[code][idx] / nv >= PRESENT_FRAC
        ]
        if not present:
            continue
        if all(kept_per_class[c] >= KEEP_PER_CLASS_PER_TASK for c in present):
            continue
        for c in present:
            kept_per_class[c] += 1
        br, bc = int(idx // nbx), int(idx % nbx)
        lon = tf.c + (bc * BLOCK + BLOCK / 2.0) * tf.a
        lat = tf.f + (br * BLOCK + BLOCK / 2.0) * tf.e
        recs.append(
            {
                "year": year,
                "lon": float(lon),
                "lat": float(lat),
                "present_ids": sorted(present),
                "source_id": f"{year}_r{row_off + br * BLOCK}_c{col_off + bc * BLOCK}",
            }
        )
        # Stop early once every class in this task has hit the cap.
        if all(
            kept_per_class[c] >= KEEP_PER_CLASS_PER_TASK for c in CODE_TO_ID.values()
        ):
            break
    return recs


def _write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    lon, lat = rec["lon"], rec["lat"]
    dst_proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    dst_transform = get_transform_from_projection_and_bounds(dst_proj, bounds)

    xs = [bounds[0] * io.RESOLUTION, bounds[2] * io.RESOLUTION]
    ys = [bounds[1] * -io.RESOLUTION, bounds[3] * -io.RESOLUTION]
    left, right = min(xs), max(xs)
    bottom, top = min(ys), max(ys)
    # Source is EPSG:4326 (lon/lat degrees).
    l2, b2, r2, t2 = transform_bounds(
        dst_proj.crs, "EPSG:4326", left, bottom, right, top
    )
    pad = 0.002  # ~200 m / ~20 native px margin so the tile is fully covered

    with rasterio.open(str(raw_path(rec["year"]))) as ds:
        win = from_bounds(l2 - pad, b2 - pad, r2 + pad, t2 + pad, ds.transform)
        src = ds.read(1, window=win, boundless=True, fill_value=NODATA_CODE)
        win_transform = ds.window_transform(win)
        src_crs = ds.crs

    dst = np.full((TILE, TILE), NODATA_CODE, np.uint8)
    reproject(
        source=src,
        destination=dst,
        src_transform=win_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_proj.crs,
        resampling=Resampling.nearest,
        src_nodata=NODATA_CODE,
        dst_nodata=NODATA_CODE,
    )
    out = _build_lut()[dst]  # remap native codes -> compact ids; nodata/unmapped -> 255

    io.write_label_geotiff(
        SLUG, sample_id, out, dst_proj, bounds, nodata=io.CLASS_NODATA
    )
    present = sorted(int(x) for x in np.unique(out) if x != io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        dst_proj,
        bounds,
        io.year_range(rec["year"]),
        source_id=rec["source_id"],
        classes_present=present,
    )


def _scan_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for year in YEARS:
        with rasterio.open(str(raw_path(year))) as ds:
            width, height = ds.width, ds.height
        for row_off in range(0, height, SUPER):
            for col_off in range(0, width, SUPER):
                tasks.append(dict(year=year, row_off=row_off, col_off=col_off))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()

    for year in YEARS:
        if not raw_path(year).exists():
            raise RuntimeError(f"missing cached raw file: {raw_path(year)}")

    tasks = _scan_tasks()
    print(f"scanning {len(tasks)} super-windows across {len(YEARS)} years...")
    all_recs: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan_super, tasks), total=len(tasks)
        ):
            all_recs.extend(recs)
    print(f"scanned {len(all_recs)} candidate windows")
    io.check_disk()

    # Candidate frequency by class (windows containing the class).
    cand_freq: Counter = Counter()
    for r in all_recs:
        for c in set(r["present_ids"]):
            cand_freq[c] += 1
    print(
        "candidate tiles per class: "
        + ", ".join(
            f"{CLASS_NAMES[c]}={cand_freq.get(c, 0)}" for c in sorted(CLASS_NAMES)
        )
    )

    selected = select_tiles_per_class(
        all_recs, classes_key="present_ids", per_class=PER_CLASS
    )
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(
        f"selected {len(selected)} windows (tiles-per-class, <= {PER_CLASS}/class, 25k cap)"
    )

    io.check_disk()
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            pass

    # Tiles-per-class counts among the selected windows.
    class_counts: dict[str, int] = defaultdict(int)
    year_counts: dict[int, int] = defaultdict(int)
    for r in selected:
        year_counts[r["year"]] += 1
        for cid in r["present_ids"]:
            class_counts[CLASS_NAMES[cid]] += 1
    print("selected tiles per class:", dict(class_counts))
    print("selected tiles per year:", dict(year_counts))

    classes_meta = [
        {
            "id": cid,
            "name": CLASS_NAMES[cid],
            "native_code": code,
            "description": CLASS_DESC[cid],
        }
        for code, cid in sorted(CODE_TO_ID.items(), key=lambda kv: kv[1])
    ]

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "NCCM (Northeast China Crop Map)",
            "task_type": "classification",
            "source": "TorchGeo / journal (You et al. 2021, Scientific Data)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.org/10.5281/zenodo.8175171",
                "paper": "https://doi.org/10.1038/s41597-021-00827-9",
                "have_locally": False,
                "annotation_method": (
                    "derived-product map (hierarchical random-forest classification of "
                    "Sentinel-2 time series) validated with field/reference samples"
                ),
                "years": YEARS,
                "raw_files": RAW_FILES,
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes_meta,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": dict(class_counts),
            "year_counts": {str(k): v for k, v in year_counts.items()},
            "notes": (
                "Tiles-per-class balanced sampling of the NCCM 10 m crop-type map for "
                "Northeast China (2017/2018/2019). The source rasters are EPSG:4326 uint8 "
                "(native codes 0=rice, 1=maize, 2=soybean, 3=others crops and lands, "
                "15=nodata), remapped to compact ids (maize=0, soybean=1, rice=2, other=3; "
                "native_code recorded per class). Non-overlapping ~64 px (640 m) native "
                f"blocks were scanned; a block is kept only if >= {int(VALID_FRAC_MIN * 100)}% "
                f"observed and a class covers >= {int(PRESENT_FRAC * 100)}% of valid pixels "
                "(high-confidence/homogeneous preference). Windows were selected "
                "tiles-per-class (rarest first) up to 1000/class under the 25k cap. Each "
                "native block was reprojected from EPSG:4326 to local UTM at 10 m with "
                "nearest resampling. Time range = the 1-year window of the block's map year. "
                "'other' (code 3) is the product's residual class and is present in most "
                "tiles; rice is the rarest crop."
            ),
        },
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
