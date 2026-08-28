"""Process LandCover.ai (Land Cover from Aerial Imagery) into open-set-segmentation patches.

Source: Boguszewski et al., CVPR EarthVision 2021 (linuxpolska), distributed as a single
public HTTP zip ``landcover.ai.v1.zip`` (~1.5 GB) at
https://landcover.ai.linuxpolska.com/. License CC-BY-NC-SA-4.0. The archive holds 41
three-channel RGB orthophotos of Poland (33 at 0.25 m, 8 at 0.5 m; ~216 km^2) under
``images/`` and their matching single-channel land-cover masks under ``masks/`` (uint8,
same georeferencing). We need only the masks — pretraining supplies its own imagery — so
only the small ``masks/*.tif`` members are pulled out of the remote zip via HTTP range
reads (fsspec + zipfile), never the multi-GB imagery.

Mask value scheme (kept as-is, ids already start at 0):
  0 = background, 1 = building, 2 = woodland, 3 = water, 4 = road.

VHR handling (task spec §4): the masks are stored in a WGS84-based Transverse Mercator
(Poland CS92 / PUWG-1992-style) at 0.25/0.5 m. Each whole orthophoto is reprojected to a
local UTM grid at 10 m with **mode** resampling (categorical majority; never bilinear) and
then tiled into <=64x64 (640 m) patches. An out-of-footprint mask (reprojected validity)
marks reprojection fill as nodata (255) so it is not confused with real background (0).

At 10 m the two fine classes are under-resolved: individual buildings (~10-20 m) and roads
(~5-10 m wide) only survive where they dominate a 10 m pixel (dense urban blocks, wide
roads/junctions), so their tile counts are far lower than woodland/water/background. Per
spec §5 we KEEP all five classes anyway (downstream assembly drops classes that end up too
small); the under-resolution is documented in the summary.

Time range: the orthophotos' per-file acquisition dates are not published; the manifest
gives a 2016-2018 window (Sentinel era). We assign a representative static 1-year window
(2017) to every patch (spec §5 static/seasonal-label rule).

Sampling: one record per non-empty 64x64 tile; tiles-per-class balanced to <=1000 tiles
per class, rarest-first, capped at 25,000 total (spec §5). All orthophotos (all source
splits) are used.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.landcover_ai
"""

import argparse
import itertools
import math
import multiprocessing
import zipfile
from typing import Any

import fsspec
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

SLUG = "landcover_ai"
NAME = "LandCover.ai"
ZIP_URL = "https://landcover.ai.linuxpolska.com/download/landcover.ai.v1.zip"
TARGET_RES = 10.0
TILE = 64
PER_CLASS = 1000
REPR_YEAR = (
    2017  # representative Sentinel-era 1-year window (dates not published per file)
)

# Output classes: source mask values are kept unchanged (already 0-based).
CLASSES = [
    (
        "background",
        "None of the four labeled cover types (residual/other surfaces); source mask value 0.",
    ),
    (
        "building",
        "Buildings / any roofed built structure; source mask value 1. Under-resolved at 10 m — "
        "individual buildings only survive where they dominate a 10 m pixel (dense urban).",
    ),
    ("woodland", "Woodlands / forest and tree cover; source mask value 2."),
    ("water", "Water bodies (rivers, lakes, ponds); source mask value 3."),
    (
        "road",
        "Roads; source mask value 4. Under-resolved at 10 m — narrow roads mostly vanish under "
        "mode resampling; survives at wide roads/junctions.",
    ),
]
NUM_CLASSES = len(CLASSES)  # 5


def _open_remote_zip() -> zipfile.ZipFile:
    fs = fsspec.filesystem("http")
    return zipfile.ZipFile(fs.open(ZIP_URL, "rb"))


def _download_masks() -> list[str]:
    """Extract only masks/*.tif from the remote zip to raw_dir (idempotent). Returns paths."""
    raw = io.raw_dir(SLUG)
    mask_dir = raw / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            f"Source: {ZIP_URL} (LandCover.ai v1, CC-BY-NC-SA-4.0).\n"
            "Full archive is ~1.5 GB (RGB orthophotos + masks). Only the small\n"
            "single-channel masks/*.tif members are extracted via HTTP range reads\n"
            "(fsspec+zipfile); the RGB imagery is never downloaded.\n"
            "Masks: uint8, WGS84-based Transverse Mercator (Poland CS92-style), 0.25/0.5 m,\n"
            "values 0=background,1=building,2=woodland,3=water,4=road.\n"
        )
    members = None
    out_paths: list[str] = []
    for_download: list[str] = []
    # First pass: figure out which masks are missing locally without opening the remote zip.
    # We don't know member names until we list, so open lazily only if something is missing.
    # Cheap check: if mask_dir already has 41 tifs, skip remote entirely.
    existing = (
        sorted(p for p in mask_dir.iterdir() if p.name.endswith(".tif"))
        if mask_dir.exists()
        else []
    )
    if len(existing) >= 41:
        return [str(p) for p in existing]

    z = _open_remote_zip()
    members = [n for n in z.namelist() if n.startswith("masks/") and n.endswith(".tif")]
    for m in members:
        name = m.rpartition("/")[2]
        dst = mask_dir / name
        out_paths.append(str(dst))
        if not dst.exists():
            for_download.append(m)
    print(f"downloading {len(for_download)} of {len(members)} masks to {mask_dir}")
    for m in tqdm.tqdm(for_download):
        name = m.rpartition("/")[2]
        dst = mask_dir / name
        data = z.read(m)
        tmp = mask_dir / (name + ".tmp")
        with tmp.open("wb") as f:
            f.write(data)
        tmp.rename(dst)
    return out_paths


def _reproject_to_utm(arr: np.ndarray, src_t: Affine, src_crs: CRS, W: int, H: int):
    """Reproject a VHR mask to local UTM at 10 m (mode). Out-of-footprint -> nodata 255.

    Returns (out_uint8[H10,W10], utm_crs_str, (col0, row0)) — col0/row0 are the integer
    10 m pixel indices of the raster's top-left under the UTM projection.
    """
    cx = src_t.c + src_t.a * W / 2.0
    cy = src_t.f + src_t.e * H / 2.0
    lon, lat = Transformer.from_crs(src_crs, 4326, always_xy=True).transform(cx, cy)
    utm_crs = get_utm_ups_projection(lon, lat, TARGET_RES, -TARGET_RES).crs
    to_utm = Transformer.from_crs(src_crs, utm_crs.to_epsg(), always_xy=True)
    xs = [src_t.c, src_t.c + src_t.a * W]
    ys = [src_t.f, src_t.f + src_t.e * H]
    pts = [to_utm.transform(X, Y) for X, Y in itertools.product(xs, ys)]
    cols = [p[0] / TARGET_RES for p in pts]
    rows = [p[1] / -TARGET_RES for p in pts]
    col0, col1 = math.floor(min(cols)), math.ceil(max(cols))
    row0, row1 = math.floor(min(rows)), math.ceil(max(rows))
    dw, dh = col1 - col0, row1 - row0
    if dw <= 0 or dh <= 0:
        return None
    dst_t = Affine(TARGET_RES, 0, col0 * TARGET_RES, 0, -TARGET_RES, row0 * -TARGET_RES)

    # Class labels: reproject with mode (categorical majority).
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
    # Validity mask so reprojection fill (outside the source footprint) becomes nodata.
    valid_src = np.ones((H, W), dtype=np.uint8)
    valid_dst = np.zeros((dh, dw), dtype=np.uint8)
    reproject(
        valid_src,
        valid_dst,
        src_transform=src_t,
        src_crs=src_crs,
        dst_transform=dst_t,
        dst_crs=utm_crs,
        resampling=Resampling.nearest,
    )
    out = np.where(valid_dst > 0, dst, io.CLASS_NODATA).astype(np.uint8)
    return out, utm_crs.to_string(), (col0, row0)


def _scan_mask(path: str) -> list[dict[str, Any]]:
    """Reproject one mask to 10 m and cut it into <=64x64 tiles; return tile records."""
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        src_t = ds.transform
        src_crs = ds.crs
        W, H = ds.width, ds.height
    res = _reproject_to_utm(arr, src_t, src_crs, W, H)
    if res is None:
        return []
    grid, crs_str, (col0, row0) = res
    gh, gw = grid.shape
    stem = path.rpartition("/")[2][: -len(".tif")]
    recs: list[dict[str, Any]] = []
    for ti in range(math.ceil(gw / TILE)):
        for tj in range(math.ceil(gh / TILE)):
            sub = grid[tj * TILE : (tj + 1) * TILE, ti * TILE : (ti + 1) * TILE]
            # Pad partial edge tiles to exactly TILE x TILE with nodata.
            if sub.shape != (TILE, TILE):
                padded = np.full((TILE, TILE), io.CLASS_NODATA, dtype=np.uint8)
                padded[: sub.shape[0], : sub.shape[1]] = sub
                sub = padded
            present = sorted(int(v) for v in np.unique(sub) if v != io.CLASS_NODATA)
            if not present:
                continue
            bx0 = col0 + ti * TILE
            by0 = row0 + tj * TILE
            recs.append(
                {
                    "array": sub,
                    "crs": crs_str,
                    "bounds": (bx0, by0, bx0 + TILE, by0 + TILE),
                    "classes_present": present,
                    "source_id": f"{stem}/{ti}_{tj}",
                }
            )
    return recs


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
        io.year_range(REPR_YEAR),
        source_id=rec["source_id"],
        classes_present=rec["classes_present"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=41)
    parser.add_argument("--write-workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    mask_paths = _download_masks()
    print(f"{len(mask_paths)} masks available")

    io.check_disk()
    all_recs: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan_mask, [dict(path=mp) for mp in mask_paths]),
            total=len(mask_paths),
        ):
            all_recs.extend(recs)
    print(f"scanned {len(all_recs)} non-empty tiles")

    selected = sampling.select_tiles_per_class(
        all_recs, classes_key="classes_present", per_class=PER_CLASS
    )
    selected.sort(key=lambda r: r["source_id"])
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} tiles (of {len(all_recs)})")

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
            "source": "CVPR EarthVision (Boguszewski et al. 2021) / landcover.ai.linuxpolska.com",
            "license": "CC-BY-NC-SA-4.0",
            "provenance": {
                "url": "https://landcover.ai.linuxpolska.com/",
                "have_locally": False,
                "annotation_method": "manual photointerpretation of 0.25/0.5 m aerial orthophotos",
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
                "VHR 0.25/0.5 m LandCover.ai masks (Poland) reprojected from a WGS84-based "
                "Transverse Mercator to local UTM at 10 m with MODE resampling and tiled into "
                "<=64x64 (640 m) patches; reprojection fill outside each orthophoto footprint "
                "set to nodata 255. Class ids unchanged from source (0=background, 1=building, "
                "2=woodland, 3=water, 4=road). building and road are under-resolved at 10 m "
                "(narrow features mostly lost under mode resampling) but retained per spec §5. "
                f"Representative static 1-year window ({REPR_YEAR}); per-file acquisition dates "
                "not published (manifest range 2016-2018, Sentinel era). Tiles-per-class "
                "balanced to <=1000/class, rarest-first, <=25k total. All orthophotos used."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("class tile counts:")
    for i in range(NUM_CLASSES):
        print(f"  {i:>2} {CLASSES[i][0]:12} {tile_counts[i]}")
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
