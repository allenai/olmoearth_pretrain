"""Process Five-Billion-Pixels / GID into open-set-segmentation land-cover patches.

Source: Xin-Yi Tong, Gui-Song Xia, Xiao Xiang Zhu (Wuhan Univ. / TU Munich), ISPRS J.
Photogramm. Remote Sens. 2023. Project page
https://x-ytong.github.io/project/Five-Billion-Pixels.html. Extends the Gaofen Image
Dataset (GID/GID-15). 150 Gaofen-2 (GF-2) multispectral scenes (~4 m, ~6900x7300 px)
over China, each with a per-pixel land-cover annotation in a 24-class system, plus an
"unlabeled" (0) class for miscellaneous/unclear areas. Distributed via Google Drive as
several sibling folders; we use only two of them:
  * ``Annotation__index``     -> ``{scene}_24label.png``  (single-band uint8 class index)
  * ``Coordinate_files``      -> ``{scene}.rpb``          (Rational Polynomial Coeffs)
The 16-bit / 8-bit GF-2 imagery folders are NOT downloaded (pretraining supplies its own
imagery; only the labels + their geolocation are needed).

GEOREFERENCING (the crux). The distributed label masks are plain **PNG** files with no
CRS/geotransform. The authors instead release per-scene ``.rpb`` files carrying the GF-2
**RPC00B** rational-polynomial coefficients ("The coordinate information ... is now
available"). We recover geolocation by attaching the RPC to the label grid and warping to
a local UTM projection at 10 m with **nearest** resampling (categorical labels; never
bilinear) via ``rasterio.warp`` (GDAL's RPC transformer, evaluated at the scene's mean
height ``HEIGHT_OFF`` -- no external DEM). RPC-without-DEM geolocation for near-nadir GF-2
is accurate to ~tens of metres over the mostly-flat annotated regions; this is the sanctioned
metadata-recovery path (task spec triage: "recover geolocation from an accompanying
metadata/RPC/worldfile"). Validated: image centres map to the RPC nominal centre and scene
footprints (~28 km) land correctly in China.

NATIVE 4 m -> 10 m. Each scene is warped straight from its 4 m label grid to the 10 m UTM
grid (nearest), i.e. ~2.5x downsample, then cut into non-overlapping <=64x64 tiles. Tiles
with < 50% labeled pixels are dropped (edge/rotation gaps + unlabeled areas).

CLASSES. Source index 0 = unlabeled -> nodata 255. Source indices 1..24 -> output ids
0..23 (order preserved from the dataset readme). 24 classes, within the 254 uint8 cap.
Some fine urban classes (overpass/railway station/square/stadium) are near the 10 m
resolution limit but are kept per spec (downstream assembly drops too-rare classes).

TIME RANGE. The distribution ships no per-scene acquisition date (only RPCs, no image
metadata XML). FBP GF-2 scenes are ~2015-2020; land cover is quasi-static, so we anchor a
single representative Sentinel-era 1-year window (2018) for every scene and document it.
change_time = null (not a change dataset).

SAMPLING. Tiles-per-class balanced, <=1000 tiles/class, rarest-first, capped at 25,000
total. A tile counts toward every class present in it.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.five_billion_pixels_gid
"""

import argparse
import multiprocessing
import pickle
import re
from typing import Any

import numpy as np
import tqdm
from affine import Affine
from PIL import Image
from rasterio.crs import CRS
from rasterio.rpc import RPC
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rslearn.utils.geometry import Projection
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import (
    download,
    io,
    manifest,
    sampling,
)

Image.MAX_IMAGE_PIXELS = (
    None  # scenes are ~50 MPix; disable PIL decompression-bomb guard
)

SLUG = "five_billion_pixels_gid"
NAME = "Five-Billion-Pixels / GID"

# Google Drive folder ids (from the project page). Listed (not bulk-downloaded) at runtime.
FOLDER_INDEX = "1InbsJG9MC60PsVSLIfjV_CUAfd1ebxzS"  # Annotation__index (*_24label.png)
FOLDER_COORD = "1IWua7zfBTyC5hCusvsUt-dPrgPdtAdvM"  # Coordinate_files (*.rpb)

TARGET_RES = 10.0
TILE = io.MAX_TILE  # 64
PER_CLASS = 1000
MIN_LABELED_FRAC = 0.5  # drop tiles that are mostly nodata/unlabeled
# Representative Sentinel-era window (per-scene dates not distributed; see module docstring).
ANCHOR_YEAR = 2018

# Output class id (0..23) -> (name, description). Source index i (1..24) maps to id i-1.
# Descriptions from the dataset readme + GB/T 21010-2017 land-use taxonomy.
CLASSES = [
    (
        "industrial area",
        "Industrial land: factories, warehouses, mining/processing sites.",
    ),
    (
        "paddy field",
        "Cropland for paddy rice, i.e. periodically flooded/irrigated rice fields.",
    ),
    (
        "irrigated field",
        "Irrigated cropland (non-paddy) with water-supply infrastructure.",
    ),
    ("dry cropland", "Rain-fed dry cropland without irrigation infrastructure."),
    (
        "garden land",
        "Garden/orchard land: perennial horticulture (orchards, tea/mulberry gardens).",
    ),
    (
        "arbor forest",
        "Arbor (tree) forest: closed-canopy woody forest, natural or planted.",
    ),
    ("shrub forest", "Shrub forest / shrubland: woody shrub-dominated vegetation."),
    ("park", "Park: managed urban green space / parkland."),
    (
        "natural meadow",
        "Natural meadow / grassland with predominantly natural herbaceous cover.",
    ),
    (
        "artificial meadow",
        "Artificial meadow: planted/managed grassland (lawns, sown pasture).",
    ),
    ("river", "River: natural flowing watercourses."),
    (
        "urban residential",
        "Urban residential built-up area (dense city housing/blocks).",
    ),
    ("lake", "Lake: natural standing inland water body."),
    ("pond", "Pond: small standing water body."),
    ("fish pond", "Fish pond / aquaculture pond (artificial water for aquaculture)."),
    ("snow", "Snow / ice cover."),
    ("bareland", "Bare land: exposed soil/rock with little or no vegetation."),
    (
        "rural residential",
        "Rural residential built-up area (villages, dispersed dwellings).",
    ),
    ("stadium", "Stadium (large sports venue), a public-service land subclass."),
    ("square", "Public square / plaza (paved open civic space)."),
    ("road", "Road / paved transportation surface."),
    ("overpass", "Overpass / elevated highway interchange."),
    ("railway station", "Railway station (station buildings + platforms/tracks)."),
    ("airport", "Airport: runways, aprons, terminal complexes."),
]
NUM_CLASSES = len(CLASSES)  # 24

# Source uint8 value -> output id. 0 (unlabeled) -> nodata; 1..24 -> 0..23.
_LUT = np.full(256, io.CLASS_NODATA, dtype=np.uint8)
for _s in range(1, 25):
    _LUT[_s] = _s - 1


def parse_rpb(path: str) -> RPC:
    """Parse a GF-2 ``.rpb`` (RPC00B) file into a ``rasterio.rpc.RPC``."""
    txt = open(path).read()

    def val(key: str) -> float:
        return float(re.search(rf"{key}\s*=\s*([-\d.eE+]+)", txt).group(1))

    def arr(key: str) -> list[float]:
        body = re.search(rf"{key}\s*=\s*\(([^)]*)\)", txt, re.S).group(1)
        return [float(x) for x in re.findall(r"[-\d.eE+]+", body)]

    return RPC(
        height_off=val("heightOffset"),
        height_scale=val("heightScale"),
        lat_off=val("latOffset"),
        lat_scale=val("latScale"),
        long_off=val("longOffset"),
        long_scale=val("longScale"),
        line_off=val("lineOffset"),
        line_scale=val("lineScale"),
        samp_off=val("sampOffset"),
        samp_scale=val("sampScale"),
        line_num_coeff=arr("lineNumCoef"),
        line_den_coeff=arr("lineDenCoef"),
        samp_num_coeff=arr("sampNumCoef"),
        samp_den_coeff=arr("sampDenCoef"),
    )


def _reproj_dir():
    return io.raw_dir(SLUG) / "reproj"


def _download_scene(scene: str, png_id: str, rpb_id: str) -> tuple[str, str] | None:
    """Download a scene's label PNG + RPC to raw_dir (atomic, idempotent)."""
    raw = io.raw_dir(SLUG)
    png_dst = raw / "index_png" / f"{scene}_24label.png"
    rpb_dst = raw / "coord" / f"{scene}.rpb"
    try:
        download.download_gdrive_file(rpb_id, rpb_dst)
        download.download_gdrive_file(png_id, png_dst)
    except Exception as e:  # noqa: BLE001 - transient Drive errors; skip this scene
        print(f"WARN download failed {scene}: {e}")
        return None
    return str(png_dst), str(rpb_dst)


def _reproject_scene(scene: str) -> list[dict[str, Any]] | None:
    """Warp one scene's label to UTM 10 m (RPC, nearest), cache the array, return tile records.

    Saves the remapped UTM label array to ``reproj/{scene}.npy`` and returns one record per
    kept <=64x64 tile: {scene, r_off, c_off, col0, row0, crs, classes_present}.
    """
    raw = io.raw_dir(SLUG)
    png = raw / "index_png" / f"{scene}_24label.png"
    rpb = raw / "coord" / f"{scene}.rpb"
    if not (png.exists() and rpb.exists()):
        return None
    npy = _reproj_dir() / f"{scene}.npy"
    meta_p = _reproj_dir() / f"{scene}.meta.pkl"
    if npy.exists() and meta_p.exists():
        with meta_p.open("rb") as f:
            return pickle.load(f)

    try:
        rpc = parse_rpb(rpb.path)
        src = np.array(Image.open(png.path))
    except Exception as e:  # noqa: BLE001
        print(f"WARN read failed {scene}: {e}")
        return None
    H, W = src.shape
    utm = get_utm_ups_projection(rpc.long_off, rpc.lat_off, TARGET_RES, -TARGET_RES).crs

    try:
        dst_t, dw, dh = calculate_default_transform(
            "EPSG:4326",
            utm,
            W,
            H,
            rpcs=rpc,
            resolution=TARGET_RES,
            RPC_HEIGHT=rpc.height_off,
        )
    except Exception as e:  # noqa: BLE001
        print(f"WARN transform failed {scene}: {e}")
        return None
    if dw <= 0 or dh <= 0:
        return None
    # Snap the output grid to integer 10 m rslearn pixels (col*10, row*-10).
    col0 = round(dst_t.c / TARGET_RES)
    row0 = round(dst_t.f / -TARGET_RES)
    snapped = Affine(
        TARGET_RES, 0, col0 * TARGET_RES, 0, -TARGET_RES, row0 * -TARGET_RES
    )

    dst = np.zeros((dh, dw), dtype=np.uint8)  # 0 = unlabeled / outside footprint
    reproject(
        source=src,
        destination=dst,
        rpcs=rpc,
        src_crs="EPSG:4326",
        dst_crs=utm,
        dst_transform=snapped,
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0,
        RPC_HEIGHT=rpc.height_off,
    )
    out = _LUT[dst]  # 0..23 valid, 255 nodata (both unlabeled and out-of-footprint)

    _reproj_dir().mkdir(parents=True, exist_ok=True)
    tmp = _reproj_dir() / f"{scene}.npy.tmp"
    with tmp.open("wb") as f:
        np.save(f, out)
    tmp.rename(npy)

    recs: list[dict[str, Any]] = []
    crs_str = utm.to_string()
    for r_off in range(0, dh - TILE + 1, TILE):
        for c_off in range(0, dw - TILE + 1, TILE):
            tile = out[r_off : r_off + TILE, c_off : c_off + TILE]
            labeled = tile != io.CLASS_NODATA
            if labeled.mean() < MIN_LABELED_FRAC:
                continue
            present = sorted(int(v) for v in np.unique(tile[labeled]))
            if not present:
                continue
            recs.append(
                {
                    "scene": scene,
                    "r_off": r_off,
                    "c_off": c_off,
                    "col0": col0,
                    "row0": row0,
                    "crs": crs_str,
                    "classes_present": present,
                }
            )
    with meta_p.open("wb") as f:
        pickle.dump(recs, f)
    return recs


def _write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    arr = np.load((_reproj_dir() / f"{rec['scene']}.npy").path, mmap_mode="r")
    r_off, c_off = rec["r_off"], rec["c_off"]
    tile = np.ascontiguousarray(arr[r_off : r_off + TILE, c_off : c_off + TILE])
    proj = Projection(CRS.from_string(rec["crs"]), TARGET_RES, -TARGET_RES)
    col0, row0 = rec["col0"], rec["row0"]
    bounds = (col0 + c_off, row0 + r_off, col0 + c_off + TILE, row0 + r_off + TILE)
    io.write_label_geotiff(SLUG, sample_id, tile, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(ANCHOR_YEAR),
        source_id=f"{rec['scene']}/r{r_off}_c{c_off}",
        classes_present=rec["classes_present"],
    )


def _build_file_map() -> dict[str, dict[str, str]]:
    """Scene -> {png_id, rpb_id}, cached to raw_dir/file_ids.pkl (reproducible listing)."""
    cache = io.raw_dir(SLUG) / "file_ids.pkl"
    if cache.exists():
        with cache.open("rb") as f:
            return pickle.load(f)
    png = {}
    for f in download.list_gdrive_folder(FOLDER_INDEX):
        name = f["path"].rsplit("/", 1)[-1]
        if name.endswith("_24label.png"):
            png[name[: -len("_24label.png")]] = f["id"]
    rpb = {}
    for f in download.list_gdrive_folder(FOLDER_COORD):
        name = f["path"].rsplit("/", 1)[-1]
        if name.endswith(".rpb"):
            rpb[name[: -len(".rpb")]] = f["id"]
    fmap = {
        s: {"png_id": png[s], "rpb_id": rpb[s]} for s in sorted(set(png) & set(rpb))
    }
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    tmp = io.raw_dir(SLUG) / "file_ids.pkl.tmp"
    with tmp.open("wb") as f:
        pickle.dump(fmap, f)
    tmp.rename(cache)
    return fmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dl-workers", type=int, default=32)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--write-workers", type=int, default=64)
    parser.add_argument("--max-scenes", type=int, default=0, help="0 = all scenes")
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Source: Five-Billion-Pixels / GID (Tong, Xia, Zhu; ISPRS J. 2023).\n"
            "https://x-ytong.github.io/project/Five-Billion-Pixels.html\n"
            "Downloaded (selectively, via public Google Drive): the Annotation__index\n"
            "class-index PNGs and the Coordinate_files .rpb (RPC00B) files. The GF-2\n"
            "16-bit/8-bit imagery folders are NOT downloaded (labels + RPC geolocation only).\n"
            "License: free for research use.\n"
        )

    fmap = _build_file_map()
    scenes = sorted(fmap)
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]
    print(f"{len(scenes)} scenes with matching PNG+RPC")

    # Phase A: download label PNGs + RPCs.
    dl_args = [
        dict(scene=s, png_id=fmap[s]["png_id"], rpb_id=fmap[s]["rpb_id"])
        for s in scenes
    ]
    ok = 0
    with multiprocessing.Pool(args.dl_workers) as p:
        for r in tqdm.tqdm(
            star_imap_unordered(p, _download_scene, dl_args),
            total=len(dl_args),
            desc="download",
        ):
            ok += r is not None
    print(f"downloaded {ok}/{len(scenes)} scenes")
    io.check_disk()

    # Phase B: reproject each scene to UTM 10 m and enumerate tiles.
    all_recs: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _reproject_scene, [dict(scene=s) for s in scenes]),
            total=len(scenes),
            desc="reproject",
        ):
            if recs:
                all_recs.extend(recs)
    print(f"enumerated {len(all_recs)} candidate tiles from {len(scenes)} scenes")

    # Phase C: tiles-per-class balanced selection (<=1000/class, <=25k total, rarest-first).
    selected = sampling.select_tiles_per_class(
        all_recs,
        classes_key="classes_present",
        per_class=PER_CLASS,
        total_cap=sampling.MAX_SAMPLES_PER_DATASET,
    )
    selected.sort(key=lambda r: (r["scene"], r["r_off"], r["c_off"]))
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} tiles")

    # Phase D: write GeoTIFFs + sidecar JSON.
    with multiprocessing.Pool(args.write_workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write",
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
            "source": "Five-Billion-Pixels / GID (Tong et al., ISPRS J. 2023)",
            "license": "free for research use",
            "provenance": {
                "url": "https://x-ytong.github.io/project/Five-Billion-Pixels.html",
                "have_locally": False,
                "annotation_method": "manual photointerpretation of 4 m Gaofen-2 imagery",
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
                "GF-2 4 m land-cover labels distributed as non-georeferenced PNG class-index "
                "masks + per-scene RPC (.rpb) coordinate files. Geolocation recovered by RPC "
                "warp (GDAL, nearest, at mean height HEIGHT_OFF, no external DEM) to local UTM "
                "at 10 m, then cut into <=64x64 tiles (tiles <50% labeled dropped). Source "
                "index 0=unlabeled -> nodata 255; source 1..24 -> ids 0..23. Per-scene "
                f"acquisition dates are not distributed; anchored a representative {ANCHOR_YEAR} "
                "Sentinel-era 1-year window (land cover is quasi-static). Tiles-per-class "
                "balanced to <=1000/class, rarest-first, capped at 25k. RPC-without-DEM "
                "geolocation is accurate to ~tens of metres for near-nadir GF-2."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("class tile counts:")
    for i in range(NUM_CLASSES):
        print(f"  {i:>2} {CLASSES[i][0]:20} {tile_counts[i]}")
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
