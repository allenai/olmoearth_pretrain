"""Process the SnowCoverage dataset into open-set-segmentation label patches.

Source: Wang, Su, Zhai, Meng, Liu, "Snow Coverage Mapping by Learning from
Sentinel-2 Satellite Multispectral Images via Machine Learning Algorithms",
Remote Sens. 2022, 14(3), 782 (DOI 10.3390/rs14030782). Data on GitHub at
``yiluyucheng/SnowCoverage`` (branch ``main``), CC-BY-4.0. It is the largest
manually-annotated Sentinel-2 snow-segmentation dataset: 40 Sentinel-2 L2A scenes
distributed across six continents (2019-2021), each a ~1000x1000 px crop that was
pixel-labelled in QGIS (Semi-Automatic Classification Plugin) and expert-checked.

We use ONLY the label rasters (``datasets/masks/*.tif``, ~19 MB total), NOT the
~1.3 GB of co-registered Sentinel-2 imagery -- pretraining supplies its own
imagery. Each mask is a single-band int16 GeoTIFF ALREADY in scene-local UTM at
10 m/pixel, north-up (nodata=-999).

Class value encoding in the source masks (values 1/2/3), verified spectrally
against the raw S2 bands (snow: high visible reflectance + very low SWIR B11 +
NDSI~0.85; cloud: high reflectance across all bands incl. SWIR; background: dark,
low reflectance):
    source 1 = background   -> id 0
    source 2 = cloud        -> id 1
    source 3 = snow         -> id 2
    -999 (and a handful of stray 0 px) -> 255 nodata/ignore.
This is dense per-pixel CLASSIFICATION.

Processing (label_type = dense_raster): each mask is already UTM 10 m north-up, so
NO reprojection -- we tile it directly into 64x64 patches, reusing the scene CRS,
and derive rslearn integer pixel bounds from the raster transform (GEE exports are
S2-grid aligned, origins multiples of 10). Tiles that are >50% nodata are dropped;
a tile counts toward a class only if it holds >= MIN_CLASS_PX px of it. Selection is
tiles-per-class balanced (spec 5) via ``sampling.select_tiles_per_class`` (<=1000
tiles/class, 25k dataset cap; rare class -- snow/cloud -- filled first).

Time range: snow / cloud / background are per-image STATES valid only for the exact
Sentinel-2 acquisition (snow is highly time-specific), NOT a diffuse yearly label.
Per spec 5 (specific-image labels) we set ``time_range`` to a ~1-hour window at the
scene's acquisition timestamp (parsed from the product-ID prefix in the filename,
e.g. ``20200804T223709`` = 2020-08-04T22:37:09 UTC) and leave ``change_time=null``.
All scenes are 2019-2021 (post-2016), so no pre-Sentinel filtering is needed.

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.snow_coverage_mapping_sentinel_2_manual
"""

import argparse
import multiprocessing
import urllib.request
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import rasterio
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import (
    download,
    io,
    manifest,
    sampling,
)

SLUG = "snow_coverage_mapping_sentinel_2_manual"
NAME = "Snow Coverage Mapping (Sentinel-2, manual)"

GH_OWNER_REPO = "yiluyucheng/SnowCoverage"
GH_BRANCH = "main"
GH_RAW = f"https://raw.githubusercontent.com/{GH_OWNER_REPO}/{GH_BRANCH}"
GH_TREE_API = (
    f"https://api.github.com/repos/{GH_OWNER_REPO}/git/trees/{GH_BRANCH}?recursive=1"
)

TILE = 64
PER_CLASS = 1000
MIN_CLASS_PX = 32  # a tile counts toward a class only with >= this many px of it
MAX_NODATA_FRAC = 0.5  # skip tiles that are more than half nodata

SRC_NODATA = -999  # int16 nodata sentinel in the source masks

# id -> (name, description). Source mask values 1/2/3 map to ids 0/1/2 (value-1).
CLASSES = [
    (
        "background",
        "Snow- and cloud-free land / dark surface (rock, soil, vegetation, water): the "
        "residual class after snow and cloud, low reflectance across the Sentinel-2 bands. "
        "Source mask value 1.",
    ),
    (
        "cloud",
        "Cloud in the Sentinel-2 acquisition: high reflectance across all bands including "
        "SWIR (B11/B12), which separates it from snow. Source mask value 2.",
    ),
    (
        "snow",
        "Snow cover: high visible reflectance with very low SWIR reflectance (high NDSI), "
        "manually delineated and expert-checked in QGIS. Source mask value 3.",
    ),
]
BACKGROUND, CLOUD, SNOW = 0, 1, 2

# source mask value -> output class id
SRC_TO_ID = {1: BACKGROUND, 2: CLOUD, 3: SNOW}


def raw_masks_dir():
    return io.raw_dir(SLUG) / "masks"


def _list_remote_masks() -> list[str]:
    """Return the repo-relative paths of the mask .tif files from the GitHub tree API."""
    import json

    with urllib.request.urlopen(GH_TREE_API, timeout=120) as r:
        tree = json.loads(r.read())
    return sorted(
        t["path"]
        for t in tree.get("tree", [])
        if t["path"].startswith("datasets/masks/") and t["path"].endswith(".tif")
    )


def download_raw() -> list[str]:
    """Download the 40 label masks (idempotent); return local mask file paths (str).

    Only ``datasets/masks/*.tif`` (~19 MB) are pulled -- NOT the ~1.3 GB of raw S2
    imagery. Falls back to whatever is already present locally if the GitHub tree
    API is unreachable (so a re-run works offline once masks are downloaded).
    """
    d = raw_masks_dir()
    d.mkdir(parents=True, exist_ok=True)
    io.check_disk()
    try:
        rel_paths = _list_remote_masks()
    except Exception as e:  # noqa: BLE001 - transient API/network issue
        print(f"  (GitHub tree API unavailable: {e}; using local masks)")
        rel_paths = []
    for rel in rel_paths:
        fn = rel.split("/")[-1]
        download.download_http(f"{GH_RAW}/{rel}", d / fn, skip_existing=True)
    local = sorted(str(p) for p in d.glob("*.tif"))
    if not local:
        raise RuntimeError("no masks downloaded and none present locally")
    return local


def _acq_time(path: str) -> datetime:
    """Parse the acquisition timestamp from the product-ID prefix of the filename."""
    stem = path.split("/")[-1]
    token = stem.split("_")[0]  # e.g. 20200804T223709
    return datetime.strptime(token, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)


def _load_mask(path: str):
    """Return (uint8 class array with 255 nodata, Projection, col0, row0) for a mask.

    The scene is already UTM 10 m north-up; we reuse its CRS and derive rslearn
    integer pixel bounds directly from the raster transform.
    """
    with rasterio.open(path) as ds:
        raw = ds.read(1)
        transform = ds.transform
        crs = ds.crs

    out = np.full(raw.shape, io.CLASS_NODATA, dtype=np.uint8)
    for src_val, cid in SRC_TO_ID.items():
        out[raw == src_val] = cid
    # everything else (source nodata -999, stray 0 px) stays 255.

    x_res, y_res = transform.a, transform.e  # 10, -10
    proj = Projection(crs, x_res, y_res)
    col0 = int(round(transform.c / x_res))
    row0 = int(round(transform.f / y_res))
    return out, proj, col0, row0


def _scan_scene(path: str) -> list[dict[str, Any]]:
    """Return one candidate record per non-mostly-nodata 64x64 tile of a scene."""
    arr, _proj, _c0, _r0 = _load_mask(path)
    nty, ntx = arr.shape[0] // TILE, arr.shape[1] // TILE
    total_px = TILE * TILE
    recs: list[dict[str, Any]] = []
    for ti in range(nty):
        for tj in range(ntx):
            sub = arr[ti * TILE : (ti + 1) * TILE, tj * TILE : (tj + 1) * TILE]
            u, c = np.unique(sub, return_counts=True)
            counts = {int(k): int(v) for k, v in zip(u, c)}
            if counts.get(io.CLASS_NODATA, 0) > MAX_NODATA_FRAC * total_px:
                continue
            present = [
                cid
                for cid, _ in enumerate(CLASSES)
                if counts.get(cid, 0) >= MIN_CLASS_PX
            ]
            if not present:
                continue
            recs.append({"path": path, "ti": ti, "tj": tj, "classes_present": present})
    return recs


def _write_scene(path: str, tiles: list[dict[str, Any]]) -> None:
    """Tile a scene and write all its selected tiles + sidecars (idempotent)."""
    arr, proj, col0, row0 = _load_mask(path)
    acq = _acq_time(path)
    tr = (acq, acq + timedelta(hours=1))  # per-image state: ~1-hour window
    stem = path.split("/")[-1].replace(".tif", "")
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
            change_time=None,
            source_id=f"{stem}_r{ti}_c{tj}",
            classes_present=present,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    print("Downloading SnowCoverage label masks (masks only, not raw imagery)...")
    paths = download_raw()
    print(f"  {len(paths)} scene masks")
    io.check_disk()

    print("Scanning scenes into 64x64 tiles...")
    with multiprocessing.Pool(args.workers) as p:
        all_recs: list[dict[str, Any]] = []
        for recs in star_imap_unordered(p, _scan_scene, [dict(path=x) for x in paths]):
            all_recs.extend(recs)
    print(f"  {len(all_recs)} candidate tiles")

    selected = sampling.select_tiles_per_class(
        all_recs, classes_key="classes_present", per_class=PER_CLASS
    )
    selected.sort(key=lambda r: (r["path"], r["ti"], r["tj"]))
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(
        f"  selected {len(selected)} tiles "
        f"(tiles-per-class balanced, <= {PER_CLASS}/class, 25k cap)"
    )

    by_scene: dict[str, list[dict[str, Any]]] = {}
    for r in selected:
        by_scene.setdefault(r["path"], []).append(r)

    io.check_disk()
    print(f"Writing tiles for {len(by_scene)} scenes...")
    write_args = [dict(path=pth, tiles=ts) for pth, ts in by_scene.items()]
    with multiprocessing.Pool(args.workers) as p:
        for _ in star_imap_unordered(p, _write_scene, write_args):
            pass

    tile_class_counts = {name: 0 for name, _ in CLASSES}
    for r in selected:
        for c in r["classes_present"]:
            tile_class_counts[CLASSES[c][0]] += 1
    print("tiles containing each class:", tile_class_counts)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "GitHub yiluyucheng/SnowCoverage (Wang et al. 2022, Remote Sens. 14(3):782)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://github.com/yiluyucheng/SnowCoverage",
                "paper_url": "https://www.mdpi.com/2072-4292/14/3/782",
                "have_locally": False,
                "annotation_method": "manual pixel labelling in QGIS (Semi-Automatic Classification Plugin), expert-checked",
                "citation": "Wang, Y.; Su, J.; Zhai, X.; Meng, F.; Liu, C. Remote Sens. 2022, 14, 782. DOI 10.3390/rs14030782",
            },
            "sensors_relevant": ["sentinel2"],
            "classes": [
                {"id": i, "name": nm, "description": desc}
                for i, (nm, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "tile_class_counts": tile_class_counts,
            "notes": (
                "40 manually-annotated Sentinel-2 L2A scenes (~1000x1000 px each) for snow "
                "coverage mapping, spanning 2019-2021 across six continents. Only the label "
                "masks were used (~19 MB); the ~1.3 GB of co-registered S2 imagery was NOT "
                "downloaded. Masks are already in scene-local UTM at 10 m north-up, so they are "
                "tiled directly into 64x64 patches (no reprojection). Source value->class-id: "
                "1->background(0), 2->cloud(1), 3->snow(2); mapping verified spectrally against "
                "the raw S2 bands (snow high-NDSI/low-SWIR, cloud bright across all bands). "
                "Tiles-per-class balanced (<=1000/class, 25k cap); rare class filled first; a "
                "tile counts toward a class with >=32 px of it; tiles >50% nodata dropped. Snow/"
                "cloud/background are per-image STATES: time_range is a ~1-hour window at each "
                "scene's acquisition timestamp (from the product-ID) and change_time is null "
                "(specific-image label, spec 5)."
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
