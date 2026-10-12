"""Process WorldFloods v2 into open-set-segmentation label patches.

Source: WorldFloods v2 (Portales-Julia, Mateo-Garcia, Purcell, Gomez-Chova,
"Global flood extent segmentation in optical satellite images", Sci. Reports 13,
20316, 2023). Data on Hugging Face ``isp-uv-es/WorldFloodsv2``. 509 pairs of
Sentinel-2 images and flood segmentation masks curated from Copernicus EMS
rapid-mapping activations over 500+ global flood events, split train/val/test
(475/16/18). Each scene is already in a local UTM projection at 10 m/pixel.

We use ONLY the label rasters (not the 76 GB of S2 imagery):
  * ``{split}/gt/{name}.tif`` (int16, 2 bands, in scene UTM at 10 m):
      band 1 = cloud layer   : 0 invalid, 1 clear, 2 cloud
      band 2 = land/water    : 0 invalid, 1 land,  2 water
  * ``{split}/PERMANENTWATERJRC/{name}.tif`` (int16, 1 band): JRC Global Surface
    Water permanent-water overlay co-registered to the scene; value 3 = permanent
    water (verified: value-3 pixel count == the meta ``pixels permanent water S2``).
  * ``dataset_metadata.csv``: per-scene split, ``s2_date`` (the paired Sentinel-2
    acquisition timestamp), crs, transform, bounds.

Class fusion (dense per-pixel CLASSIFICATION). The manifest lists a combined
``land / water/flood / cloud`` scheme; following the completed **sen1floods11**
precedent (same flood family) we split the water class into flood vs permanent
using the provided JRC layer, giving a 4-class scheme:
    id 0 = flood water      (observed water AND NOT JRC permanent)
    id 1 = permanent water  (observed water AND JRC permanent, value 3)
    id 2 = land             (land/water band == land)
    id 3 = cloud            (cloud band == cloud) -- OVERRIDES land/water, because
                             the S2 image at s2_date shows cloud there, so cloud is
                             the honest label for label<->image alignment.
    255  = nodata/ignore    (both bands invalid / unobserved).
Fusion order per pixel: nodata default; then land, then water (split by JRC);
then cloud overrides where the cloud band flags cloud.

Processing (label_type = dense_raster): each scene raster is ALREADY UTM 10 m
north-up, so no reprojection -- we tile it directly into 64x64 patches, reusing
the scene CRS. Tiles that are >50% nodata are dropped; a tile counts toward a
class only if it holds >= MIN_CLASS_PX px of it. Selection is **tiles-per-class
balanced** (spec 5) via ``sampling.select_tiles_per_class`` (<= 1000 tiles/class,
25k dataset cap; rare classes -- flood/permanent water -- filled first). All
three source splits are used (spec 5).

Time range: flood water is a per-image STATE observed in the specific S2
acquisition (NOT a diffuse yearly change). Per spec 5 (specific-image labels) we
set ``time_range`` to a short ~1-hour window at the ``s2_date`` acquisition and
leave ``change_time = null``. (This deliberately differs from sen1floods11, which
treated the mask as a change label with a year-centered window -- here the task
frames the flood mask as a specific-image observation.)

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.worldfloods_v2
"""

import argparse
import csv
import multiprocessing
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import rasterio
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, sampling

SLUG = "worldfloods_v2"
NAME = "WorldFloods v2"
HF_REPO = "isp-uv-es/WorldFloodsv2"

TILE = 64
PER_CLASS = 1000
MIN_CLASS_PX = 32  # a tile counts toward a class only with >= this many px of it
MAX_NODATA_FRAC = 0.5  # skip tiles that are more than half nodata
JRC_PERMANENT = 3  # PERMANENTWATERJRC value for permanent water

# id -> (name, description). Order mirrors sen1floods11 (flood/permanent/land) + cloud.
CLASSES = [
    (
        "flood water",
        "Surface water observed in the Sentinel-2 acquisition (WorldFloods land/water "
        "band == water) that is NOT JRC permanent water -- the flood inundation.",
    ),
    (
        "permanent water",
        "Observed surface water that the co-registered JRC Global Surface Water overlay "
        "marks as permanent (rivers, lakes, reservoirs).",
    ),
    (
        "land",
        "Land / non-water per the WorldFloods land/water band, where not cloud-covered.",
    ),
    (
        "cloud",
        "Cloud in the Sentinel-2 image (WorldFloods cloud band == cloud); overrides "
        "land/water since the optical surface is obscured at the acquisition.",
    ),
]
FLOOD, PERM, LAND, CLOUD = 0, 1, 2, 3


def raw_root():
    return io.raw_dir(SLUG)


def download_raw() -> list[dict[str, Any]]:
    """Download the label rasters + metadata csv (idempotent); return scene records.

    Only ``gt`` + ``PERMANENTWATERJRC`` rasters (~100 MB total) and the metadata
    csv are pulled -- NOT the 76 GB of Sentinel-2 imagery. Each returned record has
    name, split, s2_date (datetime), crs.
    """
    from huggingface_hub import snapshot_download

    root = raw_root()
    root.mkdir(parents=True, exist_ok=True)
    io.check_disk()
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=root.path,
        allow_patterns=[
            "*/gt/*",
            "*/PERMANENTWATERJRC/*",
            "dataset_metadata.csv",
        ],
    )

    recs: list[dict[str, Any]] = []
    with (root / "dataset_metadata.csv").open() as f:
        for row in csv.DictReader(f):
            s2 = row.get("s2_date")
            if not s2 or s2 == "NaN":
                continue  # cannot date -> skip (none observed in practice)
            dt = datetime.fromisoformat(s2)
            recs.append(
                {
                    "name": row["event id"],
                    "split": row["split"],
                    "s2_date": dt,
                    "crs": row["crs"],
                }
            )
    return recs


def _gt_path(split: str, name: str):
    return raw_root() / split / "gt" / f"{name}.tif"


def _perm_path(split: str, name: str):
    return raw_root() / split / "PERMANENTWATERJRC" / f"{name}.tif"


def _combined_label(split: str, name: str):
    """Return (uint8 4-class array, Projection, col0, row0) for a scene.

    The scene is already UTM 10 m north-up; we reuse its CRS and derive rslearn
    integer pixel bounds directly from the raster transform (origin is a multiple
    of 10, S2-grid aligned). Pixel (col0+j, row0+i) is the top-left of the array.
    """
    with rasterio.open(str(_gt_path(split, name))) as d:
        cloud_band = d.read(1)
        lw_band = d.read(2)
        transform = d.transform
        crs = d.crs
    with rasterio.open(str(_perm_path(split, name))) as d:
        perm = d.read(1)

    out = np.full(lw_band.shape, io.CLASS_NODATA, dtype=np.uint8)
    is_water = lw_band == 2
    out[lw_band == 1] = LAND
    out[is_water & (perm != JRC_PERMANENT)] = FLOOD
    out[is_water & (perm == JRC_PERMANENT)] = PERM
    out[cloud_band == 2] = CLOUD  # cloud overrides observed land/water

    x_res, y_res = transform.a, transform.e  # 10, -10
    proj = Projection(crs, x_res, y_res)
    col0 = int(round(transform.c / x_res))  # geo_x0 / 10
    row0 = int(round(transform.f / y_res))  # geo_y0 / -10  (negative, top row)
    return out, proj, col0, row0


def _scan_scene(name: str, split: str) -> list[dict[str, Any]]:
    """Return one candidate record per non-mostly-nodata 64x64 tile of a scene."""
    arr, _proj, _c0, _r0 = _combined_label(split, name)
    nty, ntx = arr.shape[0] // TILE, arr.shape[1] // TILE
    recs: list[dict[str, Any]] = []
    total_px = TILE * TILE
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
            recs.append(
                {
                    "name": name,
                    "split": split,
                    "ti": ti,
                    "tj": tj,
                    "classes_present": present,
                }
            )
    return recs


def _write_scene(
    name: str, split: str, s2_date: datetime, tiles: list[dict[str, Any]]
) -> None:
    """Tile a scene and write all its selected tiles + sidecars."""
    arr, proj, col0, row0 = _combined_label(split, name)
    # Per-image STATE: ~1-hour window at the acquisition; no change_time (spec 5).
    tr = (s2_date, s2_date + timedelta(hours=1))
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
            source_id=f"{split}/{name}_r{ti}_c{tj}",
            classes_present=present,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    from olmoearth_pretrain.open_set_segmentation_data import manifest

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    print("Downloading WorldFloods v2 label rasters (gt + PERMANENTWATERJRC)...")
    scenes = download_raw()
    date_by_name = {s["name"]: s["s2_date"] for s in scenes}
    split_by_name = {s["name"]: s["split"] for s in scenes}
    print(f"  {len(scenes)} scenes with s2_date")
    io.check_disk()

    print("Scanning scenes into 64x64 tiles...")
    with multiprocessing.Pool(args.workers) as p:
        all_recs: list[dict[str, Any]] = []
        args_list = [dict(name=s["name"], split=s["split"]) for s in scenes]
        for recs in star_imap_unordered(p, _scan_scene, args_list):
            all_recs.extend(recs)
    print(f"  {len(all_recs)} candidate tiles")

    selected = sampling.select_tiles_per_class(
        all_recs, classes_key="classes_present", per_class=PER_CLASS
    )
    selected.sort(key=lambda r: (r["split"], r["name"], r["ti"], r["tj"]))
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(
        f"  selected {len(selected)} tiles (tiles-per-class balanced, <= {PER_CLASS}/class)"
    )

    # Group selected tiles by scene for the write phase.
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for r in selected:
        by_scene.setdefault(r["name"], []).append(r)

    io.check_disk()
    print(f"Writing tiles for {len(by_scene)} scenes...")
    write_args = [
        dict(name=n, split=split_by_name[n], s2_date=date_by_name[n], tiles=ts)
        for n, ts in by_scene.items()
    ]
    with multiprocessing.Pool(args.workers) as p:
        for _ in star_imap_unordered(p, _write_scene, write_args):
            pass

    # Class tile-occurrence counts (a tile counts toward every class it contains).
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
            "source": "Hugging Face isp-uv-es/WorldFloodsv2 (Portales-Julia et al. 2023)",
            "license": "CC-BY-NC-4.0",
            "provenance": {
                "url": "https://huggingface.co/datasets/isp-uv-es/WorldFloodsv2",
                "have_locally": False,
                "annotation_method": "Copernicus EMS rapid-mapping / photointerpretation",
                "citation": "Portales-Julia et al. 2023, Sci. Reports 13:20316; DOI 10.1038/s41598-023-47595-7",
                "splits_used": ["train", "val", "test"],
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
                "509 WorldFloods v2 scenes (Copernicus EMS flood activations paired with "
                "Sentinel-2), each already in a local UTM projection at 10 m; tiled directly "
                "into 64x64 patches (no reprojection). Manifest's combined land/water-flood/cloud "
                "scheme split into flood vs permanent water via the provided JRC permanent-water "
                "overlay (following the sen1floods11 precedent): flood water = observed water not "
                "JRC-permanent; permanent water = observed water that is JRC-permanent (value 3); "
                "land = land/water band land; cloud = cloud band cloud (overrides land/water for "
                "label<->S2 alignment). Tiles-per-class balanced (<=1000/class); rare classes "
                "(flood/permanent water) filled first. Flood water is a per-image STATE: time_range "
                "is a ~1-hour window at the s2_date acquisition and change_time is null (specific-"
                "image label, spec 5) -- unlike sen1floods11 which used a year-centered change label."
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
