"""Process the Wang et al. 2023 rubber maps (SE Asia) into open-set-segmentation patches.

Source (Zenodo record 8425153, CC-BY): "High-resolution maps of rubber and
rubber-related deforestation for Southeast Asia" (Wang et al. 2023, Nature). The archive
``WangEtAl_Nature.zip`` contains two products:

- ``Rubber_10m/`` -- a 10 m binary rubber-plantation map for 2021 (value 1 = rubber,
  0 = non-rubber), tiled as many EPSG:4326 GeoTIFFs (~10 m at the equator). **We use this.**
- ``Deforestation_30m/`` -- a 30 m float layer of rubber-related deforestation. Its values
  are small floats (not clean planting years), the encoding is ambiguous, and the manifest
  classes are only rubber / non-rubber, so this layer is NOT converted. See the summary.

This is a derived-product dense raster, so per the spec we take a BOUNDED set of
spatially-homogeneous / high-confidence windows rather than full coverage:

- rubber class: 64x64 windows located over rubber-rich areas (a coarse 64x-decimated
  average locates blocks whose rubber fraction >= RUBBER_MIN_FRAC).
- non-rubber class: 64x64 windows located over rubber-free areas (decimated fraction == 0)
  that lie within the rubber bounding box of the same source tile, so they are on-land
  landscapes rather than open ocean.

Each selected block is reprojected (nearest resampling; categorical) from EPSG:4326 into a
local UTM 64x64 grid at 10 m and written as a single-band uint8 label patch carrying the
real per-pixel values (0/1). Up to PER_CLASS windows per class, spread across source tiles.
"""

import argparse
import hashlib
import multiprocessing
import os
import random
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import tqdm
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import Window
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io

SLUG = "rubber_rubber_related_deforestation_se_asia"
NAME = "Rubber & Rubber-Related Deforestation (SE Asia)"
ZENODO_RECORD = "8425153"
ZENODO_URL = "https://zenodo.org/records/8425153"
RUBBER_SUBDIR = os.path.join("WangEtAl_Nature", "Rubber_10m")

YEAR = 2021  # rubber map is the 2021 (2021-22 composite) product
PER_CLASS = 1000
TILE = 64  # output patch size and decimation factor (block == candidate window)
PAD = (
    32  # source-pixel padding read around a block so reprojection covers the footprint
)
RUBBER_MIN_FRAC = 0.5  # coarse locator threshold for a rubber-rich block
PER_TILE_CAP = 120  # cap candidates per source tile per class (geographic diversity)
SEED = 42

CLASSES = [
    (
        "non-rubber",
        "Any land/water that is not mapped as rubber plantation (forest, other "
        "crops/plantations, built-up, water, bare, etc.) in the 10 m map.",
    ),
    (
        "rubber",
        "Rubber (Hevea brasiliensis) plantation, as mapped at 10 m for 2021 by Wang "
        "et al. 2023 from Sentinel-1/2 phenology and canopy-cover differencing.",
    ),
]
NON_RUBBER_ID, RUBBER_ID = 0, 1


def _rubber_dir() -> str:
    return os.path.join(io.raw_dir(SLUG).path, RUBBER_SUBDIR)


def _tile_paths() -> list[str]:
    d = _rubber_dir()
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".tif"))


def scan_tile(path: str) -> list[dict[str, Any]]:
    """Locate rubber-rich and rubber-free candidate blocks in one source tile.

    Returns a capped, per-tile-seeded list of candidate records (with lon/lat + class).
    """
    rng = random.Random(int(hashlib.md5(path.encode()).hexdigest(), 16) % (2**32))
    with rasterio.open(path) as ds:
        oh, ow = ds.height // TILE, ds.width // TILE
        if oh < 3 or ow < 3:
            return []
        frac = ds.read(1, out_shape=(oh, ow), resampling=Resampling.average)

        def interior(bi: int, bj: int) -> bool:
            return (
                bi * TILE - PAD >= 0
                and bj * TILE - PAD >= 0
                and bi * TILE + TILE + PAD <= ds.height
                and bj * TILE + TILE + PAD <= ds.width
            )

        rubber = [
            (bi, bj)
            for bi, bj in np.argwhere(frac >= RUBBER_MIN_FRAC).tolist()
            if interior(bi, bj)
        ]
        if not rubber:
            return []
        # Restrict non-rubber to the rubber bounding box so they stay on-land.
        bis = [b for b, _ in rubber]
        bjs = [j for _, j in rubber]
        bi0, bi1, bj0, bj1 = min(bis), max(bis), min(bjs), max(bjs)
        nonrubber = [
            (bi, bj)
            for bi, bj in np.argwhere(frac == 0).tolist()
            if bi0 <= bi <= bi1 and bj0 <= bj <= bj1 and interior(bi, bj)
        ]
        rng.shuffle(rubber)
        rng.shuffle(nonrubber)

        recs: list[dict[str, Any]] = []
        for cls, blocks in (
            (RUBBER_ID, rubber[:PER_TILE_CAP]),
            (NON_RUBBER_ID, nonrubber[:PER_TILE_CAP]),
        ):
            for bi, bj in blocks:
                lon, lat = ds.xy(bi * TILE + TILE // 2, bj * TILE + TILE // 2)
                recs.append(
                    {
                        "path": path,
                        "bi": int(bi),
                        "bj": int(bj),
                        "lon": float(lon),
                        "lat": float(lat),
                        "cls": cls,
                    }
                )
        return recs


def _reproject_block(path: str, bi: int, bj: int, lon: float, lat: float):
    """Read a padded source block and reproject it into a UTM 64x64 10 m grid."""
    with rasterio.open(path) as ds:
        win = Window(bj * TILE - PAD, bi * TILE - PAD, TILE + 2 * PAD, TILE + 2 * PAD)
        src = ds.read(1, window=win, boundless=True, fill_value=0)
        src_transform = ds.window_transform(win)
    proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    dst_transform = Affine(
        proj.x_resolution,
        0,
        bounds[0] * proj.x_resolution,
        0,
        proj.y_resolution,
        bounds[1] * proj.y_resolution,
    )
    dst = np.zeros((TILE, TILE), dtype=np.uint8)
    reproject(
        source=src,
        destination=dst,
        src_transform=src_transform,
        src_crs="EPSG:4326",
        dst_transform=dst_transform,
        dst_crs=proj.crs.to_string(),
        resampling=Resampling.nearest,
    )
    return dst, proj, bounds


def _write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    arr, proj, bounds = _reproject_block(
        rec["path"], rec["bi"], rec["bj"], rec["lon"], rec["lat"]
    )
    classes_present = sorted(int(v) for v in np.unique(arr))
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(YEAR),
        source_id=f"{os.path.basename(rec['path'])}:{rec['bi']}_{rec['bj']}",
        classes_present=classes_present,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            f"Zenodo record {ZENODO_RECORD}: {ZENODO_URL}\n"
            "File: WangEtAl_Nature.zip (extracted). Using WangEtAl_Nature/Rubber_10m/*.tif "
            "(10 m binary rubber map, 2021; EPSG:4326). Deforestation_30m/ not used.\n"
        )

    tiles = _tile_paths()
    print(f"scanning {len(tiles)} rubber source tiles")
    with multiprocessing.Pool(args.workers) as p:
        candidates: list[dict[str, Any]] = []
        for recs in tqdm.tqdm(
            star_imap_unordered(p, scan_tile, [dict(path=t) for t in tiles]),
            total=len(tiles),
        ):
            candidates.extend(recs)

    rng = random.Random(SEED)
    by_cls: dict[int, list[dict[str, Any]]] = {RUBBER_ID: [], NON_RUBBER_ID: []}
    for r in candidates:
        by_cls[r["cls"]].append(r)
    selected: list[dict[str, Any]] = []
    for cls, recs in by_cls.items():
        rng.shuffle(recs)
        selected.extend(recs[:PER_CLASS])
        print(
            f"class {cls}: {len(recs)} candidates -> {min(len(recs), PER_CLASS)} selected"
        )
    rng.shuffle(selected)
    io.check_disk()
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"

    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            pass

    sel_counts = Counter(r["cls"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Zenodo / Nature (Wang et al. 2023)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": ZENODO_URL,
                "have_locally": False,
                "annotation_method": "derived-product (Sentinel-1/2 rubber map, Wang et al. 2023)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {
                "rubber_tiles": sel_counts.get(RUBBER_ID, 0),
                "non_rubber_tiles": sel_counts.get(NON_RUBBER_ID, 0),
            },
            "notes": (
                "Bounded-tile sampling from the 10 m rubber map (2021), reprojected to local "
                "UTM at 10 m (nearest). rubber tiles = rubber-rich windows (also contain "
                "non-rubber pixels); non-rubber tiles = rubber-free windows within each tile's "
                "rubber bbox. Patches carry real per-pixel values (0/1); 1-year time range "
                f"anchored on {YEAR}. Deforestation_30m layer not used (ambiguous encoding)."
            ),
        },
    )
    print(f"done: {len(selected)} samples")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
