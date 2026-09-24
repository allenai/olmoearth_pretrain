"""Process GSDP30 (Global Sand Dune Patterns) into open-set-segmentation label patches.

Source: Zhang et al. 2024, ISPRS J. Photogramm. Remote Sens. 218:781-799,
"Global perspectives on sand dune patterns: Scale-adaptable classification using Landsat
imagery and deep learning strategies" (https://zenodo.org/records/13907012, CC-BY-4.0).

GSDP30 is a global 30 m per-pixel classification of aeolian sand-dune-pattern morphology,
produced with a SegFormer deep-learning model on 2017 Landsat-8 surface-reflectance
composites (built on the earlier GSDS30 sand-dune/sheet mask). It is distributed as 331
GeoTIFF tiles (each 15,360 x 15,360 px, EPSG:3857 Web-Mercator, 30 m, uint8, nodata=255)
named by the lon/lat of their upper-left corner. Each tile covers the world's sand seas
(ergs) at ~460 km across.

The product encodes **11 sand-dune-pattern (SDP) classes**. The Zenodo description lists
them in order; the raster carries exactly the 11 values 0..10, and their global pixel
frequencies are geomorphologically self-consistent with that order (linear / network /
sand-sheet surfaces are extensive; dome and star dunes are rare), so we map value == list
index:

    0  simple crescentic dunes
    1  compound-complex crescentic dunes
    2  simple linear dunes
    3  compound-complex linear dunes
    4  dome dunes
    5  star dunes
    6  parabolic dunes
    7  dendritic dunes
    8  network dunes
    9  sand sheets           (by far the most extensive surface; ~87% of mapped pixels)
    10 others
    255 = nodata / outside the mapped sand domain (kept as CLASS_NODATA)

This is a per-pixel **classification** dense_raster. It is a large global derived-product
map, so per the spec we do bounded tiles-per-class-balanced sampling: scan every source
tile in native 30 m blocks (~630 m = a 64 px @ 10 m UTM footprint), keep blocks that
contain a meaningful fraction (>= MIN_FRAC) of one or more SDP classes and are mostly
observed (nodata <= MAX_NODATA_FRAC), then select up to PER_CLASS blocks per class
(rarest-first, a block counts toward every class present) under the 25k cap. Each selected
block is reprojected from native 30 m EPSG:3857 to a local UTM projection at 10 m with
nearest resampling (categorical labels), producing a 64x64 multi-class label patch. Time
range is a 1-year window anchored on 2017 (the Landsat epoch the map was trained on).

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.gsdp30_global_sand_dune_patterns
"""

import argparse
import glob
import multiprocessing
import os
import random
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import tqdm
from pyproj import Transformer
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds

from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    select_tiles_per_class,
)

SLUG = "gsdp30_global_sand_dune_patterns"

YEAR = 2017  # Landsat-8 epoch the GSDP30 map was produced from
TILE = 64  # output patch size (px @ 10 m -> ~640 m footprint)
BLOCK = 21  # native 30 m block ~= 630 m ~= a 64 px @ 10 m UTM tile
MIN_FRAC = 0.10  # a class must occupy >= this fraction of a block to "count"
MAX_NODATA_FRAC = 0.2  # drop blocks that are mostly outside the mapped domain
PER_CLASS = 1000  # target label patches per class
CAP_PER_TILE_PER_CLASS = (
    15  # cap candidates collected per (tile, class) to bound memory
)
N_CLASSES = 11

# Value == class id == index into this list (see module docstring for the mapping rationale).
CLASSES = [
    (
        "simple crescentic dunes",
        "Simple crescentic (barchan / transverse) dunes: single-generation crescent-shaped "
        "dunes with a gentle windward and steep slip-face, formed under a unidirectional wind.",
    ),
    (
        "compound-complex crescentic dunes",
        "Compound and complex crescentic dunes: large crescentic ridges carrying superimposed "
        "smaller dunes of the same or a different type.",
    ),
    (
        "simple linear dunes",
        "Simple linear (longitudinal / seif) dunes: single, roughly parallel elongate ridges "
        "aligned with the resultant wind; the most extensive true dune form globally.",
    ),
    (
        "compound-complex linear dunes",
        "Compound and complex linear dunes: large linear ridges with superimposed secondary "
        "dunes.",
    ),
    (
        "dome dunes",
        "Dome dunes: low, circular to elliptical mounds lacking a well-developed slip-face; "
        "rare and localized.",
    ),
    (
        "star dunes",
        "Star dunes: pyramidal dunes with three or more radiating arms, formed under "
        "multidirectional wind regimes.",
    ),
    (
        "parabolic dunes",
        "Parabolic dunes: U/V-shaped dunes with arms pointing upwind, commonly partly "
        "vegetation-anchored in semi-arid margins.",
    ),
    ("dendritic dunes", "Dendritic dunes: branching, tree-like dune networks."),
    (
        "network dunes",
        "Network (reticulate / honeycomb) dunes: interlocking dune ridges enclosing "
        "cellular interdune areas; extensive in large sand seas.",
    ),
    (
        "sand sheets",
        "Sand sheets: flat to gently undulating sand surfaces without well-developed dune "
        "forms; by far the most extensive aeolian surface in the map.",
    ),
    (
        "others",
        "Other / residual aeolian patterns not assigned to one of the ten named SDP types.",
    ),
]


def source_tiles() -> list[str]:
    return sorted(glob.glob(str(io.raw_dir(SLUG) / "GSDP30" / "*.tif")))


def _scan_tile(path: str) -> list[dict[str, Any]]:
    """Return candidate block records (tile, block idx, lon/lat, classes_present) for one tile.

    A block is BLOCK x BLOCK native (30 m) pixels. A block's ``classes_present`` are the SDP
    class ids occupying >= MIN_FRAC of the block. Blocks with > MAX_NODATA_FRAC nodata, or on
    the outer tile border (to avoid straddling adjacent source tiles when reprojected), are
    dropped. Up to CAP_PER_TILE_PER_CLASS blocks are kept per class per tile (random), then
    unioned, to bound the candidate set while keeping ample coverage for every class.
    """
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        st = ds.transform
        src_crs = ds.crs
    h, w = arr.shape
    nby, nbx = h // BLOCK, w // BLOCK
    a = arr[: nby * BLOCK, : nbx * BLOCK].reshape(nby, BLOCK, nbx, BLOCK)
    denom = float(BLOCK * BLOCK)

    # Per-class fraction per block, plus nodata fraction.
    fracs = np.empty((N_CLASSES, nby, nbx), dtype=np.float32)
    for v in range(N_CLASSES):
        fracs[v] = (a == v).sum(axis=(1, 3)) / denom
    nod = (a == io.CLASS_NODATA).sum(axis=(1, 3)) / denom

    valid = nod <= MAX_NODATA_FRAC
    # Drop the outer border ring of blocks (straddle guard).
    valid[0, :] = valid[-1, :] = False
    valid[:, 0] = valid[:, -1] = False

    rng = random.Random(hash(os.path.basename(path)) & 0xFFFFFFFF)
    keep: set[tuple[int, int]] = set()
    for v in range(N_CLASSES):
        rows, cols = np.nonzero((fracs[v] >= MIN_FRAC) & valid)
        idx = list(zip(rows.tolist(), cols.tolist()))
        if len(idx) > CAP_PER_TILE_PER_CLASS:
            idx = rng.sample(idx, CAP_PER_TILE_PER_CLASS)
        keep.update(idx)

    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    tile_name = os.path.basename(path)[:-4]
    recs: list[dict[str, Any]] = []
    for br, bc in keep:
        classes_present = [v for v in range(N_CLASSES) if fracs[v, br, bc] >= MIN_FRAC]
        if not classes_present:
            continue
        cx = bc * BLOCK + BLOCK / 2.0
        cy = br * BLOCK + BLOCK / 2.0
        x = st.c + cx * st.a
        y = st.f + cy * st.e
        lon, lat = transformer.transform(x, y)
        recs.append(
            {
                "tile": tile_name,
                "path": path,
                "lon": float(lon),
                "lat": float(lat),
                "classes_present": classes_present,
                "source_id": f"{tile_name}_r{br}_c{bc}",
            }
        )
    return recs


def _write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    lon, lat = rec["lon"], rec["lat"]
    dst_proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    dst_transform = get_transform_from_projection_and_bounds(dst_proj, bounds)

    # Geographic bbox of the UTM tile to window the source read.
    xs = [bounds[0] * io.RESOLUTION, bounds[2] * io.RESOLUTION]
    ys = [bounds[1] * -io.RESOLUTION, bounds[3] * -io.RESOLUTION]
    left, right = min(xs), max(xs)
    bottom, top = min(ys), max(ys)
    l2, b2, r2, t2 = transform_bounds(
        dst_proj.crs, "EPSG:4326", left, bottom, right, top
    )
    pad = 0.01  # ~1 km margin so the tile is fully covered before nearest-resampling

    with rasterio.open(rec["path"]) as ds:
        # Source is EPSG:3857; window the read in that CRS.
        wl, wb, wr, wt = transform_bounds(
            "EPSG:4326", ds.crs, l2 - pad, b2 - pad, r2 + pad, t2 + pad
        )
        win = from_bounds(wl, wb, wr, wt, ds.transform)
        src = ds.read(1, window=win, boundless=True, fill_value=io.CLASS_NODATA)
        win_transform = ds.window_transform(win)
        src_crs = ds.crs

    out = np.full((TILE, TILE), io.CLASS_NODATA, np.uint8)
    reproject(
        source=src,
        destination=out,
        src_transform=win_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_proj.crs,
        resampling=Resampling.nearest,
        src_nodata=io.CLASS_NODATA,
        dst_nodata=io.CLASS_NODATA,
    )

    io.write_label_geotiff(
        SLUG, sample_id, out, dst_proj, bounds, nodata=io.CLASS_NODATA
    )
    present = sorted(int(x) for x in np.unique(out) if x != io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        dst_proj,
        bounds,
        io.year_range(YEAR),
        source_id=rec["source_id"],
        classes_present=present,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--scan-workers", type=int, default=32)
    args = parser.parse_args()

    io.check_disk()

    tiles = source_tiles()
    if not tiles:
        raise RuntimeError(f"no source tiles under {io.raw_dir(SLUG)}/GSDP30/")
    print(f"scanning {len(tiles)} source tiles for candidate blocks...")

    all_recs: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.scan_workers) as p:
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan_tile, [dict(path=t) for t in tiles]),
            total=len(tiles),
        ):
            all_recs.extend(recs)
    print(f"collected {len(all_recs)} candidate blocks")

    cand_counts: Counter = Counter()
    for r in all_recs:
        for c in r["classes_present"]:
            cand_counts[c] += 1
    print(
        "candidate blocks per class:",
        {i: cand_counts.get(i, 0) for i in range(N_CLASSES)},
    )

    selected = select_tiles_per_class(
        all_recs, classes_key="classes_present", per_class=PER_CLASS
    )
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} label patches (<= {PER_CLASS}/class, 25k cap)")

    io.check_disk()
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            pass

    # Class counts over selected tiles (a tile counts toward every class present).
    sel_counts: Counter = Counter()
    for r in selected:
        for c in r["classes_present"]:
            sel_counts[c] += 1
    class_counts = {name: sel_counts.get(i, 0) for i, (name, _d) in enumerate(CLASSES)}
    print("selected tiles-per-class:", class_counts)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "GSDP30 (Global Sand Dune Patterns)",
            "task_type": "classification",
            "source": "Zenodo / ISPRS J. Photogramm. Remote Sens.",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://zenodo.org/records/13907012",
                "have_locally": False,
                "annotation_method": "SegFormer deep learning on 2017 Landsat-8 SR composites",
                "citation": (
                    "Zhang et al. 2024, ISPRS J. Photogramm. Remote Sens. 218:781-799, "
                    "doi:10.1016/j.isprsjprs.2024.10.001; data doi:10.5281/zenodo.13907012"
                ),
                "epoch_year": YEAR,
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": class_counts,
            "notes": (
                "Global derived-product dune-morphology map (30 m, EPSG:3857), 11 SDP classes. "
                "Value->class mapping follows the Zenodo description order (value == list index); "
                "frequencies are geomorphologically consistent with that order. Bounded "
                "tiles-per-class-balanced sampling: ~630 m native blocks with >= 10% of a class "
                "and <= 20% nodata, rarest-class-first up to 1000 tiles/class under the 25k cap, "
                "reprojected native 30 m -> local UTM 10 m (nearest, categorical) into 64x64 "
                "multi-class patches. Time range = 1-year window anchored on 2017 (Landsat epoch). "
                "Class 9 (sand sheets) dominates the map (~87% of pixels) and behaves like a "
                "background/extensive-surface class; class 10 (others) is a small residual."
            ),
        },
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
