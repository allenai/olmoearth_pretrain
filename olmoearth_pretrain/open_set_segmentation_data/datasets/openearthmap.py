"""Process OpenEarthMap into open-set-segmentation patches.

Source: OpenEarthMap (Xia et al., WACV 2023), a benchmark for global high-resolution
land cover mapping. Distributed on Zenodo (record 7223446, ``OpenEarthMap.zip``, 9.1 GB)
under CC-BY-4.0 (label data CC-BY-NC-SA-4.0 for public-domain-image regions). The archive
ships ``OpenEarthMap_wo_xBD/{region}/{images,labels}/{region}_N.tif``: 3500 manually
annotated 8-class land cover masks at 0.25-0.5 m GSD (1024x1024, uint8) over 97 regions in
44 countries across 6 continents. (The "_wo_xBD" archive omits the xBD-sourced RGB images
for licensing, but keeps ALL label masks; we only need the labels — pretraining supplies
its own S2/S1/Landsat imagery.)

Georeferencing (task spec §8.2 gate — PASSED): every label ``.tif`` carries a real CRS +
geotransform. CRS varies by source region (local UTM zones, EPSG:3857 Web Mercator,
EPSG:4326 geographic, national grids like EPSG:31256/custom Transverse-Mercator WKT), all
with genuine real-world coordinates (verified across kagera/accra/chisinau/coxsbazar/
houston/jeremie/pomorskie/rotterdam/santa_rosa/shanghai/vienna). Unlike LoveDA (which
strips coordinates to coordinate-free PNG), OpenEarthMap tiles ARE placeable on the S2 grid.

VHR handling (task spec §4): each 0.25-0.5 m mask (256-512 m footprint) is reprojected from
its native CRS to a local UTM grid at 10 m with **mode** resampling (categorical majority;
never bilinear), yielding one ~26-52x26-52 tile per source tile (all <= 64). Class-set
suitability: OpenEarthMap's 8 classes are already a coarse land-cover scheme, so ALL 8 are
kept. The two finest classes — road (4) and building (8) — are partially unresolvable at
10 m: mode resampling preserves them where they form contiguous majorities (dense urban
blocks, wide highways) but folds isolated buildings and narrow rural roads into the
surrounding class. They are retained (not dropped) and this coarsening is noted; downstream
assembly filters any class that ends up too sparse.

Label value mapping (source -> output): 0 unknown -> 255 nodata; 1 bareland -> 0,
2 rangeland -> 1, 3 developed space -> 2, 4 road -> 3, 5 tree -> 4, 6 water -> 5,
7 agriculture land -> 6, 8 building -> 7.

Time range: the release provides no per-tile acquisition date; imagery spans ~2016-2023
from mixed VHR sources. Land cover is treated as a persistent/static label (task spec §5
static-label rule) and assigned a representative 1-year Sentinel-era window (2020).

Labels are read directly out of the local Zenodo zip (only the ``labels/*.tif`` members are
decoded, never the imagery). Scanned tile records are cached to
``raw/{slug}/scan_cache.pkl`` so re-runs skip the reprojection scan.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.openearthmap
"""

import argparse
import itertools
import math
import multiprocessing
import pickle
import random
import zipfile
from collections import defaultdict
from io import BytesIO
from typing import Any

import numpy as np
import rasterio
import tqdm
from affine import Affine
from pyproj import Transformer
from rasterio.warp import Resampling, reproject
from rslearn.utils.geometry import Projection
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.download import download_zenodo
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    MAX_SAMPLES_PER_DATASET,
)

SLUG = "openearthmap"
NAME = "OpenEarthMap"
ZENODO_RECORD = "7223446"
ZIP_NAME = "OpenEarthMap.zip"
TARGET_RES = 10.0
PER_CLASS = 1000
REPRESENTATIVE_YEAR = 2020

# Output id -> (name, description). Source values 1..8 map to output ids 0..7;
# source value 0 (unknown) -> nodata 255.
CLASSES = [
    (
        "bareland",
        "Bare ground with no vegetation or structures: bare soil, sand, deserts, dry salt "
        "flats, beaches, exposed rock/gravel. OpenEarthMap source class 1.",
    ),
    (
        "rangeland",
        "Low herbaceous vegetation and grass: natural grassland, shrubland, lawns, parks, "
        "meadows. OpenEarthMap source class 2.",
    ),
    (
        "developed space",
        "Human-made non-building, non-road surfaces: parking lots, plazas, industrial yards, "
        "sports fields, cemeteries, artificial bare/developed ground. OpenEarthMap source class 3.",
    ),
    (
        "road",
        "Paved and unpaved transport surfaces: streets, highways, railways, parking aisles, "
        "runways. Fine/narrow roads may be unresolvable at 10 m. OpenEarthMap source class 4.",
    ),
    (
        "tree",
        "Trees and woody vegetation: forest, woodland, orchards, hedgerows, individual large "
        "trees. OpenEarthMap source class 5.",
    ),
    (
        "water",
        "Water bodies: rivers, lakes, reservoirs, ponds, sea, canals, swimming pools. "
        "OpenEarthMap source class 6.",
    ),
    (
        "agriculture land",
        "Cultivated land: cropland, farmland, paddy fields, plantations, greenhouses. "
        "OpenEarthMap source class 7.",
    ),
    (
        "building",
        "Buildings and roofed structures (residential, commercial, industrial). Isolated small "
        "buildings may be unresolvable at 10 m. OpenEarthMap source class 8.",
    ),
]
NUM_CLASSES = len(CLASSES)  # 8

# Source uint8 value -> output id lookup (0 -> nodata; 1..8 -> 0..7).
_LUT = np.full(256, io.CLASS_NODATA, dtype=np.uint8)
for _s in range(1, 9):
    _LUT[_s] = _s - 1


def _zip_path() -> Any:
    return io.raw_dir(SLUG) / ZIP_NAME


# One ZipFile handle per worker process (opening the 9 GB zip re-reads only the central
# directory; member reads are random-access).
_ZIP: zipfile.ZipFile | None = None


def _worker_init() -> None:
    global _ZIP
    _ZIP = zipfile.ZipFile(str(_zip_path()), "r")


def _reproject_mask(
    arr: np.ndarray, src_crs: Any, src_t: Affine, W: int, H: int
) -> tuple | None:
    """Reproject a VHR mask (native CRS) to local UTM 10 m (mode). Returns
    (out_uint8, utm_crs_str, (col0,row0,col1,row1)) or None if degenerate/too large.
    """
    cx = src_t.c + src_t.a * W / 2.0
    cy = src_t.f + src_t.e * H / 2.0
    lon, lat = Transformer.from_crs(src_crs, 4326, always_xy=True).transform(cx, cy)
    if not (np.isfinite(lon) and np.isfinite(lat)):
        return None
    utm = get_utm_ups_projection(lon, lat, TARGET_RES, -TARGET_RES).crs
    to_utm = Transformer.from_crs(src_crs, utm, always_xy=True)
    xs = [src_t.c, src_t.c + src_t.a * W]
    ys = [src_t.f, src_t.f + src_t.e * H]
    pts = [to_utm.transform(X, Y) for X, Y in itertools.product(xs, ys)]
    if not all(np.isfinite(p[0]) and np.isfinite(p[1]) for p in pts):
        return None
    cols = [p[0] / TARGET_RES for p in pts]
    rows = [p[1] / -TARGET_RES for p in pts]
    col0, col1 = math.floor(min(cols)), math.ceil(max(cols))
    row0, row1 = math.floor(min(rows)), math.ceil(max(rows))
    dw, dh = col1 - col0, row1 - row0
    if dw <= 0 or dh <= 0 or dw > io.MAX_TILE or dh > io.MAX_TILE:
        return None
    dst_t = Affine(TARGET_RES, 0, col0 * TARGET_RES, 0, -TARGET_RES, row0 * -TARGET_RES)
    dst = np.zeros((dh, dw), dtype=np.uint8)  # 0 = unknown -> nodata after LUT
    reproject(
        arr,
        dst,
        src_transform=src_t,
        src_crs=src_crs,
        dst_transform=dst_t,
        dst_crs=utm,
        resampling=Resampling.mode,
    )
    out = _LUT[dst]
    return out, utm.to_string(), (col0, row0, col1, row1)


def _scan_member(member: str) -> dict[str, Any] | None:
    """Read one labels/*.tif from the zip, reproject to a 10 m UTM tile, return a record."""
    try:
        data = _ZIP.read(member)  # type: ignore[union-attr]
        with rasterio.open(BytesIO(data)) as ds:
            arr = ds.read(1)
            src_crs = ds.crs
            src_t = ds.transform
            W, H = ds.width, ds.height
    except Exception as e:  # noqa: BLE001
        print(f"WARN read failed {member}: {e}")
        return None
    if src_crs is None:
        print(f"WARN no CRS {member}")
        return None
    res = _reproject_mask(arr, src_crs, src_t, W, H)
    if res is None:
        return None
    out, crs_str, bounds = res
    present = sorted(int(v) for v in np.unique(out) if v != io.CLASS_NODATA)
    if not present:
        return None
    parts = member.split("/")
    region = parts[-3]
    tile = parts[-1][: -len(".tif")]
    return {
        "array": out,
        "crs": crs_str,
        "bounds": bounds,
        "classes_present": present,
        "source_id": f"{region}/{tile}",
    }


def _list_label_members() -> list[str]:
    with zipfile.ZipFile(str(_zip_path()), "r") as z:
        return sorted(n for n in z.namelist() if "/labels/" in n and n.endswith(".tif"))


def _scan_all(workers: int) -> list[dict[str, Any]]:
    cache = io.raw_dir(SLUG) / "scan_cache.pkl"
    if cache.exists():
        print(f"loading cached scan from {cache}")
        with cache.open("rb") as f:
            return pickle.load(f)
    members = _list_label_members()
    print(f"scanning {len(members)} label masks (reproject to 10 m UTM, mode)")
    recs: list[dict[str, Any]] = []
    with multiprocessing.Pool(workers, initializer=_worker_init) as p:
        for r in tqdm.tqdm(
            star_imap_unordered(p, _scan_member, [dict(member=m) for m in members]),
            total=len(members),
        ):
            if r is not None:
                recs.append(r)
    print(f"scanned {len(recs)} non-empty tiles (of {len(members)} masks)")
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    tmp = cache.parent / "scan_cache.pkl.tmp"
    with tmp.open("wb") as f:
        pickle.dump(recs, f)
    tmp.rename(cache)
    return recs


def _select(records: list[dict[str, Any]], seed: int = 42) -> list[dict[str, Any]]:
    """Tiles-per-class balanced: <=PER_CLASS tiles per class, rarest-class-first, total
    capped at MAX_SAMPLES_PER_DATASET. A tile counts toward every class it contains.
    """
    freq: dict[int, int] = defaultdict(int)
    for r in records:
        for c in r["classes_present"]:
            freq[c] += 1
    rng = random.Random(seed)
    order = list(records)
    rng.shuffle(order)
    order.sort(key=lambda r: min(freq[c] for c in r["classes_present"]))
    counts: dict[int, int] = defaultdict(int)
    selected = []
    for r in order:
        if len(selected) >= MAX_SAMPLES_PER_DATASET:
            break
        if any(counts[c] < PER_CLASS for c in r["classes_present"]):
            selected.append(r)
            for c in r["classes_present"]:
                counts[c] += 1
    return selected


def _write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    proj = Projection(rasterio.crs.CRS.from_string(rec["crs"]), TARGET_RES, -TARGET_RES)
    bounds = tuple(rec["bounds"])
    io.write_label_geotiff(
        SLUG, sample_id, rec["array"], proj, bounds, nodata=io.CLASS_NODATA
    )
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(REPRESENTATIVE_YEAR),
        source_id=rec["source_id"],
        classes_present=rec["classes_present"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    if not _zip_path().exists():
        print("downloading OpenEarthMap.zip from Zenodo ...")
        download_zenodo(ZENODO_RECORD, raw, filenames=[ZIP_NAME])

    records = _scan_all(args.workers)
    selected = _select(records)
    selected.sort(key=lambda r: r["source_id"])
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} tiles (of {len(records)} scanned)")

    with multiprocessing.Pool(args.workers) as p:
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
            "source": "Zenodo record 7223446 (OpenEarthMap, Xia et al. WACV 2023)",
            "license": "CC-BY-NC-SA-4.0",
            "provenance": {
                "url": "https://open-earth-map.org/",
                "have_locally": False,
                "annotation_method": "manual photointerpretation of 0.25-0.5 m VHR aerial/satellite imagery",
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
                "VHR (0.25-0.5 m) OpenEarthMap 8-class land-cover masks reprojected from their "
                "native per-region CRS (mixed: local UTM, EPSG:3857, EPSG:4326, national grids) "
                "to local UTM at 10 m with MODE resampling, one ~26-52 px tile per 1024x1024 "
                "source mask. Source values 1..8 -> output 0..7; source 0 (unknown) -> nodata "
                "255. All 8 classes kept; road (3) and building (7) are only partially "
                "resolvable at 10 m (isolated buildings / narrow roads folded into neighbours "
                "by mode). No per-tile acquisition date in the release; land cover treated as "
                "static and assigned a representative 1-year window (2020). All source splits "
                "(train/val/test) used. Tiles-per-class balanced to <=1000/class, rarest-first, "
                "<=25k total. xBD-region RGB imagery omitted from the archive but all label "
                "masks retained (only labels are needed)."
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
