"""Process RiverScope into open-set-segmentation label patches.

Source: RiverScope (Ravela / Roy et al., AAAI 2025; UMass CVL), Zenodo record
15376394 (https://zenodo.org/records/15376394), CC-BY-4.0, docs at
https://github.com/cvl-umass/riverscope. RiverScope is a global-scale,
expert-labeled water-segmentation dataset built on PlanetScope imagery (3 m,
BGR+NIR) and co-registered with Sentinel-2, SWORD and SWOT. It ships 1,145 tiles
with predefined train/valid/test splits (splits are spatially disjoint by
location). Each tile has a single-band label GeoTIFF (500x500, 3 m, matching the
PlanetScope crop):

    0 = background (non-water)
    1 = river water
    2 = non-river / other water

This matches the manifest's 3-class scheme exactly. Task is dense per-pixel
**CLASSIFICATION** (river vs other water vs background). RiverScope also carries
SWORD/SWOT node widths for width estimation, but the dense raster we consume is
the water-class mask, so we treat it as classification, not width regression.

Processing (label_type = dense_raster, VHR-native 3 m -> spec §4): each label
GeoTIFF is reprojected once to its local UTM zone at 10 m with **nearest**
resampling (categorical; never bilinear) and cut into 64x64 tiles. A 500x500 3 m
crop (~1.5 km) yields a ~150x150 px 10 m array -> a handful of 64x64 tiles.
Sampling is **tiles-per-class balanced** (spec §5): a tile counts toward every
class present in it (>= MIN_CLASS_PX px); rare classes (river / other water) are
filled first up to PER_CLASS tiles. Background co-occurs in nearly every tile so
it needs no dedicated pass. All three source splits are used (spec §5).

Time range: each label is the water extent at one PlanetScope acquisition. The
PlanetScope id begins with the acquisition date (YYYYMMDD). Water extent is
seasonally variable, so we anchor a 1-year window centered on the acquisition
date (spec §5, seasonal/annual). No change_time (the river channel is a
persistent feature, not a dated change event).

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.riverscope
"""

import argparse
import csv
import multiprocessing
import re
import zipfile
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import rasterio
import shapely
import tqdm
from rasterio.warp import Resampling, reproject
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import STGeometry
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds

from olmoearth_pretrain.open_set_segmentation_data import io, manifest

SLUG = "riverscope"
NAME = "RiverScope"
ZENODO_RECORD = "15376394"

TILE = 64
PER_CLASS = 1000
MIN_CLASS_PX = 32  # a tile counts toward a class only with >= this many px of it
MAX_NODATA_FRAC = 0.5  # skip tiles that are more than half nodata

# label value -> (id, name, description). Source label values already match the
# manifest class order (0 background, 1 river, 2 other water).
CLASSES = [
    ("background", "Background / non-water land surface (RiverScope label value 0)."),
    (
        "river",
        "River water: open water belonging to the labeled river reach (RiverScope "
        "label value 1).",
    ),
    (
        "other water",
        "Non-river open water (lakes, ponds, tributaries, other water bodies within "
        "the tile that are not the target river; RiverScope label value 2).",
    ),
]
BACKGROUND, RIVER, OTHER = 0, 1, 2
DEFAULT_YEAR = 2023  # fallback when a PlanetScope acquisition date can't be parsed


def raw_root():
    return io.raw_dir(SLUG)


def extracted_root():
    return raw_root() / "RiverScope_dataset"


def ensure_extracted() -> None:
    """Extract the label GeoTIFFs + split csvs from RiverScope.zip once (idempotent).

    Only ``PlanetScope/label/**`` and the train/valid/test csvs are unpacked; the 8 GB of
    PlanetScope/Sentinel-2 imagery, SWORD shapefiles and SWOT data are not needed for the
    label patches, so we skip them.
    """
    root = extracted_root()
    if (
        root.exists()
        and (root / "train.csv").exists()
        and (root / "PlanetScope" / "label").exists()
    ):
        return
    zip_path = raw_root() / "RiverScope.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} missing; download it from "
            f"https://zenodo.org/records/{ZENODO_RECORD}/files/RiverScope.zip first."
        )
    print(f"Extracting label rasters + csvs from {zip_path} ...")
    prefix = "RiverScope_dataset/"
    with zipfile.ZipFile(zip_path.path) as zf:
        members = [
            m
            for m in zf.namelist()
            if m.startswith(prefix + "PlanetScope/label/")
            or m in (prefix + "train.csv", prefix + "valid.csv", prefix + "test.csv")
        ]
        zf.extractall(raw_root().path, members=members)


def _find_data_root():
    """Return the directory that directly contains the train/valid/test csv files."""
    root = raw_root()
    # Could be raw/{slug}/RiverScope/ or raw/{slug}/ depending on the zip layout.
    for cand in (extracted_root(), root):
        if cand.exists() and any(cand.glob("*.csv")):
            return cand
    # Fall back to a recursive search for train.csv.
    for p in root.glob("**/train.csv"):
        return p.parent
    raise FileNotFoundError(
        "could not locate RiverScope split csv files after extraction"
    )


def read_records() -> list[dict[str, str]]:
    """Read all split csvs; return list of {label_path, planetscope_id, mid_lon, mid_lat,
    reach_id, split} (absolute label paths).
    """
    data_root = _find_data_root()
    recs: list[dict[str, str]] = []
    for split in ("train", "valid", "test"):
        csv_path = data_root / f"{split}.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r") as f:
            for row in csv.DictReader(f):
                lp = row.get("label_path")
                if not lp:
                    continue
                recs.append(
                    {
                        "label_path": str(data_root / lp),
                        "planetscope_id": row.get("planetscope_id", "") or "",
                        "mid_lon": row.get("mid_lon", ""),
                        "mid_lat": row.get("mid_lat", ""),
                        "reach_id": row.get("reach_id", ""),
                        "split": split,
                    }
                )
    return recs


def _acq_year(planetscope_id: str) -> int:
    """Acquisition year from a PlanetScope id (leading YYYYMMDD); fallback DEFAULT_YEAR."""
    m = re.match(r"(20\d{2})(\d{2})(\d{2})", planetscope_id or "")
    if m:
        y = int(m.group(1))
        if 2016 <= y <= 2026:
            return y
    return DEFAULT_YEAR


def _time_range(
    planetscope_id: str,
) -> tuple[datetime | None, tuple[datetime, datetime]]:
    """1-year window centered on the PlanetScope acquisition date (or year midpoint)."""
    m = re.match(r"(20\d{2})(\d{2})(\d{2})", planetscope_id or "")
    if m:
        try:
            d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=UTC)
            if 2016 <= d.year <= 2026:
                return None, (d - timedelta(days=182), d + timedelta(days=183))
        except ValueError:
            pass
    return None, io.year_range(DEFAULT_YEAR)


def _reproject_label(label_path: str):
    """Reproject a label GeoTIFF to local UTM 10 m; return (arr, proj, col0, row0).

    ``arr`` is (H, W) uint8 with 255 nodata, sized to whole 64-px tiles.
    """
    with rasterio.open(label_path) as d:
        lab = d.read(1)
        src_transform = d.transform
        src_crs = d.crs
        src_bounds = d.bounds

    # Map any source value outside {0,1,2} to nodata (255); keep classes as-is.
    src = np.full(lab.shape, io.CLASS_NODATA, dtype=np.uint8)
    for v in (BACKGROUND, RIVER, OTHER):
        src[lab == v] = v

    # Center lon/lat for the UTM zone.
    cx = (src_bounds.left + src_bounds.right) / 2.0
    cy = (src_bounds.bottom + src_bounds.top) / 2.0
    center = (
        STGeometry(Projection_of(src_crs), shapely.Point(cx, cy), None)
        .to_projection(WGS84_PROJECTION)
        .shp
    )
    proj = io.utm_projection_for_lonlat(center.x, center.y)

    # Project the source footprint into UTM pixel coords to size the output grid.
    box = shapely.box(
        src_bounds.left, src_bounds.bottom, src_bounds.right, src_bounds.top
    )
    utm_box = STGeometry(Projection_of(src_crs), box, None).to_projection(proj).shp
    minx, miny, maxx, maxy = utm_box.bounds
    pad = 2
    col0 = int(np.floor(minx)) - pad
    row0 = int(np.floor(miny)) - pad
    w = int(np.ceil(maxx)) + pad - col0
    h = int(np.ceil(maxy)) + pad - row0
    w = ((w + TILE - 1) // TILE) * TILE
    h = ((h + TILE - 1) // TILE) * TILE
    bounds = (col0, row0, col0 + w, row0 + h)
    dst_transform = get_transform_from_projection_and_bounds(proj, bounds)

    dst = np.full((h, w), io.CLASS_NODATA, dtype=np.uint8)
    reproject(
        source=src,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=proj.crs,
        resampling=Resampling.nearest,
        src_nodata=io.CLASS_NODATA,
        dst_nodata=io.CLASS_NODATA,
    )
    return dst, proj, col0, row0


def Projection_of(crs):
    """Wrap a rasterio CRS as a Projection whose coordinates equal raw CRS units.

    Resolution 1 (not 10) so STGeometry point/box coordinates are the source CRS
    meters/degrees themselves, matching rslearn's WGS84_PROJECTION convention. Used only
    for CRS<->CRS transforms (center lon/lat, footprint -> target-UTM pixel bounds).
    """
    from rslearn.utils.geometry import Projection

    return Projection(crs, 1, 1)


def _tile_counts(arr: np.ndarray, ti: int, tj: int) -> dict[int, int]:
    sub = arr[ti * TILE : (ti + 1) * TILE, tj * TILE : (tj + 1) * TILE]
    u, c = np.unique(sub, return_counts=True)
    return {int(k): int(v) for k, v in zip(u, c)}


def _scan(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one candidate record per non-mostly-nodata 64x64 tile of a label."""
    try:
        arr, _proj, _col0, _row0 = _reproject_label(rec["label_path"])
    except Exception as e:  # noqa: BLE001
        print(f"  skip {rec['label_path']}: {e}")
        return []
    nty, ntx = arr.shape[0] // TILE, arr.shape[1] // TILE
    recs: list[dict[str, Any]] = []
    total_px = TILE * TILE
    for ti in range(nty):
        for tj in range(ntx):
            counts = _tile_counts(arr, ti, tj)
            nodata = counts.get(io.CLASS_NODATA, 0)
            if nodata > MAX_NODATA_FRAC * total_px:
                continue
            count_classes = [
                c
                for c in (BACKGROUND, RIVER, OTHER)
                if counts.get(c, 0) >= MIN_CLASS_PX
            ]
            # Require at least one water class OR skip pure-background-only tiles that
            # add nothing (kept if they still contain background for negatives).
            if not count_classes:
                continue
            recs.append(
                {
                    "label_path": rec["label_path"],
                    "planetscope_id": rec["planetscope_id"],
                    "reach_id": rec["reach_id"],
                    "split": rec["split"],
                    "ti": ti,
                    "tj": tj,
                    "count_classes": count_classes,
                }
            )
    return recs


def _select_tiles_per_class(all_recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tiles-per-class balanced selection (spec §5). Rare classes filled first."""
    import random

    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in all_recs:
        for c in rec["count_classes"]:
            by_class[c].append(rec)
    order = sorted(by_class, key=lambda c: len(by_class[c]))  # rarest first
    rng = random.Random(42)
    selected_keys: set = set()
    selected: list[dict[str, Any]] = []
    counts: dict[int, int] = defaultdict(int)
    for c in order:
        tiles = by_class[c][:]
        rng.shuffle(tiles)
        for rec in tiles:
            if counts[c] >= PER_CLASS:
                break
            key = (rec["label_path"], rec["ti"], rec["tj"])
            if key in selected_keys:
                continue
            selected_keys.add(key)
            selected.append(rec)
            for cc in rec["count_classes"]:
                counts[cc] += 1
    return selected


def _write_label(label_path: str, tiles: list[dict[str, Any]]) -> None:
    """Reproject one label once and write all its selected tiles."""
    arr, proj, col0, row0 = _reproject_label(label_path)
    for t in tiles:
        sample_id = t["sample_id"]
        if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
            continue
        ti, tj = t["ti"], t["tj"]
        sub = arr[ti * TILE : (ti + 1) * TILE, tj * TILE : (tj + 1) * TILE].copy()
        x_min = col0 + tj * TILE
        y_min = row0 + ti * TILE
        bounds = (x_min, y_min, x_min + TILE, y_min + TILE)
        change_time, tr = _time_range(t["planetscope_id"])
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
            change_time=change_time,
            source_id=f"{t['reach_id']}_{t['planetscope_id']}_r{ti}_c{tj}",
            classes_present=present,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    ensure_extracted()
    records = read_records()
    print(f"{len(records)} label tiles across splits")
    io.check_disk()

    print("Scanning labels into 64x64 tiles...")
    with multiprocessing.Pool(args.workers) as p:
        all_recs: list[dict[str, Any]] = []
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan, [dict(rec=r) for r in records]),
            total=len(records),
        ):
            all_recs.extend(recs)
    # Sort deterministically: the pool returns tiles in nondeterministic order, and the
    # class-balanced selection is only seeded-random, so a stable input order is needed for
    # reruns to pick the same tiles (hence assign the same sample ids -> truly idempotent).
    all_recs.sort(key=lambda r: (r["label_path"], r["ti"], r["tj"]))
    print(f"  {len(all_recs)} candidate tiles")

    selected = _select_tiles_per_class(all_recs)
    selected.sort(key=lambda r: (r["label_path"], r["ti"], r["tj"]))
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(
        f"  selected {len(selected)} tiles (tiles-per-class balanced, <= {PER_CLASS}/class)"
    )

    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in selected:
        by_label[r["label_path"]].append(r)

    io.check_disk()
    print(f"Writing tiles for {len(by_label)} labels...")
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(
                p,
                _write_label,
                [dict(label_path=lp, tiles=ts) for lp, ts in by_label.items()],
            ),
            total=len(by_label),
        ):
            pass

    tile_class_counts = {name: 0 for name, _ in CLASSES}
    for r in selected:
        for c in r["count_classes"]:
            tile_class_counts[CLASSES[c][0]] += 1
    print("tiles containing each class:", tile_class_counts)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "RiverScope (UMass CVL); Zenodo 15376394 / AAAI 2025",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": f"https://zenodo.org/records/{ZENODO_RECORD}",
                "code": "https://github.com/cvl-umass/riverscope",
                "have_locally": False,
                "annotation_method": "manual (15 hydrology experts); expert-labeled "
                "water segmentation on PlanetScope, co-registered with Sentinel-2/SWORD/SWOT",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "tile_class_counts": tile_class_counts,
            "notes": (
                "Expert-labeled river/water masks on PlanetScope (3 m), reprojected to "
                "local UTM at 10 m (nearest, categorical) and cut into 64x64 tiles. Source "
                "label values map directly: 0 background, 1 river water, 2 other/non-river "
                "water. Tiles-per-class balanced (<=1000/class); background co-occurs widely. "
                "All train/valid/test splits used. Time range: 1-year window centered on the "
                "PlanetScope acquisition date (from planetscope_id); no change_time. Caveat: "
                "rivers narrower than ~10 m may thin/vanish after resampling from 3 m to 10 m."
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
