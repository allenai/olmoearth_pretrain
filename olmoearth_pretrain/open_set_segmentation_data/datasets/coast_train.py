"""Process Coast Train into open-set-segmentation label patches.

Source: Coast Train (Buscombe et al. 2023, Scientific Data;
https://doi.org/10.5066/P91NP87I, USGS Pacific Coastal and Marine Science
Center), a 1.2-billion-pixel human-labeled library of coastal-environment
imagery + dense per-pixel land-cover labels. Ten data records span five imagery
sources over U.S. Pacific/Gulf/Atlantic/Great-Lakes coasts. Labels were produced
with the Doodler human-in-the-loop tool.

Each record is a ``{source}_{nclasses}_{version}.zip`` of NPZ files, one NPZ per
labeled image. An NPZ holds ``label`` (one-hot HxWxC uint8, channel k == class k
of ``classes``), ``classes`` (the class-name list), the RGB ``image``, etc.
Georeferencing lives in the release-wide ``CoastTrain_imagery_details.csv``:
per-image easting/northing footprint (XMin/XMax/YMin/YMax), ``epsg`` (a
projected UTM CRS), acquisition date, and pixel size. The doodled raster is a
resampled version of the native scene, so the CSV footprint (acc_georef ~8 m) is
the authoritative extent; we build the source affine from it.

Records processed (label_type = dense_raster, satellite + aerial at/near 10-30 m):
  * Sentinel2_11, Sentinel2_4  (native 10 m)      - best fit
  * Landsat8_11,  Landsat8_12  (native ~15-30 m)
  * NAIP_11,      NAIP_6       (1 m aerial, resampled to 10 m; coarse coastal
                                land-cover survives, fine urban detail does not)
Records SKIPPED and why:
  * Orthophoto_8/9/12 (UAS orthomosaics ~0.05 m): footprints are only ~50-100 m
    (5-10 px at 10 m) and the fine coral/sediment/anthropogenic zonation they
    capture is unresolvable at 10 m -> not useful as 10 m tiles.
  * Quadrangles_7 (USGS aerial ~6.8 m): ALL images are 2008/2012 (pre-Sentinel
    era) -> excluded by the >=2016 filter anyway.
Per-image time filter: only acquisition year >= 2016 is kept (Sentinel era);
pre-2016 NAIP/Landsat scenes are dropped.

Unified class scheme (dense per-pixel CLASSIFICATION). Coast Train uses many
per-record class sets; we reconcile them to the paper's physical superclasses,
keeping the six coherent land-cover classes and mapping the non-physical
categories (nodata / cloud / unknown / unusual / generic "other") to 255 ignore:
    0 water                 (water, sediment_plume)
    1 whitewater            (whitewater, surf)
    2 sediment              (sediment, sand, gravel, cobble_boulder, non-veg-wet)
    3 development            (development/dev/developed, buildings, pavement_road,
                             vehicles, people, coastal_defense, other_anthro)
    4 bare_natural_terrain  (other_natural_terrain, bare_ground, non-veg-dry)
    5 vegetation            (vegetated*, agricultural, marsh/terrestrial/
                             herbaceous/woody vegetation)
    255 nodata/ignore       (nodata, cloud, unknown, unusual, other)

Processing: each image's unified label raster is reprojected once to its local
UTM zone at 10 m (nearest resampling - categorical) and cut into 64x64 tiles.
Sampling is tiles-per-class balanced (spec 5): a tile counts toward every class
present in it (>= MIN_CLASS_PX px); rare classes are filled first up to
PER_CLASS tiles.

Time range: land-cover labels tied to a dated scene -> 1-year window centered on
the acquisition date (change_time left null; these are state, not change,
labels). Coastal water/whitewater are ephemeral relative to a yearly window;
noted in the summary.

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.coast_train
"""

import argparse
import csv
import multiprocessing
import random
import zipfile
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import rasterio
import shapely
import tqdm
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject, transform_bounds
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import STGeometry
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds

from olmoearth_pretrain.open_set_segmentation_data import io, manifest

SLUG = "coast_train"
NAME = "Coast Train"

TILE = 64
PER_CLASS = 1000
MIN_CLASS_PX = 32  # a tile counts toward a class only with >= this many px of it
MAX_NODATA_FRAC = 0.5  # skip tiles that are more than half nodata
MIN_YEAR = 2016  # Sentinel era; drop earlier scenes

CSV_NAME = "CoastTrain_imagery_details.csv"

# Records to process (zip basename without .zip) -> extracted dir name.
RECORD_ZIPS = [
    "Sentinel2_11_001",
    "Sentinel2_4_001",
    "Landsat8_11_001",
    "Landsat8_12_001",
    "NAIP_11_001",
    "NAIP_6_001",
]

# Unified 6-class scheme + descriptions.
CLASSES = [
    (
        "water",
        "Open water and water with suspended-sediment plumes (Coast Train 'water', "
        "'sediment_plume').",
    ),
    (
        "whitewater",
        "Wave-breaking whitewater / surf zone foam (Coast Train 'whitewater', 'surf').",
    ),
    (
        "sediment",
        "Unconsolidated coastal sediment - sand, gravel, cobble/boulder, mud, and wet "
        "non-vegetated intertidal flats (Coast Train 'sediment', 'sand', 'gravel', "
        "'cobble_boulder', 'non-vegetated-wet').",
    ),
    (
        "development",
        "Anthropogenic / developed surfaces - buildings, pavement/roads, vehicles, "
        "people, coastal defenses, and generic development (Coast Train 'development', "
        "'dev', 'developed', 'buildings', 'pavement_road', 'vehicles', 'people', "
        "'coastal_defense', 'other_anthro').",
    ),
    (
        "bare_natural_terrain",
        "Bare / non-vegetated natural terrain - bedrock, bare ground, and dry "
        "non-vegetated ground (Coast Train 'other_natural_terrain', 'bare_ground', "
        "'non-vegetated-dry').",
    ),
    (
        "vegetation",
        "Vegetated surfaces - marsh, herbaceous, woody, agricultural and generic "
        "vegetation (Coast Train 'vegetated_surface', 'vegetated', 'vegtated_ground', "
        "'agricultural', 'marsh_vegetation', 'terrestrial_vegetation', "
        "'herbaceous vegetation', 'woody vegetation').",
    ),
]
WATER, WHITEWATER, SEDIMENT, DEVELOPMENT, BARE, VEGETATION = 0, 1, 2, 3, 4, 5
IGNORE = io.CLASS_NODATA  # 255

# Coast Train class-name token -> unified id (or IGNORE). Covers every token that
# appears in any processed record; unmapped/non-physical tokens -> IGNORE.
NAME_TO_ID = {
    "water": WATER,
    "sediment_plume": WATER,
    "whitewater": WHITEWATER,
    "surf": WHITEWATER,
    "sediment": SEDIMENT,
    "sand": SEDIMENT,
    "gravel": SEDIMENT,
    "gravel_shell": SEDIMENT,
    "cobble_boulder": SEDIMENT,
    "mud_silt": SEDIMENT,
    "non-vegetated-wet": SEDIMENT,
    "development": DEVELOPMENT,
    "dev": DEVELOPMENT,
    "developed": DEVELOPMENT,
    "buildings": DEVELOPMENT,
    "pavement_road": DEVELOPMENT,
    "vehicles": DEVELOPMENT,
    "people": DEVELOPMENT,
    "coastal_defense": DEVELOPMENT,
    "other_anthro": DEVELOPMENT,
    "other_natural_terrain": BARE,
    "bare_ground": BARE,
    "bedrock": BARE,
    "non-vegetated-dry": BARE,
    "vegetated_surface": VEGETATION,
    "vegetated": VEGETATION,
    "vegtated_ground": VEGETATION,
    "agricultural": VEGETATION,
    "marsh_vegetation": VEGETATION,
    "terrestrial_vegetation": VEGETATION,
    "herbaceous vegetation": VEGETATION,
    "woody vegetation": VEGETATION,
    # non-physical -> ignore
    "nodata": IGNORE,
    "cloud": IGNORE,
    "unknown": IGNORE,
    "unusual": IGNORE,
    "ice_snow": IGNORE,
    "other": IGNORE,
}


def raw_root():
    return io.raw_dir(SLUG)


def extracted_root():
    return raw_root() / "extracted"


def load_csv_index() -> dict[str, dict[str, str]]:
    """Map basename(annotation_image_filename) -> CSV row."""
    path = raw_root() / CSV_NAME
    with path.open(encoding="latin-1") as f:
        rows = list(csv.DictReader(f))
    index: dict[str, dict[str, str]] = {}
    for r in rows:
        key = r["annotation_image_filename"].split("/")[-1]
        index[key] = r
    return index


def ensure_extracted() -> None:
    """Unzip each processed record into raw/extracted/<record>/ (idempotent)."""
    dst_root = extracted_root()
    dst_root.mkdir(parents=True, exist_ok=True)
    for rec in RECORD_ZIPS:
        out = dst_root / rec
        if out.exists() and any(out.rglob("*.npz")):
            continue
        zpath = raw_root() / f"{rec}.zip"
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath.path) as zf:
            zf.extractall(out.path)


def discover_chips(csv_index: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    """One record per processable NPZ (year >= MIN_YEAR, has CSV georeferencing)."""
    chips: list[dict[str, Any]] = []
    missing_csv = 0
    for rec in RECORD_ZIPS:
        recdir = extracted_root() / rec
        for npz in sorted(recdir.rglob("*.npz")):
            row = csv_index.get(npz.name)
            if row is None:
                missing_csv += 1
                continue
            try:
                year = int(row["year"])
            except (ValueError, KeyError):
                continue
            if year < MIN_YEAR:
                continue
            chips.append(
                {
                    "npz": str(npz),
                    "record": rec,
                    "source_id": npz.name[: -len(".npz")],
                    "row": row,
                }
            )
    if missing_csv:
        print(f"  WARNING: {missing_csv} npz files had no CSV row (skipped)")
    return chips


def _unified_label(npz_path: str) -> np.ndarray:
    """Load an NPZ and return an (H, W) uint8 array in the unified scheme (255 ignore)."""
    d = np.load(npz_path, allow_pickle=True)
    onehot = d["label"]  # (H, W, C) uint8, channel k == classes[k]
    classes = [str(c) for c in d["classes"]]
    idx = onehot.argmax(axis=2)  # channel index == class index in `classes`
    allzero = onehot.max(axis=2) == 0  # unlabeled pixel (defensive; usually none)
    out = np.full(idx.shape, IGNORE, dtype=np.uint8)
    for ci, cname in enumerate(classes):
        if ci >= onehot.shape[2]:
            break
        uid = NAME_TO_ID.get(cname, IGNORE)
        if uid == IGNORE:
            continue
        out[idx == ci] = uid
    out[allzero] = IGNORE
    return out


def _chip_geo(row: dict[str, str]):
    """Return (src_crs, src_transform, proj, center_lonlat) for a chip's footprint."""
    x0, x1 = float(row["XMin"]), float(row["XMax"])
    y0, y1 = float(row["YMin"]), float(row["YMax"])
    epsg = int(row["epsg"])
    src_crs = CRS.from_epsg(epsg)
    lon = (float(row["LonMin"]) + float(row["LonMax"])) / 2.0
    lat = (float(row["LatMin"]) + float(row["LatMax"])) / 2.0
    proj = io.utm_projection_for_lonlat(lon, lat)
    return src_crs, (x0, x1, y0, y1), proj, (lon, lat)


def _reproject_chip(chip: dict[str, Any]):
    """Reproject a chip's unified label to local UTM 10 m.

    Returns (arr, proj, col0, row0): arr is (H, W) uint8 with 255 nodata, sized to
    whole 64-px tiles; pixel (col0+j, row0+i) is the array top-left under proj.
    """
    lab = _unified_label(chip["npz"])
    h, w = lab.shape
    src_crs, (x0, x1, y0, y1), proj, _ = _chip_geo(chip["row"])
    # North-up source affine from the CSV footprint over the actual raster shape.
    src_transform = rasterio.Affine((x1 - x0) / w, 0, x0, 0, -(y1 - y0) / h, y1)

    # Size the output grid: project the footprint (via WGS84) into proj pixel coords.
    lonlat = transform_bounds(src_crs, "EPSG:4326", x0, y0, x1, y1, densify_pts=21)
    box = shapely.box(*lonlat)
    utm_box = STGeometry(WGS84_PROJECTION, box, None).to_projection(proj).shp
    minx, miny, maxx, maxy = utm_box.bounds
    pad = 2
    col0 = int(np.floor(minx)) - pad
    row0 = int(np.floor(miny)) - pad
    out_w = int(np.ceil(maxx)) + pad - col0
    out_h = int(np.ceil(maxy)) + pad - row0
    out_w = ((out_w + TILE - 1) // TILE) * TILE
    out_h = ((out_h + TILE - 1) // TILE) * TILE
    bounds = (col0, row0, col0 + out_w, row0 + out_h)
    dst_transform = get_transform_from_projection_and_bounds(proj, bounds)

    dst = np.full((out_h, out_w), IGNORE, dtype=np.uint8)
    reproject(
        source=lab,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=proj.crs,
        resampling=Resampling.nearest,
        src_nodata=IGNORE,
        dst_nodata=IGNORE,
    )
    return dst, proj, col0, row0


def _scan_chip(chip: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one candidate record per non-mostly-nodata 64x64 tile of a chip."""
    arr, _proj, _col0, _row0 = _reproject_chip(chip)
    nty, ntx = arr.shape[0] // TILE, arr.shape[1] // TILE
    total_px = TILE * TILE
    recs: list[dict[str, Any]] = []
    for ti in range(nty):
        for tj in range(ntx):
            sub = arr[ti * TILE : (ti + 1) * TILE, tj * TILE : (tj + 1) * TILE]
            u, c = np.unique(sub, return_counts=True)
            counts = {int(k): int(v) for k, v in zip(u, c)}
            if counts.get(IGNORE, 0) > MAX_NODATA_FRAC * total_px:
                continue
            count_classes = [
                cid for cid in range(len(CLASSES)) if counts.get(cid, 0) >= MIN_CLASS_PX
            ]
            if not count_classes:
                continue
            recs.append(
                {
                    "npz": chip["npz"],
                    "record": chip["record"],
                    "source_id": chip["source_id"],
                    "row": chip["row"],
                    "ti": ti,
                    "tj": tj,
                    "count_classes": count_classes,
                }
            )
    return recs


def _select_tiles_per_class(all_recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tiles-per-class balanced selection (spec 5). Rare classes filled first."""
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
            key = (rec["source_id"], rec["ti"], rec["tj"])
            if key in selected_keys:
                continue
            selected_keys.add(key)
            selected.append(rec)
            for cc in rec["count_classes"]:
                counts[cc] += 1
    return selected


def _chip_time(row: dict[str, str]):
    """1-year window centered on the acquisition date (change_time = None)."""
    y = int(row["year"])
    m = max(1, min(12, int(row["month"] or 1)))
    day = max(1, min(28, int(row["day"] or 1)))
    d = datetime(y, m, day, tzinfo=UTC)
    return (d - timedelta(days=182), d + timedelta(days=183))


def _write_chip(chip_and_tiles: dict[str, Any]) -> None:
    """Reproject one chip and write all its selected tiles."""
    chip = chip_and_tiles["chip"]
    tiles = chip_and_tiles["tiles"]
    arr, proj, col0, row0 = _reproject_chip(chip)
    tr = _chip_time(chip["row"])
    for t in tiles:
        sample_id = t["sample_id"]
        if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
            continue
        ti, tj = t["ti"], t["tj"]
        sub = arr[ti * TILE : (ti + 1) * TILE, tj * TILE : (tj + 1) * TILE].copy()
        x_min = col0 + tj * TILE
        y_min = row0 + ti * TILE
        bounds = (x_min, y_min, x_min + TILE, y_min + TILE)
        io.write_label_geotiff(SLUG, sample_id, sub, proj, bounds, nodata=IGNORE)
        present = sorted(int(x) for x in np.unique(sub) if x != IGNORE)
        io.write_sample_json(
            SLUG,
            sample_id,
            proj,
            bounds,
            tr,
            change_time=None,
            source_id=f"{chip['source_id']}_r{ti}_c{tj}",
            classes_present=present,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    print("Extracting record archives...")
    ensure_extracted()
    csv_index = load_csv_index()
    chips = discover_chips(csv_index)
    by_rec: dict[str, int] = defaultdict(int)
    for c in chips:
        by_rec[c["record"]] += 1
    print(f"  {len(chips)} processable images (year >= {MIN_YEAR}): {dict(by_rec)}")
    io.check_disk()

    print("Scanning images into 64x64 tiles...")
    with multiprocessing.Pool(args.workers) as p:
        all_recs: list[dict[str, Any]] = []
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan_chip, [dict(chip=c) for c in chips]),
            total=len(chips),
        ):
            all_recs.extend(recs)
    print(f"  {len(all_recs)} candidate tiles")

    # Stable order before selection: the scan Pool returns tiles unordered, so a
    # deterministic sort here makes the seeded shuffle (and thus the selected set
    # and sample-id assignment) reproducible across runs -> idempotent.
    all_recs.sort(key=lambda r: (r["source_id"], r["ti"], r["tj"]))
    selected = _select_tiles_per_class(all_recs)
    selected.sort(key=lambda r: (r["source_id"], r["ti"], r["tj"]))
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(
        f"  selected {len(selected)} tiles (tiles-per-class balanced, <= {PER_CLASS}/class)"
    )

    # Group selected tiles by chip for the write phase.
    chip_by_id = {c["source_id"]: c for c in chips}
    tiles_by_chip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in selected:
        tiles_by_chip[r["source_id"]].append(r)

    io.check_disk()
    print(f"Writing tiles for {len(tiles_by_chip)} images...")
    tasks = [
        dict(chip_and_tiles=dict(chip=chip_by_id[sid], tiles=ts))
        for sid, ts in tiles_by_chip.items()
    ]
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_chip, tasks), total=len(tasks)
        ):
            pass

    # Class tile-occurrence counts (a tile counts toward every class present).
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
            "source": "USGS Pacific Coastal and Marine Science Center (Coast Train)",
            "license": "public domain (U.S. Government work)",
            "provenance": {
                "url": "https://doi.org/10.5066/P91NP87I",
                "paper": "https://www.nature.com/articles/s41597-023-01929-2",
                "have_locally": False,
                "annotation_method": "manual (Doodler human-in-the-loop segmentation)",
                "records_processed": RECORD_ZIPS,
                "records_skipped": {
                    "Orthophoto_8/9/12": "UAS ~0.05 m, footprints ~50-100 m -> "
                    "unresolvable at 10 m",
                    "Quadrangles_7": "all scenes 2008/2012 -> pre-2016, filtered out",
                },
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": IGNORE,
            "num_samples": len(selected),
            "tile_class_counts": tile_class_counts,
            "notes": (
                "Coast Train dense per-pixel coastal land-cover. Six data records "
                "(Sentinel2_11/4, Landsat8_11/12, NAIP_11/6) processed; only scenes "
                f"with acquisition year >= {MIN_YEAR} kept. Each NPZ one-hot label "
                "(channel k == class k of its 'classes' list) argmax'd to a class "
                "index, mapped to a unified 6-class physical scheme (water, "
                "whitewater, sediment, development, bare_natural_terrain, "
                "vegetation); non-physical categories (nodata/cloud/unknown/unusual/"
                "generic 'other') -> 255 ignore. Georeferencing from the release CSV "
                "footprint (XMin/XMax/YMin/YMax, epsg; acc_georef ~8 m); source raster "
                "is a resampled version of the native scene, so the footprint is the "
                "authoritative extent. Reprojected to local UTM 10 m (nearest, "
                "categorical) and cut into 64x64 tiles; tiles-per-class balanced "
                "(<=1000/class). NAIP (1 m) resampled to 10 m - coarse land-cover "
                "survives, fine urban detail lost. Time range: 1-year window centered "
                "on each scene's acquisition date. Judgment calls: sediment_plume->water, "
                "agricultural->vegetation, non-vegetated-wet->sediment, "
                "non-vegetated-dry->bare_natural_terrain, and the 'other' superclass "
                "folded into ignore rather than a noise class."
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
