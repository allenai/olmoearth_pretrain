"""Process GABAM (Global Annual Burned Area Map) into open-set-segmentation labels.

Source: GABAM - "Updated 30 m resolution global annual burned area map, 2014-2024"
(Long et al.; Aerospace Information Research Institute, CAS), distributed on Zenodo
(record 13858799, CC-BY). The product is derived from Landsat surface reflectance and
maps, for each calendar year, whether each 30 m pixel burned at least once during that
year. Each year is a ZIP of ~1000 GeoTIFF tiles, one per 5x5-degree cell in EPSG:4326 at
0.00025 deg (~30 m) resolution. Per-pixel value:

    0 = not burned      1 = burned (burned at least once during the year)

These map exactly to the manifest's two classes (burned / not burned). We keep both:
class id 0 = not burned (background), class id 1 = burned.

WHY STATIC PRESENCE, NOT A DATED CHANGE LABEL (spec 5):
GABAM resolves a burn only to the YEAR (the pixel burned sometime during that calendar
year), NOT to within ~1-2 months. A dated change label would require placing the fire
event confidently inside the pretraining pairing window, which we cannot do at year
resolution. Instead we treat a burned pixel as a persistent post-fire burn-scar STATE:
after a fire the scar (charring, vegetation loss) stays visible for many months, so
"burned vs not-burned" is a legitimate static presence classification over a 1-year
window. Therefore change_time = null and time_range = the full GABAM calendar year.
(This is exactly the persistent-post-change-state exception in spec 5.)

GLOBAL DERIVED PRODUCT -> BOUNDED-TILE SAMPLING (spec 5):
GABAM is global; we do NOT attempt global coverage. We download a bounded set of year
ZIPs and extract a curated set of 5x5-deg source tiles from representative fire-prone
biomes across both hemispheres and several post-2016 years (Sub-Saharan African savannas
- which dominate global burned area - plus South American cerrado/arc-of-deforestation,
northern-Australian savanna, boreal Siberia, western North America, Central-Asian steppe,
mainland SE Asia and the Mediterranean). Only post-2016 years are used (Sentinel era).

Non-overlapping ~64px-footprint native windows are scanned; a window is a burn candidate
if it is >= BURN_MIN burned (a strong, high-confidence burn-scar signal) and a background
candidate if it is pure not-burned. Windows with a weak/ambiguous burn fraction
(0 < frac < BURN_MIN) are skipped. Tiles-per-class balanced selection (rarest class first)
draws up to 1000 tiles/class. Native 30 m EPSG:4326 windows are reprojected to a local UTM
projection at 10 m with nearest resampling (categorical labels).

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.gabam_global_annual_burned_area_map
"""

import argparse
import multiprocessing
import zipfile
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import tqdm
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds

from olmoearth_pretrain.open_set_segmentation_data import download, io
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    select_tiles_per_class,
)

SLUG = "gabam_global_annual_burned_area_map"
ZENODO_RECORD = "13858799"

PER_CLASS = 1000
BLOCK = 22  # native (30 m) block ~= 640 m ~= a 64 px @ 10 m UTM tile footprint
TILE = 64
PRESENT_FRAC = 0.05  # a class is "present" in a window if it covers >= 5% of the block
BURN_MIN = (
    0.10  # a burn window must be >= 10% burned (high-confidence burn-scar signal)
)

# Out-of-source fill sentinel (raw GABAM is only 0/1, so 255 is safe as nodata).
SRC_FILL = 255

# Curated representative fire-prone source tiles. Each entry: (tile_basename, year, region).
# tile_basename decodes as <N|S><northEdgeLat:02d><E|W><westEdgeLon:03d>; the tile spans
# 5 deg south and 5 deg east of that NW corner. Only post-2016 years are used.
TILES: list[tuple[str, int, str]] = [
    # --- Sub-Saharan African savannas (dominate global burned area) - year 2019 ---
    ("N15E000", 2019, "N Africa savanna (Sahel/Sudanian)"),
    ("N15E005", 2019, "N Africa savanna (Sahel/Sudanian)"),
    ("N15E010", 2019, "N Africa savanna (Sahel/Sudanian)"),
    ("N15E015", 2019, "N Africa savanna (Sahel/Sudanian)"),
    ("N15E020", 2019, "N Africa savanna (Sahel/Sudanian)"),
    ("N10E020", 2019, "N Africa savanna (Sahel/Sudanian)"),
    ("N10E025", 2019, "N Africa savanna (Sahel/Sudanian)"),
    ("N15E030", 2019, "N Africa savanna (Sahel/Sudanian)"),
    ("S05E015", 2019, "S Africa savanna (Angola/Zambia/DRC)"),
    ("S05E020", 2019, "S Africa savanna (Angola/Zambia/DRC)"),
    ("S10E020", 2019, "S Africa savanna (Angola/Zambia/DRC)"),
    ("S10E025", 2019, "S Africa savanna (Angola/Zambia/DRC)"),
    ("S10E030", 2019, "S Africa savanna (Angola/Zambia/DRC)"),
    ("S15E025", 2019, "S Africa savanna (Angola/Zambia/DRC)"),
    ("S15E030", 2019, "S Africa savanna (Angola/Zambia/DRC)"),
    # --- South America (cerrado / arc-of-deforestation) - year 2020 ---
    ("S05W060", 2020, "S America (Amazon arc / cerrado)"),
    ("S10W060", 2020, "S America (Amazon arc / cerrado)"),
    ("S10W055", 2020, "S America (Amazon arc / cerrado)"),
    ("S10W050", 2020, "S America (Amazon arc / cerrado)"),
    ("S15W050", 2020, "S America (Amazon arc / cerrado)"),
    # --- Northern Australia savanna - year 2020 ---
    ("S10E130", 2020, "N Australia savanna"),
    ("S15E125", 2020, "N Australia savanna"),
    ("S15E130", 2020, "N Australia savanna"),
    ("S15E135", 2020, "N Australia savanna"),
    # --- Mainland SE Asia dry forest - year 2020 ---
    ("N25E095", 2020, "Mainland SE Asia dry forest"),
    ("N20E100", 2020, "Mainland SE Asia dry forest"),
    # --- Boreal Siberia - year 2018 ---
    ("N65E100", 2018, "Boreal Siberia"),
    ("N60E110", 2018, "Boreal Siberia"),
    ("N65E120", 2018, "Boreal Siberia"),
    # --- Western North America (boreal Canada + California) - year 2018 ---
    ("N60W120", 2018, "W North America (boreal/temperate)"),
    ("N40W120", 2018, "W North America (boreal/temperate)"),
    # --- Central Asian steppe - year 2018 ---
    ("N55E060", 2018, "Central Asian steppe"),
    ("N50E070", 2018, "Central Asian steppe"),
    # --- Mediterranean (Iberia / NW Africa) - year 2018 ---
    ("N40W010", 2018, "Mediterranean"),
]

CLASSES = [
    (
        "not burned",
        "No burn detected during the calendar year (GABAM value 0): unburned land or other "
        "surface. This is the background/negative class.",
    ),
    (
        "burned",
        "Burned area (GABAM value 1): the 30 m pixel burned at least once during the calendar "
        "year, per the Landsat-derived GABAM product. Treated as a persistent post-fire "
        "burn-scar state (see module docstring).",
    ),
]


def zip_path(year: int) -> "io.UPath":
    return io.raw_dir(SLUG) / f"{year}.zip"


def extracted_tile_path(year: int, tile: str) -> "io.UPath":
    return io.raw_dir(SLUG) / "tiles" / str(year) / f"{tile}_burn_class.tif"


def download_years() -> None:
    """Download the year ZIPs we need (idempotent, disk-guarded)."""
    import json
    import urllib.request

    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    years = sorted({y for _, y, _ in TILES})
    need = [y for y in years if not zip_path(y).exists()]
    if not need:
        print("  [skip] all year ZIPs present")
        return
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{ZENODO_RECORD}") as r:
        meta = json.loads(r.read())
    links = {f["key"]: f["links"]["self"] for f in meta["files"]}
    for y in need:
        io.check_disk()
        print(f"  downloading {y}.zip")
        download.download_http(links[f"{y}.zip"], zip_path(y))


def extract_tiles() -> None:
    """Extract the curated source tiles from their year ZIPs (idempotent)."""
    by_year: dict[int, list[str]] = {}
    for tile, year, _ in TILES:
        by_year.setdefault(year, []).append(tile)
    for year, tiles in by_year.items():
        dst_dir = io.raw_dir(SLUG) / "tiles" / str(year)
        dst_dir.mkdir(parents=True, exist_ok=True)
        missing = [t for t in tiles if not extracted_tile_path(year, t).exists()]
        if not missing:
            continue
        io.check_disk()
        with zipfile.ZipFile(str(zip_path(year))) as zf:
            for t in missing:
                member = f"{t}_burn_class.tif"
                data = zf.read(member)
                dst = extracted_tile_path(year, t)
                tmp = dst.parent / (dst.name + ".tmp")
                with tmp.open("wb") as f:
                    f.write(data)
                tmp.rename(dst)
                print(f"  extracted {year}/{member} ({len(data) / 1e6:.1f} MB)")


def _scan_tile(tile: str, year: int, region: str) -> list[dict[str, Any]]:
    """Scan non-overlapping BLOCKxBLOCK native windows; return candidate records.

    A window is a candidate if it is pure not-burned (class [0]) or strongly burned
    (>= BURN_MIN burned -> classes present at >= PRESENT_FRAC). Windows with a weak/
    ambiguous burn fraction (0 < frac < BURN_MIN) are skipped to keep labels
    high-confidence. Each record lists classes_present for tiles-per-class selection.
    """
    path = str(extracted_tile_path(year, tile))
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        st = ds.transform
    h, w = arr.shape
    nby, nbx = h // BLOCK, w // BLOCK
    a = arr[: nby * BLOCK, : nbx * BLOCK].reshape(nby, BLOCK, nbx, BLOCK)
    denom = float(BLOCK * BLOCK)
    f_burn = (a == 1).sum(axis=(1, 3)).astype(np.float32) / denom
    f_bg = (a == 0).sum(axis=(1, 3)).astype(np.float32) / denom

    bg_only = f_burn == 0.0
    burned = f_burn >= BURN_MIN
    qual = bg_only | burned
    brs, bcs = np.nonzero(qual)

    cx = bcs * BLOCK + BLOCK / 2.0
    cy = brs * BLOCK + BLOCK / 2.0
    lons = st.c + cx * st.a
    lats = st.f + cy * st.e

    recs = []
    for br, bc, lon, lat in zip(
        brs.tolist(), bcs.tolist(), lons.tolist(), lats.tolist()
    ):
        present = []
        if f_bg[br, bc] >= PRESENT_FRAC:
            present.append(0)
        if f_burn[br, bc] >= PRESENT_FRAC:
            present.append(1)
        if not present:
            continue
        recs.append(
            {
                "tile": tile,
                "year": year,
                "region": region,
                "lon": float(lon),
                "lat": float(lat),
                "classes_present": present,
                "source_id": f"{year}/{tile}_r{br}_c{bc}",
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

    xs = [bounds[0] * io.RESOLUTION, bounds[2] * io.RESOLUTION]
    ys = [bounds[1] * -io.RESOLUTION, bounds[3] * -io.RESOLUTION]
    left, right = min(xs), max(xs)
    bottom, top = min(ys), max(ys)
    l2, b2, r2, t2 = transform_bounds(
        dst_proj.crs, "EPSG:4326", left, bottom, right, top
    )
    pad = 0.003  # ~330 m margin so the tile is fully covered before nearest-resampling

    with rasterio.open(str(extracted_tile_path(rec["year"], rec["tile"]))) as ds:
        win = from_bounds(l2 - pad, b2 - pad, r2 + pad, t2 + pad, ds.transform)
        src = ds.read(1, window=win, boundless=True, fill_value=SRC_FILL)
        win_transform = ds.window_transform(win)
        src_crs = ds.crs

    dst = np.full((TILE, TILE), SRC_FILL, np.uint8)
    reproject(
        source=src,
        destination=dst,
        src_transform=win_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_proj.crs,
        resampling=Resampling.nearest,
        src_nodata=SRC_FILL,
        dst_nodata=SRC_FILL,
    )
    # Raw GABAM values are 0 (not burned) / 1 (burned); out-of-source -> CLASS_NODATA.
    out = np.full((TILE, TILE), io.CLASS_NODATA, np.uint8)
    out[dst == 0] = 0
    out[dst == 1] = 1

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
        change_time=None,  # static burn-scar presence, not a dated change event
        source_id=rec["source_id"],
        classes_present=present,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()

    print("Downloading GABAM year ZIPs...")
    download_years()
    print("Extracting curated source tiles...")
    extract_tiles()
    io.check_disk()

    print(f"Scanning {len(TILES)} tiles for candidate windows...")
    tasks = [dict(tile=t, year=y, region=r) for t, y, r in TILES]
    with multiprocessing.Pool(min(len(TILES), 8)) as p:
        all_recs: list[dict[str, Any]] = []
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan_tile, tasks), total=len(tasks)
        ):
            all_recs.extend(recs)
    cand_counts: Counter = Counter()
    for r in all_recs:
        for c in r["classes_present"]:
            cand_counts[c] += 1
    print(
        f"scanned {len(all_recs)} candidate windows; "
        f"per-class candidates: {dict(cand_counts)}"
    )

    # Tiles-per-class balanced: rarest class (burned) first, <= PER_CLASS/class.
    selected = select_tiles_per_class(
        all_recs, classes_key="classes_present", per_class=PER_CLASS
    )
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} windows (tiles-per-class, <= {PER_CLASS}/class)")

    io.check_disk()
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            pass

    # Report tiles-per-class counts (a tile counts toward every class it contains).
    class_counts = {name: 0 for name, _ in CLASSES}
    region_counts: Counter = Counter()
    for r in selected:
        for c in r["classes_present"]:
            class_counts[CLASSES[c][0]] += 1
        region_counts[r["region"]] += 1
    print("tiles-per-class counts:", class_counts)
    print("selected windows per region:", dict(region_counts))

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "GABAM (Global Annual Burned Area Map)",
            "task_type": "classification",
            "source": "Zenodo",
            "license": "CC-BY",
            "provenance": {
                "url": f"https://zenodo.org/records/{ZENODO_RECORD}",
                "have_locally": False,
                "annotation_method": (
                    "derived-product (GABAM 30 m global annual burned area, "
                    "Landsat-derived)"
                ),
                "product": "GABAM v (2014-2024 release), Zenodo record 13858799",
                "years_used": sorted({y for _, y, _ in TILES}),
                "tiles": [{"tile": t, "year": y, "region": r} for t, y, r in TILES],
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": class_counts,
            "region_counts": dict(region_counts),
            "notes": (
                "Bounded-tile sampling of the global GABAM annual burned-area product "
                f"({len(TILES)} curated 5x5-deg tiles across representative fire-prone "
                "biomes and both hemispheres, post-2016 years 2018-2020). GABAM value -> "
                "class: 0 -> not burned (background), 1 -> burned. Burn windows require "
                f">= {int(BURN_MIN * 100)}% burned pixels (high-confidence); background "
                "windows are pure not-burned; ambiguous partial-burn windows are skipped. "
                "Non-overlapping ~64px-footprint windows selected tiles-per-class (rarest "
                "first). Reprojected from native 30 m EPSG:4326 to local UTM at 10 m with "
                "nearest resampling. STATIC PRESENCE (change_time=null): GABAM resolves a "
                "burn only to the year, not to within ~1-2 months, so it is NOT usable as a "
                "dated change label; instead a burned pixel is treated as a persistent "
                "post-fire burn-scar state over a 1-year window (spec 5 persistent-state "
                "exception). African savannas dominate the selection, mirroring their "
                "dominance of global burned area."
            ),
        },
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
