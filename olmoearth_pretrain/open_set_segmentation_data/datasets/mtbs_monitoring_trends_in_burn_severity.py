"""Process MTBS (Monitoring Trends in Burn Severity) into burn-severity segmentation tiles.

Source: MTBS -- the US interagency (USFS/USGS) program that maps burn severity and fire
perimeters for all large fires (>= ~1000 ac in the West, >= ~500 ac in the East) across the
United States from analyst-reviewed differenced Normalized Burn Ratio (dNBR). Public domain.

Two products are combined:

  1. **Burned Areas Boundaries** (national perimeter shapefile ``mtbs_perims_DD.shp``,
     EPSG:4269). One polygon per fire event with attributes ``event_id`` (state-prefixed id),
     ``ig_date`` (ignition date, YYYY-MM-DD -- day precision), ``incid_type``
     (Wildfire / Prescribed Fire / Other), acreage, etc. Gives us each fire's DATE and extent.
  2. **Thematic Burn Severity Mosaics** (annual national rasters, ScienceBase ver 12.0/9.0):
     30 m thematic rasters, one per (year, region) -- CONUS (ESRI:102039), AK (EPSG:3338),
     HI (Hawaii Albers). Pixel values: 0=background(outside fires), 1=Unburned to Low,
     2=Low, 3=Moderate, 4=High, 5=Increased Greenness, 6=Non-Mapping Area(mask). This is the
     per-pixel severity signal that distinguishes MTBS from a plain binary burn-scar dataset.

Task: **dense multi-class burn-severity segmentation** (label_type: dense_raster / polygons).
Class scheme (severity classes only; §5 positive-only foreground, no fabricated background):

    0 = Unburned to Low   (MTBS value 1)
    1 = Low               (MTBS value 2)
    2 = Moderate          (MTBS value 3)
    3 = High              (MTBS value 4)
    4 = Increased Greenness (MTBS value 5)
    255 = nodata/ignore   (outside this fire's perimeter, MTBS background 0, non-mapping 6)

For each fire, the year's regional severity mosaic is reprojected (nearest, categorical) to a
local UTM grid at 10 m/pixel and masked to the fire's own perimeter polygon so the tile's
severity is attributable to that fire's ignition date.

Change semantics (§5): a fire is a dated CHANGE event. ``change_time`` = ``ig_date`` (known to
day precision, well within the <=1-2 month requirement), kept as the reference for building
the windows. Instead of a single centered window, we emit two adjacent six-month windows
split at ``change_time``: ``pre_time_range`` (the <=183 days immediately before) and
``post_time_range`` (the <=183 days immediately after), with ``time_range`` set to null. The
windows are built via ``io.pre_post_time_ranges(change_time, ...)``, so pretraining pairs a
"before" image stack with an "after" stack spanning the fire and probes on their difference.
Only fires with ig_date year in 2016-2024 (Sentinel era AND covered by a downloaded
mosaic) are used; pre-2016 perimeters are filtered out, and 2025+ (no mosaic) are dropped.

Tiling: a fire fitting in a 64x64 tile (640 m) yields one centered tile; larger fires are
gridded into non-overlapping 64x64 windows, keeping windows intersecting the perimeter and
sampling up to MAX_TILES_PER_FIRE per fire. Selection is tiles-per-class balanced (rarest
severity class first -- prioritizes High and Increased Greenness) up to 1000 tiles/class,
capped at the 25,000-sample per-dataset limit (§5).

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.mtbs_monitoring_trends_in_burn_severity
Idempotent: existing locations/{id}.tif are skipped.
"""

import argparse
import math
import multiprocessing
import os
import random
import zipfile
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import numpy as np
import shapely
import shapely.geometry
import shapely.wkb
import tqdm
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds

from olmoearth_pretrain.open_set_segmentation_data import io, manifest, sampling
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    select_tiles_per_class,
)

SLUG = "mtbs_monitoring_trends_in_burn_severity"
NAME = "MTBS (Monitoring Trends in Burn Severity)"

PERIM_SHP = io.raw_dir(SLUG) / "perim_extract" / "mtbs_perims_DD.shp"
MOSAIC_ZIP_DIR = io.raw_dir(SLUG) / "mosaics"
MOSAIC_TIF_DIR = io.raw_dir(SLUG) / "mosaic_tifs"

TILE = 64
MIN_YEAR = 2016
MAX_YEAR = 2024  # last year with a downloaded severity mosaic
MAX_TILES_PER_FIRE = 20
PER_CLASS = 1000  # tiles-per-class target (25k total cap; 5 classes => plenty of room)

# MTBS thematic mosaic value -> our compact class id. Values 0 (background) and 6
# (non-mapping area) become nodata and are not classes.
MTBS_TO_ID = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
CLASSES = [
    (
        "unburned_to_low",
        "MTBS 'Unburned to Low' severity within the fire perimeter (dNBR near the unburned "
        "baseline). Includes unburned islands and very lightly affected vegetation.",
    ),
    (
        "low",
        "MTBS 'Low' burn severity: minor canopy/ground-cover loss, most vegetation survives.",
    ),
    (
        "moderate",
        "MTBS 'Moderate' burn severity: substantial but incomplete vegetation mortality / "
        "canopy consumption.",
    ),
    (
        "high",
        "MTBS 'High' burn severity: near-complete vegetation mortality and canopy consumption; "
        "extensive char/soil exposure.",
    ),
    (
        "increased_greenness",
        "MTBS 'Increased Greenness': post-fire dNBR indicates enhanced greenness relative to "
        "pre-fire (e.g. vigorous herbaceous regrowth or agricultural recovery).",
    ),
]

_UTM_CACHE: dict[tuple[int, int], Projection] = {}


def region_for_event_id(event_id: str) -> str:
    """CONUS/AK/HI region from the 2-letter state prefix of an MTBS event_id."""
    pref = (event_id or "")[:2].upper()
    if pref == "AK":
        return "AK"
    if pref == "HI":
        return "HI"
    return "CONUS"


def mosaic_tif(region: str, year: int):
    return MOSAIC_TIF_DIR / f"mtbs_{region}_{year}.tif"


# --------------------------------------------------------------------------- download / prep


def download() -> None:
    """The raw perimeter zip + mosaic zips are fetched out-of-band (see summary). Here we
    just extract the shapefile and the per-(region,year) severity GeoTIFFs to plain files.
    """
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    with (io.raw_dir(SLUG) / "SOURCE.txt").open("w") as f:
        f.write(
            "MTBS (Monitoring Trends in Burn Severity), USFS/USGS, public domain.\n"
            "Burned Areas Boundaries: https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/"
            "MTBS_Fire/data/composite_data/burned_area_extent_shapefile/mtbs_perimeter_data.zip\n"
            "Thematic Burn Severity Mosaics (annual, per region CONUS/AK/HI): ScienceBase "
            "item 5e91dee782ce172707f02cdd children (per-year mtbs_{REGION}_{YEAR}.zip).\n"
            f"Used years {MIN_YEAR}-{MAX_YEAR}.\n"
        )

    # Extract perimeter shapefile.
    perim_zip = io.raw_dir(SLUG) / "mtbs_perimeter_data.zip"
    if not PERIM_SHP.exists():
        PERIM_SHP.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(perim_zip)) as z:
            z.extractall(str(PERIM_SHP.parent))

    # Extract each mosaic GeoTIFF from its zip.
    MOSAIC_TIF_DIR.mkdir(parents=True, exist_ok=True)
    for zp in sorted(MOSAIC_ZIP_DIR.glob("*.zip")):
        try:
            zf = zipfile.ZipFile(str(zp))
        except zipfile.BadZipFile:
            print(f"  [warn] skipping unreadable zip {zp.name}")
            continue
        with zf as z:
            for member in z.namelist():
                if member.endswith(".tif"):
                    dst = MOSAIC_TIF_DIR / os.path.basename(member)
                    if dst.exists():
                        continue
                    with z.open(member) as src, dst.open("wb") as out:
                        out.write(src.read())


# --------------------------------------------------------------------------- tiling


def _tile_bounds_for_fire(px: Any, idx: int) -> list[tuple[int, int, int, int]]:
    """Candidate 64x64 pixel-bounds for a fire's perimeter (already in UTM pixel coords)."""
    from shapely.geometry import box
    from shapely.prepared import prep

    minx, miny, maxx, maxy = px.bounds
    w, h = maxx - minx, maxy - miny
    if w <= TILE and h <= TILE:
        col = round((minx + maxx) / 2.0)
        row = round((miny + maxy) / 2.0)
        return [io.centered_bounds(col, row, TILE, TILE)]

    x0, y0 = math.floor(minx), math.floor(miny)
    cells = []
    x = x0
    while x < maxx:
        y = y0
        while y < maxy:
            cells.append((x, y, x + TILE, y + TILE))
            y += TILE
        x += TILE
    rng = random.Random(idx)
    rng.shuffle(cells)
    prepared = prep(px)
    out: list[tuple[int, int, int, int]] = []
    for b in cells:
        if len(out) >= MAX_TILES_PER_FIRE:
            break
        if prepared.intersects(box(*b)):
            out.append(b)
    return out


def _reproject_severity(
    tif_path: str, proj: Projection, bounds: tuple[int, int, int, int]
) -> np.ndarray:
    """Reproject a window of a severity mosaic into a (TILE,TILE) uint8 array (values 0-6)."""
    import rasterio
    from rasterio.warp import Resampling, reproject, transform_bounds
    from rasterio.windows import from_bounds

    dst_transform = get_transform_from_projection_and_bounds(proj, bounds)
    xs = [bounds[0] * io.RESOLUTION, bounds[2] * io.RESOLUTION]
    ys = [bounds[1] * -io.RESOLUTION, bounds[3] * -io.RESOLUTION]
    left, right = min(xs), max(xs)
    bottom, top = min(ys), max(ys)
    with rasterio.open(tif_path) as ds:
        l2, b2, r2, t2 = transform_bounds(proj.crs, ds.crs, left, bottom, right, top)
        pad = 90.0  # ~3 native (30 m) px margin
        win = from_bounds(l2 - pad, b2 - pad, r2 + pad, t2 + pad, ds.transform)
        src = ds.read(1, window=win, boundless=True, fill_value=0)
        win_transform = ds.window_transform(win)
        src_crs = ds.crs
    dst = np.zeros((TILE, TILE), np.uint8)
    reproject(
        source=src,
        destination=dst,
        src_transform=win_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=proj.crs,
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0,
    )
    return dst


def _remap_lut() -> np.ndarray:
    lut = np.full(256, io.CLASS_NODATA, dtype=np.uint8)
    for v, cid in MTBS_TO_ID.items():
        lut[v] = cid
    return lut


_LUT = _remap_lut()


def _fire_tiles(fire: dict[str, Any]) -> list[dict[str, Any]]:
    """Build burn-severity label tiles for one fire (mosaic read + perimeter mask)."""
    from shapely.geometry import box

    geom = shapely.wkb.loads(fire["wkb"])
    if geom.is_empty:
        return []
    c = geom.centroid
    proj = io.utm_projection_for_lonlat(float(c.x), float(c.y))
    px = geom_to_pixels(geom, WGS84_PROJECTION, proj)
    if px.is_empty or px.area <= 0:
        return []

    tif = str(mosaic_tif(fire["region"], fire["year"]))
    if not os.path.exists(tif):
        return []

    crs = proj.crs.to_string()
    out: list[dict[str, Any]] = []
    for b in _tile_bounds_for_fire(px, fire["idx"]):
        sev = _reproject_severity(tif, proj, b)
        remapped = _LUT[sev]
        clip = px.intersection(box(*b))
        if clip.is_empty or clip.area <= 0:
            continue
        mask = rasterize_shapes(
            [(clip, 1)], b, fill=0, dtype="uint8", all_touched=False
        )[0]
        label = np.where(mask == 1, remapped, io.CLASS_NODATA).astype(np.uint8)
        present = sorted(int(v) for v in np.unique(label) if v != io.CLASS_NODATA)
        if not present:
            continue
        out.append(
            {
                "crs": crs,
                "bounds": list(b),
                "ig_ms": fire["ig_ms"],
                "source_id": fire["source_id"],
                "present_ids": present,
                "arr": label.tobytes(),
            }
        )
    return out


def _write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    proj = Projection(CRS.from_string(rec["crs"]), io.RESOLUTION, -io.RESOLUTION)
    bounds = tuple(rec["bounds"])
    label = np.frombuffer(rec["arr"], dtype=np.uint8).reshape(TILE, TILE)
    change_time = datetime.fromtimestamp(rec["ig_ms"] / 1000.0, tz=UTC)
    pre_range, post_range = io.pre_post_time_ranges(change_time)
    time_range = (pre_range[0], post_range[1])  # outer bounding span
    io.write_label_geotiff(SLUG, sample_id, label, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        time_range,
        change_time=change_time,
        source_id=rec["source_id"],
        classes_present=rec["present_ids"],
        pre_time_range=pre_range,
        post_time_range=post_range,
    )


def load_fires() -> list[dict[str, Any]]:
    import fiona

    fires: list[dict[str, Any]] = []
    with fiona.open(str(PERIM_SHP)) as src:
        for i, feat in enumerate(src):
            p = feat["properties"]
            d = p.get("ig_date")
            geom = feat.get("geometry")
            if not d or geom is None:
                continue
            year = int(d[:4])
            if year < MIN_YEAR or year > MAX_YEAR:
                continue
            region = region_for_event_id(p.get("event_id") or "")
            if not mosaic_tif(region, year).exists():
                continue
            shp = shapely.geometry.shape(geom)
            if shp.is_empty:
                continue
            try:
                ig_dt = datetime.strptime(d[:10], "%Y-%m-%d").replace(tzinfo=UTC)
            except ValueError:
                continue
            name = p.get("incid_name") or "UNNAMED"
            itype = p.get("incid_type") or "Unknown"
            fires.append(
                {
                    "idx": i,
                    "wkb": shapely.wkb.dumps(shp),
                    "year": year,
                    "region": region,
                    "ig_ms": int(ig_dt.timestamp() * 1000),
                    "source_id": f"{p.get('event_id')}:{name}:{itype}:{d[:10]}",
                }
            )
    return fires


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")
    download()
    io.check_disk()

    fires = load_fires()
    print(
        f"{len(fires)} fires with ig_date {MIN_YEAR}-{MAX_YEAR} and a covering mosaic"
    )
    regions = Counter(f["region"] for f in fires)
    print("fires by region:", dict(regions))

    # Phase B: per-fire severity tiles (parallel).
    candidates: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for tiles in tqdm.tqdm(
            star_imap_unordered(p, _fire_tiles, [dict(fire=fr) for fr in fires]),
            total=len(fires),
            desc="fire tiles",
        ):
            candidates.extend(tiles)
    print(f"{len(candidates)} candidate severity tiles")

    # Phase C: tiles-per-class balanced selection (rarest severity class first).
    selected = select_tiles_per_class(
        candidates, classes_key="present_ids", per_class=PER_CLASS
    )
    for j, r in enumerate(selected):
        r["sample_id"] = f"{j:06d}"
    print(f"selected {len(selected)} tiles (<= {PER_CLASS}/class, 25k cap)")

    # Phase D: write (parallel).
    io.check_disk()
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write tiles",
        ):
            pass

    n_written = len(list(io.locations_dir(SLUG).glob("*.tif")))
    class_counts: dict[str, int] = {}
    for r in selected:
        for cid in r["present_ids"]:
            nm = CLASSES[cid][0]
            class_counts[nm] = class_counts.get(nm, 0) + 1

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "USFS/USGS (MTBS)",
            "license": "public domain",
            "provenance": {
                "url": "https://www.mtbs.gov/",
                "boundaries": (
                    "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/"
                    "composite_data/burned_area_extent_shapefile/mtbs_perimeter_data.zip"
                ),
                "mosaics_sciencebase_parent": "5e91dee782ce172707f02cdd",
                "have_locally": False,
                "annotation_method": "analyst-reviewed dNBR (interagency USFS/USGS)",
                "mosaic_value_map": {
                    "0": "background (outside fires)",
                    "1": "unburned to low",
                    "2": "low",
                    "3": "moderate",
                    "4": "high",
                    "5": "increased greenness",
                    "6": "non-mapping area (mask)",
                },
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": nm, "description": desc}
                for i, (nm, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_written,
            "class_counts": class_counts,
            "fires_by_region": dict(regions),
            "is_change_dataset": True,
            "notes": (
                "Dense multi-class burn-severity segmentation from MTBS. 64x64 uint8 tiles, "
                "local UTM at 10 m; classes 0=unburned-to-low, 1=low, 2=moderate, 3=high, "
                "4=increased-greenness (255=nodata: outside this fire's perimeter, MTBS "
                "background=0 and non-mapping=6). Severity comes from the annual national "
                "thematic mosaics (30 m, ESRI:102039 CONUS / EPSG:3338 AK / Hawaii Albers), "
                "reprojected to UTM 10 m with nearest resampling and masked to each fire's own "
                "perimeter polygon (Burned Areas Boundaries). Fire is a dated CHANGE event: "
                "change_time = ig_date (day precision), time_range = +/-180 d centered on it. "
                f"Only ig_date years {MIN_YEAR}-{MAX_YEAR} used (pre-2016 filtered out; 2025+ "
                "dropped -- no mosaic). Small fires -> 1 centered tile; large fires gridded "
                f"into non-overlapping 64x64 windows, up to {MAX_TILES_PER_FIRE} intersecting "
                "windows/fire. Tiles-per-class balanced (rarest severity class first) up to "
                f"{PER_CLASS}/class, capped at {sampling.MAX_SAMPLES_PER_DATASET}. Both "
                "Wildfire and Prescribed Fire events are kept (both are dated burn events); "
                "incid_type recorded in source_id."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=n_written
    )
    print("class counts (tiles containing class):", class_counts)
    print("total tif on disk:", n_written)
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
