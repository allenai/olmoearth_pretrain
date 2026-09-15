"""Process CoastSat satellite-derived shorelines into open-set-segmentation labels.

Source: CoastSat (kvos/CoastSat) satellite-derived shoreline products released on
Zenodo (CC-BY-4.0). CoastSat maps the instantaneous land/water boundary of sandy
coastlines from 40 years of Landsat + Sentinel-2 imagery (horizontal accuracy
~10-15 m). We use two regional releases:

  - Pacific Rim  : Zenodo record 15614554, file ``shorelines.geojson`` (3146 beaches;
                   covers the Pacific basin incl. SE Australia, NZ, Chile, W US, Japan).
  - US East Coast: Zenodo record 18435286, file ``US_East_shorelines.geojson`` (301
                   beaches; US Atlantic + Gulf coast).

Each ``shorelines.geojson`` feature is ONE sandy-beach **reference shoreline** LineString
in WGS84 lon/lat (a median/representative position aggregated over the full 1984-2024
record; attributes: beach length, median orientation, median beach slope, tidal range,
confidence interval). These are the only files pulled (~22 MB total) — the multi-hundred-MB
``shoreline_data.zip`` holding per-transect time series is NOT needed for the label signal.

Task: **binary line segmentation** (classification):
  0 = background (land / water away from the shoreline)
  1 = sandy shoreline (the reference land/water boundary line)
The beach LineStrings are rasterized (dilated ~1 px so they are visible at 10 m) into
<=64x64 UTM 10 m tiles, tiled along their length (beaches are km-scale, median ~1.4 km
Pacific / ~10 km US East), plus background-only negative tiles inland/offshore.

Suitability at 10 m: the wet/dry sandy-shore land/water boundary is exactly what CoastSat
extracts from Sentinel-2/Landsat, so a dilated shoreline line is meaningful at 10 m/pixel.
ACCEPTED.

Signal NOT used (documented): the transect **linear trend (m/yr)** = "erosion vs accretion"
is a *multi-decadal change rate*, not a dated event and not observable within a single
1-year pretraining window, so it cannot be expressed as a per-pixel class/regression at the
pairing timescale and is dropped (see spec section 5 change-timing / observability rules).

Time range: the reference shoreline is an aggregate over ~1984-2024, i.e. effectively a
**static** label, so we assign a single representative 1-year Sentinel-era window
(REPRESENTATIVE_YEAR) and change_time=null. Caveat: the reference is a median position, so
the instantaneous shoreline in any one image can be offset by the beach's variability
(often 10-50 m); the ~1 px dilation and 10 m raster make this a coarse shoreline mask.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.coastsat_satellite_derived_shorelines
"""

import argparse
import multiprocessing
import random
from collections import Counter
from math import atan2, cos, radians, sin, sqrt
from typing import Any

import numpy as np
import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.mp import star_imap_unordered
from scipy.spatial import cKDTree

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)

SLUG = "coastsat_satellite_derived_shorelines"
NAME = "CoastSat Satellite-Derived Shorelines"

# Zenodo sources: (record id, file key, local filename, region tag).
SOURCES = [
    ("15614554", "shorelines.geojson", "pacific_rim_shorelines.geojson", "pacific_rim"),
    ("18435286", "US_East_shorelines.geojson", "us_east_shorelines.geojson", "us_east"),
]
ZENODO_FILE_URL = "https://zenodo.org/api/records/{rec}/files/{key}/content"

# Source geometries are WGS84 lon/lat degrees (res (1,1) so degree coords pass through).
SRC_PROJ = WGS84_PROJECTION

# Binary class scheme.
CID_BACKGROUND = 0
CID_SHORELINE = 1
CLASSES = [
    {
        "id": CID_BACKGROUND,
        "name": "background",
        "description": "Non-shoreline pixels: land (beach/dune/vegetation/built) or open "
        "water away from the mapped sandy shoreline reference line.",
    },
    {
        "id": CID_SHORELINE,
        "name": "shoreline",
        "description": "Sandy-coast shoreline: the instantaneous land/water boundary of a "
        "sandy beach as mapped by CoastSat from Landsat/Sentinel-2 (horizontal accuracy "
        "~10-15 m). Here the per-beach reference (median 1984-2024) position, dilated to "
        "~2-3 px (~20-30 m) so it is visible at 10 m/pixel.",
    },
]

# Static representative 1-year window (reference shoreline is a multi-decadal aggregate).
REPRESENTATIVE_YEAR = 2020

# Tiling / rasterization parameters.
TILE = 64  # 640 m tiles at 10 m.
STEP_M = 600.0  # spacing of window centers sampled along each beach line (metres).
MAX_WINDOWS_PER_LINE = 8  # cap so a few very long beaches don't dominate.
DILATE_RADIUS_PX = 1.0  # buffer the line ~1 px radius -> ~2-3 px (20-30 m) wide.
MIN_SHORELINE_PIXELS = 3  # drop windows that only clip a trivial sliver.

# Sampling budgets (total stays well under the 25k cap).
POSITIVE_BUDGET = 16000
N_NEGATIVES = 3000
NEG_MIN_DIST_M = 1500.0  # negatives must be >=1.5 km from any shoreline vertex.

_EARTH_R = 6371000.0


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    p1, p2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlmb = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(p1) * cos(p2) * sin(dlmb / 2) ** 2
    return 2 * _EARTH_R * atan2(sqrt(a), sqrt(1 - a))


# --------------------------------------------------------------------------------------
# Reading source features.
# --------------------------------------------------------------------------------------
def read_lines() -> list[dict[str, Any]]:
    """Read all reference-shoreline LineStrings from both source geojsons."""
    import json

    recs: list[dict[str, Any]] = []
    raw = io.raw_dir(SLUG)
    for _rec, _key, fname, region in SOURCES:
        with (raw / fname).open() as f:
            fc = json.load(f)
        for i, feat in enumerate(fc["features"]):
            geom = shapely.geometry.shape(feat["geometry"])
            if geom.is_empty or geom.length == 0:
                continue
            props = feat["properties"]
            recs.append(
                {
                    "region": region,
                    "beach_id": props.get("id"),
                    "geom_wkb": shapely.to_wkb(geom),
                    "src_index": i,
                }
            )
    return recs


def _line_centers(geom: Any) -> list[tuple[float, float]]:
    """Sample window-center lon/lats along a (multi)line at ~STEP_M metric spacing."""
    parts = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
    centers: list[tuple[float, float]] = []
    for part in parts:
        coords = list(part.coords)
        if len(coords) < 2:
            continue
        # Cumulative metric length along the vertices.
        cum = [0.0]
        for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
            cum.append(cum[-1] + _haversine_m(x1, y1, x2, y2))
        total = cum[-1]
        if total == 0:
            continue
        n = max(1, int(total // STEP_M) + 1)
        targets = [min(total, (k + 0.5) * (total / n)) for k in range(n)]
        j = 0
        for t in targets:
            while j < len(cum) - 2 and cum[j + 1] < t:
                j += 1
            seg = cum[j + 1] - cum[j]
            frac = 0.0 if seg == 0 else (t - cum[j]) / seg
            (x1, y1), (x2, y2) = coords[j], coords[j + 1]
            centers.append((x1 + frac * (x2 - x1), y1 + frac * (y2 - y1)))
    return centers


def build_windows(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand beach-line records into candidate positive window records."""
    windows: list[dict[str, Any]] = []
    for rec in recs:
        geom = shapely.from_wkb(rec["geom_wkb"])
        centers = _line_centers(geom)
        rng = random.Random(hash((rec["region"], rec["src_index"])) & 0xFFFFFFFF)
        if len(centers) > MAX_WINDOWS_PER_LINE:
            centers = rng.sample(centers, MAX_WINDOWS_PER_LINE)
        for j, (lon, lat) in enumerate(centers):
            windows.append(
                {
                    "kind": "positive",
                    "lon": lon,
                    "lat": lat,
                    "geom_wkb": rec["geom_wkb"],
                    "source_id": f"{rec['region']}/{rec['beach_id']}/w{j}",
                }
            )
    return windows


# --------------------------------------------------------------------------------------
# Writers (run in worker processes).
# --------------------------------------------------------------------------------------
def _write_positive(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    _, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"], proj)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    geom = shapely.from_wkb(rec["geom_wkb"])
    pix = geom_to_pixels(geom, SRC_PROJ, proj)
    dilated = pix.buffer(DILATE_RADIUS_PX)
    arr = rasterize_shapes(
        [(dilated, CID_SHORELINE)],
        bounds,
        fill=CID_BACKGROUND,
        dtype="uint8",
        all_touched=True,
    )
    if int((arr == CID_SHORELINE).sum()) < MIN_SHORELINE_PIXELS:
        return "empty"
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(REPRESENTATIVE_YEAR),
        source_id=rec["source_id"],
        classes_present=sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA}),
    )
    return "positive"


def _write_negative(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    _, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"], proj)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    arr = np.full((1, TILE, TILE), CID_BACKGROUND, dtype=np.uint8)
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(REPRESENTATIVE_YEAR),
        source_id=rec["source_id"],
        classes_present=[CID_BACKGROUND],
    )
    return "negative"


def _dispatch(rec: dict[str, Any]) -> str:
    if rec["kind"] == "positive":
        return _write_positive(rec)
    return _write_negative(rec)


# --------------------------------------------------------------------------------------
# Negatives: background-only tiles offset well away from any shoreline.
# --------------------------------------------------------------------------------------
def _make_negatives(
    recs: list[dict[str, Any]], n: int, seed: int = 7
) -> list[dict[str, Any]]:
    verts: list[tuple[float, float]] = []
    for rec in recs:
        geom = shapely.from_wkb(rec["geom_wkb"])
        coords = (
            list(geom.coords)
            if geom.geom_type == "LineString"
            else [c for part in geom.geoms for c in part.coords]
        )
        for c in coords[::5]:  # decimate vertices
            verts.append((c[0], c[1]))
    verts_arr = np.array(verts, dtype=float)
    tree = cKDTree(verts_arr)  # nearest in degree space (approx; refined by haversine)
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    attempts = 0
    while len(out) < n and attempts < n * 100:
        attempts += 1
        blon, blat = verts_arr[rng.randrange(len(verts_arr))]
        ang = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(3000, 30000)  # 3-30 km offset
        dlat = (dist * sin(ang)) / 110540.0
        coslat = max(0.2, cos(radians(blat)))
        dlon = (dist * cos(ang)) / (111320.0 * coslat)
        lon, lat = blon + dlon, blat + dlat
        if not (-180 < lon < 180 and -85 < lat < 85):
            continue
        d, idx = tree.query([lon, lat])
        nlon, nlat = verts_arr[idx]
        if _haversine_m(lon, lat, nlon, nlat) < NEG_MIN_DIST_M:
            continue
        out.append(
            {
                "kind": "negative",
                "lon": lon,
                "lat": lat,
                "source_id": f"negative/{len(out)}",
            }
        )
    return out


# --------------------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------------------
def _download_sources() -> None:
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    for rec, key, fname, _region in SOURCES:
        dst = raw / fname
        if not dst.exists():
            print(f"downloading {fname} from Zenodo record {rec} ...")
            download.download_http(ZENODO_FILE_URL.format(rec=rec, key=key), dst)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "CoastSat satellite-derived shorelines (kvos/CoastSat), CC-BY-4.0.\n"
            "Per-beach reference shoreline LineStrings (WGS84), shorelines.geojson only:\n"
            "  Pacific Rim  : https://zenodo.org/records/15614554 (shorelines.geojson)\n"
            "  US East Coast: https://zenodo.org/records/18435286 (US_East_shorelines.geojson)\n"
            "The large shoreline_data.zip (per-transect time series) is not needed and not "
            "downloaded; the transect linear-trend (erosion/accretion m/yr) signal is a "
            "multi-decadal rate and is intentionally not used (not observable in a 1-yr window).\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    _download_sources()

    print("reading reference shoreline lines ...")
    recs = read_lines()
    region_counts = Counter(r["region"] for r in recs)
    print(f"  {len(recs)} beach lines: {dict(region_counts)}")

    io.check_disk()

    windows = build_windows(recs)
    print(f"  {len(windows)} candidate positive windows")
    rng = random.Random(42)
    rng.shuffle(windows)
    positives = windows[:POSITIVE_BUDGET]

    negatives = _make_negatives(recs, N_NEGATIVES)
    print(f"selected {len(positives)} positives, {len(negatives)} negatives")

    selected = positives + negatives
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"

    results: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _dispatch, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            results[res] += 1
    print("write results:", dict(results))

    io.check_disk()

    num_samples = sum(1 for _ in io.locations_dir(SLUG).glob("*.tif"))
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Zenodo (CoastSat)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://github.com/kvos/CoastSat",
                "have_locally": False,
                "annotation_method": "derived-product (algorithmic, validated); CoastSat "
                "satellite-derived shorelines from Landsat/Sentinel-2, ~10-15 m accuracy.",
                "zenodo_records": [
                    "15614554 (Pacific Rim)",
                    "18435286 (US East Coast)",
                ],
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": num_samples,
            "class_counts": {
                "positive_tiles_with_shoreline": results.get("positive", 0)
                + results.get("skip", 0),
                "background_negative_tiles": len(negatives),
            },
            "notes": (
                "Binary sandy-shoreline line segmentation. Per-beach reference shoreline "
                "LineStrings (WGS84) from CoastSat Zenodo releases (Pacific Rim rec "
                f"15614554: {region_counts.get('pacific_rim', 0)} beaches; US East Coast "
                f"rec 18435286: {region_counts.get('us_east', 0)} beaches) rasterized "
                "(buffered ~1 px -> ~20-30 m wide, all_touched) into 64x64 UTM 10 m tiles; "
                "class 1 = shoreline, class 0 = background. Beaches are km-scale so each is "
                "tiled into up to 8 windows sampled at 600 m spacing along its length; "
                f"positives capped at {POSITIVE_BUDGET} (random subsample) plus "
                f"{N_NEGATIVES} background-only negatives >=1.5 km from any shoreline. "
                "Reference shorelines are median 1984-2024 aggregates (effectively static), "
                f"so a single representative 1-year window ({REPRESENTATIVE_YEAR}) is used "
                "with change_time=null. Only shorelines.geojson (~22 MB) was pulled; the "
                "per-transect linear-trend (erosion/accretion m/yr) signal is a multi-decadal "
                "rate, not observable in a 1-year window, and is intentionally NOT used. "
                "Caveat: the reference line is a median position, so the instantaneous "
                "shoreline in any single image may be offset by beach variability (~10-50 m)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=num_samples
    )
    print(f"done: {num_samples} samples")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
