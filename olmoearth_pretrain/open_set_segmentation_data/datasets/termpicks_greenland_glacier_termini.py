"""Process TermPicks (Greenland glacier termini) into open-set-segmentation labels.

Source: TermPicks V2 (Zenodo record 6557981), "A century of Greenland glacier
terminus data for use in machine learning applications" (Goliber et al., 2022, The
Cryosphere). It is a compiled, QC'd set of manually-digitized terminus traces for
Greenland's marine-terminating glaciers, one LineString/MultiLineString per
glacier-per-date, with GlacierID, Date/Year/Month/Day, satellite/image id, author,
and a quality flag. CRS is EPSG:3413 (NSIDC polar stereographic north, metres).

  https://zenodo.org/records/6557981  (file: TermPicks_V2.zip -> TermPicks_V2.shp)

Task: rasterize the terminus line into a thin dilated mask -> **binary segmentation**
(0 = background, 1 = glacier terminus / calving front) in <=64x64 UTM 10 m tiles,
plus background-only negative tiles.

Suitability at 10 m: a glacier calving front is a sharp, physically-real ice/ocean (or
ice/rock) boundary that is clearly resolvable in Sentinel-2 / Landsat imagery, so a
line dilated to ~20-30 m (2-3 px) is meaningful at 10 m. ACCEPTED. The termini lines
are km-scale (median ~5 km), far larger than a 640 m tile, so each line is TILED into
multiple <=64x64 windows sampled along its length; the full line is rasterized and
clipped to each window.

Time range: each trace is dated. We keep only 2016-2020 traces (Sentinel-2 / recent
Landsat era, matching the manifest time_range) and assign a 1-year window anchored on
the observation year. The exact date is recorded in the sample source_id.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.termpicks_greenland_glacier_termini
"""

import argparse
import multiprocessing
import random
from collections import Counter
from typing import Any

import fiona
import numpy as np
import shapely
import tqdm
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered
from scipy.spatial import cKDTree

from olmoearth_pretrain.open_set_segmentation_data import download, io
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)

SLUG = "termpicks_greenland_glacier_termini"
NAME = "TermPicks (Greenland glacier termini)"
ZENODO_RECORD = "6557981"
ZIP_NAME = "TermPicks_V2.zip"
SHP_NAME = "TermPicks_V2.shp"

# Binary class scheme.
CID_BACKGROUND = 0
CID_TERMINUS = 1
CLASSES = [
    {
        "id": CID_BACKGROUND,
        "name": "background",
        "description": "Non-terminus pixels: glacier ice interior, ocean/fjord, sea "
        "ice, bare rock, or land away from the mapped calving front.",
    },
    {
        "id": CID_TERMINUS,
        "name": "glacier_terminus",
        "description": "Marine-terminating glacier terminus / calving front, i.e. the "
        "ice-ocean (or ice-rock) boundary at the glacier front on the observation date. "
        "Manually digitized terminus trace, dilated to ~20-30 m so it is visible at "
        "10 m/pixel.",
    },
]

# Source CRS is EPSG:3413 (metres). Projection res (1,1) => geometry coords are treated
# as pixel==metre, matching the rasterize.geom_to_pixels convention (see GRW script).
SRC_EPSG = 3413
SRC_PROJ = Projection(CRS.from_epsg(SRC_EPSG), 1, 1)

# Sentinel/recent-Landsat era window matching the manifest time_range.
YEAR_MIN = 2016
YEAR_MAX = 2020

# Tiling / rasterization parameters.
TILE = 64  # 640 m tiles.
STEP_M = 600.0  # spacing of window centers sampled along each line (metres).
MAX_WINDOWS_PER_LINE = 4  # cap so a few huge glaciers don't dominate.
DILATE_RADIUS_PX = 1.0  # buffer the line by ~1 px radius -> ~2-3 px (20-30 m) wide.
MIN_TERMINUS_PIXELS = 3  # drop windows that only clip a trivial sliver of terminus.

# Sampling budgets (total stays well under the 25k cap).
POSITIVE_BUDGET = 15000
N_NEGATIVES = 3000
NEG_MIN_DIST_M = 2000.0  # negatives must be >=2 km from any terminus vertex.

_TO_WGS84 = None  # lazily-built pyproj transformer (per process).


def _lonlat(x: float, y: float) -> tuple[float, float]:
    """EPSG:3413 (x, y) metres -> (lon, lat) degrees."""
    global _TO_WGS84
    if _TO_WGS84 is None:
        from pyproj import Transformer

        _TO_WGS84 = Transformer.from_crs(SRC_EPSG, 4326, always_xy=True)
    lon, lat = _TO_WGS84.transform(x, y)
    return lon, lat


# --------------------------------------------------------------------------------------
# Reading source features and generating candidate windows.
# --------------------------------------------------------------------------------------
def read_lines() -> list[dict[str, Any]]:
    """Read 2016-2020 terminus traces into records (geometry WKB + attributes)."""
    path = io.raw_dir(SLUG) / SHP_NAME
    recs: list[dict[str, Any]] = []
    with fiona.open(path.path) as src:
        for i, feat in enumerate(src):
            p = feat["properties"]
            year = p.get("Year")
            if year is None or year < YEAR_MIN or year > YEAR_MAX:
                continue
            geom = shapely.geometry.shape(feat["geometry"])
            if geom.is_empty or geom.length == 0:
                continue
            recs.append(
                {
                    "glacier_id": p.get("GlacierID"),
                    "year": int(year),
                    "date": p.get("Date"),
                    "qual": p.get("QualFlag"),
                    "satellite": p.get("Satellite"),
                    "geom_wkb": shapely.to_wkb(geom),
                    "src_index": i,
                }
            )
    return recs


def _line_centers(geom: Any) -> list[tuple[float, float]]:
    """Sample window-center points along a (multi)line at STEP_M spacing (EPSG:3413)."""
    parts = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
    centers: list[tuple[float, float]] = []
    for part in parts:
        L = part.length
        if L == 0:
            continue
        n = max(1, int(L // STEP_M) + 1)
        for k in range(n):
            d = min(L, (k + 0.5) * (L / n))
            pt = part.interpolate(d)
            centers.append((pt.x, pt.y))
    return centers


def build_windows(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand line records into candidate window records (one per sampled center)."""
    windows: list[dict[str, Any]] = []
    for rec in recs:
        geom = shapely.from_wkb(rec["geom_wkb"])
        centers = _line_centers(geom)
        rng = random.Random(hash((rec["glacier_id"], rec["src_index"])) & 0xFFFFFFFF)
        if len(centers) > MAX_WINDOWS_PER_LINE:
            centers = rng.sample(centers, MAX_WINDOWS_PER_LINE)
        for j, (cx, cy) in enumerate(centers):
            lon, lat = _lonlat(cx, cy)
            windows.append(
                {
                    "kind": "positive",
                    "lon": lon,
                    "lat": lat,
                    "year": rec["year"],
                    "geom_wkb": rec["geom_wkb"],
                    "source_id": f"glacier{rec['glacier_id']}/{rec['date']}/w{j}",
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
        [(dilated, CID_TERMINUS)],
        bounds,
        fill=CID_BACKGROUND,
        dtype="uint8",
        all_touched=True,
    )
    if int((arr == CID_TERMINUS).sum()) < MIN_TERMINUS_PIXELS:
        return "empty"
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
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
        io.year_range(rec["year"]),
        source_id=rec["source_id"],
        classes_present=[CID_BACKGROUND],
    )
    return "negative"


def _dispatch(rec: dict[str, Any]) -> str:
    if rec["kind"] == "positive":
        return _write_positive(rec)
    return _write_negative(rec)


# --------------------------------------------------------------------------------------
# Negatives.
# --------------------------------------------------------------------------------------
def _make_negatives(
    recs: list[dict[str, Any]], n: int, seed: int = 7
) -> list[dict[str, Any]]:
    """Background-only tiles: offset random terminus vertices by 3-20 km, reject any
    center within NEG_MIN_DIST_M of a (decimated) terminus vertex.
    """
    verts: list[tuple[float, float]] = []
    years: list[int] = []
    for rec in recs:
        geom = shapely.from_wkb(rec["geom_wkb"])
        coords = (
            list(geom.coords)
            if geom.geom_type == "LineString"
            else [c for part in geom.geoms for c in part.coords]
        )
        for c in coords[::5]:  # decimate vertices
            verts.append((c[0], c[1]))
            years.append(rec["year"])
    verts_arr = np.array(verts, dtype=float)
    tree = cKDTree(verts_arr)
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        idx = rng.randrange(len(verts_arr))
        bx, by = verts_arr[idx]
        ang = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(3000, 20000)
        x = bx + dist * np.cos(ang)
        y = by + dist * np.sin(ang)
        if tree.query([x, y])[0] < NEG_MIN_DIST_M:
            continue
        lon, lat = _lonlat(x, y)
        out.append(
            {
                "kind": "negative",
                "lon": lon,
                "lat": lat,
                "year": years[idx],
                "source_id": f"negative/{len(out)}",
            }
        )
    return out


# --------------------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    if not (raw / SHP_NAME).exists():
        print("downloading TermPicks_V2.zip from Zenodo ...")
        download.download_http(
            f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{ZIP_NAME}/content",
            raw / ZIP_NAME,
        )
        import zipfile

        with zipfile.ZipFile((raw / ZIP_NAME).path) as z:
            z.extractall(raw.path)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "TermPicks V2 (Goliber et al. 2022, The Cryosphere), Zenodo record "
            f"{ZENODO_RECORD}.\n"
            f"https://zenodo.org/records/{ZENODO_RECORD}  (file {ZIP_NAME})\n"
            "Manually-digitized Greenland glacier terminus traces, EPSG:3413.\n"
        )

    print("reading terminus lines (2016-2020) ...")
    recs = read_lines()
    print(f"  {len(recs)} terminus traces in {YEAR_MIN}-{YEAR_MAX}")

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

    n_pos = results.get("positive", 0) + results.get(
        "skip", 0
    )  # approximate if resumed
    num_samples = sum(1 for _ in io.locations_dir(SLUG).glob("*.tif"))
    year_hist = Counter(r["year"] for r in recs)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Zenodo",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": f"https://zenodo.org/records/{ZENODO_RECORD}",
                "have_locally": False,
                "annotation_method": "manual digitization (compiled + QC'd)",
                "citation": "Goliber et al. 2022, The Cryosphere (TermPicks V2).",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": num_samples,
            "class_counts": {
                "positive_tiles_with_terminus": len(positives),
                "background_negative_tiles": len(negatives),
            },
            "notes": (
                "Binary terminus segmentation. Glacier terminus LineString/"
                "MultiLineString traces (EPSG:3413) rasterized (buffered ~1 px -> "
                "~20-30 m wide, all_touched) into 64x64 UTM 10 m tiles; class 1 = "
                "terminus, class 0 = background. Lines are km-scale so each is tiled "
                "into up to 4 windows sampled along its length (600 m spacing). "
                "Positives capped at 15000 (random subsample of candidate windows) "
                "plus 3000 background-only negatives >=2 km from any terminus. Kept "
                f"{YEAR_MIN}-{YEAR_MAX} traces only; 1-year time window anchored on the "
                "observation year (exact date in source_id). Year distribution of "
                f"source traces: {dict(sorted(year_hist.items()))}. Caveat: termini are "
                "dated to a specific image within the year, and calving fronts can shift "
                "seasonally, so the yearly window is an approximation."
            ),
        },
    )
    print(f"done: {num_samples} samples")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
