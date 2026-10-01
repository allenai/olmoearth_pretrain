"""Process IceLines (Antarctic ice-shelf / glacier fronts) into open-set-seg labels.

Source: IceLines (Baumhoer et al. 2023, Scientific Data 10, 138), the DLR EOC
GeoService monitoring service. >19,400 automatically extracted Antarctic ice-shelf /
glacier calving-front positions derived from Sentinel-1 (2014-today) with the HED-UNet
deep network + post-processing (elevation thresholding, morphological filtering,
vectorization). One (Multi)LineString front position per ice shelf per acquisition.

  https://geoservice.dlr.de/web/maps/eoc:icelines
  download: https://download.geoservice.dlr.de/icelines/files/  (open, no auth)

Layout: one subfolder per ice shelf/basin (51 total: LarsenC, Ross1/2/3, Ronne1/2,
Filchner, PineIsland, Thwaites1/2, Amery, Getz*, ... ). Each has
{daily,monthly,quarterly,annual}/{fronts,fronts-eliminated}/*.gpkg. We use the
**daily/fronts** product only:
  - "fronts"           = confidently-extracted calving-front positions (reliable).
  - "fronts-eliminated"= flagged as potentially unreliable, need manual checks -> SKIPPED.
  - daily              = one front per individual Sentinel-1 acquisition -> gives an EXACT
                         observation date (monthly/seasonal/annual are temporal averages).
GeoPackage attrs: DATE_ (YYYYMMDD), name (shelf), s1name (Sentinel-1 scene id, daily/
monthly only), version, updated. CRS EPSG:3031 (Antarctic polar stereographic, metres).
Each file's front is one line repeated over a few identical features -> deduped by union.

Task: each front is a dated per-date STATE (a position, not a dated change mask), so we
rasterize the front line into a thin dilated mask -> **binary segmentation**
(0 = background, 1 = calving_front) in <=64x64 UTM/UPS 10 m tiles, change_time=null,
with a TIGHT time window anchored on the exact observation date (+/- 45 days). This
follows the completed termpicks_greenland_glacier_termini precedent (Greenland termini);
the difference is the tight (not 1-year) window, chosen because Antarctic fronts advance/
calve over the multi-year record so a narrow window keeps the label aligned with the
imagery pretraining will pair to it.

Suitability at 10 m: an ice-shelf / glacier calving front is a sharp, physically-real
ice/ocean boundary spanning tens-to-hundreds of km, clearly resolvable in Sentinel-1/-2
and Landsat. A line dilated to ~30-50 m (few px) is meaningful at 10 m. ACCEPTED.
Fronts are km-scale (100s of km), far larger than a 640 m tile, so each front is TILED
into several <=64x64 windows sampled along its length; the full line is rasterized and
clipped to each window.

Time filter: source spans 2014-2023; per spec we keep only >= 2016 (Sentinel era).

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.icelines_antarctic_ice_shelf_glacier_fronts
"""

import argparse
import multiprocessing
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

import fiona
import numpy as np
import shapely
import shapely.ops
import tqdm
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered
from scipy.spatial import cKDTree

from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)

SLUG = "icelines_antarctic_ice_shelf_glacier_fronts"
NAME = "IceLines (Antarctic ice-shelf & glacier fronts)"
BASE_URL = "https://download.geoservice.dlr.de/icelines/files/"
UA = {"User-Agent": "Mozilla/5.0 (olmoearth-open-set-seg data build)"}

# Binary class scheme (mirrors termpicks).
CID_BACKGROUND = 0
CID_FRONT = 1
CLASSES = [
    {
        "id": CID_BACKGROUND,
        "name": "background",
        "description": "Non-front pixels: ice-shelf / glacier ice interior, open ocean, "
        "sea ice / ice melange, or bare rock away from the mapped calving front.",
    },
    {
        "id": CID_FRONT,
        "name": "calving_front",
        "description": "Antarctic ice-shelf / glacier calving front, i.e. the ice-ocean "
        "boundary at the seaward front on the observation date. Automatically extracted "
        "from a single Sentinel-1 acquisition (HED-UNet + post-processing), dilated to "
        "~30-50 m so it is visible at 10 m/pixel.",
    },
]

# Source CRS EPSG:3031 (metres). Projection res (1,1) => geometry coords treated as
# pixel==metre, matching rasterize.geom_to_pixels convention (see termpicks / GRW).
SRC_EPSG = 3031
SRC_PROJ = Projection(CRS.from_epsg(SRC_EPSG), 1, 1)

YEAR_MIN = 2016  # Sentinel era; source spans 2014-2023, keep >= 2016.

# Tiling / rasterization parameters.
TILE = io.MAX_TILE  # 64 -> 640 m tiles at 10 m.
MAX_WINDOWS_PER_LINE = 4  # cap so long fronts don't dominate.
DILATE_RADIUS_PX = 1.5  # buffer line ~1.5 px radius -> ~3-5 px (30-50 m) wide.
MIN_FRONT_PIXELS = 5  # drop windows clipping only a trivial sliver of front.

# Balance across shelves + limit download volume (~51 shelves).
MAX_FILES_PER_SHELF = 150

# Sampling budgets (total well under the 25k cap).
POSITIVE_BUDGET = 18000
N_NEGATIVES = 3000
NEG_MIN_DIST_M = 3000.0  # negatives must be >= 3 km from any front vertex.

# Tight time window (+/- this many days) anchored on the exact observation date.
TIME_HALFWIDTH_DAYS = 45

_TO_WGS84 = None


def _lonlat(x: float, y: float) -> tuple[float, float]:
    """EPSG:3031 (x, y) metres -> (lon, lat) degrees."""
    global _TO_WGS84
    if _TO_WGS84 is None:
        from pyproj import Transformer

        _TO_WGS84 = Transformer.from_crs(SRC_EPSG, 4326, always_xy=True)
    return _TO_WGS84.transform(x, y)


def _time_range(date: datetime) -> tuple[datetime, datetime]:
    d = timedelta(days=TIME_HALFWIDTH_DAYS)
    return (date - d, date + d)


# --------------------------------------------------------------------------------------
# HTTP helpers (gentle: the DLR service 503s under high concurrency).
# --------------------------------------------------------------------------------------
def _http_get(url: str, retries: int = 5) -> bytes:
    last: Exception | None = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (i + 1) + random.random())
                continue
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(1.5 * (i + 1) + random.random())
    raise RuntimeError(f"GET failed after {retries} tries: {url} ({last})")


def _list_gpkg(shelf: str) -> list[str]:
    html = _http_get(f"{BASE_URL}{shelf}/daily/fronts/").decode("utf-8", "ignore")
    return re.findall(r'href="([^"/?]+\.gpkg)"', html)


def list_shelves() -> list[str]:
    html = _http_get(BASE_URL).decode("utf-8", "ignore")
    dirs = re.findall(r'href="([^"/?][^"]*)/"', html)
    return [d for d in dirs if not d.startswith("http") and d != ".."]


def _file_date(fname: str) -> datetime | None:
    m = re.search(r"_(\d{8})_", fname)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d").replace(tzinfo=UTC)
    except ValueError:
        return None


# --------------------------------------------------------------------------------------
# Download phase (one worker per file; gentle concurrency).
# --------------------------------------------------------------------------------------
def _download_one(shelf: str, fname: str) -> str:
    dst = io.raw_dir(SLUG) / shelf / fname
    if dst.exists() and dst.stat().st_size > 0:
        return "skip"
    data = _http_get(f"{BASE_URL}{shelf}/daily/fronts/{fname}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / (fname + ".tmp")
    with tmp.open("wb") as f:
        f.write(data)
    tmp.rename(dst)
    return "download"


# --------------------------------------------------------------------------------------
# Read phase: one gpkg -> candidate window records (no geometry carried in IPC).
# --------------------------------------------------------------------------------------
def _read_front(path: str) -> Any:
    """Return the deduped union (Multi)LineString of a front gpkg, or None."""
    layers = fiona.listlayers(path)
    geoms = []
    with fiona.open(path, layer=layers[0]) as src:
        for feat in src:
            g = shapely.geometry.shape(feat["geometry"])
            if not g.is_empty and g.length > 0:
                geoms.append(g)
    if not geoms:
        return None
    return shapely.ops.unary_union(geoms)


def _sample_centers(geom: Any, n: int, seed: int) -> list[tuple[float, float]]:
    """Sample up to n window-center points along a (multi)line (EPSG:3031 coords)."""
    total = geom.length
    if total == 0:
        return []
    rng = random.Random(seed)
    # Evenly-spaced fractions with a little jitter so windows spread along the front.
    fracs = [
        min(1.0, max(0.0, (k + 0.5) / n + rng.uniform(-0.3, 0.3) / n)) for k in range(n)
    ]
    pts = [geom.interpolate(f, normalized=True) for f in fracs]
    return [(p.x, p.y) for p in pts]


def build_windows(path: str, shelf: str, fname: str) -> list[dict[str, Any]]:
    date = _file_date(fname)
    if date is None or date.year < YEAR_MIN:
        return []
    geom = _read_front(path)
    if geom is None:
        return []
    seed = hash((shelf, fname)) & 0xFFFFFFFF
    centers = _sample_centers(geom, MAX_WINDOWS_PER_LINE, seed)
    out: list[dict[str, Any]] = []
    for j, (cx, cy) in enumerate(centers):
        lon, lat = _lonlat(cx, cy)
        out.append(
            {
                "kind": "positive",
                "path": path,
                "lon": lon,
                "lat": lat,
                "cx": cx,
                "cy": cy,
                "date": date.isoformat(),
                "source_id": f"{shelf}/{fname[:-5]}/w{j}",
            }
        )
    return out


# --------------------------------------------------------------------------------------
# Writers (workers).
# --------------------------------------------------------------------------------------
def _write_positive(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    _, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"], proj)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    geom = _read_front(rec["path"])
    if geom is None:
        return "empty"
    pix = geom_to_pixels(geom, SRC_PROJ, proj)
    dilated = pix.buffer(DILATE_RADIUS_PX)
    arr = rasterize_shapes(
        [(dilated, CID_FRONT)],
        bounds,
        fill=CID_BACKGROUND,
        dtype="uint8",
        all_touched=True,
    )
    if int((arr == CID_FRONT).sum()) < MIN_FRONT_PIXELS:
        return "empty"
    date = datetime.fromisoformat(rec["date"])
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        _time_range(date),
        change_time=None,
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
    date = datetime.fromisoformat(rec["date"])
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        _time_range(date),
        change_time=None,
        source_id=rec["source_id"],
        classes_present=[CID_BACKGROUND],
    )
    return "negative"


def _dispatch(rec: dict[str, Any]) -> str:
    if rec["kind"] == "positive":
        return _write_positive(rec)
    return _write_negative(rec)


def _make_negatives(
    pos: list[dict[str, Any]], n: int, seed: int = 7
) -> list[dict[str, Any]]:
    """Background-only tiles: offset positive centers (EPSG:3031) by 5-30 km and reject
    any center within NEG_MIN_DIST_M of a front center.
    """
    pts = np.array([(r["cx"], r["cy"]) for r in pos], dtype=float)
    dates = [r["date"] for r in pos]
    tree = cKDTree(pts)
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    attempts = 0
    while len(out) < n and attempts < n * 60:
        attempts += 1
        idx = rng.randrange(len(pts))
        bx, by = pts[idx]
        ang = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(5000, 30000)
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
                "date": dates[idx],
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
    parser.add_argument("--dl-workers", type=int, default=12)
    args = parser.parse_args()

    io.check_disk()
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)

    # 1. Enumerate shelves + daily/fronts files (gentle threaded listing).
    print("listing ice shelves ...")
    shelves = list_shelves()
    print(f"  {len(shelves)} shelves")

    import concurrent.futures as cf

    file_tasks: list[tuple[str, str]] = []  # (shelf, fname)

    def _listing(shelf: str) -> tuple[str, list[str]]:
        try:
            files = _list_gpkg(shelf)
        except Exception as e:  # noqa: BLE001
            print(f"  WARN list {shelf}: {e}")
            files = []
        return shelf, files

    with cf.ThreadPoolExecutor(8) as ex:
        for shelf, files in tqdm.tqdm(ex.map(_listing, shelves), total=len(shelves)):
            post = [
                f for f in files if (_file_date(f) and _file_date(f).year >= YEAR_MIN)
            ]
            rng = random.Random(hash(shelf) & 0xFFFFFFFF)
            if len(post) > MAX_FILES_PER_SHELF:
                post = rng.sample(sorted(post), MAX_FILES_PER_SHELF)
            for f in post:
                file_tasks.append((shelf, f))
    print(
        f"  {len(file_tasks)} daily/front files selected (>= {YEAR_MIN}, "
        f"<= {MAX_FILES_PER_SHELF}/shelf)"
    )

    io.check_disk()

    # 2. Download (gentle threaded).
    print("downloading gpkgs ...")
    results: Counter = Counter()
    with cf.ThreadPoolExecutor(args.dl_workers) as ex:
        futs = {ex.submit(_download_one, s, f): (s, f) for s, f in file_tasks}
        for fut in tqdm.tqdm(cf.as_completed(futs), total=len(futs)):
            try:
                results[fut.result()] += 1
            except Exception as e:  # noqa: BLE001
                results["error"] += 1
                if results["error"] <= 5:
                    print(f"  WARN download {futs[fut]}: {e}")
    print("  download:", dict(results))

    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "IceLines (Baumhoer et al. 2023, Scientific Data 10:138). DLR EOC GeoService.\n"
            f"  {BASE_URL}  (open download, CC-BY-4.0)\n"
            "Antarctic ice-shelf / glacier calving-front positions from Sentinel-1 "
            "(HED-UNet).\n"
            "Used daily/fronts/*.gpkg (confident positions, exact S1 date). CRS EPSG:3031.\n"
            "fronts-eliminated (flagged unreliable) and monthly/quarterly/annual "
            "(temporal averages) NOT used.\n"
        )

    io.check_disk()

    # 3. Read + build candidate windows (pool).
    print("reading fronts + building windows ...")
    read_tasks = [
        dict(path=(raw / s / f).path, shelf=s, fname=f)
        for s, f in file_tasks
        if (raw / s / f).exists()
    ]
    windows: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for recs in tqdm.tqdm(
            star_imap_unordered(p, build_windows, read_tasks), total=len(read_tasks)
        ):
            windows.extend(recs)
    print(f"  {len(windows)} candidate positive windows")

    # 4. Select positives + build negatives (deterministic).
    windows.sort(key=lambda r: r["source_id"])
    rng = random.Random(42)
    rng.shuffle(windows)
    positives = windows[:POSITIVE_BUDGET]
    negatives = _make_negatives(positives, N_NEGATIVES)
    print(f"selected {len(positives)} positives, {len(negatives)} negatives")

    selected = positives + negatives
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"

    io.check_disk()

    # 5. Write tiles (pool).
    wres: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _dispatch, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            wres[res] += 1
    print("write results:", dict(wres))

    io.check_disk()

    num_samples = sum(1 for _ in io.locations_dir(SLUG).glob("*.tif"))
    shelf_hist = Counter(r["source_id"].split("/")[0] for r in positives)
    year_hist = Counter(r["date"][:4] for r in positives)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "DLR EOC GeoService",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://geoservice.dlr.de/web/maps/eoc:icelines",
                "download_url": BASE_URL,
                "have_locally": False,
                "annotation_method": "derived (HED-UNet on Sentinel-1) + post-processing",
                "citation": "Baumhoer et al. 2023, Scientific Data 10:138 (IceLines).",
            },
            "sensors_relevant": ["sentinel1", "sentinel2", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": num_samples,
            "class_counts": {
                "positive_tiles_with_front": len(positives),
                "background_negative_tiles": len(negatives),
            },
            "shelf_counts": dict(sorted(shelf_hist.items())),
            "year_counts": dict(sorted(year_hist.items())),
            "notes": (
                "Binary calving-front segmentation. IceLines daily/fronts (Multi)LineString "
                "front positions (EPSG:3031, one front per Sentinel-1 acquisition) rasterized "
                f"(buffered ~{DILATE_RADIUS_PX} px -> ~30-50 m wide, all_touched) into 64x64 "
                "UTM/UPS 10 m tiles; class 1 = calving_front, 0 = background. Fronts are "
                "100s of km so each is tiled into up to "
                f"{MAX_WINDOWS_PER_LINE} windows sampled along its length. Positives capped "
                f"at {POSITIVE_BUDGET} (shuffled subsample; <= {MAX_FILES_PER_SHELF} daily "
                f"files/shelf to balance across the 51 shelves) plus {N_NEGATIVES} "
                "background-only negatives >= 3 km from any front. Kept only >= "
                f"{YEAR_MIN} (Sentinel era). Each front is a per-date STATE (change_time=null) "
                f"with a TIGHT +/-{TIME_HALFWIDTH_DAYS}-day window anchored on the exact "
                "observation date (from the filename / S1 scene) -- narrower than the "
                "1-year termpicks window because Antarctic fronts advance and calve over "
                "the record, so a tight window keeps the label aligned with paired imagery. "
                "Used 'fronts' (confident) not 'fronts-eliminated' (flagged unreliable), and "
                "daily (exact date) not monthly/quarterly/annual (temporal averages). "
                f"Shelf distribution of positives: {dict(sorted(shelf_hist.items()))}."
            ),
        },
    )
    print(f"done: {num_samples} samples")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
