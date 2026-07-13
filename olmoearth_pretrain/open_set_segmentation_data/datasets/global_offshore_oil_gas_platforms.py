"""Process the Global Offshore Oil & Gas Platforms (OOGPs) dataset into detection tiles.

Source: "The Offshore Oil and Gas Platforms (OOGPs) dataset based on satellite data
spanning 2017 to 2023", Zenodo (https://doi.org/10.5281/zenodo.18350974, CC-BY-4.0). A
vector inventory of offshore oil/gas platforms across six major offshore basins (Gulf of
Mexico, Persian Gulf, North Sea, Caspian Sea, Gulf of Guinea, Gulf of Thailand), produced
from satellite (Sentinel-1 SAR) observations. We download only the 977 KB label archive
``OOGPs_v1.0.0.zip`` -- NO imagery; pretraining supplies its own S1/S2/Landsat.

We use ``OOGPs_all_v1.0.0.gpkg`` (layer ``platforms``, 9,334 Point features, EPSG:4326).
Fields: Latitude, Longitude, Area, Country, EEZ, Installation_date (YYYYMM, sparse),
Removal_date (YYYYMM, sparse), Flaring_status (0/1), Year_label (comma-separated list of
the calendar years 2017-2023 in which the platform was detected/present). Year_label is
the authoritative per-year presence signal (it is derived from the install/removal
history), so we drive the time model off it rather than the sparse month-precision dates.

Task type: positive-only object DETECTION, encoded as per-pixel classes (spec section 4).
A single positive class (offshore oil/gas platform) resolvable at 10 m against open water
(the source is derived at Sentinel-1 10 m). Class scheme:
    0 = background (open water), 1 = platform ; 255 = nodata / ignore (detection buffer
    rings; ambiguous in-tile neighbors handled by leaving them background/ignore).

Time / change handling (spec section 5). Platforms are PERSISTENT structures, not change
events. Year_label resolves presence only to a calendar year, and month-precision install
dates exist for only ~4% of platforms -- coarser/sparser than the ~1-2 month change-timing
bar -- so we do NOT emit dated change labels. Instead each platform is treated as a
persistent structure: a positive is emitted only for a calendar year in its Year_label,
guaranteeing the structure is present across the whole 1-year label window. change_time is
null and the time range is that calendar year (io.year_range). This mirrors the GFW SAR
fixed-infrastructure and DeepOWT persistent-structure precedents.

Overlap note: this source partially overlaps the GFW SAR fixed-infrastructure dataset
(both are SAR-derived offshore oil/gas detections). That is acceptable -- downstream
assembly handles dedup; we process this source on its own terms (six named basins, a
distinct 2017-2023 span, and per-year presence labels).

Encoding: one 32x32 (DET_TILE) UTM 10 m context tile per selected platform, centered on
its point; a 1 px positive ringed by a 10 px nodata buffer, rest background. Every OTHER
platform present the same year that falls inside the tile is also encoded as a positive
(platforms cluster in fields). Background NEGATIVE tiles are geolocated open-water sites
obtained by offsetting a random real platform by 3-8 km in a random bearing and confirming
(KD-tree) no platform lies within ~1.1 km, so they are real offshore open water in the
same basins as the positives.

Sampling: to avoid over-representing long-lived platforms, each physical platform
contributes at most one positive tile, at a randomly chosen year from its Year_label; up
to 1000 positives stratified across years 2017-2023 for temporal diversity, plus up to
1000 background negatives.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_offshore_oil_gas_platforms
"""

import argparse
import math
import multiprocessing
import random
from collections import Counter
from typing import Any

import numpy as np
import tqdm
from rslearn.utils.mp import star_imap_unordered
from scipy.spatial import cKDTree

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.download import (
    download_zenodo,
    extract_zip,
)
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    balance_by_class,
    encode_detection_tile,
)

SLUG = "global_offshore_oil_gas_platforms"
NAME = "Global Offshore Oil & Gas Platforms"
ZENODO = "https://doi.org/10.5281/zenodo.18350974"
ZENODO_RECORD = "18350974"
ZIP_FILE = "OOGPs_v1.0.0.zip"
GPKG_FILE = "OOGPs_all_v1.0.0.gpkg"
GPKG_LAYER = "platforms"

CID_BACKGROUND = 0
CID_PLATFORM = 1
POSITIVE_CIDS = (CID_PLATFORM,)

CLASSES = [
    {
        "id": CID_BACKGROUND,
        "name": "background",
        "description": "Open water / ocean surface with no offshore oil/gas platform.",
    },
    {
        "id": CID_PLATFORM,
        "name": "offshore_oil_gas_platform",
        "description": "Fixed offshore oil or gas platform (production/drilling platforms, "
        "wellheads and related fixed structures) across six major offshore basins (Gulf of "
        "Mexico, Persian Gulf, North Sea, Caspian Sea, Gulf of Guinea, Gulf of Thailand), "
        "detected from satellite (Sentinel-1 SAR) observations spanning 2017-2023.",
    },
]

YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023]
PER_CLASS = 1000
PER_YEAR = math.ceil(PER_CLASS / len(YEARS))  # 143 -> up to 1000 per class
N_NEGATIVES = 1000
NEG_PER_YEAR = math.ceil(N_NEGATIVES / len(YEARS))
SEED = 42

# Detection encoding parameters (spec section 4). Platforms ~1 px at 10 m.
DET_TILE = 32
DET_POS_SIZE = 1
DET_BUFFER = 10

# Neighbor search radius (deg): 32 px * 10 m = 320 m tile; ~0.006 deg (~660 m) covers the
# tile safely at all latitudes; precise filter is by tile pixel bounds.
NEIGHBOR_RADIUS_DEG = 0.006
# Negative open-water offset from a real platform, and min clearance from any platform
# required for a negative center (deg; ~1.1 km at the equator).
NEG_OFFSET_M = (3000.0, 8000.0)
NEG_MIN_CLEAR_DEG = 0.01


def _parse_years(year_label: Any) -> list[int]:
    """Parse the comma-separated Year_label into a sorted list of years in range."""
    if year_label is None:
        return []
    years: list[int] = []
    for tok in str(year_label).split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            y = int(float(tok))
        except ValueError:
            continue
        if y in YEARS:
            years.append(y)
    return sorted(set(years))


def _load_platforms() -> list[dict[str, Any]]:
    """Load platforms (idx, lon, lat, years present) from the OOGPs_all gpkg."""
    import geopandas as gpd

    raw = io.raw_dir(SLUG)
    gpkg = raw / "extracted" / GPKG_FILE
    gdf = gpd.read_file(str(gpkg), layer=GPKG_LAYER)
    pts: list[dict[str, Any]] = []
    for i, row in enumerate(gdf.itertuples(index=False)):
        lon = float(row.Longitude)
        lat = float(row.Latitude)
        years = _parse_years(row.Year_label)
        if not years:
            continue
        pts.append({"idx": i, "lon": lon, "lat": lat, "years": years})
    return pts


def _build_candidates(
    pts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (positive_candidates, negative_candidates).

    One positive per physical platform, at a randomly chosen year from its Year_label (so
    long-lived platforms are not over-represented by ~7 near-identical tiles), spread
    across years for the year-stratified balancing. Negatives are open-water sites offset
    from a real platform with a KD-tree clearance check.
    """
    rng = random.Random(SEED)
    pos: list[dict[str, Any]] = []
    for p in pts:
        pos.append(
            {
                "kind": "pos",
                "class": CID_PLATFORM,
                "year": rng.choice(p["years"]),
                "lon": p["lon"],
                "lat": p["lat"],
                "source_id": f"oogps/{p['idx']}",
            }
        )

    coords = np.array([[p["lon"], p["lat"]] for p in pts], dtype=float)
    tree = cKDTree(coords)
    neg: list[dict[str, Any]] = []
    n_target = N_NEGATIVES * 3
    attempts = 0
    while len(neg) < n_target and attempts < n_target * 50:
        attempts += 1
        p = pts[rng.randrange(len(pts))]
        lon, lat = p["lon"], p["lat"]
        dist = rng.uniform(*NEG_OFFSET_M)
        bearing = rng.uniform(0, 2 * math.pi)
        dlat = (dist * math.cos(bearing)) / 111320.0
        dlon = (dist * math.sin(bearing)) / (
            111320.0 * max(0.1, math.cos(math.radians(lat)))
        )
        nlon, nlat = lon + dlon, lat + dlat
        if not (-180 <= nlon <= 180 and -85 <= nlat <= 85):
            continue
        d, _ = tree.query([nlon, nlat], k=1)
        if d < NEG_MIN_CLEAR_DEG:
            continue
        neg.append(
            {
                "kind": "neg",
                "class": CID_BACKGROUND,
                "year": rng.choice(YEARS),
                "lon": nlon,
                "lat": nlat,
                "source_id": f"oogps_neg/{p['idx']}",
            }
        )
    return pos, neg


# Globals for worker processes (populated in main before the write pool).
_COORDS: np.ndarray | None = None
_YEARS_SET: list[frozenset[int]] = []
_NEI_TREE: cKDTree | None = None


def _init_worker(coords: list[tuple[float, float]], years_set: list[list[int]]) -> None:
    global _COORDS, _YEARS_SET, _NEI_TREE
    _COORDS = np.array(coords, dtype=float)
    _YEARS_SET = [frozenset(ys) for ys in years_set]
    _NEI_TREE = cKDTree(_COORDS)


def _write_tile(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    _, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"], proj)
    bounds = io.centered_bounds(col, row, DET_TILE, DET_TILE)
    x_min, y_min = bounds[0], bounds[1]
    year = rec["year"]

    positives: list[tuple[int, int, int]] = []
    if rec["kind"] == "pos":
        positives.append((row - y_min, col - x_min, rec["class"]))

    # In-tile neighbor platforms present the same year -> also positive.
    if _NEI_TREE is not None and _COORDS is not None:
        idxs = _NEI_TREE.query_ball_point(
            [rec["lon"], rec["lat"]], r=NEIGHBOR_RADIUS_DEG
        )
        for j in idxs:
            nlon, nlat = _COORDS[j]
            if nlon == rec["lon"] and nlat == rec["lat"]:
                continue
            if year not in _YEARS_SET[j]:
                continue  # not present that year -> leave as background
            _, c, r = io.lonlat_to_utm_pixel(float(nlon), float(nlat), proj)
            lc, lr = c - x_min, r - y_min
            if 0 <= lc < DET_TILE and 0 <= lr < DET_TILE:
                positives.append((lr, lc, CID_PLATFORM))

    arr = encode_detection_tile(
        positives,
        tile_size=DET_TILE,
        positive_size=DET_POS_SIZE,
        buffer_size=DET_BUFFER,
        nodata=io.CLASS_NODATA,
        background=CID_BACKGROUND,
    )
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(year),
        change_time=None,
        source_id=rec["source_id"],
        classes_present=sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA}),
    )
    return "pos" if rec["kind"] == "pos" else "neg"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    if not (raw / "extracted" / GPKG_FILE).exists():
        print(f"downloading {ZIP_FILE} from Zenodo ...", flush=True)
        download_zenodo(ZENODO_RECORD, raw, filenames=[ZIP_FILE])
        extract_zip(raw / ZIP_FILE, raw / "extracted")
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "The Offshore Oil and Gas Platforms (OOGPs) dataset based on satellite data "
            "spanning 2017 to 2023.\n"
            f"{ZENODO}\n"
            f"Zenodo record {ZENODO_RECORD}, file {ZIP_FILE}.\n"
            f"Used {GPKG_FILE} (layer '{GPKG_LAYER}'): 9334 Point features (EPSG:4326) of "
            "offshore oil/gas platforms across six basins (Gulf of Mexico, Persian Gulf, "
            "North Sea, Caspian Sea, Gulf of Guinea, Gulf of Thailand). Fields incl. "
            "Latitude/Longitude/Country/EEZ/Installation_date/Removal_date/Flaring_status/"
            "Year_label (comma-separated years 2017-2023 the platform is present). "
            "License CC-BY-4.0. NO imagery downloaded.\n"
        )

    pts = _load_platforms()
    print(f"loaded {len(pts)} platforms with >=1 in-range year", flush=True)

    pos_cands, neg_cands = _build_candidates(pts)
    print(
        f"positive candidates={len(pos_cands)}; negative candidates={len(neg_cands)}",
        flush=True,
    )

    selected: list[dict[str, Any]] = []
    sel_pos = balance_by_class(pos_cands, "year", per_class=PER_YEAR, seed=SEED)[
        :PER_CLASS
    ]
    selected.extend(sel_pos)
    print(f"  platform: selected {len(sel_pos)}", flush=True)
    neg_sel = balance_by_class(neg_cands, "year", per_class=NEG_PER_YEAR, seed=SEED)[
        :N_NEGATIVES
    ]
    selected.extend(neg_sel)
    print(f"  negatives: selected {len(neg_sel)}", flush=True)

    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"

    io.check_disk()

    coords = [(p["lon"], p["lat"]) for p in pts]
    years_set = [p["years"] for p in pts]
    results: Counter = Counter()
    with multiprocessing.Pool(
        args.workers, initializer=_init_worker, initargs=(coords, years_set)
    ) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_tile, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            results[res] += 1
    print("write results:", dict(results), flush=True)

    io.check_disk()

    class_counts = {
        "offshore_oil_gas_platform": len(sel_pos),
        "background_negative_tiles": len(neg_sel),
    }
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",  # detection encoded as per-pixel classes
            "source": "Zenodo / ESSD (OOGPs v1.0.0)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": ZENODO,
                "have_locally": False,
                "annotation_method": "derived-product (Sentinel-1 SAR) + validation",
                "file": GPKG_FILE,
            },
            "sensors_relevant": ["sentinel1", "sentinel2", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "detection_encoding": {
                "tile_size": DET_TILE,
                "positive_size": DET_POS_SIZE,
                "buffer_size": DET_BUFFER,
            },
            "num_samples": len(selected),
            "class_counts": class_counts,
            "notes": (
                "Offshore oil/gas platform DETECTION from the OOGPs satellite (Sentinel-1 "
                "SAR) inventory (2017-2023, six major basins). Single positive class: "
                "0=background(open water), 1=offshore_oil_gas_platform; 255=nodata. "
                "Detection encoding: 32x32 UTM 10 m context tile per platform, 1 px "
                "positive + 10 px nodata buffer (21x21 ignore), rest background; other "
                "in-tile platforms present the same year encoded as positives. "
                "Persistent-structure time model: a positive is emitted only for a calendar "
                "year listed in the platform's Year_label, so the structure is present "
                "across the whole 1-year window; change_time=null (Year_label is "
                "year-resolved and month-precision install dates cover only ~4% of "
                "platforms, both coarser/sparser than the ~1-2 month change-label bar, so "
                "NOT encoded as dated change). Each physical platform contributes at most "
                "one positive tile (random year from its Year_label) to avoid "
                "over-representing long-lived platforms. Negatives: open-water tiles offset "
                "3-8 km from real platforms with no platform within ~1.1 km. Up to 1000 "
                "positive tiles stratified across 2017-2023 + up to 1000 negatives. All "
                "labels post-2016. Partially overlaps the GFW SAR fixed-infrastructure "
                "dataset (both SAR-derived offshore oil/gas); downstream assembly dedups."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print(f"done: {len(selected)} samples", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
