"""Process HydroWASTE (Global Wastewater Treatment Plants) into detection tiles.

Source: Ehalt Macedo, H., Lehner, B., Nicell, J. A., Grill, G., Li, J., Limtong, A.,
Shakya, R. "Distribution and characteristics of wastewater treatment plants within the
global river network." Earth Syst. Sci. Data 14, 559-577 (2022),
https://doi.org/10.5194/essd-14-559-2022. Data: HydroWASTE version 1.0 (HydroSHEDS),
figshare https://doi.org/10.6084/m9.figshare.14847786.v1, license CC-BY-4.0. One zip
containing HydroWASTE_v10.csv (58,502 WWTPs) + README.txt.

We build a single-class, positive-only **object-detection** dataset of wastewater
treatment plants (label_type "points" that mark presence; spec section 4). Each WWTP is a
point; its aeration/settling ponds and clarifier tanks are readily discernible at 10-30 m.

Coordinate precision (spec note): HydroWASTE reports a geocoded plant location
(LAT_WWTP/LON_WWTP) with a per-record quality flag QUAL_LOC (1=high >80% of a
country/region's points accurate, 2=medium 50-80%, 3=low <50%, 4=not analysed). Most
records are 3-decimal (~110 m) precision. To keep positives reliable we place tile centers
only on **well-located** plants (QUAL_LOC in {1,2}) whose STATUS implies a built plant
(exclude Projected/Proposed/Under Construction/Construction Completed), and we ring each
positive with a **generous nodata buffer** (buffer_size=12 => 250 m ignore ring) to absorb
geocoding error and the plant's real footprint. We use the reported PLANT location, NOT
the estimated river-outfall location (LAT_OUT/LON_OUT), since the physical infrastructure
sits at the plant.

Encoding (tunable detection, spec section 4): each plant point is a 1 px positive, ringed
by a 12 px nodata (255) buffer, with background (0) filling the rest of a 48x48 (480 m)
context tile. All other well-located plants inside a tile are also marked positive. Per
spec section 4 we additionally emit background-only NEGATIVE tiles away from any plant so
the class has spatially-meaningful negatives.

Class scheme (id 0 = background; 255 = nodata/ignore = detection buffer rings):
  0 background
  1 wastewater_treatment_plant

Time range: WWTPs are persistent structures (undated in the source). Per spec section 5
(static labels) each sample gets a 1-year window at a representative Sentinel-era year,
spread pseudo-randomly across 2016-2022 (the manifest time_range) for temporal diversity.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.hydrowaste_global_wastewater_treatment_plants
"""

import argparse
import csv
import multiprocessing
import random
import zipfile
from collections import Counter
from typing import Any

import numpy as np
import tqdm
from rslearn.utils.mp import star_imap_unordered
from scipy.spatial import cKDTree

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import encode_detection_tile

SLUG = "hydrowaste_global_wastewater_treatment_plants"
NAME = "HydroWASTE (Global Wastewater Treatment Plants)"
FIGSHARE_URL = "https://doi.org/10.6084/m9.figshare.14847786.v1"
DOWNLOAD_FILE_URL = "https://ndownloader.figshare.com/files/31910714"
ZIP_NAME = "HydroWASTE_v10.zip"
CSV_NAME = "HydroWASTE_v10.csv"

CID_BACKGROUND = 0
CID_WWTP = 1
CLASSES = [
    {
        "id": 0,
        "name": "background",
        "description": "Negative / non-plant land: pixels outside any mapped WWTP.",
    },
    {
        "id": 1,
        "name": "wastewater_treatment_plant",
        "description": "Wastewater/sewage treatment plant location from HydroWASTE (reported "
        "plant location LAT_WWTP/LON_WWTP, compiled from national/regional datasets). Physical "
        "infrastructure (aeration/settling ponds, clarifier tanks) discernible at 10-30 m.",
    },
]

# STATUS values that imply the plant is NOT (yet) a built, visible facility.
NOT_BUILT_STATUS = {
    "Projected",
    "Proposed",
    "Under Construction",
    "Construction Completed",
}
# Location-quality flags we trust for placing positive tile centers (1=high, 2=medium).
GOOD_QUAL_LOC = {"1", "2"}

# Sampling / encoding parameters.
PER_CLASS = 1000  # positive WWTP tiles (spec section 5, single class)
N_NEGATIVES = 500  # background-only tiles
YEARS = list(range(2016, 2023))

DET_TILE = 48
DET_POS_SIZE = 1
DET_BUFFER = 12  # generous ignore ring (~250 m) for geocoding imprecision

NEIGHBOR_RADIUS_M = 700.0  # 3857 prefilter radius for in-tile neighbor plants
NEG_MIN_DIST_M = 1000.0  # min distance a negative tile center keeps from any plant
NEG_OFFSET_MIN_M = 3000.0
NEG_OFFSET_MAX_M = 20000.0

_TO_3857 = None
_TO_4326 = None


def _to_3857(lon: float, lat: float) -> tuple[float, float]:
    global _TO_3857
    if _TO_3857 is None:
        from pyproj import Transformer

        _TO_3857 = Transformer.from_crs(4326, 3857, always_xy=True)
    return _TO_3857.transform(lon, lat)


def _to_4326(x: float, y: float) -> tuple[float, float]:
    global _TO_4326
    if _TO_4326 is None:
        from pyproj import Transformer

        _TO_4326 = Transformer.from_crs(3857, 4326, always_xy=True)
    return _TO_4326.transform(x, y)


def csv_path() -> Any:
    return io.raw_dir(SLUG) / CSV_NAME


def ensure_downloaded() -> None:
    """Download + extract HydroWASTE_v10.zip into raw_dir if the CSV is not present."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    if csv_path().exists():
        return
    zip_path = download.download_http(DOWNLOAD_FILE_URL, raw / ZIP_NAME)
    with zipfile.ZipFile(str(zip_path)) as z:
        z.extractall(str(raw))


def read_plants() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read HydroWASTE rows.

    Returns (all_plants, good_plants): every record with valid coords, and the subset that
    is well-located (QUAL_LOC in {1,2}) and built (STATUS not in NOT_BUILT_STATUS). Each
    record has lon/lat + source_id.
    """
    all_plants: list[dict[str, Any]] = []
    good_plants: list[dict[str, Any]] = []
    with csv_path().open(encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row["LAT_WWTP"])
                lon = float(row["LON_WWTP"])
            except (TypeError, ValueError):
                continue
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue
            if lat == 0.0 and lon == 0.0:
                continue
            rec = {
                "lon": lon,
                "lat": lat,
                "source_id": f"WASTE_ID/{row['WASTE_ID']}",
                "qual_loc": row["QUAL_LOC"],
                "status": row["STATUS"],
            }
            all_plants.append(rec)
            if (
                rec["qual_loc"] in GOOD_QUAL_LOC
                and rec["status"] not in NOT_BUILT_STATUS
            ):
                good_plants.append(rec)
    return all_plants, good_plants


# --------------------------------------------------------------------------------------
# Writers (worker processes).
# --------------------------------------------------------------------------------------
def _write_positive(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    _, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"], proj)
    bounds = io.centered_bounds(col, row, DET_TILE, DET_TILE)
    x_min, y_min, _, _ = bounds
    positives: list[tuple[int, int, int]] = []
    cands = [(rec["lon"], rec["lat"])] + rec.get("neighbors", [])
    for lon, lat in cands:
        _, c, r = io.lonlat_to_utm_pixel(lon, lat, proj)
        lc, lr = c - x_min, r - y_min
        if 0 <= lc < DET_TILE and 0 <= lr < DET_TILE:
            positives.append((lr, lc, CID_WWTP))
    arr = encode_detection_tile(
        positives,
        tile_size=DET_TILE,
        positive_size=DET_POS_SIZE,
        buffer_size=DET_BUFFER,
        nodata=io.CLASS_NODATA,
        background=CID_BACKGROUND,
    )[np.newaxis]
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
    bounds = io.centered_bounds(col, row, DET_TILE, DET_TILE)
    arr = encode_detection_tile(
        [],
        tile_size=DET_TILE,
        positive_size=DET_POS_SIZE,
        buffer_size=DET_BUFFER,
        nodata=io.CLASS_NODATA,
        background=CID_BACKGROUND,
    )[np.newaxis]
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
    if rec["kind"] == "negative":
        return _write_negative(rec)
    return _write_positive(rec)


# --------------------------------------------------------------------------------------
# Negatives.
# --------------------------------------------------------------------------------------
def make_negatives(
    tree: cKDTree, plants: list[dict[str, Any]], n: int, seed: int = 7
) -> list[dict[str, Any]]:
    """Background-only tile centers offset from plants, guaranteed plant-free."""
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    attempts = 0
    while len(out) < n and attempts < n * 100:
        attempts += 1
        base = plants[rng.randrange(len(plants))]
        ang = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(NEG_OFFSET_MIN_M, NEG_OFFSET_MAX_M)
        bx, by = _to_3857(base["lon"], base["lat"])
        x, y = bx + dist * np.cos(ang), by + dist * np.sin(ang)
        if tree.query_ball_point([x, y], r=NEG_MIN_DIST_M):
            continue
        lon, lat = _to_4326(x, y)
        if not (-58 <= lat <= 74):
            continue
        out.append(
            {
                "kind": "negative",
                "lon": float(lon),
                "lat": float(lat),
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
    manifest.write_registry_entry(SLUG, "in_progress")

    ensure_downloaded()

    raw = io.raw_dir(SLUG)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "HydroWASTE version 1.0 (Global Wastewater Treatment Plants). Ehalt Macedo "
            "et al., Earth Syst. Sci. Data 14, 559-577 (2022). HydroSHEDS. License "
            "CC-BY-4.0.\n"
            f"figshare: {FIGSHARE_URL}\n"
            f"file: {DOWNLOAD_FILE_URL} -> {ZIP_NAME} -> {CSV_NAME} (58,502 WWTPs) + "
            "README.txt.\n"
            "Positives use reported plant location LAT_WWTP/LON_WWTP (not the estimated "
            "river outfall LAT_OUT/LON_OUT).\n"
        )

    print("reading WWTP points ...", flush=True)
    all_plants, good_plants = read_plants()
    print(
        f"  {len(all_plants)} total plants; {len(good_plants)} well-located & built "
        "(QUAL_LOC in {1,2}, STATUS built)",
        flush=True,
    )

    io.check_disk()

    # KDTree over ALL plants (EPSG:3857) so negatives avoid every reported plant.
    all_xy = np.array([_to_3857(d["lon"], d["lat"]) for d in all_plants], dtype=float)
    all_tree = cKDTree(all_xy)
    # KDTree over well-located plants for in-tile neighbor marking (reliable positives).
    good_xy = np.array([_to_3857(d["lon"], d["lat"]) for d in good_plants], dtype=float)
    good_tree = cKDTree(good_xy)

    # Select positive tile centers from well-located, built plants.
    rng = random.Random(42)
    idxs = list(range(len(good_plants)))
    rng.shuffle(idxs)
    selected = [dict(good_plants[i]) for i in idxs[:PER_CLASS]]

    # Mark neighboring well-located plants that fall inside each positive tile.
    for r in selected:
        x, y = _to_3857(r["lon"], r["lat"])
        near = good_tree.query_ball_point([x, y], r=NEIGHBOR_RADIUS_M)
        r["neighbors"] = [
            (good_plants[i]["lon"], good_plants[i]["lat"])
            for i in near
            if good_plants[i]["source_id"] != r["source_id"]
        ][:200]

    negatives = make_negatives(all_tree, all_plants, N_NEGATIVES)
    print(
        f"selected {len(selected)} positive tiles + {len(negatives)} negatives",
        flush=True,
    )

    for r in selected:
        r["kind"] = "positive"
    yrng = random.Random(123)
    all_recs = selected + negatives
    for r in all_recs:
        r["year"] = YEARS[yrng.randrange(len(YEARS))]
    for i, r in enumerate(all_recs):
        r["sample_id"] = f"{i:06d}"

    results: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _dispatch, [dict(rec=r) for r in all_recs]),
            total=len(all_recs),
        ):
            results[res] += 1
    print("write results:", dict(results), flush=True)

    io.check_disk()

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "HydroSHEDS / HydroWASTE v1.0 (Ehalt Macedo et al. 2022, ESSD)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": FIGSHARE_URL,
                "paper": "https://doi.org/10.5194/essd-14-559-2022",
                "have_locally": False,
                "annotation_method": "authoritative + modeled (national/regional WWTP "
                "registries geocoded and completed with auxiliary data)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "detection_encoding": {
                "applies_to": "WWTP points (single foreground class)",
                "tile_size": DET_TILE,
                "positive_size": DET_POS_SIZE,
                "buffer_size": DET_BUFFER,
                "location_field": "LAT_WWTP/LON_WWTP (reported plant location)",
                "positive_center_filter": "QUAL_LOC in {1,2} and STATUS is a built plant",
            },
            "num_samples": len(all_recs),
            "class_counts": {
                "wwtp_positive_tiles": len(selected),
                "background_negative_tiles": len(negatives),
            },
            "notes": (
                "Positive-only WWTP object detection. 1 px positive at each reported plant "
                "location (LAT_WWTP/LON_WWTP) + 12 px nodata buffer ring (~250 m, absorbs "
                "geocoding imprecision), background fill in a 48x48 (480 m) context tile; "
                "all well-located plants inside a tile are marked positive. Positive tile "
                "centers restricted to QUAL_LOC in {1,2} (>50% located-accurately) and "
                "built STATUS (Projected/Proposed/Under Construction/Construction Completed "
                "excluded). 500 background-only negative tiles emitted away from any of the "
                "58,502 plants (>=1 km). 1000 of ~52k well-located plants sampled as tile "
                "centers (spec section 5 per-class cap). Persistent features -> 1-year "
                "window at a representative Sentinel-era year (2016-2022)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(all_recs)
    )
    print("done:", len(all_recs), "samples", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
