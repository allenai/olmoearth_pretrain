"""Process USGS MRDS (Mineral Resources Data System) into open-set-segmentation labels.

Source: USGS Mineral Resources Data System (MRDS), a global point database of mineral
deposits, mines, prospects and occurrences with commodity / deposit-type attributes.
Public domain. Downloaded as the national CSV export from mrdata.usgs.gov:

  https://mrdata.usgs.gov/mrds/mrds-csv.zip   (project page https://mrdata.usgs.gov/mrds/)

MRDS is a **positive-only point** dataset (a point marks a mineral site; absence is
everywhere else), so per spec 4 we use the tunable DETECTION encoding: a 1 px positive at
the site, a nodata buffer ring, and background fill in a context tile, plus background-only
negative tiles. The class is the **primary commodity**.

Observability (spec 8) — the crux for MRDS:
  * MRDS site coordinates are frequently LOW-PRECISION (many are PLSS-section-derived, so
    true positional error is often 100-400 m even though lon/lat are stored to 5 decimals),
    and many records are sub-pixel exploration points with no surface expression. We
    therefore (a) keep only development statuses with a physical ground disturbance
    (Producer, Past Producer, Prospect) and DROP Occurrence / Plant / Unknown (a documented
    mineral occurrence or a processing plant is not a resolvable mine footprint); (b) use a
    generous detection buffer (buffer_size=12 -> a 25x25 = ~250 m ignore ring) inside a
    48x48 context tile so a site that lands a couple hundred metres off is ignored, not
    penalised; (c) prefer Producer > Past Producer > Prospect when selecting per class, so
    the strongest-signal actual mines are chosen first. Even so, these are WEAK
    presence-detection targets (a "mineral mine is present near here" signal), not precise
    footprints -- flagged in the summary. Compare usgs_usmin_mine_features (map-digitised
    mine symbols, better positional accuracy, feature-type classes).

Class scheme: id 0 = background; commodity classes are ids 1..N in descending frequency;
255 = nodata/ignore (detection buffer rings). Non-observable fluid/energy commodities
(geothermal, natural gas, petroleum, helium, CO2, water, brine halogens) are dropped -- no
surface mine footprint.

Time range: persistent, undated mine sites -> per spec 5 (static labels) a 1-year window at
a representative Sentinel-era year, pseudo-randomly spread across 2016-2022.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.usgs_mrds_mineral_resources_data_system
"""

import argparse
import csv
import multiprocessing
import random
import re
import zipfile
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import tqdm
from rslearn.utils.mp import star_imap_unordered
from scipy.spatial import cKDTree

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import encode_detection_tile

SLUG = "usgs_mrds_mineral_resources_data_system"
NAME = "USGS MRDS (Mineral Resources Data System)"
DOWNLOAD_URL = "https://mrdata.usgs.gov/mrds/mrds-csv.zip"
UA = {"User-Agent": "Mozilla/5.0 (OlmoEarth research data pipeline)"}
CSV_NAME = "mrds.csv"

# Development statuses kept: sites with a physical ground disturbance observable at 10-30 m.
OBSERVABLE_DEV_STAT = {"Producer", "Past Producer", "Prospect"}
# Prefer stronger-signal actual mines first when selecting per class.
DEV_STAT_RANK = {"Producer": 0, "Past Producer": 1, "Prospect": 2}

# Sampling / detection-encoding parameters.
POS_BUDGET = 24000  # positives budget; +N_NEGATIVES stays under the 25k hard cap
N_NEGATIVES = 1000
YEARS = list(range(2016, 2023))  # representative Sentinel-era 1-year windows

DET_TILE = 48
DET_POS_SIZE = 1
DET_BUFFER = 12  # ~250 m ignore ring; MRDS coords are lower precision than USMIN

CID_BACKGROUND = 0

# Non-observable fluid/energy commodities (no surface mine footprint) -> dropped.
DROP_COMMODITIES = {
    "geothermal",
    "natural_gas",
    "petroleum",
    "carbon_dioxide",
    "helium",
    "water",
    "iodine",
    "bromine",
    "chlorine",
    "nitrogen_nitrates",
    "oil_shale",
    "oil_sands",
    "rock_asphalt",
}

# Merge obvious primary-commodity synonyms / hyphenated pairs into one canonical class.
MERGE = {
    "barium_barite": "barite",
    "fluorine_fluorite": "fluorite",
    "phosphorus_phosphates": "phosphate",
    "gypsum_anhydrite": "gypsum",
    "talc_soapstone": "talc",
    "boron_borates": "boron",
    "sulfur_pyrite": "sulfur",
    "iron_pyrite": "iron",
    "pyrite": "sulfur",
    "ree": "rare_earths",
    "pge": "platinum_group",
    "semiprecious_gemstone": "gemstone",
    "sand": "sand_and_gravel",
    "aggregate": "stone",
    "coal": "coal",
    "lignite": "coal",
    "subbituminous": "coal",
    "bituminous": "coal",
    "halite": "salt",
    "sodium_carbonate": "soda_ash",
    "soda_ash": "soda_ash",
    "titanium_heavy_minerals": "titanium",
    "titanium_ilmenite": "titanium",
    "titanium_rutile": "titanium",
    "copper_oxide": "copper",
    "copper_sulfide": "copper",
}


def _slug_commod(token: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", token.lower()).strip("_")
    return re.sub(r"_+", "_", s)


def primary_commodity(commod1: str | None) -> str | None:
    """Canonical commodity class name from the raw commod1 field (primary = first token)."""
    if not commod1:
        return None
    tok = commod1.split(",")[0]
    tok = re.sub(r"\(.*?\)", "", tok).strip()
    if not tok:
        return None
    name = _slug_commod(tok)
    name = MERGE.get(name, name)
    if name in DROP_COMMODITIES or not name:
        return None
    return name


# --------------------------------------------------------------------------------------
# Read source.
# --------------------------------------------------------------------------------------
def read_sites() -> list[dict[str, Any]]:
    """Read observable MRDS sites with a valid coordinate + primary commodity."""
    zip_path = io.raw_dir(SLUG) / "mrds-csv.zip"
    recs: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path.path) as zf, zf.open(CSV_NAME) as fh:
        reader = csv.DictReader(line.decode("latin-1") for line in fh)
        for row in reader:
            if row.get("dev_stat") not in OBSERVABLE_DEV_STAT:
                continue
            commod = primary_commodity(row.get("commod1"))
            if commod is None:
                continue
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
            except (TypeError, ValueError):
                continue
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue
            if lat == 0.0 and lon == 0.0:
                continue
            recs.append(
                {
                    "lon": lon,
                    "lat": lat,
                    "commodity": commod,
                    "dev_stat": row["dev_stat"],
                    "source_id": f"dep_id={row.get('dep_id')};dev_stat={row.get('dev_stat')}",
                }
            )
    return recs


# --------------------------------------------------------------------------------------
# Selection: per-commodity, prefer Producer > Past Producer > Prospect.
# --------------------------------------------------------------------------------------
def build_class_map(recs: list[dict[str, Any]]) -> dict[str, int]:
    """Assign class ids 1..N to commodities in descending frequency (honors 254 cap)."""
    freq = Counter(r["commodity"] for r in recs)
    ordered = [c for c, _ in freq.most_common()]
    # id 0 is background; commodity ids start at 1; uint8 cap => max id 254 => 254 commodities.
    ordered = ordered[:254]
    return {c: i + 1 for i, c in enumerate(ordered)}


def select_records(
    recs: list[dict[str, Any]], class_map: dict[str, int], seed: int = 42
) -> list[dict[str, Any]]:
    """Up to per_class records per commodity, preferring stronger dev_stat first."""
    n_classes = len(class_map)
    per_class = min(1000, max(1, POS_BUDGET // n_classes))
    rng = random.Random(seed)
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in recs:
        cid = class_map.get(r["commodity"])
        if cid is not None:
            r["class_id"] = cid
            by_class[cid].append(r)
    selected: list[dict[str, Any]] = []
    for cid in sorted(by_class):
        items = by_class[cid]
        # Stable shuffle then stable sort by dev_stat rank => strongest signal first,
        # random within a rank.
        rng.shuffle(items)
        items.sort(key=lambda r: DEV_STAT_RANK.get(r["dev_stat"], 9))
        selected.extend(items[:per_class])
    return selected, per_class


# --------------------------------------------------------------------------------------
# Writers (worker processes).
# --------------------------------------------------------------------------------------
def _encode(positives: list[tuple[int, int, int]]) -> np.ndarray:
    return encode_detection_tile(
        positives,
        tile_size=DET_TILE,
        positive_size=DET_POS_SIZE,
        buffer_size=DET_BUFFER,
        nodata=io.CLASS_NODATA,
        background=CID_BACKGROUND,
    )[np.newaxis]


def _write_point(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    _, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"], proj)
    bounds = io.centered_bounds(col, row, DET_TILE, DET_TILE)
    x_min, y_min, _, _ = bounds
    positives: list[tuple[int, int, int]] = []
    cands = [(rec["lon"], rec["lat"], rec["class_id"])] + rec.get("neighbors", [])
    for lon, lat, cid in cands:
        _, c, r = io.lonlat_to_utm_pixel(lon, lat, proj)
        lc, lr = c - x_min, r - y_min
        if 0 <= lc < DET_TILE and 0 <= lr < DET_TILE:
            positives.append((lr, lc, cid))
    arr = _encode(positives)
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
    return "point"


def _write_negative(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    _, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"], proj)
    bounds = io.centered_bounds(col, row, DET_TILE, DET_TILE)
    arr = _encode([])
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
    return _write_point(rec)


# --------------------------------------------------------------------------------------
# Negatives: background-only tiles offset from any site, guaranteed site-free.
# --------------------------------------------------------------------------------------
_TO_3857 = None
_TO_4326 = None


def _to_3857(lon, lat):
    global _TO_3857
    if _TO_3857 is None:
        from pyproj import Transformer

        _TO_3857 = Transformer.from_crs(4326, 3857, always_xy=True)
    return _TO_3857.transform(lon, lat)


def _to_4326(x, y):
    global _TO_4326
    if _TO_4326 is None:
        from pyproj import Transformer

        _TO_4326 = Transformer.from_crs(3857, 4326, always_xy=True)
    return _TO_4326.transform(x, y)


def make_negatives(tree, recs, n, seed=7):
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    attempts = 0
    while len(out) < n and attempts < n * 100:
        attempts += 1
        base = recs[rng.randrange(len(recs))]
        ang = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(3000, 15000)
        bx, by = _to_3857(base["lon"], base["lat"])
        x, y = bx + dist * np.cos(ang), by + dist * np.sin(ang)
        if tree.query_ball_point([x, y], r=1000.0):
            continue
        lon, lat = _to_4326(x, y)
        if not (-60 <= lat <= 75):
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

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    download.download_http(DOWNLOAD_URL, raw / "mrds-csv.zip", headers=UA)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "USGS Mineral Resources Data System (MRDS), national CSV export. "
            "Public domain.\n"
            f"{DOWNLOAD_URL}\nProject page https://mrdata.usgs.gov/mrds/\n"
            "Positive-only mineral-site points; detection-encoded. Kept dev_stat in "
            "{Producer, Past Producer, Prospect}; class = primary commodity.\n"
        )

    io.check_disk()
    print("reading MRDS sites ...")
    recs = read_sites()
    print(f"  {len(recs)} observable sites with primary commodity + coords")

    class_map = build_class_map(recs)
    id_to_name = {v: k for k, v in class_map.items()}
    id_to_name[CID_BACKGROUND] = "background"
    print(f"  {len(class_map)} commodity classes")

    selected, per_class = select_records(recs, class_map)
    print(f"  per_class cap = {per_class}; selected {len(selected)} positive tiles")

    # KDTree over ALL classified sites (EPSG:3857) for neighbor marking + negative avoidance.
    xy = np.array([_to_3857(r["lon"], r["lat"]) for r in recs], dtype=float)
    tree = cKDTree(xy)

    negatives = make_negatives(tree, recs, N_NEGATIVES)
    print(f"  {len(negatives)} background-only negatives")

    rng = random.Random(123)
    for r in selected:
        r["kind"] = "point"
        r["year"] = YEARS[rng.randrange(len(YEARS))]
        x, y = _to_3857(r["lon"], r["lat"])
        idxs = tree.query_ball_point([x, y], r=1000.0)
        nb = []
        for i in idxs:
            o = recs[i]
            if o["source_id"] == r["source_id"]:
                continue
            cid = class_map.get(o["commodity"])
            if cid is not None:
                nb.append((o["lon"], o["lat"], cid))
        r["neighbors"] = nb[:200]
    for r in negatives:
        r["year"] = YEARS[rng.randrange(len(YEARS))]

    all_recs = selected + negatives
    for i, r in enumerate(all_recs):
        r["sample_id"] = f"{i:06d}"

    sel_counts: Counter = Counter(r["class_id"] for r in selected)
    print(
        f"selected {len(selected)} positives + {len(negatives)} negatives "
        f"= {len(all_recs)} total"
    )
    for cid in sorted(sel_counts, key=lambda c: -sel_counts[c])[:20]:
        print(f"  {sel_counts[cid]:5d}  {id_to_name[cid]}")

    results: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _dispatch, [dict(rec=r) for r in all_recs]),
            total=len(all_recs),
        ):
            results[res] += 1
    print("write results:", dict(results))
    io.check_disk()

    classes = [
        {
            "id": 0,
            "name": "background",
            "description": "Negative / non-mine land: pixels outside any mineral site.",
        }
    ]
    for cid in sorted(id_to_name):
        if cid == 0:
            continue
        classes.append(
            {
                "id": cid,
                "name": id_to_name[cid],
                "description": f"Mineral site whose primary commodity is "
                f"{id_to_name[cid].replace('_', ' ')} (MRDS commod1).",
            }
        )

    class_counts = {id_to_name[cid]: sel_counts[cid] for cid in sorted(sel_counts)}
    class_counts["background_negative_tiles"] = len(negatives)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "USGS (mrdata.usgs.gov)",
            "license": "public domain",
            "provenance": {
                "url": "https://mrdata.usgs.gov/mrds/",
                "download_url": DOWNLOAD_URL,
                "have_locally": False,
                "annotation_method": "manual compilation of mineral deposit records",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes,
            "nodata_value": io.CLASS_NODATA,
            "detection_encoding": {
                "applies_to": "all mineral-site points",
                "tile_size": DET_TILE,
                "positive_size": DET_POS_SIZE,
                "buffer_size": DET_BUFFER,
            },
            "num_samples": len(all_recs),
            "class_counts": class_counts,
            "notes": (
                "Positive-only mineral-site point dataset, detection-encoded (1 px positive "
                f"+ {DET_BUFFER} px nodata buffer ring in a {DET_TILE}x{DET_TILE} background "
                "context tile; nearby sites of any commodity also marked positive via a "
                "global KDTree). Class = primary commodity (commod1 first token, "
                "normalized/merged; ids 1..N by descending frequency, 254-class cap). Kept "
                "dev_stat in {Producer, Past Producer, Prospect} (physical ground "
                "disturbance); dropped Occurrence/Plant/Unknown and non-observable "
                "fluid/energy commodities (geothermal, natural gas, petroleum, helium, CO2, "
                "water, brine halogens). Within a class, Producer preferred over Past "
                "Producer over Prospect. CAVEAT: MRDS coordinates are frequently "
                "low-precision (PLSS-section-derived; true error often 100-400 m), so these "
                "are WEAK presence-detection targets, not precise footprints -- hence the "
                "wide ignore ring. Persistent sites -> 1-year window at a representative "
                "Sentinel-era year (2016-2022)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(all_recs)
    )
    print("done:", len(all_recs), "samples")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
