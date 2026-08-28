"""Process USGS MRDS (Mineral Resources Data System) into presence-only POINTS.

Source: USGS Mineral Resources Data System (MRDS), a global point database of mineral
deposits, mines, prospects and occurrences with commodity / deposit-type attributes.
Public domain. Downloaded as the national CSV export from mrdata.usgs.gov:

  https://mrdata.usgs.gov/mrds/mrds-csv.zip   (project page https://mrdata.usgs.gov/mrds/)

MRDS is a **positive-only point** dataset (a point marks a mineral site; absence is
everywhere else). We emit each site as one presence POINT carrying its primary-commodity
class into a dataset-wide ``points.geojson`` (joining the other presence-only point
datasets); cross-dataset negatives are supplied by assembly. The earlier per-detection
GeoTIFF tile encoding (1 px positive + nodata buffer ring + background fill + fabricated
background-only negative tiles) is dropped.

Observability (spec 8) — the crux for MRDS:
  * MRDS site coordinates are frequently LOW-PRECISION (many are PLSS-section-derived, so
    true positional error is often 100-400 m even though lon/lat are stored to 5 decimals),
    and many records are sub-pixel exploration points with no surface expression. We
    therefore keep only development statuses with a physical ground disturbance (Producer,
    Past Producer, Prospect) and DROP Occurrence / Plant / Unknown (a documented mineral
    occurrence or a processing plant is not a resolvable mine footprint). Even so, these are
    WEAK presence targets (a "mineral mine is present near here" signal), not precise
    footprints. Compare usgs_usmin_mine_features (map-digitised mine symbols, better
    positional accuracy, feature-type classes).

Class scheme: primary-commodity classes, ids 0..N-1 in descending frequency (254-class
cap). Non-observable fluid/energy commodities (geothermal, natural gas, petroleum, helium,
CO2, water, brine halogens) are dropped -- no surface mine footprint.

Time range: persistent, undated mine sites -> per spec 5 (static labels) a 1-year window at
a representative Sentinel-era year, pseudo-randomly spread across 2016-2022.

Run (idempotent; reuses cached raw):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.usgs_mrds_mineral_resources_data_system
"""

import argparse
import csv
import multiprocessing
import random
import re
import zipfile
from collections import Counter
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "usgs_mrds_mineral_resources_data_system"
NAME = "USGS MRDS (Mineral Resources Data System)"
DOWNLOAD_URL = "https://mrdata.usgs.gov/mrds/mrds-csv.zip"
UA = {"User-Agent": "Mozilla/5.0 (OlmoEarth research data pipeline)"}
CSV_NAME = "mrds.csv"

# Development statuses kept: sites with a physical ground disturbance observable at 10-30 m.
OBSERVABLE_DEV_STAT = {"Producer", "Past Producer", "Prospect"}

# Sampling parameters.
PER_CLASS = 1000  # up to 1000 presence points per commodity class (spec section 5)
YEARS = list(range(2016, 2023))  # representative Sentinel-era 1-year windows

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


def build_class_map(recs: list[dict[str, Any]]) -> dict[str, int]:
    """Assign class ids 0..N-1 to commodities in descending frequency (honors 254 cap)."""
    freq = Counter(r["commodity"] for r in recs)
    ordered = [c for c, _ in freq.most_common()]
    # uint8 label ids => keep the top 254 commodities by frequency.
    ordered = ordered[:254]
    return {c: i for i, c in enumerate(ordered)}


# --------------------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.parse_args()

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
            "Positive-only mineral-site points -> presence-only points. Kept dev_stat in "
            "{Producer, Past Producer, Prospect}; class = primary commodity.\n"
        )

    io.check_disk()
    print("reading MRDS sites ...")
    recs = read_sites()
    print(f"  {len(recs)} observable sites with primary commodity + coords")

    class_map = build_class_map(recs)
    id_to_name = {v: k for k, v in class_map.items()}
    print(f"  {len(class_map)} commodity classes")

    # Attach class id ("label") and drop sites whose commodity fell outside the 254-cap.
    labeled = []
    for r in recs:
        cid = class_map.get(r["commodity"])
        if cid is None:
            continue
        r["label"] = cid
        labeled.append(r)

    selected = balance_by_class(labeled, "label", per_class=PER_CLASS)

    rng = random.Random(123)
    for r in selected:
        r["year"] = YEARS[rng.randrange(len(YEARS))]

    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": r["label"],
                "time_range": io.year_range(r["year"]),
                "change_time": None,
                "source_id": r["source_id"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    sel_counts: Counter = Counter(r["label"] for r in selected)
    print(f"selected {len(selected)} presence points")
    for cid in sorted(sel_counts, key=lambda c: -sel_counts[c])[:20]:
        print(f"  {sel_counts[cid]:5d}  {id_to_name[cid]}")

    io.check_disk()

    classes = [
        {
            "id": cid,
            "name": id_to_name[cid],
            "description": f"Mineral site whose primary commodity is "
            f"{id_to_name[cid].replace('_', ' ')} (MRDS commod1).",
        }
        for cid in sorted(id_to_name)
    ]

    class_counts = {id_to_name[cid]: sel_counts[cid] for cid in sorted(sel_counts)}

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
            "num_samples": len(selected),
            "class_counts": class_counts,
            "notes": (
                "Positive-only mineral-site point dataset -> presence-only POINTS (converted "
                "from the earlier per-detection GeoTIFF tile encoding; negatives now come from "
                "assembly). Each kept site is one presence point carrying its primary-commodity "
                "class id in a dataset-wide points.geojson. Class = primary commodity (commod1 "
                "first token, normalized/merged; ids 0..N-1 by descending frequency, 254-class "
                "cap). Kept dev_stat in {Producer, Past Producer, Prospect} (physical ground "
                "disturbance); dropped Occurrence/Plant/Unknown and non-observable fluid/energy "
                "commodities (geothermal, natural gas, petroleum, helium, CO2, water, brine "
                "halogens). Balanced up to 1000/class (25k total cap). CAVEAT: MRDS coordinates "
                "are frequently low-precision (PLSS-section-derived; true error often 100-400 "
                "m), so these are WEAK presence targets, not precise footprints. Persistent "
                "sites -> 1-year window at a representative Sentinel-era year (2016-2022)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done:", len(selected), "samples")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
