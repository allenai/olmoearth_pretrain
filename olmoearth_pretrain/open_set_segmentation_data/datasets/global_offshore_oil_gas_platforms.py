"""Process the Global Offshore Oil & Gas Platforms (OOGPs) dataset into presence points.

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

Task type: presence-only classification POINTS (spec section 2a). Each selected platform is
emitted as a single presence point in one dataset-wide ``points.geojson``. Single foreground
class (offshore oil/gas platform, id 0). Negatives are supplied downstream by the assembly
step from other datasets.

Time / change handling (spec section 5). Platforms are PERSISTENT structures, not change
events. Year_label resolves presence only to a calendar year, and month-precision install
dates exist for only ~4% of platforms -- coarser/sparser than the ~1-2 month change-timing
bar -- so we do NOT emit dated change labels. Instead each platform is treated as a
persistent structure: a positive is emitted only for a calendar year in its Year_label,
guaranteeing the structure is present across the whole 1-year label window. change_time is
null and the time range is that calendar year (io.year_range).

Overlap note: this source partially overlaps the GFW SAR fixed-infrastructure dataset
(both are SAR-derived offshore oil/gas detections). That is acceptable -- downstream
assembly handles dedup.

Sampling: to avoid over-representing long-lived platforms, each physical platform
contributes at most one point, at a randomly chosen year from its Year_label; up to 1000
points selected via balance_by_class.

Run (reuses cached raw):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_offshore_oil_gas_platforms
"""

import argparse
import multiprocessing
import random
from collections import Counter
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.download import (
    download_zenodo,
    extract_zip,
)
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "global_offshore_oil_gas_platforms"
NAME = "Global Offshore Oil & Gas Platforms"
ZENODO = "https://doi.org/10.5281/zenodo.18350974"
ZENODO_RECORD = "18350974"
ZIP_FILE = "OOGPs_v1.0.0.zip"
GPKG_FILE = "OOGPs_all_v1.0.0.gpkg"
GPKG_LAYER = "platforms"

# Single foreground class: offshore oil/gas platform (id 0). No background class.
CID_PLATFORM = 0

CLASSES = [
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
SEED = 42


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


def _build_records(pts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One presence record per physical platform, at a randomly chosen year from its
    Year_label (so long-lived platforms are not over-represented), spread across years.
    """
    rng = random.Random(SEED)
    recs: list[dict[str, Any]] = []
    for p in pts:
        recs.append(
            {
                "label": CID_PLATFORM,
                "year": rng.choice(p["years"]),
                "lon": p["lon"],
                "lat": p["lat"],
                "source_id": f"oogps/{p['idx']}",
            }
        )
    return recs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.parse_args()

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

    recs = _build_records(pts)
    print(f"built {len(recs)} presence records", flush=True)
    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class)", flush=True)

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

    counts = Counter(r["label"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
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
            "num_samples": len(points),
            "class_counts": {"offshore_oil_gas_platform": counts.get(CID_PLATFORM, 0)},
            "notes": (
                "Presence-only classification POINTS converted from the old detection-tile "
                "encoding. Each selected offshore oil/gas platform is emitted as a single "
                "presence point (no fabricated GeoTIFF context, no background/negative tiles); "
                "single foreground class (id 0 = offshore_oil_gas_platform) from the OOGPs "
                "satellite (Sentinel-1 SAR) inventory (2017-2023, six major basins). "
                "Persistent-structure time model: a positive is emitted only for a calendar "
                "year listed in the platform's Year_label, so the structure is present across "
                "the whole 1-year window; change_time=null (Year_label is year-resolved and "
                "month-precision install dates cover only ~4% of platforms, both coarser/"
                "sparser than the ~1-2 month change-label bar, so NOT encoded as dated change). "
                "Each physical platform contributes at most one point (random year from its "
                "Year_label) to avoid over-representing long-lived platforms. up to 1000 "
                "points/class (balance_by_class). All labels post-2016. Partially overlaps the "
                "GFW SAR fixed-infrastructure dataset (both SAR-derived offshore oil/gas); "
                "downstream assembly dedups. Negatives are supplied downstream by the assembly "
                "step from other datasets."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(points)
    )
    print(f"done: {len(points)} points", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
