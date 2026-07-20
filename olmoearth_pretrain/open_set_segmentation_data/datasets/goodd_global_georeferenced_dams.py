"""Process GOODD (Global Georeferenced Dams) into presence-only points.

Source: Mulligan, M., van Soesbergen, A. & Saenz, L. "GOODD, a global dataset of more
than 38,000 georeferenced dams." Scientific Data 7, 31 (2020). Distributed by Global Dam
Watch (https://www.globaldamwatch.org/goodd), license CC0. Downloaded as a zip of two
ESRI shapefiles:
  - GOOD2_dams.shp      -> 38,667 dam-wall POINTS (EPSG:4326), digitized by manual
                           photointerpretation from Landsat/SPOT imagery.
  - GOOD2_catchments.shp -> upstream drainage catchment POLYGONS, one per dam.

We build a single-class, presence-only classification POINT dataset of dam walls (spec
section 2a). Each selected dam point is emitted as a single presence point in one
dataset-wide ``points.geojson``. The catchment polygons are DROPPED: they delineate the
full upstream hydrological drainage basin of each dam (often thousands of km2), not a
feature observable/segmentable at the dam location from S2/S1/Landsat at 10-30 m. Negatives
are supplied downstream by the assembly step from other datasets.

Class scheme: single foreground class (id 0 = dam).

Time range: dams are persistent structures (undated in the source). Per spec section 5
(static labels) each point gets a 1-year window at a representative Sentinel-era year,
spread pseudo-randomly across 2016-2022 for temporal diversity. change_time is null.

Run (reuses cached raw):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.goodd_global_georeferenced_dams
"""

import argparse
import multiprocessing
import random
from collections import Counter
from typing import Any

import fiona

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "goodd_global_georeferenced_dams"
NAME = "GOODD (Global Georeferenced Dams)"
DOWNLOAD_URL = "https://www.globaldamwatch.org/goodd"
DAMS_SHP = "Data/GOOD2_dams.shp"

# Single foreground class: dam (id 0). No background class.
CID_DAM = 0
CLASSES = [
    {
        "id": CID_DAM,
        "name": "dam",
        "description": "Dam wall location from GOODD, digitized by manual photointerpretation "
        "of Landsat/SPOT imagery. Marks a barrier/dam wall on a watercourse (all dam types; "
        "the source records only a point, no dam-type attribute).",
    },
]

PER_CLASS = 1000  # dam points selected (spec section 5, single class)
YEARS = list(range(2016, 2023))
SEED = 42


def dams_path() -> str:
    return str(io.raw_dir(SLUG) / DAMS_SHP)


def ensure_extracted() -> None:
    """Extract GOODD_data.zip into raw_dir if the dam shapefile is not present."""
    import zipfile

    raw = io.raw_dir(SLUG)
    if (raw / DAMS_SHP).exists():
        return
    zip_path = raw / "GOODD_data.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"{zip_path} missing; download GOODD_data.zip from {DOWNLOAD_URL}"
        )
    with zipfile.ZipFile(str(zip_path)) as z:
        z.extractall(str(raw))


def read_dams() -> list[dict[str, Any]]:
    """Read GOODD dam points into records with lon/lat + source id."""
    recs: list[dict[str, Any]] = []
    with fiona.open(dams_path()) as src:
        for i, feat in enumerate(src):
            if feat["geometry"] is None:
                continue
            lon, lat = feat["geometry"]["coordinates"][:2]
            dam_id = feat["properties"].get("DAM_ID")
            recs.append(
                {
                    "label": CID_DAM,
                    "lon": float(lon),
                    "lat": float(lat),
                    "source_id": f"DAM_ID/{int(dam_id)}"
                    if dam_id is not None
                    else f"row/{i}",
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
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "GOODD (Global Georeferenced Dams). Mulligan, van Soesbergen & Saenz, "
            "Sci Data 7, 31 (2020). Global Dam Watch. License CC0.\n"
            f"{DOWNLOAD_URL}\n"
            "GOODD_data.zip -> Data/GOOD2_dams.shp (38,667 dam points, EPSG:4326) + "
            "Data/GOOD2_catchments.shp (upstream drainage catchments; DROPPED for this "
            "dataset - not observable/segmentable at the dam location at 10-30 m).\n"
        )

    ensure_extracted()
    print("reading dam points ...", flush=True)
    dams = read_dams()
    print(f"  {len(dams)} dam points", flush=True)

    selected = balance_by_class(dams, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class)", flush=True)

    # Persistent, undated structures -> a representative 1-year window spread across the
    # Sentinel era for temporal diversity (deterministic per point).
    yrng = random.Random(123)
    points = []
    for i, r in enumerate(selected):
        year = YEARS[yrng.randrange(len(YEARS))]
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": r["label"],
                "time_range": io.year_range(year),
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
            "source": "Global Dam Watch / Scientific Data (Mulligan et al. 2020)",
            "license": "CC0",
            "provenance": {
                "url": DOWNLOAD_URL,
                "paper": "https://doi.org/10.1038/s41597-020-0362-5",
                "have_locally": False,
                "annotation_method": "manual photointerpretation of Landsat/SPOT imagery",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(points),
            "class_counts": {"dam": counts.get(CID_DAM, 0)},
            "notes": (
                "Presence-only classification POINTS converted from the old detection-tile "
                "encoding. Each selected dam wall is emitted as a single presence point (no "
                "fabricated GeoTIFF context, no background/negative tiles); single foreground "
                "class (id 0 = dam). 1000 of 38,667 dams sampled (balance_by_class, spec "
                "section 5 per-class cap). Catchment polygons (GOOD2_catchments) dropped: "
                "upstream drainage basins, not observable at the dam location. Persistent "
                "features -> 1-year window at a representative Sentinel-era year (2016-2022), "
                "change_time=null. Negatives are supplied downstream by the assembly step from "
                "other datasets."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(points)
    )
    print("done:", len(points), "points", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
