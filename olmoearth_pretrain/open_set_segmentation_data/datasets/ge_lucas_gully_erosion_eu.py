"""Process GE-LUCAS Gully Erosion (EU) into open-set-segmentation point labels.

Source: EC JRC / ESDAC "Gully erosion in the EU" (Borrelli et al. 2025, Nature Scientific
Data 12, 755). The ESDAC download portal is registration-gated, but the very same dataset
is deposited openly on Figshare (DOI 10.6084/m9.figshare.27211473), so we pull from there
and avoid the credential wall.

We use "Data 1 - LUCAS2022 original" (the full LUCAS 2022 feature-detection survey of
399,591 grid locations, in WGS84 lon/lat), which carries the gully-erosion attributes
directly:
  - SURVEY_GULLY_SIGNS: 1 = gully channel present (3,116 pts), 2 = absent (396,475 pts)
  - SURVEY_GULLY_TYPE : 1 ephemeral, 2 permanent, 3 badlands (NaN when absent)

This is a pure sparse-point dataset (each label is a single 10 m pixel), so per spec 2a we
write ONE dataset-wide point table (points.json), not per-point GeoTIFFs. We fold gully
presence and erosion class into one unified 4-class scheme (0 = no gully, 1 = ephemeral,
2 = permanent, 3 = badlands), balanced to <=1000 per class (spec 5).

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.ge_lucas_gully_erosion_eu
"""

import argparse
import csv
import io as _io
import multiprocessing
import zipfile
from collections import Counter
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.download import download_http
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "ge_lucas_gully_erosion_eu"
NAME = "GE-LUCAS Gully Erosion (EU)"
YEAR = 2022
PER_CLASS = 1000

# Figshare open deposit (mirror of the ESDAC registration-gated release).
FIGSHARE_DOI = "10.6084/m9.figshare.27211473"
DATA1_URL = (
    "https://ndownloader.figshare.com/files/53415074"  # Data 1 - LUCAS2022 original.zip
)
DATA1_CSV = "Data 1 - LUCAS2022 original/LUCAS2022_original.csv"

# Unified presence/erosion-class scheme. id 0 is the background/no-gully class.
CLASSES = [
    (
        "No gully channel",
        "LUCAS 2022 location surveyed for gully erosion where no gully channel was observed (SURVEY_GULLY_SIGNS=2).",
    ),
    (
        "Ephemeral gully",
        "Ephemeral gully channel affecting only the topsoil, <0.5 m deep (SURVEY_GULLY_TYPE=1).",
    ),
    (
        "Permanent gully",
        "Permanent gully channel affecting both topsoil and subsoil, 0.5-30 m deep (SURVEY_GULLY_TYPE=2).",
    ),
    ("Badlands", "Badlands, i.e. gullied landscape areas (SURVEY_GULLY_TYPE=3)."),
]
# gully type code -> class id for present gullies
TYPE_TO_ID = {1: 1, 2: 2, 3: 3}


def _read_records(csv_bytes: bytes) -> list[dict[str, Any]]:
    """Parse the LUCAS2022 CSV into flat point records with a unified class id."""
    recs: list[dict[str, Any]] = []
    reader = csv.DictReader(_io.TextIOWrapper(_io.BytesIO(csv_bytes), encoding="utf-8"))
    for row in reader:
        try:
            lon = float(row["POINT_LONG"])
            lat = float(row["POINT_LAT"])
        except (KeyError, ValueError, TypeError):
            continue
        signs = (row.get("SURVEY_GULLY_SIGNS") or "").strip()
        if signs == "1":  # gully present -> class by type
            gt = (row.get("SURVEY_GULLY_TYPE") or "").strip()
            try:
                cid = TYPE_TO_ID[int(float(gt))]
            except (ValueError, KeyError):
                continue
        elif signs == "2":  # no gully
            cid = 0
        else:
            continue
        recs.append(
            {
                "lon": lon,
                "lat": lat,
                "label": cid,
                "source_id": str(row.get("POINT_ID", "")),
            }
        )
    return recs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    # Download raw source (Figshare mirror) to weka raw/.
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / "Data1_LUCAS2022_original.zip"
    if not zip_path.exists():
        download_http(DATA1_URL, zip_path)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "ESDAC 'Gully erosion in the EU' (Borrelli et al. 2025, Nature Sci Data 12, 755).\n"
            "ESDAC portal is registration-gated; open Figshare mirror used instead.\n"
            f"Figshare DOI: {FIGSHARE_DOI}\n"
            f"Data 1 - LUCAS2022 original: {DATA1_URL}\n"
        )

    with zipfile.ZipFile(str(zip_path)) as zf:
        csv_bytes = zf.read(DATA1_CSV)
    recs = _read_records(csv_bytes)
    print(
        f"parsed {len(recs)} LUCAS points; raw class counts: {Counter(r['label'] for r in recs)}"
    )

    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(
        f"selected {len(selected)} (<= {PER_CLASS}/class): {Counter(r['label'] for r in selected)}"
    )

    points = [
        {
            "id": f"{i:06d}",
            "lon": r["lon"],
            "lat": r["lat"],
            "label": r["label"],
            "time_range": io.year_range(YEAR),
            "source_id": r["source_id"],
        }
        for i, r in enumerate(selected)
    ]
    io.write_points_table(SLUG, "classification", points)

    counts = Counter(r["label"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "EC JRC / ESDAC (Figshare mirror)",
            "license": "free (ESDAC registration; open Figshare mirror CC BY 4.0)",
            "provenance": {
                "url": "https://esdac.jrc.ec.europa.eu/content/gully-erosion-eu",
                "figshare_doi": FIGSHARE_DOI,
                "have_locally": False,
                "annotation_method": "manual/expert LUCAS 2022 in-situ + on-screen survey",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {
                name: counts.get(i, 0) for i, (name, _d) in enumerate(CLASSES)
            },
            "notes": (
                "Sparse point classification from the LUCAS 2022 gully feature-detection "
                "survey. Unified scheme folds gully presence (0=absent) and erosion type "
                "(1=ephemeral, 2=permanent, 3=badlands). 1-year time range anchored on 2022. "
                "The gully-occurrence probability RASTER (Data 4) is not encoded here (point-only)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
