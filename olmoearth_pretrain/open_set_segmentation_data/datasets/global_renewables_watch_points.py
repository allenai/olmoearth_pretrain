"""Global Renewables Watch wind turbines -> presence-only point dataset.

Companion to ``global_renewables_watch`` (which handles the solar PV polygons). The wind
turbines are an isolated global point inventory; previously they were encoded with the
fabricated detection encoding (positive square + nodata buffer + synthetic background/negative
tiles). Per spec 4/5 such point inventories are now emitted as **presence-only points** (one
point per turbine, class ``wind_turbine``), with negatives supplied downstream by the assembly
step rather than fabricated here.

Source: GRW v1.0 2024-Q2 ``wind_all_2024q2_v1.gpkg`` (shared raw dir with
``global_renewables_watch``). Per-feature ``construction_year`` sets a ~1-year time range.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_renewables_watch_points
"""

import argparse
from collections import Counter
from typing import Any

import fiona
import shapely

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.datasets.global_renewables_watch import (
    WIND_FILE,
    _lonlat,
)
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "global_renewables_watch_points"
NAME = "Global Renewables Watch (wind turbines)"
# Raw files live in the solar dataset's raw dir (same product download).
RAW_SLUG = "global_renewables_watch"
PER_CLASS = 1000
YEARS = list(range(2017, 2025))
PER_YEAR = PER_CLASS // len(YEARS)

CID_WIND = 0
CLASSES = [
    {
        "id": CID_WIND,
        "name": "wind_turbine",
        "description": "Individual wind turbine location (point), from GRW PlanetScope "
        "deep-learning detections with human QC. Presence-only point.",
    },
]


def read_wind() -> list[dict[str, Any]]:
    path = io.raw_dir(RAW_SLUG) / WIND_FILE
    recs: list[dict[str, Any]] = []
    with fiona.open(path.path) as src:
        for i, feat in enumerate(src):
            props = feat["properties"]
            year = props.get("construction_year")
            if year is None:
                continue
            geom = shapely.geometry.shape(feat["geometry"])
            if geom.is_empty:
                continue
            x, y = feat["geometry"]["coordinates"]
            lon, lat = _lonlat(x, y)
            recs.append(
                {"year": int(year), "lon": lon, "lat": lat, "source_id": f"wind/{i}"}
            )
    return recs


def main() -> None:
    argparse.ArgumentParser().parse_args()
    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Global Renewables Watch v1.0 (2024 Q2) wind turbines; raw GeoPackage under "
            f"raw/{RAW_SLUG}/{WIND_FILE}.\n"
        )

    wind = read_wind()
    print(f"{len(wind)} wind turbine points")
    sel = balance_by_class(wind, "year", per_class=PER_YEAR, seed=42)[:PER_CLASS]
    points = [
        {
            "id": f"{i:06d}",
            "lon": r["lon"],
            "lat": r["lat"],
            "label": CID_WIND,
            "time_range": io.year_range(r["year"]),
            "change_time": None,
            "source_id": r["source_id"],
        }
        for i, r in enumerate(sel)
    ]
    io.write_points_table(SLUG, "classification", points)
    counts = Counter(p["label"] for p in points)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "GitHub (Microsoft/Planet/TNC)",
            "license": "MIT",
            "provenance": {
                "url": "https://github.com/microsoft/global-renewables-watch",
                "have_locally": False,
                "annotation_method": "deep learning (PlanetScope) + human QC",
                "release": "v1.0 (2024 Q2)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(points),
            "class_counts": {"wind_turbine": counts.get(CID_WIND, 0)},
            "notes": (
                "Presence-only wind-turbine points (converted from the old detection-tile "
                "encoding; negatives supplied by the assembly step). 1-year time_range on "
                "each turbine's construction_year (2017-2024), change_time=null. Solar PV "
                "footprints from the same product are the separate global_renewables_watch "
                "dataset."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(points)
    )
    print(f"done: {len(points)} wind points")


if __name__ == "__main__":
    main()
