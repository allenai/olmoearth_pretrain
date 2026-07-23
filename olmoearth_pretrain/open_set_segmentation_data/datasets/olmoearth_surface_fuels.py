"""Process OlmoEarth surface fuels into open-set-segmentation label points.

Source: local rslearn eval at ``olmoearth_evals/surface_fuels``. Each window is a
64x64 (10 m, local UTM) tile, but the ``label_raster`` layer carries a **single**
labeled pixel (all other pixels are nodata=255): the LANDFIRE FBFM40 (Scott & Burgan
40 Fire Behavior Fuel Models) surface-fuel class at one 10 m point. So despite the
manifest ``label_type: dense_raster``, the real label footprint is 1x1 -> this is a
sparse-point classification dataset, and we write ONE dataset-wide point table
(``points.geojson``, spec 2a) instead of 14k near-empty per-sample GeoTIFFs.

Each window's FBFM40 code (metadata ``options.category`` == ``data.csv``'s ``fbfm40``)
maps 1:1 to the label-raster class id 0..28 (ascending code order). Lon/lat come from
``data.csv`` (verified 1:1 with the windows). Labels describe the 2024 fuel state, so
each point gets a 1-year time range (2024). All source splits (train/val/test) are used.
Balanced to <= 1000/class (spec 5; total_cap=25000 lowers the effective per-class limit
to 25000 // 29 = 862).
"""

import argparse
import csv
import multiprocessing
from collections import Counter

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "olmoearth_surface_fuels"
NAME = "OlmoEarth surface fuels"
SOURCE = "/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/surface_fuels"
PER_CLASS = 1000
YEAR = 2024

# FBFM40 code -> (short name, description). Order defines class id (0..28), matching the
# source label_raster encoding (ascending FBFM40 code). LANDFIRE / Scott & Burgan (2005)
# 40 Fire Behavior Fuel Models. NB = non-burnable, GR = grass, GS = grass-shrub,
# SH = shrub, TU = timber-understory, TL = timber litter, SB = slash-blowdown.
FBFM40 = [
    (
        91,
        "NB1 Urban/Developed",
        "Non-burnable: urban or developed land, insufficient wildland fuel to carry fire.",
    ),
    (
        93,
        "NB3 Agricultural",
        "Non-burnable: maintained agricultural land, tilled or irrigated.",
    ),
    (98, "NB8 Open Water", "Non-burnable: open water."),
    (
        99,
        "NB9 Bare Ground",
        "Non-burnable: bare ground, rock, or sparsely vegetated barren.",
    ),
    (
        101,
        "GR1 Short sparse dry-climate grass",
        "Grass: short, sparse, dry-climate grass; low load.",
    ),
    (
        102,
        "GR2 Low-load dry-climate grass",
        "Grass: low load, dry-climate grass; moderately coarse and continuous.",
    ),
    (
        103,
        "GR3 Low-load coarse humid-climate grass",
        "Grass: low load, very coarse, humid-climate grass.",
    ),
    (
        121,
        "GS1 Low-load dry-climate grass-shrub",
        "Grass-shrub: low load, dry-climate grass and shrub mixture.",
    ),
    (
        122,
        "GS2 Moderate-load dry-climate grass-shrub",
        "Grass-shrub: moderate load, dry-climate grass and shrub mixture.",
    ),
    (
        141,
        "SH1 Low-load dry-climate shrub",
        "Shrub: low load, dry-climate shrub with some grass.",
    ),
    (
        142,
        "SH2 Moderate-load dry-climate shrub",
        "Shrub: moderate load, dry-climate shrub; no grass.",
    ),
    (
        143,
        "SH3 Moderate-load humid-climate shrub",
        "Shrub: moderate load, humid-climate shrub.",
    ),
    (
        144,
        "SH4 Low-load humid-climate timber-shrub",
        "Shrub: low load, humid-climate timber-shrub with woody understory.",
    ),
    (
        145,
        "SH5 High-load dry-climate shrub",
        "Shrub: high load, dry-climate shrub; tall, heavy fuel.",
    ),
    (
        161,
        "TU1 Low-load dry-climate timber-grass-shrub",
        "Timber-understory: low load, dry-climate timber with grass and shrub.",
    ),
    (
        162,
        "TU2 Moderate-load humid-climate timber-shrub",
        "Timber-understory: moderate load, humid-climate timber-shrub.",
    ),
    (
        163,
        "TU3 Moderate-load humid-climate timber-grass-shrub",
        "Timber-understory: moderate load, humid-climate timber-grass-shrub.",
    ),
    (
        165,
        "TU5 Very-high-load dry-climate timber-shrub",
        "Timber-understory: very high load, dry-climate timber-shrub; heavy forest litter with shrub.",
    ),
    (
        181,
        "TL1 Low-load compact conifer litter",
        "Timber litter: low load, compact conifer litter.",
    ),
    (
        182,
        "TL2 Low-load broadleaf litter",
        "Timber litter: low load, broadleaf litter.",
    ),
    (
        183,
        "TL3 Moderate-load conifer litter",
        "Timber litter: moderate load conifer litter.",
    ),
    (184, "TL4 Small downed logs", "Timber litter: moderate load, small downed logs."),
    (
        185,
        "TL5 High-load conifer litter",
        "Timber litter: high load conifer litter with small downed logs.",
    ),
    (
        186,
        "TL6 Moderate-load broadleaf litter",
        "Timber litter: moderate load, less compact broadleaf litter.",
    ),
    (187, "TL7 Large downed logs", "Timber litter: heavy load with large downed logs."),
    (
        188,
        "TL8 Long-needle litter",
        "Timber litter: moderate load long-needle pine litter.",
    ),
    (
        189,
        "TL9 Very-high-load broadleaf litter",
        "Timber litter: very high load, fluffy broadleaf litter.",
    ),
    (
        201,
        "SB1 Low-load activity fuel",
        "Slash-blowdown: low load activity fuel / fine fuel load from thinning.",
    ),
    (
        202,
        "SB2 Moderate-load activity fuel",
        "Slash-blowdown: moderate load activity fuel or light blowdown.",
    ),
]
CODE_TO_ID = {code: i for i, (code, _n, _d) in enumerate(FBFM40)}


def scan_records() -> list[dict]:
    """Read data.csv (lon/lat + FBFM40 code per window) into flat records."""
    recs = []
    with open(f"{SOURCE}/data.csv") as f:
        for r in csv.DictReader(f):
            code = int(r["fbfm40"])
            cid = CODE_TO_ID.get(code)
            if cid is None:
                continue
            recs.append(
                {
                    "lon": float(r["longitude"]),
                    "lat": float(r["latitude"]),
                    "label": cid,
                    "source_id": r["task_name"],
                }
            )
    return recs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    _ = args

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(f"local rslearn dataset: {SOURCE}\n")

    recs = scan_records()
    print(f"scanned {len(recs)} labeled points across {len(CODE_TO_ID)} classes")
    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} points (<= {PER_CLASS}/class, 25k total cap)")

    tr = io.year_range(YEAR)
    points = [
        {
            "id": f"{i:06d}",
            "lon": r["lon"],
            "lat": r["lat"],
            "label": r["label"],
            "time_range": tr,
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
            "source": "olmoearth",
            "license": "internal",
            "provenance": {
                "url": SOURCE,
                "have_locally": True,
                "annotation_method": "derived-product (LANDFIRE FBFM40 surface fuel model)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (_code, name, desc) in enumerate(FBFM40)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {
                name: counts.get(i, 0) for i, (_c, name, _d) in enumerate(FBFM40)
            },
            "notes": (
                "1x1 point-segmentation labels (each source 64x64 window carried a single "
                "labeled FBFM40 pixel). 29 LANDFIRE Scott & Burgan FBFM40 surface fuel-model "
                "classes over the US. All source splits (train/val/test) used. 1-year time "
                "range (2024, the labeled fuel-state year). Balanced to <=1000/class; "
                "total_cap=25000 lowers the effective per-class limit to 25000//29=862."
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
