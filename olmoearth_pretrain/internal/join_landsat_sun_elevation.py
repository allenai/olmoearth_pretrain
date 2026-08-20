r"""Merge acquired Landsat sun elevation into the monthly modality CSV.

``acquire_landsat_sun_elevation.py`` produces a CSV keyed by
``(crs, col, row, tile_time, image_idx)`` holding the per-timestep
``sun_elevation`` (degrees) and ``platform`` (``LC08`` / ``LC09``) for the
existing Landsat monthly tiles. This script left-joins that onto the existing
``10_landsat_monthly.csv`` (which only has ``start_time`` / ``end_time``) and
writes a new monthly CSV whose columns match ``METADATA_COLUMNS`` -- i.e. with
``sun_elevation`` and ``platform`` appended.

The new CSV is what ``ConvertToH5py`` reads (via ``parse_modality_csv``) to
convert DN to reflectance / brightness temperature at h5-creation. Rows with no
acquired sun elevation get empty strings (the loader leaves those timesteps as
MISSING_VALUE), so the join never drops rows.

Usage:
    python -m olmoearth_pretrain.internal.join_landsat_sun_elevation \
        --landsat_csv /weka/dfive-default/helios/dataset/osm_sampling/10_landsat_monthly.csv \
        --sun_elevation_csv /weka/dfive-default/yawenz/landsat_refl_work/sun_elevation.csv \
        --output_csv /weka/dfive-default/helios/dataset/osm_sampling_landsat_refl/10_landsat_monthly.csv
"""

import argparse
import csv
import logging
import sys

from upath import UPath

from olmoearth_pretrain.dataset_creation.constants import METADATA_COLUMNS

logger = logging.getLogger(__name__)

csv.field_size_limit(sys.maxsize)


def _load_sun_elevation(
    path: str,
) -> dict[tuple[str, str, str, str, str], tuple[str, str]]:
    """Load the acquired sun-elevation CSV into a lookup keyed by the join tuple."""
    lookup: dict[tuple[str, str, str, str, str], tuple[str, str]] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (r["crs"], r["col"], r["row"], r["tile_time"], r["image_idx"])
            lookup[key] = (
                r.get("sun_elevation", "") or "",
                r.get("platform", "") or "",
            )
    return lookup


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Join acquired Landsat sun elevation into the monthly modality CSV",
    )
    parser.add_argument(
        "--landsat_csv", required=True, help="existing 10_landsat_monthly.csv"
    )
    parser.add_argument(
        "--sun_elevation_csv",
        required=True,
        help="output of acquire_landsat_sun_elevation.py",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="new monthly CSV with sun_elevation/platform",
    )
    args = parser.parse_args()

    logger.info(f"loading sun elevation from {args.sun_elevation_csv}")
    lookup = _load_sun_elevation(args.sun_elevation_csv)
    logger.info(f"loaded {len(lookup)} (window, image_idx) sun-elevation entries")

    out_path = UPath(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    matched = 0
    empty_elev = 0
    with (
        open(args.landsat_csv) as in_f,
        open(args.output_csv, "w", newline="") as out_f,
    ):
        reader = csv.DictReader(in_f)
        writer = csv.DictWriter(out_f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        for row in reader:
            total += 1
            key = (
                row["crs"],
                row["col"],
                row["row"],
                row["tile_time"],
                row["image_idx"],
            )
            sun_elevation, platform = lookup.get(key, ("", ""))
            if key in lookup:
                matched += 1
            if sun_elevation == "":
                empty_elev += 1
            out_row = {col: row.get(col, "") for col in METADATA_COLUMNS}
            out_row["sun_elevation"] = sun_elevation
            out_row["platform"] = platform
            writer.writerow(out_row)

    logger.info(
        f"wrote {total} rows to {args.output_csv} "
        f"(matched={matched}, empty_sun_elevation={empty_elev})"
    )


if __name__ == "__main__":
    main()
