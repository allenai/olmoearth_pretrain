"""Triage Tallo global tree allometry database for open-set-segmentation labels.

Source: Tallo (Jucker et al. 2022, Global Change Biology), Zenodo record 6637599, a
global database of ~499k georeferenced individual-tree records (stem diameter, height,
crown radius) across 5,163 species / 1,453 genera / 187 families, compiled from ~69 field
allometry / forest-inventory sources.

OUTCOME: REJECTED (does not fit the open-set-segmentation label bank). Two decisive,
compounding reasons, both discovered by downloading and analyzing the actual table:

1. Pre-2016 rule (primary). The published ``Tallo.csv`` contains **no per-record
   measurement date** at all (columns: tree_id, division, family, genus, species,
   latitude, longitude, stem_diameter_cm, height_m, crown_radius_m, height_outlier,
   crown_radius_outlier, reference_id). The manifest's "records are dated; filter to
   Sentinel-2 era" note and its time_range [2016, 2022] do not hold for this release. The
   only temporal signal is the *publication* year of each record's reference, which is not
   a valid measurement-era filter (field allometry campaigns predate publication, often by
   years to decades). Even using publication year as a generous upper bound, ~200k records
   come from pre-2016 publications and ~101k references have no parseable year; the dataset
   is a compilation of largely pre-2016 field measurements. Because no record can be
   confidently placed in the post-2016 Sentinel era, there is no usable post-2016 subset to
   keep, so the pre-2016 rule (reject if not resolvable to post-2016) applies.

2. Georeferencing / observability at 10-30 m (compounding). Coordinates are plot-centroid,
   not individual-tree GPS: 61,856 unique lon/lat points for 498,838 records (mean 8.1
   trees per point; one point stacks 23,249 trees), rounded to ~0.001-1 deg (~100 m to
   >10 km), and include FIA (~1 mile coordinate fuzzing) and NEON plot data. Individual
   trees at plot-rounded coordinates are not reliably observable or placeable on the 10 m
   S2 grid (the spec explicitly lists "individual small trees" and "coordinate-fuzzed
   points like FIA ~1 mi" as not observable at 10-30 m). This applies equally to a species
   classification target and to a height/biomass regression target (one tree != a 10 m
   pixel's canopy height).

This script downloads the (label-only) CSVs, prints the diagnostics behind the decision,
and records the rejection via the per-dataset registry entry. It writes nothing to weka
``datasets/tallo/`` except ``registry_entry.json`` (per spec, rejected datasets write only
that plus the repo summary). Re-run if a dated / individual-GPS Tallo release appears.

Reproduce: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.tallo
"""

import argparse
import re

import pandas as pd

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest

SLUG = "tallo"
NAME = "Tallo"
ZENODO_RECORD = "6637599"
FILES = ["Tallo.csv", "Tallo_metadata.csv", "Tallo_references.csv"]

REJECT_NOTES = (
    "pre-2016: Tallo.csv has no per-record measurement date; compilation of largely "
    "pre-2016 field allometry (only reference publication year available, an invalid "
    "measurement-era proxy) so no usable post-2016 subset. Compounding: plot-centroid "
    "coordinates (61.9k unique points for 499k trees, rounded ~0.001-1deg, incl. FIA "
    "~1mi-fuzzed & NEON) -> individual trees not observable/placeable at 10-30 m."
)


def _pub_year(s: str) -> int | None:
    m = re.search(r"\((\d{4})\)", str(s))
    if m:
        return int(m.group(1))
    m = re.search(r"(19\d{2}|20\d{2})", str(s))
    return int(m.group(1)) if m else None


def analyze() -> None:
    """Download label CSVs and print the diagnostics behind the rejection."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Tallo database (Jucker et al. 2022, Global Change Biology).\n"
            f"Zenodo record {ZENODO_RECORD}: https://doi.org/10.5281/zenodo.6637599\n"
            "Files used (label-only metadata table; no imagery): "
            + ", ".join(FILES)
            + "\n"
        )
    for fn in FILES:
        download.download_http(
            f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{fn}/content",
            raw / fn,
        )

    df = pd.read_csv(str(raw / "Tallo.csv"), low_memory=False)
    refs = pd.read_csv(str(raw / "Tallo_references.csv"), encoding="latin-1")
    print(f"rows={len(df)} cols={list(df.columns)}")
    has_date = any(
        c.lower() in {"year", "date", "measurement_year", "observation_year"}
        for c in df.columns
    )
    print(f"has per-record measurement date column: {has_date}")

    refs["pubyear"] = refs["source"].map(_pub_year)
    ymap = dict(zip(refs["reference_id"], refs["pubyear"]))
    df["pubyear"] = df["reference_id"].map(ymap)
    print(
        f"reference pubyear (proxy only): <2016={int((df['pubyear'] < 2016).sum())} "
        f">=2016={int((df['pubyear'] >= 2016).sum())} "
        f"undatable={int(df['pubyear'].isna().sum())}"
    )

    uc = df.groupby(["latitude", "longitude"]).size()
    print(
        f"georeferencing: {len(uc)} unique coord points for {len(df)} records "
        f"(mean {df.shape[0] / len(uc):.1f} trees/point, max {int(uc.max())})"
    )
    print(f"species={df['species'].nunique()} genera={df['genus'].nunique()}")
    print("DECISION: rejected -> " + REJECT_NOTES)


def main() -> None:
    argparse.ArgumentParser().parse_args()
    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")
    analyze()
    manifest.write_registry_entry(SLUG, "rejected", notes=REJECT_NOTES)
    print("done (rejected)")


if __name__ == "__main__":
    main()
