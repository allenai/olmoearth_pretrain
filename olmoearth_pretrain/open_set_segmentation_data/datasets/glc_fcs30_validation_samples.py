"""Process the GLC_FCS30 global land-cover validation samples into open-set-segmentation
labels.

Source: Zenodo record 10.5281/zenodo.3551994 (concept) -> 3551995 (version), "A Dataset of
Global Land Cover Validation Samples" (Zhang et al., used to validate the GLC_FCS30 30 m
global land-cover product, ESSD). One RAR holds an ESRI point shapefile
``GLC_ValidationSampleSet.shp`` (EPSG:4326) with ~44,514 points, each carrying a single
integer field ``sample_lab`` = the LCCS *fine* land-cover code (24 distinct codes). Points
were interpreted/re-checked on high-resolution Google Earth imagery.

This is a pure sparse-point classification dataset (each label is a single 10 m pixel with a
class id), so per spec 2a we write ONE dataset-wide GeoJSON point table (points.geojson) via
``io.write_points_table`` rather than per-point GeoTIFFs. Balanced to <=1000 per class with
the 25k per-dataset cap (24 classes -> at most 24k, well under the cap); every class is kept.

Time range: the shapefile has no per-point acquisition date, and the validation set is a
static/stable reference (points were chosen to be homogeneous, persistent land cover, then
visually re-checked). GLC_FCS30 spans the ~2015-2020 epochs, so per spec 5 (static labels ->
representative 1-year Sentinel-era window) we assign a single representative 1-year window in
2018 (mid-point of the 2015-2020 span, firmly inside the Sentinel-2 era) to every point.
``change_time`` is null (no dated event).
"""

import argparse
import os
from collections import Counter
from typing import Any

import fiona

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "glc_fcs30_validation_samples"
SOURCE_RECORD = "3551995"  # version record under concept DOI 10.5281/zenodo.3551994
SHP_REL = "extracted/GLC_ValidationSampleSet_v1/GLC_ValidationSampleSet.shp"
PER_CLASS = 1000
# Static/stable reference labels, no per-point date -> one representative Sentinel-era year.
REPRESENTATIVE_YEAR = 2018

# GLC_FCS30 LCCS *fine* classification system (from the record's "Data description.docx",
# Table 1). Ordered by LCCS code ascending -> class id 0..23. Each entry:
# (lccs_code, name, level1_group, description).
CLASSES: list[tuple[int, str, str, str]] = [
    (
        10,
        "Rainfed cropland",
        "Cropland",
        "Cropland dependent on rainfall (no irrigation).",
    ),
    (
        11,
        "Herbaceous cover cropland",
        "Cropland",
        "Rainfed cropland with herbaceous crop cover.",
    ),
    (
        12,
        "Tree or shrub cover (orchard) cropland",
        "Cropland",
        "Rainfed cropland of tree/shrub crops (orchards).",
    ),
    (20, "Irrigated cropland", "Cropland", "Cropland supplied by irrigation."),
    (
        50,
        "Evergreen broadleaved forest",
        "Forest",
        "Forest dominated by evergreen broadleaved trees.",
    ),
    (
        60,
        "Deciduous broadleaved forest",
        "Forest",
        "Forest dominated by deciduous broadleaved trees.",
    ),
    (
        70,
        "Evergreen needleleaved forest",
        "Forest",
        "Forest dominated by evergreen needleleaved trees.",
    ),
    (
        80,
        "Deciduous needleleaved forest",
        "Forest",
        "Forest dominated by deciduous needleleaved trees.",
    ),
    (
        90,
        "Mixed leaf forest",
        "Forest",
        "Forest mixing broadleaved and needleleaved trees.",
    ),
    (120, "Shrubland", "Shrubland", "Woody vegetation dominated by shrubs."),
    (
        121,
        "Evergreen shrubland",
        "Shrubland",
        "Shrubland dominated by evergreen shrubs.",
    ),
    (
        122,
        "Deciduous shrubland",
        "Shrubland",
        "Shrubland dominated by deciduous shrubs.",
    ),
    (130, "Grassland", "Grassland", "Herbaceous, grass-dominated vegetation."),
    (140, "Lichens and mosses", "Tundra", "Lichen/moss-dominated (tundra) cover."),
    (150, "Sparse vegetation", "Bare areas", "Sparse vegetation (<15% cover)."),
    (152, "Sparse shrubland", "Bare areas", "Sparse shrub cover (<15%)."),
    (153, "Sparse herbaceous cover", "Bare areas", "Sparse herbaceous cover (<15%)."),
    (
        180,
        "Wetlands",
        "Wetlands",
        "Land periodically flooded / saturated (marsh, swamp).",
    ),
    (
        190,
        "Impervious surfaces",
        "Impervious surfaces",
        "Human-made impervious surfaces (urban, roads, buildings).",
    ),
    (200, "Bare areas", "Bare areas", "Bare land with little or no vegetation."),
    (
        201,
        "Consolidated bare areas",
        "Bare areas",
        "Consolidated bare surfaces (rock, hardpan).",
    ),
    (
        202,
        "Unconsolidated bare areas",
        "Bare areas",
        "Unconsolidated bare surfaces (sand, gravel).",
    ),
    (210, "Water body", "Water body", "Open water (rivers, lakes, reservoirs, sea)."),
    (
        220,
        "Permanent ice and snow",
        "Permanent ice and snow",
        "Perennial ice / snow cover (glaciers, ice caps).",
    ),
]
CODE_TO_ID = {code: i for i, (code, _n, _g, _d) in enumerate(CLASSES)}


def scan_records() -> list[dict[str, Any]]:
    """Read the validation-sample shapefile into flat records (one per point).

    A single ~1.2 MB shapefile of ~44.5k points reads in ~1 s serially, so no pool needed.
    """
    shp = os.path.join(io.raw_dir(SLUG).path, SHP_REL)
    recs: list[dict[str, Any]] = []
    with fiona.open(shp) as src:
        for fid, feat in enumerate(src):
            props = feat["properties"]
            code = int(props["sample_lab"])
            if code not in CODE_TO_ID:
                continue
            geom = feat["geometry"]["coordinates"]
            lon = float(props.get("lon", geom[0]))
            lat = float(props.get("lat", geom[1]))
            recs.append(
                {
                    "lon": lon,
                    "lat": lat,
                    "code": code,
                    "label": CODE_TO_ID[code],
                    "source_id": f"sample_{fid}",
                }
            )
    return recs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    if not (raw / SHP_REL).exists():
        from olmoearth_pretrain.open_set_segmentation_data import download

        download.download_zenodo(SOURCE_RECORD, raw)
        # Extract the RAR (bsdtar handles rar) into raw/extracted/.
        import subprocess

        rar = raw / "GLC_ValidationSampleSet_v1.rar"
        extracted = raw / "extracted"
        extracted.mkdir(parents=True, exist_ok=True)
        subprocess.run(["bsdtar", "-xf", rar.path, "-C", extracted.path], check=True)

    recs = scan_records()
    print(
        f"scanned {len(recs)} labeled points across {len({r['label'] for r in recs})} classes"
    )

    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class, 25k cap)")

    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": r["label"],
                "time_range": io.year_range(REPRESENTATIVE_YEAR),
                "change_time": None,
                "source_id": r["source_id"],
                "lccs_code": r["code"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    counts = Counter(r["label"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "GLC_FCS30 Validation Samples",
            "task_type": "classification",
            "source": "Zenodo / ESSD",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.org/10.5281/zenodo.3551994",
                "have_locally": False,
                "annotation_method": "manual interpretation + visual checking on Google Earth",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {
                    "id": i,
                    "name": name,
                    "description": f"[{group}] {desc} (LCCS fine code {code}).",
                }
                for i, (code, name, group, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {
                name: counts.get(i, 0) for i, (_c, name, _g, _d) in enumerate(CLASSES)
            },
            "notes": (
                "Sparse-point (1x1) classification; label = LCCS fine land-cover class "
                "(24 classes). Global validation samples for GLC_FCS30, re-checked on Google "
                "Earth. No per-point date; static/stable reference labels assigned a single "
                f"representative 1-year Sentinel-era window ({REPRESENTATIVE_YEAR}). Balanced "
                "to <=1000/class (25k cap); all 24 classes kept. Raw lccs_code retained per "
                "point in points.geojson properties."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    main()
