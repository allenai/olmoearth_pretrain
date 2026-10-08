"""Process OlmoEarth Mozambique LULC + crop-type field surveys into open-set-seg points.

Source: OlmoEarth Mozambique LULC project (internal), staged locally. The authoritative
labels are field-survey **point** samples over three Mozambique provinces (Gaza, Manica,
Zambezia), distributed as GeoPackages at
``/weka/dfive-default/yawenz/datasets/mozambique/train_test_samples``:
  - LULC:      {gaza,manica,zambezia}_{train,test}.gpkg   (col ``class`` = int 0..6)
  - crop type: {training,test}_gaza_zambezia_manica.gpkg  (col ``crop1`` = crop name str)
Each feature is a single labelled Point (photo-/field-interpreted reference), so this is a
**sparse-point** dataset (spec 2a): we emit ONE dataset-wide ``points.geojson`` table, not
per-sample GeoTIFFs. (The manifest lists ``label_type: polygons`` but every source feature
is a Point; the paired rslearn project also treats each label as a single 10 m pixel — see
``create_label_raster.py`` draws only the centre 1x1 px.)

Georeferencing (PASTIS-style dummy-bounds check): we do NOT read the staged rslearn window
bounds at all — we read lon/lat straight from the source GPKGs (LULC in EPSG:4326, crop in
EPSG:3036 = Moznet / UTM 36S) and reproject to WGS84. All centroids land in Mozambique
(lon ~31.3-38.6, lat ~-25.3 to -15.3), so geolocation is real and needs no recovery.

Unified class scheme (spec 5 multi-target -> ONE dataset with a unified class map). The
LULC classes 0..6 keep their native ids; the 7 crop types are appended as ids 7..13 (they
refine the generic ``Cropland`` LULC class with the surveyed crop). 14 classes, uint8, well
under the 254 cap. nodata=255 (unused: every point carries a real class).

Time range: the surveyed growing season 2024-10-23 -> 2025-06-20 (UTC, ~240 days <= 1 yr,
post-2016), from the project's per-province window time ranges. change_time=null (this is
state classification, not a dated change event).

Sampling: <=1000 points per class (spec 5), 25k total cap. All source splits used.

Run (idempotent; rewrites points.geojson + metadata deterministically):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_mozambique_lulc
"""

import argparse
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "olmoearth_mozambique_lulc"
NAME = "OlmoEarth Mozambique LULC"
GPKG_DIR = Path("/weka/dfive-default/yawenz/datasets/mozambique/train_test_samples")
STAGED_RSLEARN = (
    "/weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc/20251202"
)
PER_CLASS = 1000

# Surveyed growing season shared by all three provinces (project GROUP_TIME); <= 1 year.
TIME_RANGE = (
    datetime(2024, 10, 23, tzinfo=UTC),
    datetime(2025, 6, 20, tzinfo=UTC),
)

# LULC source ``class`` int -> (id, name, description). Ids kept native (0..6).
LULC_CLASSES: list[tuple[str, str]] = [
    ("Water", "Open water: rivers, lakes, reservoirs, ponds and coastal water."),
    ("Bare Ground", "Bare soil, sand, or rock with little to no vegetation."),
    (
        "Rangeland",
        "Grass- and shrub-dominated rangeland / natural herbaceous vegetation.",
    ),
    (
        "Flooded Vegetation",
        "Seasonally or permanently flooded vegetation (wetland, marsh, mangrove).",
    ),
    ("Trees", "Tree-dominated land cover: forest, woodland, and tree plantations."),
    (
        "Cropland",
        "Cultivated / managed agricultural land (generic cropland, crop unspecified).",
    ),
    (
        "Buildings",
        "Human-made built-up structures and impervious surfaces (settlements, buildings).",
    ),
]

# Crop-type source ``crop1`` string -> appended id (offset by len(LULC_CLASSES)).
CROP_CLASSES: list[tuple[str, str, str]] = [
    ("corn", "corn", "Maize / corn (Zea mays), a staple summer cereal."),
    (
        "cassava",
        "cassava",
        "Cassava (Manihot esculenta), a perennial root/tuber staple.",
    ),
    ("rice", "rice", "Rice (Oryza sativa), often grown in lowland / flooded fields."),
    ("sesame", "sesame", "Sesame (Sesamum indicum), an oilseed cash crop."),
    ("beans", "beans", "Beans / common legumes (Phaseolus and related pulses)."),
    ("millet", "millet", "Millet (pearl / finger millet), a drought-tolerant cereal."),
    ("sorghum", "sorghum", "Sorghum (Sorghum bicolor), a drought-tolerant cereal."),
]

LULC_OFFSET = 0
CROP_OFFSET = len(LULC_CLASSES)  # crop ids start at 7

# Unified class list (id -> name, description).
CLASSES: list[tuple[str, str]] = [(n, d) for (n, d) in LULC_CLASSES] + [
    (n, d) for (_key, n, d) in CROP_CLASSES
]
CROP_NAME_TO_ID = {key: CROP_OFFSET + i for i, (key, _n, _d) in enumerate(CROP_CLASSES)}

LULC_FILES = [
    "gaza_train",
    "gaza_test",
    "manica_train",
    "manica_test",
    "zambezia_train",
    "zambezia_test",
]
CROP_FILES = ["training_gaza_zambezia_manica", "test_gaza_zambezia_manica"]


def scan_records() -> list[dict[str, Any]]:
    """Read all source GPKGs -> flat point records with unified class id + lon/lat (WGS84).

    Only 8 small GPKG files are read (a few thousand points each), so a direct read is
    fast and correct; the spec's Pool(64) guidance targets tens-of-thousands of small
    weka window files, which does not apply here.
    """
    recs: list[dict[str, Any]] = []
    # ---- LULC points (native class ids 0..6) --------------------------------------
    for stem in LULC_FILES:
        gdf = gpd.read_file(str(GPKG_DIR / f"{stem}.gpkg")).to_crs(4326)
        split = "test" if stem.endswith("_test") else "train"
        province = stem.rsplit("_", 1)[0]
        for fid, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            cid = int(row["class"])
            if not 0 <= cid < len(LULC_CLASSES):
                continue
            pt = geom if geom.geom_type == "Point" else geom.centroid
            recs.append(
                {
                    "lon": float(pt.x),
                    "lat": float(pt.y),
                    "label": LULC_OFFSET + cid,
                    "source_id": f"lulc/{province}/{split}/{fid}",
                }
            )
    # ---- Crop-type points (ids 7..13) ---------------------------------------------
    for stem in CROP_FILES:
        gdf = gpd.read_file(str(GPKG_DIR / f"{stem}.gpkg")).to_crs(4326)
        split = "test" if stem.startswith("test") else "train"
        for fid, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            crop = row["crop1"]
            if crop not in CROP_NAME_TO_ID:
                continue
            pt = geom if geom.geom_type == "Point" else geom.centroid
            recs.append(
                {
                    "lon": float(pt.x),
                    "lat": float(pt.y),
                    "label": CROP_NAME_TO_ID[crop],
                    "source_id": f"crop_type/{split}/{fid}",
                }
            )
    return recs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "OlmoEarth Mozambique LULC + crop-type field surveys (internal), staged locally.\n"
            f"Source label GPKGs: {GPKG_DIR}\n"
            "  LULC:      {gaza,manica,zambezia}_{train,test}.gpkg (col 'class' int 0..6)\n"
            "  crop type: {training,test}_gaza_zambezia_manica.gpkg (col 'crop1' str)\n"
            f"Paired staged rslearn dataset (not read for geoloc): {STAGED_RSLEARN}\n"
            "Project code: olmoearth_projects/projects/mozambique_lulc\n"
            "Only labels used; pretraining supplies imagery.\n"
        )

    recs = scan_records()
    print(f"scanned {len(recs)} labelled points")

    # Geolocation sanity: all points must land in Mozambique's bbox (else dummy bounds).
    lons = [r["lon"] for r in recs]
    lats = [r["lat"] for r in recs]
    print(
        f"lon range {min(lons):.3f}..{max(lons):.3f}  lat range {min(lats):.3f}..{max(lats):.3f}"
    )
    assert 30.0 <= min(lons) and max(lons) <= 41.0, "lon outside Mozambique"
    assert -27.0 <= min(lats) and max(lats) <= -10.0, "lat outside Mozambique"

    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} points (<= {PER_CLASS}/class, 25k cap)")

    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": r["label"],
                "time_range": TIME_RANGE,
                "change_time": None,
                "source_id": r["source_id"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    counts = Counter(r["label"] for r in selected)
    classes_meta = [
        {"id": i, "name": name, "description": desc}
        for i, (name, desc) in enumerate(CLASSES)
    ]
    class_counts = {CLASSES[i][0]: int(counts.get(i, 0)) for i in range(len(CLASSES))}
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "olmoearth",
            "license": "internal",
            "provenance": {
                "url": "olmoearth_projects/projects/mozambique_lulc",
                "have_locally": True,
                "local_path": str(GPKG_DIR),
                "staged_rslearn": STAGED_RSLEARN,
                "annotation_method": "manual field-survey reference points (Gaza, Manica, Zambezia)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes_meta,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": class_counts,
            "notes": (
                "Sparse-point dataset -> points.geojson (spec 2a). Unified class scheme: "
                "LULC ids 0-6 (Water, Bare Ground, Rangeland, Flooded Vegetation, Trees, "
                "Cropland, Buildings) + crop types ids 7-13 (corn, cassava, rice, sesame, "
                "beans, millet, sorghum). Field-survey reference points over Gaza/Manica/"
                "Zambezia provinces, Mozambique. lon/lat read from source GPKGs (LULC "
                "EPSG:4326; crop EPSG:3036 Moznet/UTM36S reprojected to WGS84), not from the "
                "staged rslearn window bounds. All source train/val/test splits used. Time "
                "range = surveyed growing season 2024-10-23..2025-06-20 (<=1yr, post-2016); "
                "change_time=null. <=1000 points/class."
            ),
        },
    )

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    main()
