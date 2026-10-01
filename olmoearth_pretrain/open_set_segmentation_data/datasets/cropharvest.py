"""Process the CropHarvest global agricultural point dataset into an open-set-segmentation
point table.

Source: CropHarvest (Tseng et al., NeurIPS 2021 Datasets & Benchmarks), Zenodo record
7257688, file ``labels.geojson`` (~95k harmonized agricultural samples aggregating 25
source datasets, strong Africa/Asia coverage). Each label carries a representative lon/lat,
an ``is_crop`` flag, a harmonized FAO crop group (``classification_label``, present for
~33k crop samples), and an ``export_end_date`` marking the end of the ~1-year EO window the
label describes.

This is a pure sparse-point classification dataset (each label is a single ~10 m location),
so we emit ONE dataset-wide point table (points.json, spec 2a) rather than per-point tifs.

Unified class scheme (multiclass crop labels where available, crop/non-crop fallback):
  0 non_crop            (is_crop == 0)
  1..8 harmonized FAO crop groups from ``classification_label``
  9 other_crop          (classification_label == 'other'; a crop with no FAO group)
  10 crop_unspecified   (is_crop == 1 but no harmonized group)

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.cropharvest
"""

import argparse
import json
import multiprocessing
from collections import Counter
from datetime import UTC, datetime, timedelta

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest

SLUG = "cropharvest"
ZENODO_RECORD = "7257688"
LABELS_URL = "https://zenodo.org/api/records/7257688/files/labels.geojson/content"
PER_CLASS = 1000

# Unified class scheme. Harmonized FAO crop groups come from CropHarvest's
# ``classification_label`` field; non_crop / crop_unspecified are the crop/non-crop fallback
# for the ~62k samples with no harmonized group.
CLASSES = [
    (
        "non_crop",
        "Non-agricultural land (is_crop == 0): natural vegetation, water, built-up, bare, etc.",
    ),
    (
        "cereals",
        "Cereal crops (FAO group): wheat, rice, maize, sorghum, millet, barley, etc.",
    ),
    (
        "fruits_nuts",
        "Fruit and nut crops (FAO group): orchards, vines, tree fruits, nuts.",
    ),
    ("vegetables_melons", "Vegetables and melons (FAO group)."),
    ("leguminous", "Leguminous crops / pulses (FAO group): beans, peas, lentils, etc."),
    (
        "oilseeds",
        "Oilseed crops (FAO group): soybean, sunflower, rapeseed, groundnut, etc.",
    ),
    (
        "beverage_spice",
        "Beverage and spice crops (FAO group): coffee, tea, cocoa, spices.",
    ),
    ("sugar", "Sugar crops (FAO group): sugarcane, sugar beet."),
    ("root_tuber", "Root and tuber crops (FAO group): potato, cassava, yam, etc."),
    (
        "other_crop",
        "Cropland labeled as a crop but not mapped to a specific FAO group ('other').",
    ),
    (
        "crop_unspecified",
        "Cropland (is_crop == 1) with no harmonized crop-group label available.",
    ),
]
NAME_TO_ID = {name: i for i, (name, _d) in enumerate(CLASSES)}

# FAO harmonized-group classification_label -> our class name (identity for named groups).
GROUP_TO_NAME = {
    "cereals": "cereals",
    "fruits_nuts": "fruits_nuts",
    "vegetables_melons": "vegetables_melons",
    "leguminous": "leguminous",
    "oilseeds": "oilseeds",
    "beverage_spice": "beverage_spice",
    "sugar": "sugar",
    "root_tuber": "root_tuber",
    "other": "other_crop",
    "non_crop": "non_crop",
}


def _class_name(props: dict) -> str | None:
    """Map a CropHarvest label record to our unified class name (or None to drop)."""
    is_crop = props.get("is_crop")
    cl = props.get("classification_label")
    if cl is not None:
        return GROUP_TO_NAME.get(cl)
    # No harmonized group: fall back to crop/non-crop.
    if is_crop == 0:
        return "non_crop"
    if is_crop == 1:
        return "crop_unspecified"
    return None


def _one_year_window(export_end_date: str) -> tuple[datetime, datetime]:
    """1-year time range ending at CropHarvest's export_end_date (its EO window end)."""
    end = datetime.fromisoformat(export_end_date)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    start = end - timedelta(days=365)
    return (start, end)


def load_records() -> list[dict]:
    raw = io.raw_dir(SLUG)
    path = raw / "labels.geojson"
    with path.open() as f:
        gj = json.load(f)
    recs = []
    for feat in gj["features"]:
        props = feat.get("properties", {})
        lon, lat = props.get("lon"), props.get("lat")
        if lon is None or lat is None:
            continue
        name = _class_name(props)
        if name is None:
            continue
        eed = props.get("export_end_date")
        if not eed:
            continue
        recs.append(
            {
                "lon": lon,
                "lat": lat,
                "label": name,
                "time_range": _one_year_window(eed),
                "source_id": f"{props.get('dataset')}/{props.get('index')}",
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
    download.download_http(LABELS_URL, raw / "labels.geojson")

    recs = load_records()
    print(f"loaded {len(recs)} usable labeled points")
    raw_counts = Counter(r["label"] for r in recs)
    print("raw class counts:")
    for name, _ in CLASSES:
        print(f"  {name}: {raw_counts.get(name, 0)}")

    from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class)")

    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": NAME_TO_ID[r["label"]],
                "time_range": r["time_range"],
                "source_id": r["source_id"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    counts = Counter(r["label"] for r in selected)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "CropHarvest",
            "task_type": "classification",
            "source": "Zenodo / NeurIPS (CropHarvest)",
            "license": "CC-BY-SA-4.0",
            "provenance": {
                "url": "https://zenodo.org/records/7257688 ; https://github.com/nasaharvest/cropharvest",
                "have_locally": False,
                "annotation_method": "aggregated field survey / declaration (25 source datasets, FAO-harmonized)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {name: counts.get(name, 0) for name, _ in CLASSES},
            "notes": (
                "Sparse 1x1 point classification (points.json, spec 2a). Multiclass FAO crop "
                "groups from CropHarvest 'classification_label' where available (~33k), else "
                "crop/non-crop from 'is_crop'. Time range = 1-year window ending at each "
                "label's export_end_date (CropHarvest's EO-export window). Balanced to "
                f"<= {PER_CLASS}/class; all source splits (is_test true/false) used."
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
