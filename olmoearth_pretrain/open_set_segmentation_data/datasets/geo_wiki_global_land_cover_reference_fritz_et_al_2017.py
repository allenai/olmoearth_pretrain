"""Process the Geo-Wiki Global Land Cover Reference (Fritz et al. 2017) into open-set
segmentation label points.

Source: PANGAEA doi 10.1594/PANGAEA.869680, "A global dataset of crowdsourced land cover
and land use reference data (2011-2012)" (the *Global Crowd* campaign file
``globalcrowd.csv``). Each row is one crowdsourced visual interpretation of a location by
one volunteer: lon/lat plus up to three land-cover classes (``LC1/LC2/LC3``, codes 1-10)
with their fractional cover (``perc1/perc2/perc3``). Classification: label = dominant
land-cover class.

This is a **pure sparse-point** dataset, so we write one dataset-wide GeoJSON point table
(``points.geojson``, spec 2a), not per-point GeoTIFFs.

Aggregation (see summary for full rationale):
- ``pixelID`` is NOT a stable global (nor per-competition) location key in this file, so we
  key locations on rounded (lon, lat) instead. 79,848 unique locations.
- Per crowd record, the *dominant* class is the ``LC`` with the largest ``perc`` among the
  three (falling back to LC1). Per location, we take the majority vote of the per-record
  dominant classes across all volunteers, breaking ties by lowest class id.
- Source class codes 1-10 map to ids 0-9 (id = code - 1), matching the manifest class order.

Time range: the interpreted imagery pre-dates the Sentinel era (``googleimagedate`` is
mostly 2003-2012). Land cover is a comparatively static class, so per spec §5 (static
labels) we assign a representative Sentinel-era 1-year window (2016) to every point and
record the caveat in the summary.
"""

import argparse
from collections import Counter

import numpy as np
import pandas as pd

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "geo_wiki_global_land_cover_reference_fritz_et_al_2017"
NAME = "Geo-Wiki Global Land Cover Reference (Fritz et al. 2017)"
CSV_URL = "https://store.pangaea.de/Publications/FritzS-etal_2016/globalcrowd.csv"
META_URL = (
    "https://store.pangaea.de/Publications/FritzS-etal_2016/globalcrowd_metadata.xlsx"
)
PER_CLASS = 1000
# Representative Sentinel-era year (interpreted imagery is mostly 2003-2012; land cover is
# static enough to anchor to a Sentinel-era window; see summary caveat).
REPRESENTATIVE_YEAR = 2016

# Source LC code (1-10) -> (manifest class name, source definition). id = code - 1.
CLASSES = [
    ("tree cover", "Tree-dominated cover (source Land Cover code 1)."),
    ("shrub cover", "Shrub-dominated cover (source code 2)."),
    ("herbaceous/grassland", "Herbaceous vegetation / grassland (source code 3)."),
    ("cultivated & managed", "Cultivated and managed land / cropland (source code 4)."),
    (
        "mosaic cultivated/natural",
        "Mosaic of cultivated & managed and natural vegetation (source code 5).",
    ),
    ("flooded/wetland", "Regularly flooded land / wetland (source code 6)."),
    ("urban", "Urban / built-up (source code 7)."),
    ("snow & ice", "Snow and ice (source code 8)."),
    ("barren", "Barren land (source code 9)."),
    ("open water", "Open water (source code 10)."),
]


def _download_raw():
    from olmoearth_pretrain.open_set_segmentation_data import download

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    csv_path = download.download_http(
        CSV_URL, raw / "globalcrowd.csv", headers=headers, timeout=300
    )
    download.download_http(
        META_URL, raw / "globalcrowd_metadata.xlsx", headers=headers, timeout=120
    )
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "PANGAEA doi 10.1594/PANGAEA.869680 (Fritz et al. 2017, Global Crowd campaign)\n"
            f"{CSV_URL}\n{META_URL}\n"
        )
    return csv_path


def _record_dominant(df: pd.DataFrame) -> np.ndarray:
    """Per-record dominant LC code: argmax perc among (LC1..3), fallback LC1."""
    lc = df[["LC1", "LC2", "LC3"]].to_numpy()
    pc = df[["perc1", "perc2", "perc3"]].fillna(0).to_numpy()
    lc = np.where(np.isnan(lc), 0, lc)
    best = np.argmax(pc, axis=1)
    dom = lc[np.arange(len(df)), best]
    dom = np.where(dom == 0, lc[:, 0], dom)
    return dom.astype(int)


def _majority(codes: list[int]) -> int:
    c = Counter(codes)
    m = max(c.values())
    return sorted(k for k, v in c.items() if v == m)[0]


def scan_records(csv_path: str) -> list[dict]:
    """Aggregate crowd records into one record per unique location."""
    df = pd.read_csv(csv_path)
    df["dom"] = _record_dominant(df)
    df = df[(df["dom"] >= 1) & (df["dom"] <= 10)].copy()
    df["lonr"] = df["lon"].round(6)
    df["latr"] = df["lat"].round(6)
    recs = []
    for (lonr, latr), sub in df.groupby(["lonr", "latr"]):
        codes = sub["dom"].tolist()
        code = _majority(codes)
        recs.append(
            {
                "lon": float(lonr),
                "lat": float(latr),
                "label": code - 1,  # 0-based class id
                "n_votes": len(codes),
                "source_id": f"{lonr:.6f}_{latr:.6f}",
            }
        )
    return recs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    if args.skip_download:
        csv_path = str(io.raw_dir(SLUG) / "globalcrowd.csv")
    else:
        csv_path = str(_download_raw())

    recs = scan_records(csv_path)
    print(f"aggregated {len(recs)} unique locations from crowd records")

    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} (<= {PER_CLASS}/class, 25k total cap)")

    tr = io.year_range(REPRESENTATIVE_YEAR)
    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": r["label"],
                "time_range": tr,
                "source_id": r["source_id"],
                "n_votes": r["n_votes"],
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
            "source": "PANGAEA / Sci Data (Fritz et al. 2017)",
            "license": "CC-BY-3.0",
            "provenance": {
                "url": "https://doi.pangaea.de/10.1594/PANGAEA.869680",
                "have_locally": False,
                "annotation_method": "manual (crowdsourced visual interpretation)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {
                CLASSES[cid][0]: counts.get(cid, 0) for cid in range(len(CLASSES))
            },
            "notes": (
                "Sparse land-cover reference points (1x1). Label = dominant land-cover "
                "class from crowdsourced visual interpretation. Locations keyed on rounded "
                "(lon,lat); dominant class = majority vote across volunteers. Interpreted "
                "imagery is mostly 2003-2012 (pre-Sentinel); a representative 2016 window is "
                "assigned since land cover is comparatively static (spec 5)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    main()
