"""Process "Detailed Vegetation Maps of the Brazilian Cerrado" (Bendini et al. 2021).

Source: PANGAEA doi:10.1594/PANGAEA.932642 (CC-BY-4.0). Direct file download, no
credential. The record ships:
  - Cerrado_Vegetation_Map_level1_Bendini-etal_2021.tif  (RF map, 3 level-1 classes)
  - Cerrado_Vegetation_Map_level2_Bendini-etal_2021.tif  (RF map, 12 level-2 classes)
        30 m random-forest maps of Cerrado savanna physiognomies (EPSG:4326, uint16,
        nodata=0, class values 1..12 for level-2 / 1..3 for level-1) covering the biome.
  - Samples_for_Vegetation_Mapping_Bendini-etal_2021.csv  (2,828 field ground samples)
  - two QGIS .qml style files.

DECISION (recorded in the summary): the manifest label_type is "dense_raster + 2,828
ground samples". The two rasters are DERIVED random-forest products (level-2 overall
accuracy 0.77, with poor per-class F1 for Vereda 0.36 / Campo rupestre 0.53), while the
CSV holds the actual in-situ FIELD samples (WGS84 lon/lat + level-1 and level-2
physiognomy class + provenance). Spec §0 explicitly prefers manual/in-situ REFERENCE data
over derived-product maps (maps are a fallback). The field samples carry lon/lat + class,
so per the task's §1 preference we use the ground samples as high-confidence sparse points
rather than cropping windows from the RF map. This yields a sparse-point dataset ->
one dataset-wide points.geojson (spec §2a), NOT per-point GeoTIFFs.

Label = the finest hierarchy (level-2, 12 Cerrado physiognomies), which is exactly the
"fine savanna-physiognomy classes" the manifest highlights. Class id = source level-2 code
minus 1 (ids 0..11). Each point also carries level-1 name/id, level-2 name, and the source
origin as auxiliary properties.

Time range: Cerrado vegetation physiognomy is a persistent (static) land-cover type. Per
spec §5 (static labels) we assign a representative 1-year Sentinel-era window; the maps and
study period fall in 2016-2020, so we anchor on 2018. change_time=null.

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.detailed_vegetation_maps_of_the_brazilian_cerrado
"""

import argparse
import csv
import multiprocessing
import re
from collections import Counter
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "detailed_vegetation_maps_of_the_brazilian_cerrado"

PANGAEA_BASE = "https://download.pangaea.de/dataset/932642/files"
FILES = [
    "Cerrado_Vegetation_Map_level1_Bendini-etal_2021.tif",
    "Cerrado_Vegetation_Map_level2_Bendini-etal_2021.tif",
    "Samples_for_Vegetation_Mapping_Bendini-etal_2021.csv",
    "Style_Vegetation_level1_Bendini-etal_2021.qml",
    "Style_Vegetation_level2_Bendini-etal_2021.qml",
]
CSV_NAME = "Samples_for_Vegetation_Mapping_Bendini-etal_2021.csv"

PER_CLASS = 1000
# Persistent vegetation physiognomy -> static label, representative Sentinel-era year
# within the maps' 2016-2020 study window (spec §5).
LABEL_YEAR = 2018

# Level-2 physiognomy classes in source-code order (source code = id + 1). Descriptions
# follow the Ribeiro & Walter (2008) Cerrado physiognomy classification used by the source.
CLASSES: list[tuple[str, str]] = [
    (
        "Campo limpo",
        "Open grassland (grassland formation): predominantly herbaceous cover of grasses and "
        "forbs with virtually no shrubs and no trees. Source level-2 code 1.",
    ),
    (
        "Campo rupestre",
        "Rupestrian grassland on quartzite/sandstone rock outcrops, typically above ~900 m: "
        "herbaceous-shrubby vegetation with rupicolous, highly endemic flora. Code 2.",
    ),
    (
        "Campo sujo",
        "'Dirty field' — grassland with sparse, scattered shrubs and subshrubs above a "
        "continuous grass layer (grassland formation). Code 3.",
    ),
    (
        "Cerradao",
        "Cerradao: the densest savanna form, a closed woodland/forest with ~50-90% canopy "
        "cover and 8-15 m trees; structurally forest but floristically savanna. Code 4.",
    ),
    (
        "Cerrado rupestre",
        "Savanna on rocky substrate: scattered trees and shrubs rooted among rock outcrops, "
        "with a discontinuous grass layer. Code 5.",
    ),
    (
        "Cerrado sensu stricto",
        "Typical cerrado — savanna of trees and shrubs (~20-50% woody cover) over a "
        "continuous grass layer; the most characteristic Cerrado physiognomy. Code 6.",
    ),
    (
        "Ipuca",
        "Ipuca: seasonally flooded forest 'islands' (murundus/covoais) within the flooded "
        "grasslands of the Araguaia/Tocantins plains. Code 7.",
    ),
    (
        "Mata riparia",
        "Riparian/gallery forest: evergreen forest along watercourses and drainage lines "
        "(forest formation). Code 8.",
    ),
    (
        "Mata seca",
        "Dry seasonal forest (deciduous/semideciduous) on more fertile soils, not linked to "
        "watercourses; sheds leaves in the dry season. Code 9.",
    ),
    (
        "Palmeiral",
        "Palm grove: savanna/grove where a single palm species (e.g. babacu, buriti, other "
        "palms) is dominant. Code 10.",
    ),
    (
        "Parque de cerrado",
        "'Cerrado park': savanna with trees clustered on slightly raised earth mounds "
        "(murundus/campos de murundus) amid grassland, often seasonally waterlogged. Code 11.",
    ),
    (
        "Vereda",
        "Vereda: buriti (Mauritia flexuosa) palm swamp along drainage lines and humid valley "
        "bottoms on hydromorphic soils. Code 12.",
    ),
]
N_CLASSES = len(CLASSES)

# Level-1 (hierarchy-1) codes -> name, kept as an auxiliary per-point property.
LEVEL1_NAME = {
    1: "Nat. Arbustivo (savanna)",
    2: "Nat. Campestre (grassland)",
    3: "Nat. Florestal (forest)",
}

_WKT_RE = re.compile(r"Point\s*\(\s*([-0-9.]+)\s+([-0-9.]+)\s*\)")


def _download_raw() -> None:
    """Download all PANGAEA files to raw/{slug}/ (idempotent) and write SOURCE.txt."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    for fname in FILES:
        download.download_http(
            f"{PANGAEA_BASE}/{fname}",
            raw / fname,
            headers={"User-Agent": "Mozilla/5.0"},
        )
    (raw / "SOURCE.txt").write_text(
        "Detailed Vegetation Maps of the Brazilian Savanna (Cerrado) biome, Bendini et al. "
        "(2021).\nPANGAEA doi:10.1594/PANGAEA.932642 — https://doi.org/10.1594/PANGAEA.932642\n"
        "License: CC-BY-4.0. Direct file download (no credential):\n"
        f"  {PANGAEA_BASE}/<filename>\n"
        "Files: two 30 m random-forest vegetation maps (level-1 = 3 classes, level-2 = 12\n"
        "classes; EPSG:4326, uint16, nodata=0), a CSV of 2,828 field ground samples, and two\n"
        "QGIS style files.\n\n"
        "This dataset is built from the CSV FIELD ground samples (in-situ reference, spec "
        "§0-preferred) as sparse points, not from the derived RF map.\n"
    )


def read_samples() -> list[dict[str, Any]]:
    """Parse the field-sample CSV into flat point records (lon/lat + level-2 class)."""
    path = io.raw_dir(SLUG) / CSV_NAME
    recs: list[dict[str, Any]] = []
    with path.open("r", encoding="latin-1", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            wkt = row.get("wkt_geom") or ""
            m = _WKT_RE.search(wkt)
            if m is None:
                continue
            lon, lat = float(m.group(1)), float(m.group(2))
            try:
                l2 = int(row["lvl2_nm"])
                l1 = int(row["lvl1_nm"])
            except (KeyError, ValueError, TypeError):
                continue
            if not (1 <= l2 <= N_CLASSES):
                continue
            # Basic geographic sanity (Cerrado is in central Brazil).
            if not (-75.0 < lon < -30.0 and -35.0 < lat < 6.0):
                continue
            recs.append(
                {
                    "lon": lon,
                    "lat": lat,
                    "label": l2 - 1,  # class id 0..11
                    "level2_name": CLASSES[l2 - 1][0],
                    "level1_id": l1 - 1,
                    "level1_name": LEVEL1_NAME.get(l1, str(l1)),
                    "origin": (row.get("orign") or "").strip() or None,
                    "source_id": (row.get("ID") or "").strip() or None,
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

    _download_raw()
    io.check_disk()

    recs = read_samples()
    print(f"parsed {len(recs)} valid field samples")
    raw_counts = Counter(r["label"] for r in recs)
    print(
        "raw per-class counts:",
        {CLASSES[i][0]: raw_counts.get(i, 0) for i in range(N_CLASSES)},
    )

    # Sparse points -> tiles-per-class balance to <=1000/class under the 25k cap. With 12
    # classes and max 580 samples/class nothing is truncated; balancing is applied for
    # determinism and cap-compliance.
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
                "time_range": io.year_range(LABEL_YEAR),
                "change_time": None,
                "source_id": r["source_id"],
                # auxiliary fields copied verbatim into feature properties
                "level2_name": r["level2_name"],
                "level1_id": r["level1_id"],
                "level1_name": r["level1_name"],
                "origin": r["origin"],
            }
        )
    io.write_points_table(SLUG, "classification", points)

    sel_counts = Counter(r["label"] for r in selected)
    class_counts = {CLASSES[i][0]: sel_counts.get(i, 0) for i in range(N_CLASSES)}
    print("selected per-class counts:", class_counts)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "Detailed Vegetation Maps of the Brazilian Cerrado",
            "task_type": "classification",
            "source": "PANGAEA",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.org/10.1594/PANGAEA.932642",
                "have_locally": False,
                "annotation_method": (
                    "in-situ field ground samples (2,828 points compiled from SEMA, FIP "
                    "fieldwork, LAPIG, IFN-DF and other Cerrado field surveys); used to "
                    "train/validate a 30 m random-forest map (Bendini et al. 2021). We use "
                    "the field samples directly (spec §0 prefers in-situ reference over the "
                    "derived RF map)."
                ),
                "citation": (
                    "Bendini, H.N. et al. (2021): Detailed vegetation maps of the Brazilian "
                    "Savanna (Cerrado) biome produced with a semi-automatic approach. "
                    "PANGAEA, https://doi.org/10.1594/PANGAEA.932642"
                ),
                "hierarchy": "Ribeiro & Walter (2008) Cerrado physiognomy classification",
                "level2_code_to_name": {
                    str(i + 1): CLASSES[i][0] for i in range(N_CLASSES)
                },
                "level1_code_to_name": {
                    str(k): v for k, v in sorted(LEVEL1_NAME.items())
                },
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": class_counts,
            "notes": (
                "Sparse-point (1x1) segmentation from the 2,828 in-situ field ground samples "
                "of Bendini et al. (2021); written as one points.geojson (spec §2a). Label = "
                "level-2 Cerrado physiognomy (12 classes, id = source code - 1); level-1 "
                "name/id and sample origin retained as auxiliary point properties. The two "
                "30 m random-forest maps in the source are DERIVED products (level-2 OA 0.77) "
                "and were NOT used for labels — spec §0 prefers in-situ reference over derived "
                "maps; they remain in raw/ for a possible future dense-raster reprocess. "
                "Vegetation physiognomy is a persistent land-cover type: static label, "
                f"change_time=null, 1-year window on {LABEL_YEAR} (spec §5 static-label rule; "
                "maps/study period 2016-2020, within the Sentinel era)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG,
        "completed",
        task_type="classification",
        num_samples=len(selected),
        notes=(
            f"{len(selected)} in-situ field ground points, 12 level-2 Cerrado physiognomy "
            "classes (points.geojson, spec §2a). Derived RF map not used (in-situ preferred)."
        ),
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
