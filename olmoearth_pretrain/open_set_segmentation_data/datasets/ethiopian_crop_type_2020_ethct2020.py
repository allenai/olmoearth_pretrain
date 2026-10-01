"""Process the Ethiopian Crop Type 2020 (EthCT2020) dataset into a point table.

Source: CIMMYT / Data in Brief (Kerner et al. 2024), Mendeley Data record mfpvmk8cnm,
CC-BY-4.0. 2,793 quality-controlled, georeferenced in-situ crop-type samples collected in
smallholder wheat-based farming systems across Ethiopia for the Meher (main) season
2020/21. Distributed as an ESRI shapefile (EPSG:32637 / UTM 37N) whose geometries are
small (~20 m) circular field plots buffered around field centroids. Each plot carries a
hierarchical crop taxonomy (7 crop groups, 22 crop classes).

label_type = points -> sparse point classification, so we emit one dataset-wide point
table (points.json, spec 2a), not per-point GeoTIFFs. Each plot -> one point at its
centroid lon/lat with the crop-class id. Crop type is a seasonal/annual label, so each
point gets a 1-year time range anchored on 2020 (the Meher 2020/21 growing season).
Balanced to <=1000 per class (wheat, 2077 raw, is the only class capped).

NOTE on the shapefile's "lat"/"long" columns: they hold UTM (northing/easting) values, not
WGS84 degrees, so we ignore them and reproject each geometry centroid to WGS84 ourselves.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.ethiopian_crop_type_2020_ethct2020
"""

import argparse
import os
from collections import Counter

from olmoearth_pretrain.open_set_segmentation_data import io, manifest

SLUG = "ethiopian_crop_type_2020_ethct2020"
NAME = "Ethiopian Crop Type 2020 (EthCT2020)"
SOURCE_URL = "https://data.mendeley.com/datasets/mfpvmk8cnm/1"
PER_CLASS = 1000
ANCHOR_YEAR = 2020  # Meher (main) season 2020/21.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    import geopandas as gpd

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    shp = os.path.join(raw.path, "EthCT2020.shp")

    gdf = gpd.read_file(shp)
    # Reproject centroids to WGS84 lon/lat (source is UTM 37N; the "lat"/"long"
    # attribute columns are actually UTM coords, so we do not use them).
    cent = gpd.GeoSeries(gdf.geometry.centroid, crs=gdf.crs).to_crs(4326)
    lons = cent.x.tolist()
    lats = cent.y.tolist()

    # Build class id map, ordered by descending frequency (ids 0..n-1). Description
    # carries the source crop group for context.
    class_counts_raw = Counter(gdf["c_class"])
    group_of: dict[str, str] = {}
    for cls, grp in zip(gdf["c_class"], gdf["c_group"]):
        group_of.setdefault(cls, grp)
    ordered = [c for c, _ in class_counts_raw.most_common()]
    name_to_id = {name: i for i, name in enumerate(ordered)}

    records = []
    for i in range(len(gdf)):
        cls = gdf["c_class"].iloc[i]
        records.append(
            {
                "lon": float(lons[i]),
                "lat": float(lats[i]),
                "cls": cls,
                "label": name_to_id[cls],
                "source": str(gdf["sorc_nm"].iloc[i]),
                "src_id": int(gdf["id"].iloc[i]),
            }
        )

    from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

    selected = balance_by_class(records, "label", per_class=PER_CLASS)
    print(f"read {len(records)} plots; selected {len(selected)} (<= {PER_CLASS}/class)")

    tr = io.year_range(ANCHOR_YEAR)
    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": r["label"],
                "time_range": tr,
                "source_id": f"{r['source']}/{r['src_id']}",
            }
        )
    io.write_points_table(SLUG, "classification", points)

    sel_counts = Counter(r["cls"] for r in selected)
    classes_meta = [
        {
            "id": name_to_id[name],
            "name": name,
            "description": f"Crop group: {group_of[name]}.",
        }
        for name in ordered
    ]
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "CIMMYT / Data in Brief",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": SOURCE_URL,
                "have_locally": False,
                "annotation_method": "manual field survey (in-situ), quality-controlled",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes_meta,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {name: sel_counts.get(name, 0) for name in ordered},
            "notes": (
                "Sparse point crop-type segmentation (1x1); ~20 m field plots reduced to "
                "centroid points. 22 crop classes, ids by descending frequency. wheat "
                "capped at 1000 (2077 raw); all other classes kept in full. 1-year time "
                "range anchored on 2020 (Meher season 2020/21). Coords from geometry "
                "centroid reprojected UTM37N->WGS84 (shapefile lat/long cols are UTM)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done:", len(selected), "points")


if __name__ == "__main__":
    main()
