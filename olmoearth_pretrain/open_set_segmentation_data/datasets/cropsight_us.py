"""Process CropSight-US into open-set-segmentation label patches.

Source: Zenodo record 15702415 (CROPSIGHT-US). The manifest record's files are
access-restricted (HTTP 403), but the latest open-access version of the same record
concept (recid 19501943, v1.0.1, CC-BY-4.0) publishes the identical product as a single
open ZIP, so we pull that instead.

The product is one WGS84 ESRI shapefile of 124,419 cropland field polygons across CONUS.
Each polygon carries a per-field crop-type ``Label`` (17 crops), the ``Year``/``Month`` of
the Google Street View image used to audit it, and confidence metrics. Crop-type labels
are year-specific (a field rotates crops), so we keep only fields with ``Year >= 2016``
(the Sentinel era, matching the manifest's 2016-2023 range) and drop the pre-2016 fields
that cannot be paired with Sentinel imagery.

Each field polygon is rasterized (label_type=polygons) into a local-UTM 10 m label tile,
sized to the field footprint and hard-capped at 64x64 (fields larger than 640 m are
cropped to a 64x64 window centered on the field). Value = class id inside the field;
outside the field = 255 (nodata / unobserved, since neighboring land use is unknown).
Balanced to <=1000 fields per class. Time range is the labeled year (1-year window).
"""

import argparse
import multiprocessing
from collections import Counter
from typing import Any

import numpy as np
import shapely
import shapely.wkb
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, rasterize
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "cropsight_us"
NAME = "CropSight-US"
PER_CLASS = 1000
MIN_YEAR = 2016  # keep Sentinel-era fields only (drops 2013-2015)

# Zenodo: the manifest record (15702415, v1.0.0) is access-restricted; its latest open
# version publishes the same product.
ZENODO_OPEN_RECORD = "19501943"
ZIP_NAME = "cropsight-us_app_dat_v1.0.1.zip"
SHP_NAME = "cropsight-us_app_dat_v1.0.1.shp"
ZIP_URL = (
    f"https://zenodo.org/api/records/{ZENODO_OPEN_RECORD}/files/{ZIP_NAME}/content"
)

# Canonical class order (manifest order) -> id. The shapefile spells two crops
# differently ("peanuts", "potatoes"); map them to the manifest names.
CLASSES = [
    "alfalfa",
    "almond",
    "canola",
    "cereal",
    "corn",
    "cotton",
    "grape",
    "orange",
    "peanut",
    "pistachio",
    "potato",
    "sorghum",
    "soybean",
    "sugarbeet",
    "sugarcane",
    "sunflower",
    "walnut",
]
NAME_TO_ID = {name: i for i, name in enumerate(CLASSES)}
# shapefile Label -> canonical class name
LABEL_ALIASES = {"peanuts": "peanut", "potatoes": "potato"}


def _canon(label: str) -> str | None:
    label = label.strip().lower()
    label = LABEL_ALIASES.get(label, label)
    return label if label in NAME_TO_ID else None


def scan_records() -> list[dict[str, Any]]:
    """Read all field polygons, keep Year>=2016 with a known crop, into flat records."""
    import geopandas as gpd

    shp = io.raw_dir(SLUG) / ZIP_NAME.replace(".zip", ".shp")
    gdf = gpd.read_file(shp.path)
    recs: list[dict[str, Any]] = []
    for idx, row in enumerate(gdf.itertuples(index=False)):
        cls = _canon(row.Label)
        if cls is None:
            continue
        year = int(row.Year)
        if year < MIN_YEAR:
            continue
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        recs.append(
            {
                "wkb": shapely.wkb.dumps(geom),
                "lon": geom.centroid.x,
                "lat": geom.centroid.y,
                "label": cls,
                "year": year,
                "source_id": f"field_{idx}",
            }
        )
    return recs


def _write_one(rec: dict[str, Any]) -> str | None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return rec["label"]
    cid = NAME_TO_ID[rec["label"]]
    geom = shapely.wkb.loads(rec["wkb"])

    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    pix = rasterize.geom_to_pixels(geom, WGS84_PROJECTION, proj)
    minx, miny, maxx, maxy = pix.bounds
    w = min(io.MAX_TILE, max(1, int(np.ceil(maxx - minx))))
    h = min(io.MAX_TILE, max(1, int(np.ceil(maxy - miny))))
    col = int(round((minx + maxx) / 2))
    row = int(round((miny + maxy) / 2))
    bounds = io.centered_bounds(col, row, w, h)

    arr = rasterize.rasterize_shapes(
        [(pix, cid)], bounds, fill=io.CLASS_NODATA, dtype="uint8", all_touched=True
    )
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        source_id=rec["source_id"],
        classes_present=[cid],
    )
    return rec["label"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()

    # Ensure raw source present (download + unzip if needed).
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    shp = raw / SHP_NAME
    if not shp.exists():
        import zipfile

        from olmoearth_pretrain.open_set_segmentation_data import download

        zip_path = raw / ZIP_NAME
        download.download_http(ZIP_URL, zip_path)
        io.check_disk()
        with zipfile.ZipFile(zip_path.path) as z:
            z.extractall(raw.path)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "manifest: https://zenodo.org/records/15702415 (v1.0.0, access-restricted)\n"
            f"downloaded open version: https://zenodo.org/records/{ZENODO_OPEN_RECORD} "
            "(v1.0.1, CC-BY-4.0)\n"
            f"file: {ZIP_NAME}\n"
        )

    recs = scan_records()
    print(f"scanned {len(recs)} fields (Year>={MIN_YEAR}, known crop)")
    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} fields (<= {PER_CLASS}/class)")

    counts: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for label in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            if label is not None:
                counts[label] += 1

    io.check_disk()
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Zenodo",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://zenodo.org/records/15702415",
                "download_url": f"https://zenodo.org/records/{ZENODO_OPEN_RECORD}",
                "have_locally": False,
                "annotation_method": "street-view virtual audit + Sentinel-2 delineation",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {
                    "id": i,
                    "name": name,
                    "description": (
                        f"Cropland fields identified as {name} via Google Street View "
                        "virtual audit with Sentinel-2 field-boundary delineation "
                        "(CropSight-US)."
                    ),
                }
                for i, name in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {name: counts.get(name, 0) for name in CLASSES},
            "notes": (
                "Per-field crop-type polygons rasterized to local-UTM 10 m tiles, sized "
                "to the field footprint, capped at 64x64 (larger fields cropped to a "
                "centered 64x64 window). Inside field = class id; outside = 255 (nodata, "
                "neighboring land use unknown). Kept Year>=2016 only (Sentinel era); "
                "pre-2016 fields dropped since crop-type labels are year-specific. "
                "1-year time range anchored on the labeled year. All confidence levels "
                "included. Downloaded open v1.0.1 (recid 19501943) because the manifest "
                "record v1.0.0 is access-restricted."
            ),
        },
    )
    print("class counts:", dict(sorted(counts.items())))
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
