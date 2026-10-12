"""Process Great African Food Company Crop Type Tanzania into open-set-segmentation labels.

Source: "Great African Food Company Crop Type Tanzania" (Radiant Earth Foundation & Great
African Food Company, 2018), originally on Radiant MLHub (retired; DOI
10.34911/rdnt.5vx40r) and now mirrored openly on Source Cooperative at
``radiantearth/african-crops-tanzania-01`` (public, unsigned S3 via the
``https://data.source.coop`` proxy; bucket ``radiantearth``). Licensed CC-BY-4.0.
Field-level crop-type reference collected in-field with the Farmforce app: a surveyor
recorded a point in each field plus field boundary and properties (Village, Region, Plot
Area, Planting Date, estimated Harvest Date, Crop). 392 field-boundary **polygons** across
Tanzania (Arusha, Simiyu, ...). Growing season = 2018 (planting Jan-May 2018, mostly March;
harvest Jul-Dec 2018).

The mirror ships the labels twice (identical 392 fields): 24 top-level STAC label items
``ref_african_crops_tanzania_01_tile_XXX.geojson`` (WGS84) and per-imagery-chip
``label/{NN}/{NN}_label.geojson`` (UTM). We use the top-level WGS84 tiles (the STAC label
layer). The bundled Sentinel-2 ``imagery/`` (~46k COGs) is NOT downloaded -- pretraining
supplies its own imagery; we pull only the geojson labels + docs.

Task: per-pixel **classification** (crop type). EuroCrops/LEM/AgriFieldNet-style: one label
patch per field -- the field polygon rasterized into a <=64x64 UTM 10 m tile centered on the
polygon, with the crop class id burned inside the polygon and 255 (nodata/ignore) outside.
We only have ground truth inside surveyed fields, so unlabeled land is ignore, not a
background class (spec 5 positive-only handling).

Classes = the 6 crop types actually present, ids assigned 0..5 by descending field count
(the manifest's guessed classes [wheat, maize, sorghum, vegetables] do not match the data;
the source crops are used instead, keeping every class incl. sparse ones per spec 5):
    Bush Bean       156 -> 0
    Dry Bean        137 -> 1
    Sunflower        51 -> 2
    Safflower        24 -> 3
    White Sorghum    15 -> 4
    Yellow Maize      9 -> 5

Time range: crop type is a seasonal label; each field carries a real Planting Date, so we
anchor a per-field 1-year window = [Planting Date, Planting Date + 360 days], which spans
the field's growing/harvest cycle (all within the 2018 season). change_time is null.

Sampling: class-balanced (<=1000 fields/class, 25k cap); with only 392 fields all are kept.

Run (idempotent; skips already-written {sample_id}.tif):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.great_african_food_company_crop_type_tanzania
"""

import argparse
import json
import multiprocessing
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import shapely
import shapely.geometry
import tqdm
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "great_african_food_company_crop_type_tanzania"
NAME = "Great African Food Company Crop Type Tanzania"
URL = "https://source.coop/radiantearth/african-crops-tanzania-01"
DOI = "10.34911/rdnt.5vx40r"

S3_ENDPOINT = "https://data.source.coop"
S3_BUCKET = "radiantearth"
S3_PREFIX = "african-crops-tanzania-01/"

PER_CLASS = 1000
MAX_TILE = io.MAX_TILE  # 64
MAX_WINDOW_DAYS = 360  # spec: time_range must be <= ~1 year

# Short per-class definitions (source is a Tanzanian smallholder crop field survey).
CLASS_DESCRIPTIONS = {
    "Bush Bean": "Bush-type common bean (Phaseolus vulgaris), a determinate short-stature bean.",
    "Dry Bean": "Dry common bean (Phaseolus vulgaris) grown to mature/dry grain (pulses).",
    "Sunflower": "Sunflower (Helianthus annuus), grown for oilseed.",
    "Safflower": "Safflower (Carthamus tinctorius), an oilseed crop.",
    "White Sorghum": "White-grain sorghum (Sorghum bicolor), a cereal.",
    "Yellow Maize": "Yellow maize / corn (Zea mays).",
}

_WGS84_SRC = Projection(CRS.from_epsg(4326), 1, 1)


def list_label_keys() -> list[str]:
    """List the top-level STAC label geojson keys (WGS84 field polygons)."""
    import boto3
    import botocore

    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        config=botocore.config.Config(signature_version=botocore.UNSIGNED),
    )
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
        for o in page.get("Contents", []):
            k = o["Key"]
            rel = k[len(S3_PREFIX) :]
            if rel.startswith("ref_") and rel.endswith(".geojson"):
                keys.append(k)
    return sorted(keys)


def ensure_data() -> list[str]:
    """Download label geojsons + docs into raw_dir; return local geojson paths."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    # docs (small, useful provenance)
    for name in ("Tanzania_Documentation.pdf", "Tanzania_properties.csv"):
        try:
            download.download_s3_unsigned(
                S3_BUCKET, S3_PREFIX + name, raw / name, endpoint_url=S3_ENDPOINT
            )
        except Exception as e:  # noqa: BLE001 - docs are optional
            print(f"warn: could not fetch {name}: {e}")
    keys = list_label_keys()
    local: list[str] = []
    for k in keys:
        name = k[len(S3_PREFIX) :]
        dst = raw / name
        download.download_s3_unsigned(S3_BUCKET, k, dst, endpoint_url=S3_ENDPOINT)
        local.append(str(dst))
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Great African Food Company Crop Type Tanzania (Radiant Earth). "
            f"CC-BY-4.0. DOI {DOI}.\n{URL}\n"
            "Source Cooperative mirror bucket radiantearth, prefix "
            f"{S3_PREFIX} (unsigned S3 via {S3_ENDPOINT}).\n"
            "Downloaded: top-level ref_african_crops_tanzania_01_tile_*.geojson "
            "(24 STAC label items, WGS84; 392 field-boundary crop polygons) + "
            "Tanzania_Documentation.pdf + Tanzania_properties.csv. Bundled Sentinel-2 "
            "imagery/ (~46k COGs) intentionally NOT downloaded (pretraining supplies "
            "imagery).\n"
        )
    return local


def parse_planting(pr: dict[str, Any]) -> datetime | None:
    """Parse the field's Planting Date (YYYY-MM-DD) to a UTC datetime, or None."""
    v = pr.get("Planting Date")
    if not v:
        return None
    try:
        d = datetime.strptime(str(v)[:10], "%Y-%m-%d")
        return d.replace(tzinfo=UTC)
    except ValueError:
        return None


def field_time_range(planting: datetime | None) -> tuple[datetime, datetime]:
    """Per-field 1-year window anchored on the planting date (spanning the season).

    [planting, planting + 360d]. Falls back to calendar-year 2018 if no planting date.
    """
    if planting is None:
        return io.year_range(2018)
    return planting, planting + timedelta(days=MAX_WINDOW_DAYS)


def _write_tile(rec: dict[str, Any]) -> tuple[str, str, int]:
    sample_id = rec["sample_id"]
    class_id = int(rec["class_id"])
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return sample_id, "skip", class_id
    try:
        geom = shapely.from_wkb(rec["geom_wkb"])  # WGS84 lon/lat
        proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
        pix = geom_to_pixels(geom, _WGS84_SRC, proj)
        minx, miny, maxx, maxy = pix.bounds
        cx = int(round((minx + maxx) / 2))
        cy = int(round((miny + maxy) / 2))
        w = min(MAX_TILE, max(1, int(np.ceil(maxx - minx))))
        h = min(MAX_TILE, max(1, int(np.ceil(maxy - miny))))
        bounds = io.centered_bounds(cx, cy, w, h)
        arr = rasterize_shapes(
            [(pix, class_id)],
            bounds,
            fill=io.CLASS_NODATA,
            dtype="uint8",
            all_touched=True,
        )
        if not (arr != io.CLASS_NODATA).any():
            return sample_id, "empty", class_id
        start = datetime.fromisoformat(rec["t_start"])
        end = datetime.fromisoformat(rec["t_end"])
        io.write_label_geotiff(
            SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA
        )
        io.write_sample_json(
            SLUG,
            sample_id,
            proj,
            bounds,
            (start, end),
            source_id=rec["source_id"],
            classes_present=sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA}),
        )
        return sample_id, "ok", class_id
    except Exception as e:  # noqa: BLE001
        print(f"error on {sample_id}: {e}")
        return sample_id, "error", class_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    geojson_paths = ensure_data()
    print(f"downloaded {len(geojson_paths)} label geojson tiles")

    # ---- Read all field polygons; compute class frequency --------------------------
    raw_records: list[dict[str, Any]] = []
    freq: Counter = Counter()
    for path in geojson_paths:
        with open(path) as f:
            fc = json.load(f)
        tile_name = path.split("/")[-1].replace(".geojson", "")
        for idx, feat in enumerate(fc.get("features", [])):
            geom = shapely.geometry.shape(feat["geometry"])
            if geom is None or geom.is_empty:
                continue
            if not geom.is_valid:
                geom = geom.buffer(0)
                if geom.is_empty:
                    continue
            crop = feat["properties"].get("Crop")
            if not crop:
                continue
            crop = str(crop).strip()
            cent = geom.centroid
            if not (np.isfinite(cent.x) and np.isfinite(cent.y)):
                continue
            freq[crop] += 1
            planting = parse_planting(feat["properties"])
            start, end = field_time_range(planting)
            raw_records.append(
                {
                    "label": crop,
                    "lon": float(cent.x),
                    "lat": float(cent.y),
                    "geom_wkb": shapely.to_wkb(geom),
                    "t_start": start.isoformat(),
                    "t_end": end.isoformat(),
                    "source_id": f"{tile_name}/feat_{idx}",
                }
            )
    print(f"total field polygons: {len(raw_records)} across {len(freq)} classes")

    # ---- Assign class ids by descending field frequency ----------------------------
    ranked = [lbl for lbl, _ in freq.most_common()]
    label_to_id = {lbl: i for i, lbl in enumerate(ranked)}
    for r in raw_records:
        r["class_id"] = label_to_id[r["label"]]
    print("class frequency:", {lbl: freq[lbl] for lbl in ranked})

    # ---- Class-balanced selection (<=1000/class, 25k cap) --------------------------
    selected = balance_by_class(
        raw_records, key="class_id", per_class=PER_CLASS, total_cap=25000
    )
    print(f"selected {len(selected)} fields after balancing")

    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"

    # ---- Write tiles in parallel ---------------------------------------------------
    results: Counter = Counter()
    written_by_class: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for sample_id, res, class_id in tqdm.tqdm(
            star_imap_unordered(p, _write_tile, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            results[res] += 1
            if res in ("ok", "skip"):
                written_by_class[class_id] += 1
    print("write results:", dict(results))
    io.check_disk()

    # ---- Metadata ------------------------------------------------------------------
    classes = [
        {"id": cid, "name": lbl, "description": CLASS_DESCRIPTIONS.get(lbl)}
        for lbl, cid in sorted(label_to_id.items(), key=lambda kv: kv[1])
    ]
    class_counts = {
        lbl: int(written_by_class.get(cid, 0))
        for lbl, cid in sorted(label_to_id.items(), key=lambda kv: kv[1])
    }
    num_written = int(results.get("ok", 0) + results.get("skip", 0))
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Source Cooperative (radiantearth/african-crops-tanzania-01)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": URL,
                "doi": DOI,
                "have_locally": False,
                "annotation_method": (
                    "in-field survey via Farmforce app (point + field boundary + crop/"
                    "planting-date properties per field), Great African Food Company / "
                    "Radiant Earth Foundation"
                ),
                "region": "Tanzania",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": num_written,
            "class_counts": class_counts,
            "notes": (
                "392 in-field crop-type reference field polygons in Tanzania (Farmforce "
                "app survey), growing season 2018. Each field polygon is rasterized into a "
                "<=64x64 UTM 10 m tile centered on the polygon: crop class id inside, "
                "255=nodata/ignore outside (no background class -- unlabeled land is "
                "ignore, spec 5 positive-only). Classes are the 6 crops actually present "
                "(manifest's guessed classes did not match the data); ids 0..5 by "
                "descending field count, all classes kept incl. sparse (Yellow Maize=9). "
                "Per-field time_range = [Planting Date, +360 days] spanning the growing "
                "season. Labels sourced from the top-level STAC geojson tiles on the Source "
                "Cooperative mirror; bundled Sentinel-2 imagery not downloaded. "
                "Class-balanced, <=1000/class, 25k cap (all 392 fields kept)."
            ),
        },
    )

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=num_written
    )
    print(f"done: {num_written} samples across {len(classes)} classes")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
