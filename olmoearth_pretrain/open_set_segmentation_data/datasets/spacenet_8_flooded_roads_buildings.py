"""SpaceNet 8 (flooded roads & buildings) -> open-set-segmentation flood masks.

Source: SpaceNet 8 Flood Detection Challenge, hosted on the public AWS Open Data bucket
``s3://spacenet-dataset/spacenet/SN8_floods/`` (unsigned/anonymous access; CC-BY-SA-4.0).
Two labeled AOIs are public: ``Germany_Training_Public`` (2021 Western-Europe floods,
Ahr/Erft valleys) and ``Louisiana-East_Training_Public`` (Hurricane Ida, Aug 2021). A
third AOI (``Louisiana-West_Test_Public``) is imagery-only (no ``annotations/``) and is
skipped. Labels are per-tile GeoJSONs of building footprints (Polygon, ``building=yes``)
and road centerlines (LineString, ``highway=<type>``), each flagged ``flooded=yes`` (post-
event inundated) or not (``flooded`` null / "no"). Geometries are WGS84 (CRS84); the
paired imagery is Maxar VHR (~0.3-0.8 m). We need only the small GeoJSON labels --
pretraining supplies its own S2/S1 imagery -- so no imagery TIFFs are downloaded.

Class scheme (unified building+road x flooded/non-flooded, spec S5 multi-target rule):
  0 = non_flooded_building, 1 = flooded_building,
  2 = non_flooded_road,     3 = flooded_road.

VHR handling (spec S4): individual buildings (~10-20 m) and road centerlines (~5-10 m
wide) are at/under one 10 m pixel. Each source label tile (~350-650 m across, i.e. roughly
one 64x64 tile at 10 m) is rasterized directly into a local-UTM 10 m grid with
all_touched=True so every touched pixel is marked; flooded structures cluster, so the
FLOODED classes aggregate into flood-extent patches (the salvageable signal). Paint order
puts flooded classes last so the flood signal wins overlaps. Un-labeled ground stays nodata
(255): this is a positive-only dataset (spec S5) -- assembly supplies negatives from other
datasets; we do NOT fabricate a background class.

Change label (spec S5 change-timing rule): flooding is a transient/event state (water
recedes), so a persistent-state recast is NOT valid; we use the dated-event approach. Both
events are resolvable to well within ~1-2 months: Germany = mid-July 2021 (2021-07-15),
Louisiana = Hurricane Ida landfall (2021-08-30). ``change_time`` is set per AOI and kept as
the reference date used to build two adjacent windows via ``io.pre_post_time_ranges``:
``pre_time_range`` (the ~6 months, <=183 days, immediately before ``change_time``) and
``post_time_range`` (the ~6 months, <=183 days, immediately after); ``time_range`` is null.
Pretraining pairs a "before" image stack with an "after" stack and probes on their
difference, so it always straddles the flood.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.spacenet_8_flooded_roads_buildings
"""

import argparse
import math
import multiprocessing
from datetime import UTC, datetime
from typing import Any

import numpy as np
import shapely
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from olmoearth_pretrain.open_set_segmentation_data import (
    download,
    io,
    manifest,
    rasterize,
    sampling,
)

SLUG = "spacenet_8_flooded_roads_buildings"
NAME = "SpaceNet 8 (flooded roads & buildings)"
BUCKET = "spacenet-dataset"
S3_ROOT = "spacenet/SN8_floods"
TILE = 64
PER_CLASS = 1000

# Public labeled AOIs -> dated flood event (change_time). Louisiana-West is test/imagery-
# only (no annotations) and is excluded.
AOIS = {
    "Germany_Training_Public": datetime(2021, 7, 15, tzinfo=UTC),
    "Louisiana-East_Training_Public": datetime(2021, 8, 30, tzinfo=UTC),
}

CLASSES = [
    (
        "non_flooded_building",
        "Building footprint (OSM building=yes) that was NOT inundated in the post-event VHR "
        "image. Source geometry is a polygon; at 10 m an isolated building is ~1-2 px "
        "(all_touched rasterization).",
    ),
    (
        "flooded_building",
        "Building footprint flagged flooded=yes in the post-event image (standing floodwater "
        "in/around the structure). Flooded buildings cluster, aggregating into flood-extent "
        "patches at 10 m.",
    ),
    (
        "non_flooded_road",
        "Road centerline (OSM highway=*) that was NOT inundated. Rasterized from the line with "
        "all_touched=True; narrow roads are under-resolved at 10 m.",
    ),
    (
        "flooded_road",
        "Road centerline flagged flooded=yes (submerged / impassable due to floodwater).",
    ),
]
NUM_CLASSES = len(CLASSES)

# Paint order (rasterize burns shapes in list order; later overwrites earlier). Ordered so
# flooded classes win overlaps, emphasizing the flood signal: non_flooded_road <
# non_flooded_building < flooded_road < flooded_building.
PAINT_RANK = {2: 0, 0: 1, 3: 2, 1: 3}


def _classify(props: dict[str, Any]) -> int | None:
    """Map a feature's properties to a class id, or None to skip."""
    is_building = props.get("building") is not None
    is_road = props.get("highway") is not None
    flooded = str(props.get("flooded")).strip().lower() == "yes"
    if is_building:
        return 1 if flooded else 0
    if is_road:
        return 3 if flooded else 2
    return None


def _list_annotations(aoi: str) -> list[str]:
    import boto3
    import botocore

    s3 = boto3.client(
        "s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED)
    )
    prefix = f"{S3_ROOT}/{aoi}/annotations/"
    keys: list[str] = []
    for pg in s3.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=prefix
    ):
        for o in pg.get("Contents", []):
            if o["Key"].endswith(".geojson"):
                keys.append(o["Key"])
    return keys


def _download_aoi(aoi: str) -> list[str]:
    """Download an AOI's annotation GeoJSONs to raw_dir (idempotent). Returns local paths."""
    dst_dir = io.raw_dir(SLUG) / aoi / "annotations"
    dst_dir.mkdir(parents=True, exist_ok=True)
    keys = _list_annotations(aoi)
    paths: list[str] = []
    for key in keys:
        name = key.rsplit("/", 1)[1]
        dst = dst_dir / name
        if not dst.exists():
            download.download_s3_unsigned(BUCKET, key, dst)
        paths.append(str(dst))
    return paths


def _scan_file(path: str, aoi: str) -> list[dict[str, Any]]:
    """Rasterize one label tile's features into <=64x64 UTM 10 m patches."""
    import json

    with UPath(path).open("r") as f:
        gj = json.load(f)
    feats = gj.get("features", [])
    shapes: list[tuple[Any, int]] = []  # (wgs84 shapely geom, class_id)
    for feat in feats:
        cid = _classify(feat.get("properties", {}))
        if cid is None:
            continue
        geom = feat.get("geometry")
        if not geom:
            continue
        try:
            shapes.append((shapely.geometry.shape(geom), cid))
        except Exception:
            continue
    if not shapes:
        return []

    # Local UTM projection at 10 m from the union centroid of all labeled geometries.
    union = shapely.unary_union([g for g, _ in shapes])
    c = union.centroid
    proj = io.utm_projection_for_lonlat(float(c.x), float(c.y))

    # Reproject each geometry to UTM pixel coords; drop any that fail/are empty.
    px_shapes: list[tuple[Any, int]] = []
    xs: list[float] = []
    ys: list[float] = []
    for geom, cid in shapes:
        try:
            pg = rasterize.geom_to_pixels(geom, WGS84_PROJECTION, proj)
        except Exception:
            continue
        if pg.is_empty:
            continue
        px_shapes.append((pg, cid))
        x0, y0, x1, y1 = pg.bounds
        xs += [x0, x1]
        ys += [y0, y1]
    if not px_shapes:
        return []
    # Sort by paint precedence so flooded classes are burned last.
    px_shapes.sort(key=lambda s: PAINT_RANK[s[1]])

    x_min = int(math.floor(min(xs)))
    y_min = int(math.floor(min(ys)))
    x_max = int(math.ceil(max(xs)))
    y_max = int(math.ceil(max(ys)))
    w = max(1, x_max - x_min)
    h = max(1, y_max - y_min)
    stem = path.rsplit("/", 1)[1][: -len(".geojson")]

    recs: list[dict[str, Any]] = []
    for ti in range(math.ceil(w / TILE)):
        for tj in range(math.ceil(h / TILE)):
            bx0 = x_min + ti * TILE
            by0 = y_min + tj * TILE
            bounds = (bx0, by0, bx0 + TILE, by0 + TILE)
            arr = rasterize.rasterize_shapes(
                px_shapes, bounds, fill=io.CLASS_NODATA, dtype="uint8", all_touched=True
            )[0]
            present = sorted(int(v) for v in np.unique(arr) if v != io.CLASS_NODATA)
            if not present:
                continue
            recs.append(
                {
                    "array": arr,
                    "crs": proj.crs.to_string(),
                    "bounds": bounds,
                    "classes_present": present,
                    "aoi": aoi,
                    "source_id": f"{aoi}/{stem}/{ti}_{tj}",
                }
            )
    return recs


def _write_one(rec: dict[str, Any]) -> int:
    from rasterio.crs import CRS

    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return 0
    proj = Projection(CRS.from_string(rec["crs"]), io.RESOLUTION, -io.RESOLUTION)
    bounds = tuple(rec["bounds"])
    change_time = AOIS[rec["aoi"]]
    pre_range, post_range = io.pre_post_time_ranges(change_time)
    time_range = (pre_range[0], post_range[1])  # outer bounding span
    io.write_label_geotiff(
        SLUG, sample_id, rec["array"], proj, bounds, nodata=io.CLASS_NODATA
    )
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        time_range,
        change_time=change_time,
        source_id=rec["source_id"],
        classes_present=rec["classes_present"],
        pre_time_range=pre_range,
        post_time_range=post_range,
    )
    return 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument(
        "--probe", action="store_true", help="scan/report only, no writes"
    )
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    # Download labels (small GeoJSONs only).
    jobs: list[dict[str, Any]] = []
    for aoi in AOIS:
        paths = _download_aoi(aoi)
        print(f"{aoi}: {len(paths)} annotation files")
        jobs += [dict(path=p, aoi=aoi) for p in paths]
    with (io.raw_dir(SLUG) / "SOURCE.txt").open("w") as f:
        f.write(
            "SpaceNet 8 Flood Detection Challenge labels.\n"
            f"Source: s3://{BUCKET}/{S3_ROOT}/ (public AWS Open Data, unsigned). "
            "License CC-BY-SA-4.0.\n"
            "Only the per-tile annotation GeoJSONs (building footprints + road centerlines, "
            "flooded flag) are downloaded; VHR imagery is NOT pulled. AOIs: "
            "Germany_Training_Public, Louisiana-East_Training_Public. Louisiana-West is "
            "imagery-only (no labels) and excluded.\n"
        )

    io.check_disk()
    all_recs: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for recs in star_imap_unordered(p, _scan_file, jobs):
            all_recs.extend(recs)
    print(f"scanned {len(all_recs)} non-empty tiles from {len(jobs)} label files")

    selected = sampling.select_tiles_per_class(
        all_recs, classes_key="classes_present", per_class=PER_CLASS
    )
    selected.sort(key=lambda r: r["source_id"])
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} tiles (of {len(all_recs)})")

    tile_counts = {i: 0 for i in range(NUM_CLASSES)}
    aoi_counts: dict[str, int] = {}
    for r in selected:
        aoi_counts[r["aoi"]] = aoi_counts.get(r["aoi"], 0) + 1
        for c in r["classes_present"]:
            tile_counts[c] += 1
    print(
        "tiles-per-class:", {CLASSES[i][0]: tile_counts[i] for i in range(NUM_CLASSES)}
    )
    print("tiles-per-aoi:", aoi_counts)

    if args.probe:
        print("probe only; exiting before writes")
        return

    written = 0
    with multiprocessing.Pool(args.workers) as p:
        for n in star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]):
            written += n
    print(f"wrote {written} new tiles ({len(selected)} total selected)")

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "SpaceNet 8 / AWS Open Data",
            "license": "CC-BY-SA-4.0",
            "provenance": {
                "url": "https://spacenet.ai/sn8-challenge/",
                "have_locally": False,
                "annotation_method": "manual annotation of Maxar VHR pre/post-event imagery",
                "s3": f"s3://{BUCKET}/{S3_ROOT}/",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "tiles_per_class": {
                CLASSES[i][0]: tile_counts[i] for i in range(NUM_CLASSES)
            },
            "tiles_per_aoi": aoi_counts,
            "change_time_by_aoi": {a: t.isoformat() for a, t in AOIS.items()},
            "notes": (
                "VHR (~0.3-0.8 m) SpaceNet 8 flood labels rasterized to local-UTM 10 m "
                "<=64x64 patches (all_touched). Unified 4-class scheme "
                "(building/road x flooded/non-flooded); flooded classes painted last so the "
                "flood signal wins overlaps. Positive-only (non-structure ground = nodata "
                "255); assembly supplies negatives. Individual buildings/narrow roads are "
                "under-resolved at 10 m but flooded structures aggregate into flood-extent "
                "patches (retained per spec S5). Change label: change_time set per dated "
                "flood event (Germany 2021-07-15; Louisiana Hurricane Ida 2021-08-30), "
                "time_range = 360-day window centered on it (flood is transient, so no "
                "persistent-state recast). Louisiana-West test AOI excluded (imagery only)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("class tile counts:")
    for i in range(NUM_CLASSES):
        print(f"  {i} {CLASSES[i][0]:22} {tile_counts[i]}")
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
