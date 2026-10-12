"""SpaceNet 3 & 5 road centerlines -> open-set-segmentation road masks.

Source: SpaceNet Roads, hosted on the public AWS Open Data bucket
``s3://spacenet-dataset/spacenet/`` (unsigned/anonymous access; CC-BY-SA-4.0).
Two challenges cover roads:
  - SpaceNet 3 (SN3_roads): AOI_2_Vegas, AOI_3_Paris, AOI_4_Shanghai, AOI_5_Khartoum.
  - SpaceNet 5 (SN5_roads): AOI_7_Moscow, AOI_8_Mumbai (San Juan is test/imagery-only).
Labels are hand-digitized road **centerlines** (LineString), one small GeoJSON per ~400 m
image chip, carrying route/type + inferred-speed attributes (SN3: ``road_type``,
``lane_number``, ``paved``, ``bridge_type``, ``inferred_speed_mph``; SN5: OSM-style
``highway``, ``surface``, ``lanes``, ``inferred_speed_mph``). Geometries are WGS84 (CRS84
lon/lat); the paired imagery is DigitalGlobe/Maxar VHR (~0.3 m). We need only the small
label GeoJSONs -- pretraining supplies its own S2/S1/Landsat imagery -- so the multi-GB
imagery tarballs are NOT downloaded (only the per-chip ``geojson_roads_speed`` files, and
only from the labeled ``train/`` splits; ``test_public`` labels are withheld).

Recipe (spec S4 "lines"): each road centerline is rasterized into a thin dilated mask so
it is visible at 10 m/pixel. Single foreground class:
  0 = road (a mapped road centerline; dilated to ~20-30 m so it registers at 10 m).
This is a **positive-only** dataset (spec S5): non-road pixels are left as nodata/ignore
(255) -- we do NOT fabricate a background class; the assembly step supplies negatives from
other datasets. Type/surface/speed attributes are retained as summary provenance but not
split into classes (the task specifies a single "road" foreground class).

Suitability at 10 m: paved roads (esp. multi-lane arterials/highways in these dense urban
AOIs) are sharp, physically-real linear features clearly resolvable in Sentinel-2 / Landsat
imagery; a centerline dilated to ~2-3 px is a meaningful 10 m label. ACCEPTED. Narrow
residential streets are under-resolved at 10 m but aggregate into the road network signal
(retained per S5).

Tiling: each ~400 m chip (~40 px at 10 m) fits inside one 64x64 (640 m) tile. One tile per
chip, centered on the chip's road-union centroid; chips whose rasterized road mask has
< MIN_ROAD_PIXELS pixels (empty / trivial slivers) are dropped.

Time range (spec S5): road networks are static/persistent features. SpaceNet 3 VHR was
collected ~2015-2016 and SpaceNet 5 ~2017-2018; we anchor each program to a representative
1-year window in the Sentinel era within the manifest range [2016, 2019] (SN3 -> 2017,
SN5 -> 2018). ``change_time`` is null.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.spacenet_3_5_roads
"""

import argparse
import json
import math
import multiprocessing
from typing import Any

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

SLUG = "spacenet_3_5_roads"
NAME = "SpaceNet 3/5 Roads"
BUCKET = "spacenet-dataset"

TILE = 64  # 640 m tiles (one per ~400 m chip).
DILATE_RADIUS_PX = 1.0  # buffer the centerline by ~1 px -> ~2-3 px (20-30 m) wide.
MIN_ROAD_PIXELS = 3  # drop chips whose road mask is a trivial sliver / empty.

CID_ROAD = 0
CLASSES = [
    {
        "id": CID_ROAD,
        "name": "road",
        "description": (
            "A mapped road centerline (SpaceNet 3/5 hand-digitized route network, with "
            "type/surface/lane/inferred-speed attributes in the source). Rasterized from "
            "the LineString and dilated to ~20-30 m (2-3 px) so it is visible at "
            "10 m/pixel. Non-road pixels are nodata (255): this is a positive-only mask."
        ),
    }
]

# Labeled train AOIs per program -> representative Sentinel-era year (roads are static).
PROGRAMS = {
    "SN3": {
        "year": 2017,
        "aois": ["AOI_2_Vegas", "AOI_3_Paris", "AOI_4_Shanghai", "AOI_5_Khartoum"],
    },
    "SN5": {
        "year": 2018,
        "aois": ["AOI_7_Moscow", "AOI_8_Mumbai"],
    },
}


def _s3():
    import boto3
    import botocore

    return boto3.client(
        "s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED)
    )


def _list_chip_keys() -> list[dict[str, Any]]:
    """List every per-chip road-label GeoJSON key across the labeled AOIs."""
    s3 = _s3()
    jobs: list[dict[str, Any]] = []
    for prog, cfg in PROGRAMS.items():
        for aoi in cfg["aois"]:
            prefix = f"spacenet/{prog}_roads/train/{aoi}/geojson_roads_speed/"
            for pg in s3.get_paginator("list_objects_v2").paginate(
                Bucket=BUCKET, Prefix=prefix
            ):
                for o in pg.get("Contents", []):
                    if o["Key"].endswith(".geojson"):
                        jobs.append({"key": o["Key"], "program": prog, "aoi": aoi})
    return jobs


def _local_path(key: str) -> UPath:
    """raw/{slug}/<key-relative-to-spacenet/> for a chip GeoJSON."""
    rel = key[len("spacenet/") :]
    return io.raw_dir(SLUG) / rel


def _scan_chip(key: str, program: str, aoi: str) -> list[dict[str, Any]]:
    """Download (idempotent) + rasterize one chip's road centerlines into a 64x64 tile.

    Returns a one-element record list (or [] if the chip has no usable road pixels).
    """
    dst = _local_path(key)
    if not dst.exists():
        download.download_s3_unsigned(BUCKET, key, dst)

    with dst.open("r") as f:
        gj = json.load(f)

    lines: list[Any] = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry")
        if not geom:
            continue
        try:
            shp = shapely.geometry.shape(geom)
        except Exception:
            continue
        if shp.is_empty or shp.length == 0:
            continue
        lines.append(shp)
    if not lines:
        return []

    union = shapely.unary_union(lines)
    c = union.centroid
    if c.is_empty:
        return []
    proj = io.utm_projection_for_lonlat(float(c.x), float(c.y))

    # Reproject each line to UTM pixel coords and dilate to a thin mask.
    px_shapes: list[tuple[Any, int]] = []
    xs: list[float] = []
    ys: list[float] = []
    for shp in lines:
        try:
            pg = rasterize.geom_to_pixels(shp, WGS84_PROJECTION, proj)
        except Exception:
            continue
        if pg.is_empty:
            continue
        dil = pg.buffer(DILATE_RADIUS_PX)
        if dil.is_empty:
            continue
        px_shapes.append((dil, CID_ROAD))
        x0, y0, x1, y1 = dil.bounds
        xs += [x0, x1]
        ys += [y0, y1]
    if not px_shapes:
        return []

    # Center a 64x64 tile on the road-union pixel bbox center.
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    col = int(math.floor(cx))
    row = int(math.floor(cy))
    bounds = io.centered_bounds(col, row, TILE, TILE)

    arr = rasterize.rasterize_shapes(
        px_shapes, bounds, fill=io.CLASS_NODATA, dtype="uint8", all_touched=True
    )[0]
    if int((arr == CID_ROAD).sum()) < MIN_ROAD_PIXELS:
        return []

    stem = key.rsplit("/", 1)[1][: -len(".geojson")]
    return [
        {
            "array": arr,
            "crs": proj.crs.to_string(),
            "bounds": bounds,
            "program": program,
            "aoi": aoi,
            "source_id": f"{program}/{aoi}/{stem}",
        }
    ]


def _write_one(rec: dict[str, Any]) -> int:
    from rasterio.crs import CRS

    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return 0
    proj = Projection(CRS.from_string(rec["crs"]), io.RESOLUTION, -io.RESOLUTION)
    bounds = tuple(rec["bounds"])
    io.write_label_geotiff(
        SLUG, sample_id, rec["array"], proj, bounds, nodata=io.CLASS_NODATA
    )
    year = PROGRAMS[rec["program"]]["year"]
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(year),
        change_time=None,
        source_id=rec["source_id"],
        classes_present=[CID_ROAD],
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

    jobs = _list_chip_keys()
    print(
        f"listed {len(jobs)} road-label chips across {sum(len(c['aois']) for c in PROGRAMS.values())} AOIs"
    )

    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    with (io.raw_dir(SLUG) / "SOURCE.txt").open("w") as f:
        f.write(
            "SpaceNet 3 & 5 Roads labels (hand-digitized road centerlines with type/"
            "surface/speed attributes).\n"
            f"Source: s3://{BUCKET}/spacenet/SN3_roads/ and .../SN5_roads/ "
            "(public AWS Open Data, unsigned). License CC-BY-SA-4.0.\n"
            "Only the per-chip geojson_roads_speed/*.geojson label files from the labeled "
            "train/ splits are downloaded; the multi-GB VHR imagery tarballs and the "
            "unlabeled test_public splits are NOT pulled.\n"
            "AOIs: SN3 = Vegas, Paris, Shanghai, Khartoum; SN5 = Moscow, Mumbai.\n"
        )

    all_recs: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for recs in star_imap_unordered(p, _scan_chip, jobs):
            all_recs.extend(recs)
    print(f"scanned {len(all_recs)} non-empty road tiles from {len(jobs)} chips")

    io.check_disk()

    # Single positive-only class; keep all tiles, honoring the 25k hard cap.
    all_recs.sort(key=lambda r: r["source_id"])
    if len(all_recs) > sampling.MAX_SAMPLES_PER_DATASET:
        all_recs = all_recs[: sampling.MAX_SAMPLES_PER_DATASET]
        print(f"capped to {sampling.MAX_SAMPLES_PER_DATASET}")
    for i, r in enumerate(all_recs):
        r["sample_id"] = f"{i:06d}"

    aoi_counts: dict[str, int] = {}
    for r in all_recs:
        aoi_counts[r["aoi"]] = aoi_counts.get(r["aoi"], 0) + 1
    print("tiles-per-aoi:", aoi_counts)

    if args.probe:
        print("probe only; exiting before writes")
        return

    written = 0
    with multiprocessing.Pool(args.workers) as p:
        for n in star_imap_unordered(p, _write_one, [dict(rec=r) for r in all_recs]):
            written += n
    print(f"wrote {written} new tiles ({len(all_recs)} total selected)")

    io.check_disk()
    num_samples = sum(1 for _ in io.locations_dir(SLUG).glob("*.tif"))

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "AWS Open Data (SpaceNet)",
            "license": "CC-BY-SA-4.0",
            "provenance": {
                "url": "https://spacenet.ai/spacenet-roads-dataset/",
                "have_locally": False,
                "annotation_method": "manual digitization of DigitalGlobe/Maxar VHR imagery",
                "s3": f"s3://{BUCKET}/spacenet/SN3_roads/ , SN5_roads/",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": num_samples,
            "tiles_per_aoi": aoi_counts,
            "year_by_program": {p: c["year"] for p, c in PROGRAMS.items()},
            "notes": (
                "Positive-only road-centerline segmentation. SpaceNet 3/5 hand-digitized "
                "road centerlines (WGS84 LineStrings, per-chip geojson_roads_speed) "
                "rasterized (buffered ~1 px -> ~20-30 m wide, all_touched) into 64x64 UTM "
                "10 m tiles; class 0 = road, non-road = nodata (255). One tile per ~400 m "
                "chip centered on the chip's road-union centroid; chips with < "
                f"{MIN_ROAD_PIXELS} road px dropped. Only label GeoJSONs from labeled "
                "train/ splits downloaded (no imagery, no unlabeled test_public). AOIs: "
                "SN3 Vegas/Paris/Shanghai/Khartoum, SN5 Moscow/Mumbai. Roads are static; "
                "1-year window per program (SN3->2017, SN5->2018) within manifest range "
                "[2016,2019]. Type/surface/lane/inferred-speed attributes exist in source "
                "but are collapsed to a single road class per the task spec. Caveat: narrow "
                "residential streets are under-resolved at 10 m."
            ),
        },
    )

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=num_samples
    )
    print(f"done: {num_samples} samples; tiles-per-aoi={aoi_counts}")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
