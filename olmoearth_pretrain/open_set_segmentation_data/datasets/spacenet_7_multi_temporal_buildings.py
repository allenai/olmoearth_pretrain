"""SpaceNet 7 (multi-temporal buildings) -> open-set-segmentation building-presence masks.

Source: SpaceNet 7 Multi-Temporal Urban Development Challenge, hosted on the public AWS
Open Data bucket ``s3://spacenet-dataset/spacenet/SN7_buildings/`` (unsigned/anonymous
access; CC-BY-SA-4.0). 60 training AOIs worldwide (``train/``), each ~4 km x 4 km, imaged
as monthly PlanetScope mosaics (~4 m/px, EPSG:3857) for ~25 months (2018-01 .. 2020-01).
Per AOI/month there is a ``labels/..._Buildings.geojson`` of manually-digitized building
footprints (CRS84 polygons) plus a ``..._UDM.geojson`` unusable-data mask. The 20
``test_public/`` AOIs are imagery-only (labels withheld) and are excluded. We need only
the small label GeoJSONs (+ each image's header for the AOI extent) -- pretraining supplies
its own S2/S1/Landsat imagery -- so no PlanetScope mosaic rasters are downloaded.

ENCODING CHOICE (spec S4 polygons / S2 GeoTIFF): **building-presence classification**.
Native imagery is 4 m and individual buildings are sub-10 m, but building FOOTPRINT
PRESENCE aggregates to a built-up-vs-not signal that is observable at 10 m. We take ONE
representative month per AOI (the latest available month => most-complete built-up state),
rasterize the union of that month's building footprints into a local-UTM 10 m grid
(building=1, background=0), and tile the full AOI extent into <=64x64 patches. This is a
genuine dense 2-class raster: the whole AOI is annotated, so background (0) is a real
observed "not-built" class, NOT a fabricated negative. UDM (unusable-data) polygons, when
present, are burned as nodata (255).

We deliberately DO NOT use the SpaceNet 7 change/tracking task: although the headline task
is building change and construction is monthly-resolved (within the ~1-2 month tolerance),
the presence encoding is simpler and robust and captures the salvageable 10 m signal.
change_time is therefore null and time_range is a static 360-day window centered on the
chosen month (spec S5 seasonal/annual rule). All labels are 2018-2020 (post-2016 OK).

Balancing (spec S5): building-present tiles vs background-only tiles are balanced
(up to 1000 each) via balance_by_class on a per-tile presence category.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.spacenet_7_multi_temporal_buildings
"""

import argparse
import json
import math
import multiprocessing
import os
import re
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

# rasterio /vsis3/ header reads use anonymous access.
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")

SLUG = "spacenet_7_multi_temporal_buildings"
NAME = "SpaceNet 7 (multi-temporal buildings)"
BUCKET = "spacenet-dataset"
S3_ROOT = "spacenet/SN7_buildings"
TRAIN_PREFIX = f"{S3_ROOT}/train"
TILE = 64
PER_CLASS = 1000

CLASSES = [
    (
        "background",
        "Non-building background: pixels inside the densely-annotated SN7 AOI mosaic extent "
        "not covered by any building footprint (open land, vegetation, water, roads, bare "
        "ground). A genuinely observed absence-of-building class, not a fabricated negative -- "
        "the entire ~4 km x 4 km AOI is manually annotated.",
    ),
    (
        "building",
        "Building footprint presence: a 10 m pixel touched by any manually-digitized building "
        "polygon in the chosen monthly PlanetScope mosaic. Individual buildings are sub-10 m, "
        "but footprints aggregate into built-up / settlement extent that is observable at "
        "10 m (built-up vs not).",
    ),
]
NUM_CLASSES = len(CLASSES)
BG, BUILDING = 0, 1

_FNAME_RE = re.compile(
    r"global_monthly_(\d{4})_(\d{2})_mosaic_(.+)_Buildings\.geojson$"
)


def _s3_client():
    import boto3
    import botocore

    return boto3.client(
        "s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED)
    )


def list_aois() -> list[str]:
    """List the 60 training AOI names (those with building labels)."""
    s3 = _s3_client()
    aois: list[str] = []
    for pg in s3.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=f"{TRAIN_PREFIX}/", Delimiter="/"
    ):
        for cp in pg.get("CommonPrefixes", []):
            name = cp["Prefix"].rstrip("/").rsplit("/", 1)[1]
            aois.append(name)
    return sorted(aois)


def _latest_month(aoi: str) -> tuple[int, int, str] | None:
    """Find the latest (year, month, key) Buildings.geojson for an AOI."""
    s3 = _s3_client()
    prefix = f"{TRAIN_PREFIX}/{aoi}/labels/"
    best: tuple[int, int, str] | None = None
    for pg in s3.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=prefix
    ):
        for o in pg.get("Contents", []):
            m = _FNAME_RE.search(o["Key"])
            if not m:
                continue
            y, mo = int(m.group(1)), int(m.group(2))
            if best is None or (y, mo) > (best[0], best[1]):
                best = (y, mo, o["Key"])
    return best


def _aoi_extent_wgs84(
    aoi: str, year: int, month: int
) -> tuple[float, float, float, float]:
    """Return (min_lon, min_lat, max_lon, max_lat) of the AOI mosaic from the image header."""
    import rasterio
    from rasterio.warp import transform_bounds

    stem = f"global_monthly_{year:04d}_{month:02d}_mosaic_{aoi}"
    url = f"/vsis3/{BUCKET}/{TRAIN_PREFIX}/{aoi}/images/{stem}.tif"
    with rasterio.open(url) as ds:
        return tuple(transform_bounds(ds.crs, "EPSG:4326", *ds.bounds))  # type: ignore[return-value]


def _process_aoi(aoi: str) -> list[dict[str, Any]]:
    """Download the latest-month labels for an AOI and rasterize presence tiles."""
    found = _latest_month(aoi)
    if found is None:
        print(f"{aoi}: no Buildings.geojson found")
        return []
    year, month, key = found
    stem = f"global_monthly_{year:04d}_{month:02d}_mosaic_{aoi}"

    dst_dir = io.raw_dir(SLUG) / aoi
    b_dst = dst_dir / f"{stem}_Buildings.geojson"
    download.download_s3_unsigned(BUCKET, key, b_dst)
    udm_key = key.replace("_Buildings.geojson", "_UDM.geojson")
    udm_dst = dst_dir / f"{stem}_UDM.geojson"
    try:
        download.download_s3_unsigned(BUCKET, udm_key, udm_dst)
    except Exception:
        udm_dst = None  # type: ignore[assignment]

    with UPath(b_dst).open("r") as f:
        bgj = json.load(f)
    bshapes: list[Any] = []
    for feat in bgj.get("features", []):
        geom = feat.get("geometry")
        if not geom:
            continue
        try:
            g = shapely.geometry.shape(geom)
        except Exception:
            continue
        if not g.is_empty and g.is_valid:
            bshapes.append(g)
    if not bshapes:
        print(f"{aoi}: 0 building polygons in {stem}")
        return []

    udm_shapes: list[Any] = []
    if udm_dst is not None:
        try:
            with UPath(udm_dst).open("r") as f:
                ugj = json.load(f)
            for feat in ugj.get("features", []):
                geom = feat.get("geometry")
                if not geom:
                    continue
                g = shapely.geometry.shape(geom)
                if not g.is_empty and g.is_valid:
                    udm_shapes.append(g)
        except Exception:
            udm_shapes = []

    # AOI extent (WGS84) from the image header; local UTM projection from its centroid.
    min_lon, min_lat, max_lon, max_lat = _aoi_extent_wgs84(aoi, year, month)
    c_lon = (min_lon + max_lon) / 2.0
    c_lat = (min_lat + max_lat) / 2.0
    proj = io.utm_projection_for_lonlat(c_lon, c_lat)

    # Reproject the AOI bbox to UTM pixel coords to get the full extent to tile.
    extent_box = shapely.geometry.box(min_lon, min_lat, max_lon, max_lat)
    px_extent = rasterize.geom_to_pixels(extent_box, WGS84_PROJECTION, proj)
    ex0, ey0, ex1, ey1 = px_extent.bounds
    x_min, y_min = int(math.floor(ex0)), int(math.floor(ey0))
    x_max, y_max = int(math.ceil(ex1)), int(math.ceil(ey1))

    # Reproject building + UDM polygons to UTM pixel coords.
    b_px: list[tuple[Any, int]] = []
    for g in bshapes:
        try:
            pg = rasterize.geom_to_pixels(g, WGS84_PROJECTION, proj)
        except Exception:
            continue
        if not pg.is_empty:
            b_px.append((pg, BUILDING))
    udm_px: list[tuple[Any, int]] = []
    for g in udm_shapes:
        try:
            pg = rasterize.geom_to_pixels(g, WGS84_PROJECTION, proj)
        except Exception:
            continue
        if not pg.is_empty:
            udm_px.append((pg, io.CLASS_NODATA))
    if not b_px:
        return []
    # Paint buildings (1) then UDM nodata (255) last so unusable regions win.
    paint = b_px + udm_px

    center = datetime(year, month, 15, tzinfo=UTC)
    recs: list[dict[str, Any]] = []
    for ti in range(math.ceil((x_max - x_min) / TILE)):
        for tj in range(math.ceil((y_max - y_min) / TILE)):
            bx0 = x_min + ti * TILE
            by0 = y_min + tj * TILE
            bounds = (bx0, by0, bx0 + TILE, by0 + TILE)
            arr = rasterize.rasterize_shapes(
                paint, bounds, fill=BG, dtype="uint8", all_touched=True
            )[0]
            present = sorted(int(v) for v in np.unique(arr) if v != io.CLASS_NODATA)
            if not present:
                continue  # entirely UDM/nodata
            recs.append(
                {
                    "array": arr,
                    "crs": proj.crs.to_string(),
                    "bounds": bounds,
                    "classes_present": present,
                    "category": BUILDING if BUILDING in present else BG,
                    "aoi": aoi,
                    "year": year,
                    "month": month,
                    "center": center.isoformat(),
                    "source_id": f"{aoi}/{year:04d}_{month:02d}/{ti}_{tj}",
                }
            )
    print(f"{aoi}: {stem} -> {len(recs)} tiles ({len(b_px)} bldg polys)")
    return recs


def _write_one(rec: dict[str, Any]) -> int:
    from datetime import timedelta

    from rasterio.crs import CRS

    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists() and (
        io.locations_dir(SLUG) / f"{sample_id}.json"
    ).exists():
        return 0
    proj = Projection(CRS.from_string(rec["crs"]), io.RESOLUTION, -io.RESOLUTION)
    bounds = tuple(rec["bounds"])
    center = datetime.fromisoformat(rec["center"])
    time_range = (center - timedelta(days=180), center + timedelta(days=180))
    io.write_label_geotiff(
        SLUG, sample_id, rec["array"], proj, bounds, nodata=io.CLASS_NODATA
    )
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        time_range,
        change_time=None,
        source_id=rec["source_id"],
        classes_present=rec["classes_present"],
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

    aois = list_aois()
    print(f"{len(aois)} training AOIs")

    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    with (io.raw_dir(SLUG) / "SOURCE.txt").open("w") as f:
        f.write(
            "SpaceNet 7 Multi-Temporal Urban Development Challenge labels.\n"
            f"Source: s3://{BUCKET}/{S3_ROOT}/ (public AWS Open Data, unsigned). "
            "License CC-BY-SA-4.0.\n"
            "Only the latest-month per-AOI building-footprint GeoJSONs (train/, 60 AOIs) "
            "and each AOI image header (for extent) are used; PlanetScope mosaics are NOT "
            "downloaded. test_public/ AOIs are imagery-only (labels withheld) and excluded.\n"
        )

    all_recs: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for recs in star_imap_unordered(p, _process_aoi, [dict(aoi=a) for a in aois]):
            all_recs.extend(recs)
    io.check_disk()
    n_bldg = sum(1 for r in all_recs if r["category"] == BUILDING)
    n_bg = len(all_recs) - n_bldg
    print(
        f"scanned {len(all_recs)} candidate tiles ({n_bldg} building, {n_bg} background)"
    )

    selected = sampling.balance_by_class(
        all_recs, key=lambda r: r["category"], per_class=PER_CLASS
    )
    selected.sort(key=lambda r: r["source_id"])
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    sel_bldg = sum(1 for r in selected if r["category"] == BUILDING)
    sel_bg = len(selected) - sel_bldg
    print(f"selected {len(selected)} tiles ({sel_bldg} building, {sel_bg} background)")

    # tiles-per-pixel-class (a building tile also contains background pixels).
    tile_counts = {BG: 0, BUILDING: 0}
    aoi_counts: dict[str, int] = {}
    for r in selected:
        aoi_counts[r["aoi"]] = aoi_counts.get(r["aoi"], 0) + 1
        for c in r["classes_present"]:
            tile_counts[c] += 1

    if args.probe:
        print(
            "tiles-per-class:",
            {CLASSES[i][0]: tile_counts[i] for i in range(NUM_CLASSES)},
        )
        print("tiles-per-aoi:", aoi_counts)
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
            "source": "SpaceNet 7 / AWS Open Data",
            "license": "CC-BY-SA-4.0",
            "provenance": {
                "url": "https://spacenet.ai/sn7-challenge/",
                "have_locally": False,
                "annotation_method": "manual digitization of monthly PlanetScope mosaics",
                "s3": f"s3://{BUCKET}/{S3_ROOT}/",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "num_building_tiles": sel_bldg,
            "num_background_tiles": sel_bg,
            "tiles_per_class": {
                CLASSES[i][0]: tile_counts[i] for i in range(NUM_CLASSES)
            },
            "tiles_per_aoi": aoi_counts,
            "notes": (
                "SpaceNet 7 building footprints (60 worldwide train AOIs, ~4 km each) "
                "encoded as building-PRESENCE classification at 10 m. One representative "
                "month per AOI (latest available, most-complete built-up state) is "
                "rasterized (all_touched) into local-UTM 10 m <=64x64 tiles: background=0, "
                "building=1, UDM unusable-data=255 (nodata). Dense fully-annotated scene, "
                "so background is a real observed class (not a fabricated negative). "
                "Building-present vs background-only tiles balanced up to 1000 each. "
                "change_time=null; time_range = 360-day window centered on the chosen "
                "month (presence, not change). Multi-temporal/change task deliberately not "
                "used (presence is simpler + robust; native 4 m buildings are sub-10 m but "
                "footprint presence aggregates to a built-up signal at 10 m). All labels "
                "2018-2020 (post-2016). test_public AOIs excluded (labels withheld)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("class tile counts:")
    for i in range(NUM_CLASSES):
        print(f"  {i} {CLASSES[i][0]:12} {tile_counts[i]}")
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
