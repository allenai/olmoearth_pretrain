"""Process OpenStreetMap Leisure/Tourism extracts (Geofabrik) into leisure/tourism
land-use segmentation label tiles.

Source: OpenStreetMap, packaged by Geofabrik as regional ``*-latest-free.shp.zip``
shapefile extracts (https://download.geofabrik.de/). Each extract bundles thematic
ESRI-shapefile layers in WGS84 (EPSG:4326); Geofabrik assigns every feature an ``fclass``
string derived from its OSM tags. We use the two *area* (polygon) layers that carry
leisure/tourism land-use features:

  * ``gis_osm_pois_a_free_1``   -- POI polygons (park, golf_course, pitch, stadium,
                                   marina, camp_site, caravan_site, nature_reserve, ...)
  * ``gis_osm_landuse_a_free_1``-- land-use polygons (park, cemetery, nature_reserve, ...)

Geofabrik publishes ``.shp.zip`` only for regions under a size cap (large regions such as
Great Britain, California, Japan, Brazil, whole-Germany have **no** shapefile download and
were skipped); we therefore sample a **bounded, globally diverse set of regional extracts**
(spec 5: large global crowdsourced source -> sample representative regions, not global
coverage). Regions used span 6 continents (see REGIONS).

Unified class scheme (spec 5: mixed points+polygons -> ONE class scheme). All target
classes are areal land-use features; OSM represents them as polygons in the *_a_ layers, so
this is processed as a **polygons -> rasterized GeoTIFF** dataset (no separate point table):

    0 park            leisure=park
    1 golf_course     leisure=golf_course
    2 stadium_pitch   leisure=stadium + leisure=pitch      (manifest "stadium/pitch")
    3 cemetery        landuse=cemetery + amenity=grave_yard
    4 marina          leisure=marina
    5 camp_site       tourism=camp_site + tourism=caravan_site
    6 nature_reserve  leisure=nature_reserve

Observability at 10-30 m / coarsening (spec 4 VHR judgment, manifest note):
  * ``ski piste`` (manifest class) is DROPPED -- OSM piste:type features are not present in
    Geofabrik's free shapefile layers (there is no piste layer), so there is no geometry to
    rasterize from this source.
  * Sub-pixel features (individual tennis courts, pocket parks, small graveyards) are not
    resolvable at 10 m. We drop every polygon smaller than MIN_AREA_M2 (0.25 ha ~ 25 px),
    keeping only footprints discernible from S2/S1/Landsat (full sports pitches, golf
    courses, urban parks, large cemeteries, marinas, camp sites, nature reserves).

Positive-only (spec 5): OSM tags presence, not absence -- an untagged pixel is not a
verified negative. So each tile rasterizes its polygon to the class id and leaves all other
pixels as nodata (255); no synthetic background is fabricated. Assembly adds negatives from
other datasets.

Tiling: each kept polygon -> one tile in a local UTM projection at 10 m/pixel. The tile is
sized down to the polygon footprint (padded), capped at 64x64; polygons larger than 640 m
yield a 64x64 window centered on the centroid (a representative chunk). all_touched
rasterization so thin/small resolvable features still register.

Time range: OSM features here are persistent land use (static). Per spec 5 we assign a
representative 1-year Sentinel-era window (REP_YEAR).

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.openstreetmap_leisure_tourism_extracts
Idempotent: existing locations/{id}.tif are skipped; re-running re-uses downloaded zips.
"""

import argparse
import math
import multiprocessing
from collections import Counter
from typing import Any

import geopandas as gpd
import numpy as np
import shapely
import shapely.wkb
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "openstreetmap_leisure_tourism_extracts"
NAME = "OpenStreetMap Leisure/Tourism Extracts"

# Globally diverse, Geofabrik-shapefile-available regional extracts (6 continents).
REGIONS = [
    "europe/spain",
    "europe/switzerland",
    "europe/netherlands",
    "europe/ireland-and-northern-ireland",
    "north-america/us/florida",
    "north-america/us/washington",
    "north-america/us/arizona",
    "asia/south-korea",
    "asia/philippines",
    "asia/india",
    "australia-oceania/australia",
    "africa/south-africa",
    "africa/kenya",
    "south-america/chile",
    "south-america/argentina",
]

LAYERS = ["gis_osm_pois_a_free_1", "gis_osm_landuse_a_free_1"]

# Unified class scheme: (name, description, {source fclass values}).
CLASSES: list[tuple[str, str, set[str]]] = [
    (
        "park",
        "Public / urban park or green recreation ground (OSM leisure=park).",
        {"park"},
    ),
    (
        "golf_course",
        "Golf course, incl. fairways/greens/roughs (OSM leisure=golf_course).",
        {"golf_course"},
    ),
    (
        "stadium_pitch",
        "Outdoor sports stadium or sports pitch/field (OSM leisure=stadium, leisure=pitch); "
        "manifest 'stadium/pitch'.",
        {"stadium", "pitch"},
    ),
    (
        "cemetery",
        "Cemetery / graveyard (OSM landuse=cemetery, amenity=grave_yard).",
        {"cemetery", "graveyard"},
    ),
    (
        "marina",
        "Marina / boat harbour with berths and docks (OSM leisure=marina).",
        {"marina"},
    ),
    (
        "camp_site",
        "Camp site or caravan/RV site (OSM tourism=camp_site, tourism=caravan_site).",
        {"camp_site", "caravan_site"},
    ),
    (
        "nature_reserve",
        "Protected nature reserve (OSM leisure=nature_reserve).",
        {"nature_reserve"},
    ),
]
NAME_TO_ID = {name: i for i, (name, _d, _f) in enumerate(CLASSES)}
FCLASS_TO_ID = {fc: i for i, (_n, _d, fcs) in enumerate(CLASSES) for fc in fcs}

MIN_AREA_M2 = 2500.0  # 0.25 ha (~25 px at 10 m): drop sub-pixel/unresolvable features
EQUAL_AREA_CRS = "EPSG:6933"  # global cylindrical equal-area (metres) for area filter
MIN_TILE = 8
MAX_TILE = io.MAX_TILE  # 64
PAD = 2
PER_CLASS = 1000
REP_YEAR = (
    2024  # representative Sentinel-era year for these static OSM land-use features
)


def region_zip(region: str) -> Any:
    fn = region.replace("/", "_") + "-latest-free.shp.zip"
    return io.raw_dir(SLUG) / fn


def scan_region(region: str) -> list[dict[str, Any]]:
    """Read the leisure/tourism polygons from one region's extract -> record dicts.

    Each record: wkb (WGS84 geometry), class name, source_id. Applies fclass filter and
    the MIN_AREA_M2 resolvability filter (equal-area).
    """
    zp = region_zip(region)
    if not zp.exists():
        print(f"  [skip] missing extract for {region}")
        return []
    recs: list[dict[str, Any]] = []
    for layer in LAYERS:
        path = f"/vsizip/{zp.path}/{layer}.shp"
        try:
            gdf = gpd.read_file(path)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] {region}/{layer}: {e}")
            continue
        if "fclass" not in gdf.columns or len(gdf) == 0:
            continue
        gdf = gdf[gdf["fclass"].isin(FCLASS_TO_ID)]
        if len(gdf) == 0:
            continue
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
        if len(gdf) == 0:
            continue
        # Equal-area filter for 10 m resolvability.
        area_m2 = gdf.geometry.to_crs(EQUAL_AREA_CRS).area
        gdf = gdf[area_m2.values >= MIN_AREA_M2]
        for osm_id, fclass, geom in zip(gdf["osm_id"], gdf["fclass"], gdf.geometry):
            recs.append(
                {
                    "osm_id": str(osm_id),
                    "wkb": shapely.wkb.dumps(geom),
                    "class": CLASSES[FCLASS_TO_ID[fclass]][0],  # unified class name
                    "source_id": f"{region}:{layer}:osm_id={osm_id}:{fclass}",
                }
            )
    print(f"  {region}: {len(recs)} polygons kept")
    return recs


def _write_one(rec: dict[str, Any]) -> str | None:
    from shapely.geometry import box

    from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
        geom_to_pixels,
        rasterize_shapes,
    )

    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return "skip"

    geom = shapely.wkb.loads(rec["wkb"])
    if geom.is_empty:
        return None
    c = geom.centroid
    proj = io.utm_projection_for_lonlat(float(c.x), float(c.y))
    px = geom_to_pixels(geom, WGS84_PROJECTION, proj)
    if px.is_empty or px.area <= 0:
        return None
    minx, miny, maxx, maxy = px.bounds
    w, h = maxx - minx, maxy - miny
    tw = min(MAX_TILE, max(MIN_TILE, int(math.ceil(w)) + 2 * PAD))
    th = min(MAX_TILE, max(MIN_TILE, int(math.ceil(h)) + 2 * PAD))
    col = round((minx + maxx) / 2.0)
    row = round((miny + maxy) / 2.0)
    bounds = io.centered_bounds(col, row, tw, th)

    # Clip geometry to the tile box (avoids huge geoms; rasterize also clips).
    clip = px.intersection(box(*bounds))
    if clip.is_empty or clip.area <= 0:
        return None

    cid = NAME_TO_ID[rec["class"]]
    label = rasterize_shapes(
        [(clip, cid)], bounds, fill=io.CLASS_NODATA, dtype="uint8", all_touched=True
    )[0]
    present = sorted(int(v) for v in np.unique(label) if v != io.CLASS_NODATA)
    if not present:  # polygon slipped entirely into nodata (degenerate) -> skip
        return None

    io.write_label_geotiff(SLUG, sample_id, label, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(REP_YEAR),
        source_id=rec["source_id"],
        classes_present=present,
    )
    return rec["class"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    # ---- Phase A: scan every region for leisure/tourism polygons (parallel over regions)
    io.check_disk()
    records: list[dict[str, Any]] = []
    with multiprocessing.Pool(min(len(REGIONS), 16)) as p:
        for recs in star_imap_unordered(
            p, scan_region, [dict(region=r) for r in REGIONS]
        ):
            records.extend(recs)
    if not records:
        raise RuntimeError(
            "no polygons scanned -- region extracts missing? (download step failed)"
        )
    # Dedup: leisure=park / landuse=cemetery are emitted in BOTH pois_a and landuse_a with
    # the same osm_id, and border features recur across regional extracts. Keep one per
    # osm_id so a location is not tiled twice.
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for r in records:
        if r["osm_id"] in seen:
            continue
        seen.add(r["osm_id"])
        deduped.append(r)
    print(f"deduped {len(records)} -> {len(deduped)} polygons by osm_id")
    records = deduped
    raw_counts = Counter(r["class"] for r in records)
    print(f"scanned {len(records)} polygons; raw per-class: {dict(raw_counts)}")

    # ---- Phase B: class-balanced selection (<=1000/class, 25k total cap)
    selected = balance_by_class(records, "class", per_class=PER_CLASS)
    for j, r in enumerate(selected):
        r["sample_id"] = f"{j:06d}"
    sel_counts = Counter(r["class"] for r in selected)
    print(f"selected {len(selected)} tiles; per-class: {dict(sel_counts)}")

    # ---- Phase C: rasterize + write tiles (parallel)
    io.check_disk()
    written: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write tiles",
        ):
            if res is not None:
                written[res] += 1

    n_written = len(list(io.locations_dir(SLUG).glob("*.tif")))
    print("written by class:", dict(written))
    print("total tif on disk:", n_written)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "OpenStreetMap / Geofabrik",
            "license": "ODbL",
            "provenance": {
                "url": "https://download.geofabrik.de/",
                "have_locally": False,
                "annotation_method": "OSM-crowdsourced",
                "regions": REGIONS,
                "layers": LAYERS,
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc, _fc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_written,
            "class_counts": {name: sel_counts.get(name, 0) for name, _d, _f in CLASSES},
            "min_area_m2": MIN_AREA_M2,
            "notes": (
                "Leisure/tourism land-use polygons from OSM Geofabrik regional shapefile "
                "extracts (gis_osm_pois_a_free_1 + gis_osm_landuse_a_free_1), rasterized to "
                "<=64x64 uint8 tiles in local UTM at 10 m. Unified 7-class scheme combining "
                "the manifest's points+polygons classes; outside-polygon = nodata (255), "
                "positive-only (no fabricated negatives). Manifest class 'ski piste' DROPPED "
                "(OSM piste features absent from Geofabrik free shapefiles). Sub-0.25-ha "
                "polygons dropped as unresolvable at 10 m. Bounded global sample of 15 "
                "Geofabrik regions across 6 continents (large regions lack .shp.zip and were "
                "skipped). Static land use -> representative 1-year window (%d)."
                % REP_YEAR
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=n_written
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
