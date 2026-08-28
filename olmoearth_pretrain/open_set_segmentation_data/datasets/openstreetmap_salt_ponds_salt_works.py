"""Process OpenStreetMap salt ponds / salt works into single-class segmentation tiles.

Source: OpenStreetMap (ODbL), queried live via the Overpass API. We fetch ONLY the salt
features globally by tag -- NOT bulk Geofabrik/planet extracts (a sibling OSM dataset was
rejected for a ~14 GB whole-region download runaway; spec 8 "impractical-download"). The
Overpass query pulls the geometry of every way/relation carrying one of:

    landuse=salt_pond
    man_made=salt_pond | man_made=salt_works | man_made=saltern

which returns ~21k features / ~33 MB -- just the thin label layer, no imagery. (In
practice every matched feature also carries landuse=salt_pond, so this is genuinely one
class.)

Unified single class (spec 5, manifest "salt pond / salt works"):

    0 salt_pond   Human-made solar salt evaporation / production ponds and salt works
                  (OSM landuse=salt_pond, man_made=salt_pond/salt_works/saltern).

These are large, geometric, human-made pond complexes clearly discernible at 10 m
(distinct from natural salars). Ways are closed area polygons; multipolygon relations are
reconstructed from their outer/inner member ways (shapely polygonize).

Positive-only (spec 5): OSM tags presence, not absence -- an untagged pixel is not a
verified negative. Each tile rasterizes its polygon footprint to the class id (0) and
leaves all other pixels as nodata (255); no synthetic background. Assembly adds negatives
from other datasets.

Resolvability (spec 4): polygons smaller than MIN_AREA_M2 (0.25 ha ~ 25 px at 10 m) are
dropped as unresolvable. In practice almost none are removed -- salt ponds are large.

Tiling: each kept polygon -> one tile in a local UTM projection at 10 m/pixel. The tile is
sized to the polygon footprint (padded), capped at 64x64; footprints larger than 640 m
yield a 64x64 window centered on the centroid (a representative chunk of the complex).
all_touched rasterization so thin/small resolvable features still register.

Sampling (spec 5): classification, up to 1000 locations per class. One class -> up to 1000
tiles, drawn (seeded shuffle) from the ~21k global polygons -> naturally globally diverse.

Time range: salt ponds are persistent land use (static). Per spec 5 we assign a
representative 1-year Sentinel-era window (REP_YEAR).

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.openstreetmap_salt_ponds_salt_works
Idempotent: existing locations/{id}.tif are skipped; the raw Overpass response is cached
in raw/{slug}/ and re-used.
"""

import argparse
import json
import math
import multiprocessing
import time
import urllib.parse
import urllib.request
from typing import Any

import numpy as np
import shapely
import shapely.ops
import shapely.wkb
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.mp import star_imap_unordered
from shapely.geometry import LineString, Polygon, box

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "openstreetmap_salt_ponds_salt_works"
NAME = "OpenStreetMap Salt Ponds / Salt Works"

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_QUERY = """[out:json][timeout:540];
(
  way["landuse"="salt_pond"];
  relation["landuse"="salt_pond"];
  way["man_made"="salt_pond"];
  relation["man_made"="salt_pond"];
  way["man_made"="salt_works"];
  relation["man_made"="salt_works"];
  way["man_made"="saltern"];
  relation["man_made"="saltern"];
);
out geom;"""

CLASS_NAME = "salt_pond"
CLASS_DESC = (
    "Human-made solar salt evaporation / production ponds and salt works "
    "(OSM landuse=salt_pond, man_made=salt_pond/salt_works/saltern). Large geometric "
    "pond complexes clearly discernible at 10 m, distinct from natural salars."
)
CLASS_ID = 0

MIN_AREA_M2 = 2500.0  # 0.25 ha (~25 px at 10 m): drop sub-pixel / unresolvable features
EQUAL_AREA_CRS = "EPSG:6933"  # global cylindrical equal-area (metres) for area filter
MIN_TILE = 8
MAX_TILE = io.MAX_TILE  # 64
PAD = 2
PER_CLASS = 1000
REP_YEAR = 2024  # representative Sentinel-era year for these static OSM features


def raw_json_path() -> Any:
    return io.raw_dir(SLUG) / "osm_salt_features.json"


def download() -> None:
    """Fetch salt features from Overpass into raw/{slug}/ (idempotent, atomic)."""
    out = raw_json_path()
    if out.exists():
        print(f"raw already present: {out}")
        return
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    data = urllib.parse.urlencode({"data": OVERPASS_QUERY}).encode()
    req = urllib.request.Request(
        OVERPASS_URL,
        data=data,
        headers={"User-Agent": "olmoearth-dataset/1.0 favyenb@allenai.org"},
    )
    t = time.time()
    with urllib.request.urlopen(req, timeout=540) as r:
        body = r.read()
    print(f"downloaded {len(body)} bytes in {time.time() - t:.1f}s")
    tmp = io.raw_dir(SLUG) / "osm_salt_features.json.tmp"
    with tmp.open("wb") as f:
        f.write(body)
    tmp.rename(out)


def _way_polygon(geometry: list[dict[str, float]]) -> Polygon | None:
    coords = [(p["lon"], p["lat"]) for p in geometry]
    if len(coords) < 3:
        return None
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    try:
        poly = Polygon(coords)
    except Exception:  # noqa: BLE001
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0:
        return None
    return poly


def _relation_polygon(members: list[dict[str, Any]]) -> Any:
    """Reconstruct a (multi)polygon from a multipolygon relation's member ways."""
    outer_lines, inner_lines = [], []
    for m in members:
        if m.get("type") != "way" or "geometry" not in m:
            continue
        coords = [(p["lon"], p["lat"]) for p in m["geometry"]]
        if len(coords) < 2:
            continue
        (inner_lines if m.get("role") == "inner" else outer_lines).append(
            LineString(coords)
        )
    if not outer_lines:
        return None
    try:
        outer = shapely.ops.unary_union(
            list(shapely.ops.polygonize(shapely.ops.unary_union(outer_lines)))
        )
    except Exception:  # noqa: BLE001
        return None
    if outer.is_empty:
        return None
    if inner_lines:
        try:
            inner = shapely.ops.unary_union(
                list(shapely.ops.polygonize(shapely.ops.unary_union(inner_lines)))
            )
            outer = outer.difference(inner)
        except Exception:  # noqa: BLE001
            pass
    if outer.is_empty or outer.area <= 0:
        return None
    return outer


def parse_records() -> list[dict[str, Any]]:
    """Load the Overpass response and build geometry records (WGS84 wkb)."""
    with raw_json_path().open() as f:
        j = json.load(f)
    recs: list[dict[str, Any]] = []
    n_bad = 0
    for e in j["elements"]:
        if e["type"] == "way":
            geom = _way_polygon(e.get("geometry", []))
        elif e["type"] == "relation":
            geom = _relation_polygon(e.get("members", []))
        else:
            geom = None
        if geom is None:
            n_bad += 1
            continue
        recs.append(
            {
                "wkb": shapely.wkb.dumps(geom),
                "class": CLASS_NAME,
                "source_id": f"osm:{e['type']}/{e['id']}",
            }
        )
    print(f"parsed {len(recs)} geometries ({n_bad} skipped as degenerate)")
    return recs


def area_filter(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop polygons smaller than MIN_AREA_M2 (equal-area, vectorized)."""
    import geopandas as gpd

    geoms = [shapely.wkb.loads(r["wkb"]) for r in recs]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326")
    area_m2 = gdf.geometry.to_crs(EQUAL_AREA_CRS).area.values
    kept = [r for r, a in zip(recs, area_m2) if a >= MIN_AREA_M2]
    print(f"area filter (>= {MIN_AREA_M2} m2): {len(recs)} -> {len(kept)}")
    return kept


def _write_one(rec: dict[str, Any]) -> str | None:
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

    clip = px.intersection(box(*bounds))
    if clip.is_empty or clip.area <= 0:
        return None

    label = rasterize_shapes(
        [(clip, CLASS_ID)],
        bounds,
        fill=io.CLASS_NODATA,
        dtype="uint8",
        all_touched=True,
    )[0]
    present = sorted(int(v) for v in np.unique(label) if v != io.CLASS_NODATA)
    if not present:
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

    download()
    records = parse_records()
    if not records:
        raise RuntimeError("no salt-pond geometries parsed -- download failed?")
    records = area_filter(records)

    io.check_disk()
    selected = balance_by_class(records, "class", per_class=PER_CLASS)
    for j, r in enumerate(selected):
        r["sample_id"] = f"{j:06d}"
    print(f"selected {len(selected)} tiles (of {len(records)} candidates)")

    io.check_disk()
    n_ok = 0
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write tiles",
        ):
            if res is not None:
                n_ok += 1

    n_written = len(list(io.locations_dir(SLUG).glob("*.tif")))
    print(f"tiles written (this run, non-skip): {n_ok}; total tif on disk: {n_written}")

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "OpenStreetMap (Overpass API)",
            "license": "ODbL",
            "provenance": {
                "url": "https://wiki.openstreetmap.org/wiki/Tag:landuse=salt_pond",
                "have_locally": False,
                "annotation_method": "OSM manual community mapping",
                "access": "Overpass API tag query (not bulk extracts)",
                "overpass_tags": [
                    "landuse=salt_pond",
                    "man_made=salt_pond",
                    "man_made=salt_works",
                    "man_made=saltern",
                ],
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": CLASS_ID, "name": CLASS_NAME, "description": CLASS_DESC}
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_written,
            "min_area_m2": MIN_AREA_M2,
            "notes": (
                "Global OSM salt ponds / salt works fetched via Overpass API by tag "
                "(~21k features, ~33 MB) -- label-only, no bulk extracts. Ways are closed "
                "polygons; multipolygon relations reconstructed from member ways. Rasterized "
                "to <=64x64 uint8 tiles in local UTM at 10 m; footprints > 640 m yield a "
                "64x64 chunk centered on the centroid. Outside-polygon = nodata (255), "
                "positive-only single class (no fabricated negatives). Sub-0.25-ha polygons "
                "dropped as unresolvable at 10 m. Static land use -> representative 1-year "
                "window (%d). Classification cap: up to %d tiles for the one class."
                % (REP_YEAR, PER_CLASS)
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
