"""Process the ESA WorldCereal Reference Data Module (RDM) into open-set-segmentation labels.

Source: the WorldCereal RDM public OGC-style REST API (no authentication needed for the
public reference collections):

  base:        https://ewoc-rdm-api.iiasa.ac.at
  collections: GET /collections?PageNumber=&PageSize=
  features:    GET /collections/{collectionId}/items?MaxResultCount=&SkipCount=

The RDM is a global online repository of harmonized, curated in-situ crop-type and
land-cover reference datasets. Each feature carries a harmonized WorldCereal legend code
``ewoc_code`` (10-digit hierarchical), an ``irrigation_status``, a single ``valid_time``
date, and land-cover / crop-type quality scores. 260 public collections (130 Point,
130 Polygon), 84M features total -- far more than the 25k cap allows, so we sample a
bounded number of features per collection (PER_COLLECTION_CAP) for geographic + class
diversity, then balance by class.

Harmonized class scheme: we map each ``ewoc_code`` to a class using the official
WorldCereal class-mapping tables (fetched from the worldcereal-classification repo):
  * CROPTYPE24 gives a concrete crop type (maize, wheat, rice, barley, soybean, ...)
    when the code is a recognised crop; else
  * LANDCOVER10 gives the broad land-cover class (grasslands, trees, built_up, water,
    wetlands, shrubland, bare_sparsely_vegetated, permanent_crops, temporary_crops,
    temporary_grasses).
Codes that map to "ignore"/"no_crop"/unknown in both tables are dropped. The
``irrigation_status`` attribute (irrigated vs rainfed) is available in the source but is
NOT used as the primary class (it would multiply the class count); it is preserved in the
per-sample source_id / point rows for downstream use.

Mixed geometry:
  * Point collections -> sparse point segmentation -> one dataset-wide point table
    (points.json, spec 2a): one row per point with label=class_id.
  * Polygon collections -> field parcels rasterized into <=64x64 UTM 10 m tiles
    (spec 2 / 4): polygon interior = class_id, outside-polygon = 255 (nodata) since only
    the labeled parcel's class is known.
Both share ONE unified class map (metadata.json). Selection balances the combined set to
<=1000/class subject to the 25k total cap (balance_by_class lowers the per-class limit to
25000 // n_classes when needed).

Time range: crop/land-cover labels are seasonal/annual, so each sample gets a 1-year
window anchored on the year of its ``valid_time`` (clamped to the Sentinel era, >=2016).

Run:
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.worldcereal_reference_data_module_rdm
"""

import argparse
import json
import math
import multiprocessing
import urllib.request
from collections import Counter
from typing import Any

import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.mp import star_imap_unordered
from shapely.geometry import shape as shapely_shape

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "worldcereal_reference_data_module_rdm"
NAME = "WorldCereal Reference Data Module (RDM)"

API_BASE = "https://ewoc-rdm-api.iiasa.ac.at"
UA = "Mozilla/5.0 (OlmoEarth open-set-segmentation data pipeline; research)"
MAPPINGS_URL = (
    "https://raw.githubusercontent.com/WorldCereal/worldcereal-classification/"
    "main/src/worldcereal/data/croptype_mappings/class_mappings.json"
)

PER_COLLECTION_CAP = (
    2000  # bounded sample per collection (of up to millions of features)
)
PAGE = (
    500  # features per API request (large requests + high concurrency -> server 500s)
)
DOWNLOAD_WORKERS = 16  # keep concurrency modest so the RDM API does not 500
PER_CLASS = 1000
MIN_YEAR = 2016  # clamp valid_time year into the Sentinel era

# Short definitions for the unified (CROPTYPE24 crop + LANDCOVER10 land-cover) classes.
CLASS_DESCRIPTIONS = {
    # crop types (WorldCereal CROPTYPE24 harmonized legend)
    "maize": "Maize / corn (Zea mays), grain and silage.",
    "wheat": "Wheat (Triticum spp.), winter and spring.",
    "rice": "Rice (Oryza sativa), incl. paddy.",
    "barley": "Barley (Hordeum vulgare).",
    "soy_soybeans": "Soybean (Glycine max).",
    "sunflower": "Sunflower (Helianthus annuus).",
    "rapeseed_rape": "Rapeseed / canola (Brassica napus).",
    "fibre_crops": "Fibre crops (cotton, flax, hemp, etc.).",
    "sugar_cane": "Sugar cane (Saccharum officinarum).",
    "potatoes": "Potato (Solanum tuberosum).",
    "vegetables": "Vegetable crops (mixed field / market-garden vegetables).",
    "sorghum": "Sorghum (Sorghum bicolor).",
    "millet": "Millet (various small-grained cereals).",
    "oats": "Oats (Avena sativa).",
    "rye": "Rye (Secale cereale).",
    "triticale": "Triticale (wheat x rye hybrid cereal).",
    "beet": "Beet (sugar beet / fodder beet).",
    "cassava": "Cassava / manioc (Manihot esculenta).",
    "groundnuts": "Groundnut / peanut (Arachis hypogaea).",
    "dry_pulses_legumes": "Dry pulses & grain legumes (beans, peas, lentils, chickpeas).",
    "grass_fodder_crops": "Grass and fodder crops (temporary grass leys, fodder).",
    "other_oilseed": "Other oilseed crops not separately listed.",
    "tobacco": "Tobacco (Nicotiana tabacum).",
    # land-cover fallback (WorldCereal LANDCOVER10)
    "temporary_crops": "Unspecified annual/temporary cropland (crop type not resolved).",
    "temporary_grasses": "Temporary grasses / grass leys in a crop rotation.",
    "permanent_crops": "Permanent crops (orchards, vineyards, fruit & nut trees, plantations).",
    "grasslands": "Natural / semi-natural grassland and herbaceous non-cropland.",
    "trees": "Tree-dominated land cover (forest, woodland).",
    "shrubland": "Shrub-dominated land cover.",
    "built_up": "Built-up / artificial impervious surfaces.",
    "water": "Open water.",
    "wetlands": "Wetland (herbaceous / flooded vegetation).",
    "bare_sparsely_vegetated": "Bare or sparsely vegetated ground.",
}

# Values in the mapping tables that mean "no usable class".
_DROP = {"ignore", "no_crop", "no_temporary_crop", None, ""}


def fetch_json(url: str, timeout: int = 120, retries: int = 4) -> Any:
    import time

    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last  # type: ignore[misc]


def load_mappings() -> dict[str, str]:
    """ewoc_code (str int) -> unified class name (CROPTYPE24 crop, else LANDCOVER10)."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    dst = raw / "class_mappings.json"
    if dst.exists():
        m = json.loads(dst.read_text())
    else:
        m = fetch_json(MAPPINGS_URL)
        dst.write_text(json.dumps(m))
    ct = m["CROPTYPE24"]
    lc = m["LANDCOVER10"]
    codes = set(ct) | set(lc)
    mapping: dict[str, str] = {}
    for code in codes:
        c = ct.get(code)
        if c not in _DROP:
            mapping[code] = c
            continue
        c = lc.get(code)
        if c not in _DROP:
            mapping[code] = c
    return mapping


def fetch_collections() -> list[dict[str, Any]]:
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    dst = raw / "collections_index.json"
    if dst.exists():
        return json.loads(dst.read_text())
    # The /collections endpoint paginates with SkipCount/MaxResultCount (ABP-style),
    # NOT PageNumber/PageSize (those are ignored and re-return the first page).
    items: list[dict[str, Any]] = []
    skip = 0
    while True:
        d = fetch_json(f"{API_BASE}/collections?MaxResultCount=100&SkipCount={skip}")
        its = d.get("items", [])
        if not its:
            break
        items += its
        skip += len(its)
        if skip >= d.get("totalCount", 0):
            break
    # Defensive de-dup by collectionId.
    seen: set[str] = set()
    uniq = []
    for c in items:
        cid = c["collectionId"]
        if cid not in seen:
            seen.add(cid)
            uniq.append(c)
    dst.write_text(json.dumps(uniq))
    return uniq


def _fetch_collection_items(collection_id: str) -> str:
    """Fetch up to PER_COLLECTION_CAP features for one collection; save raw geojson."""
    items_dir = io.raw_dir(SLUG) / "items"
    items_dir.mkdir(parents=True, exist_ok=True)
    dst = items_dir / f"{collection_id}.geojson"
    if dst.exists():
        return collection_id
    feats: list[dict[str, Any]] = []
    skip = 0
    while skip < PER_COLLECTION_CAP:
        want = min(PAGE, PER_COLLECTION_CAP - skip)
        url = (
            f"{API_BASE}/collections/{collection_id}/items"
            f"?MaxResultCount={want}&SkipCount={skip}"
        )
        try:
            d = fetch_json(url)
        except Exception as e:  # noqa: BLE001
            print(f"  WARN fetch failed {collection_id} @skip={skip}: {e}")
            break
        page_feats = d.get("features", [])
        feats += page_feats
        matched = d.get("NumberMatched") or 0
        if len(page_feats) < want or skip + len(page_feats) >= matched:
            break
        skip += len(page_feats)
    tmp = items_dir / f"{collection_id}.geojson.tmp"
    tmp.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    tmp.rename(dst)
    return collection_id


def build_records(
    collections: list[dict[str, Any]], mapping: dict[str, str]
) -> list[dict[str, Any]]:
    items_dir = io.raw_dir(SLUG) / "items"
    recs: list[dict[str, Any]] = []
    for col in collections:
        cid = col["collectionId"]
        path = items_dir / f"{cid}.geojson"
        if not path.exists():
            continue
        try:
            gj = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        for feat in gj.get("features", []):
            geom = feat.get("geometry")
            props = feat.get("properties", {})
            if not geom:
                continue
            cls = mapping.get(str(props.get("ewoc_code")))
            if cls is None:
                continue
            gtype = geom.get("type")
            vt = props.get("valid_time") or ""
            try:
                year = max(MIN_YEAR, int(str(vt)[:4]))
            except (ValueError, TypeError):
                year = 2019
            rec: dict[str, Any] = {
                "class_name": cls,
                "year": year,
                "collection": cid,
                "source_id": f"{cid}/{props.get('sample_id')}",
                "irrigation_status": props.get("irrigation_status"),
            }
            if gtype == "Point":
                lon, lat = geom["coordinates"][0], geom["coordinates"][1]
                rec["kind"] = "point"
                rec["lon"] = float(lon)
                rec["lat"] = float(lat)
            elif gtype in ("Polygon", "MultiPolygon"):
                shp = shapely_shape(geom)
                if shp.is_empty:
                    continue
                c = shp.representative_point()
                rec["kind"] = "polygon"
                rec["lon"] = float(c.x)
                rec["lat"] = float(c.y)
                rec["geometry"] = geom
            else:
                continue
            recs.append(rec)
    return recs


def _write_polygon(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return
    geom = shapely_shape(rec["geometry"])
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    px_geom = geom_to_pixels(geom, WGS84_PROJECTION, proj)
    minx, miny, maxx, maxy = px_geom.bounds
    # Tile sized to the polygon footprint (+2 px pad), capped at 64, centered on centroid.
    col = int(math.floor((minx + maxx) / 2.0))
    row = int(math.floor((miny + maxy) / 2.0))
    tw = min(io.MAX_TILE, max(4, int(math.ceil(maxx - minx)) + 2))
    th = min(io.MAX_TILE, max(4, int(math.ceil(maxy - miny)) + 2))
    bounds = io.centered_bounds(col, row, tw, th)
    arr = rasterize_shapes(
        [(px_geom, rec["class_id"])],
        bounds,
        fill=io.CLASS_NODATA,
        dtype="uint8",
        all_touched=True,
    )
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        source_id=rec["source_id"],
        classes_present=[rec["class_id"]],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "ESA WorldCereal Reference Data Module (RDM) public REST API\n"
            f"collections: {API_BASE}/collections\n"
            f"features:    {API_BASE}/collections/{{collectionId}}/items"
            f"?MaxResultCount={PER_COLLECTION_CAP}&SkipCount=0\n"
            f"class mapping: {MAPPINGS_URL}\n"
            "No authentication required for public collections.\n"
        )

    mapping = load_mappings()
    print(f"class mapping: {len(mapping)} ewoc_codes -> unified classes")

    collections = fetch_collections()
    print(f"public collections: {len(collections)}")

    # Download phase (parallel, idempotent: skips already-downloaded collections).
    ids = [c["collectionId"] for c in collections]
    done = 0
    with multiprocessing.Pool(DOWNLOAD_WORKERS) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(
                p, _fetch_collection_items, [dict(collection_id=i) for i in ids]
            ),
            total=len(ids),
            desc="download",
        ):
            done += 1
            if done % 64 == 0:
                io.check_disk()  # periodic disk guard during downloads

    recs = build_records(collections, mapping)
    print(f"built {len(recs)} labeled records (points + polygons)")
    raw_counts = Counter(r["class_name"] for r in recs)
    print(f"available classes: {len(raw_counts)}")

    selected = balance_by_class(recs, "class_name", per_class=PER_CLASS)
    print(f"selected {len(selected)} records (<= {PER_CLASS}/class, 25k cap)")

    # Assign class ids by descending frequency among selected records.
    sel_counts = Counter(r["class_name"] for r in selected)
    ordered = [name for name, _ in sel_counts.most_common()]
    name_to_id = {name: i for i, name in enumerate(ordered)}
    for r in selected:
        r["class_id"] = name_to_id[r["class_name"]]

    points = [r for r in selected if r["kind"] == "point"]
    polygons = [r for r in selected if r["kind"] == "polygon"]
    print(f"  points: {len(points)}  polygons: {len(polygons)}")

    # Points -> dataset-wide point table (spec 2a).
    point_rows = []
    for i, r in enumerate(points):
        point_rows.append(
            {
                "id": f"p{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": r["class_id"],
                "time_range": io.year_range(r["year"]),
                "source_id": r["source_id"],
            }
        )
    io.write_points_table(SLUG, "classification", point_rows)

    # Polygons -> rasterized <=64x64 tiles (spec 2/4).
    for i, r in enumerate(polygons):
        r["sample_id"] = f"{i:06d}"
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_polygon, [dict(rec=r) for r in polygons]),
            total=len(polygons),
            desc="rasterize",
        ):
            pass

    classes_meta = [
        {
            "id": i,
            "name": name,
            "description": CLASS_DESCRIPTIONS.get(name),
        }
        for i, name in enumerate(ordered)
    ]
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "ESA WorldCereal RDM",
            "license": "mixed (public reference collections; many CC-BY-4.0)",
            "provenance": {
                "url": "https://rdm.esa-worldcereal.org/",
                "api": API_BASE,
                "have_locally": False,
                "annotation_method": "mostly in-situ field survey (harmonized reference)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes_meta,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {name: sel_counts[name] for name in ordered},
            "n_points": len(points),
            "n_polygons": len(polygons),
            "notes": (
                "Harmonized in-situ crop-type / land-cover reference from the ESA "
                "WorldCereal RDM public API (260 public collections, 130 point + 130 "
                f"polygon). Bounded sample of up to {PER_COLLECTION_CAP} features per "
                "collection for geographic+class diversity; balanced to <=1000/class "
                "under the 25k cap. ewoc_code mapped to a unified class via WorldCereal "
                "CROPTYPE24 (crop detail) with LANDCOVER10 fallback (broad land cover); "
                "codes mapping to ignore/no_crop dropped. Point collections -> points.json "
                "(1x1 point segmentation); polygon collections -> rasterized <=64x64 UTM "
                "10 m tiles (parcel interior = class, outside = 255 nodata). "
                "irrigation_status attribute preserved in source but not used as a class. "
                "1-year time window anchored on each feature's valid_time year (>=2016)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
