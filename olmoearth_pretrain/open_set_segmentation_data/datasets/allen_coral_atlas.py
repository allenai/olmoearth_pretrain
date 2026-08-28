"""Process the Allen Coral Atlas (ACA) global coral-reef maps into label tiles.

Source: Allen Coral Atlas (Arizona State University / Planet / Univ. of Queensland),
https://allencoralatlas.org/ -- global 5 m maps of shallow tropical coral-reef
*geomorphic zonation* and *benthic habitat*, produced from Planet Dove mosaics
(~2018-2021) trained/validated with extensive field photo-quadrats and contextual
editing. License CC-BY-4.0.

ACCESS (no credential needed): the ACA website's bulk download is behind a free login,
BUT the same vector maps are served openly (CC-BY-4.0, Fees NONE) from the ACA GeoServer
WFS at https://allencoralatlas.org/geoserver/ows with two global polygon layers:
  * coral-atlas:benthic_data_verbose    -- benthic cover polygons (class_name)
  * coral-atlas:geomorphic_data_verbose -- geomorphic zone polygons (class_name)
Both are EPSG:4326 MultiPolygons with attributes {class_name, area_sqkm}. (The GEE mirror
ACA/reef_habitat/v2_0 needs a GEE key, which is not available in this environment; the
open WFS is used instead.) This is a global derived-product, so we do BOUNDED REGIONAL
sampling (spec section 5): we pull WFS polygons for ~21 reef regions spanning the major
reef provinces (Indo-Pacific, Coral Triangle, Red Sea, Persian Gulf, Caribbean/Atlantic,
Central & South Pacific, Indian Ocean), tile each into 64x64 (640 m) UTM patches at 10 m,
and rasterize.

10 m SUITABILITY: the ACA benthic (6) and geomorphic (~11) classes ARE the product's
*top-level* legend -- broad cover / zonation categories that occupy large contiguous
areas of reef, not sub-metre zonation. They are resolvable at 10 m (native product is
5 m; we resample to 10 m by nearest-effect polygon rasterization). We therefore keep the
full top-level legend for both families and do NOT attempt any finer sub-classes.

CLASS SCHEME: benthic and geomorphic are two orthogonal segmentations of the same
pixels, so they cannot share one per-pixel raster. We keep ONE dataset with a unified
legend (17 class ids); each output tile is rasterized from a SINGLE family (benthic OR
geomorphic), so its pixels only carry that family's ids. classes_present in each sample
JSON records which. These are positive-only reef maps: non-reef / unmapped pixels are
left as nodata/ignore (255) (spec section 5 -- assembly supplies negatives from other
datasets); there is no background class.

TIME RANGE: benthic cover / geomorphic zonation are persistent habitat/geological
features; the maps are a 2018-2021 composite. We assign a representative 1-year window
(2020) with change_time=null (static label, spec section 5).

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.allen_coral_atlas
"""

import argparse
import json
import math
import multiprocessing
from collections import Counter
from typing import Any

import numpy as np
import shapely
from pyproj import Transformer
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest
from olmoearth_pretrain.open_set_segmentation_data.rasterize import rasterize_shapes
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    select_tiles_per_class,
)

SLUG = "allen_coral_atlas"
NAME = "Allen Coral Atlas"

WFS_BASE = "https://allencoralatlas.org/geoserver/ows"
BENTHIC_LAYER = "benthic_data_verbose"
GEOMORPHIC_LAYER = "geomorphic_data_verbose"

TILE = io.MAX_TILE  # 64 px @ 10 m = 640 m
NODATA = io.CLASS_NODATA  # 255; positive-only reef maps -> outside polygons = ignore
PER_CLASS = 1000
YEAR = 2020  # representative 1-year window within the 2018-2021 mapping period

# Unified legend: benthic ids 0-5, geomorphic ids 6-16. Source class_name -> id.
BENTHIC_MAP = {
    "Coral/Algae": 0,
    "Seagrass": 1,
    "Sand": 2,
    "Rubble": 3,
    "Rock": 4,
    "Microalgal Mats": 5,
}
GEOMORPHIC_MAP = {
    "Reef Slope": 6,
    "Sheltered Reef Slope": 7,
    "Reef Crest": 8,
    "Outer Reef Flat": 9,
    "Inner Reef Flat": 10,
    "Terrestrial Reef Flat": 11,
    "Back Reef Slope": 12,
    "Plateau": 13,
    "Patch Reef": 14,
    "Deep Lagoon": 15,
    "Shallow Lagoon": 16,
}

CLASS_DESCRIPTIONS = {
    0: "Coral/Algae: hard substrate dominated by living coral and/or algae (turf, macro-, "
    "coralline) -- ACA benthic class.",
    1: "Seagrass: benthos dominated by seagrass beds (soft sediment with rooted marine "
    "angiosperms) -- ACA benthic class.",
    2: "Sand: soft, unconsolidated fine sediment with sparse/no cover -- ACA benthic class.",
    3: "Rubble: loose fragments of dead coral / coarse consolidated debris -- ACA benthic class.",
    4: "Rock: exposed consolidated hard substrate / bare rock (little living cover) -- ACA "
    "benthic class.",
    5: "Microalgal Mats: benthos covered by microalgal / cyanobacterial mats (common in "
    "turbid, high-nutrient shallows, e.g. Persian Gulf) -- ACA benthic class.",
    6: "Reef Slope: seaward-facing reef flank descending from the crest into deeper water "
    "-- ACA geomorphic zone.",
    7: "Sheltered Reef Slope: reef slope on a leeward / protected aspect with lower wave "
    "energy -- ACA geomorphic zone.",
    8: "Reef Crest: shallow, wave-breaking outermost ridge of the reef -- ACA geomorphic zone.",
    9: "Outer Reef Flat: seaward part of the horizontal reef-flat platform behind the crest "
    "-- ACA geomorphic zone.",
    10: "Inner Reef Flat: landward part of the reef-flat platform -- ACA geomorphic zone.",
    11: "Terrestrial Reef Flat: reef flat adjoining land / intertidal reef fringing an island "
    "-- ACA geomorphic zone.",
    12: "Back Reef Slope: landward-facing slope behind the crest descending into a lagoon "
    "-- ACA geomorphic zone.",
    13: "Plateau: elevated flat-topped submerged reef structure -- ACA geomorphic zone.",
    14: "Patch Reef: small isolated reef body, typically within a lagoon -- ACA geomorphic zone.",
    15: "Deep Lagoon: deeper enclosed/semi-enclosed basin water behind the reef -- ACA "
    "geomorphic zone.",
    16: "Shallow Lagoon: shallow enclosed/semi-enclosed basin behind the reef -- ACA "
    "geomorphic zone.",
}

# Bounded regional sampling boxes: (name, min_lon, min_lat, max_lon, max_lat). Chosen to
# span the major reef provinces and habitat types; each verified non-empty against the WFS.
REGIONS = [
    ("gbr_lizard", 145.35, -14.80, 145.55, -14.60),
    ("gbr_cairns", 145.90, -16.90, 146.15, -16.65),
    ("gbr_capricorn", 151.85, -23.50, 152.10, -23.30),
    ("maldives_male", 73.30, 4.05, 73.60, 4.35),
    ("redsea_farasan", 41.60, 16.60, 41.90, 16.90),
    ("redsea_egypt", 34.15, 27.75, 34.45, 28.05),
    ("persian_gulf_qatar", 51.40, 25.85, 51.70, 26.15),
    ("belize_barrier", -88.20, 17.10, -87.90, 17.45),
    ("bahamas_exuma", -76.55, 23.45, -76.25, 23.75),
    ("florida_keys", -80.35, 24.95, -80.05, 25.25),
    ("hawaii_kaneohe", -157.85, 21.40, -157.65, 21.55),
    ("moorea", -149.95, -17.60, -149.70, -17.40),
    ("tuamotu_rangiroa", -147.85, -15.25, -147.55, -14.95),
    ("new_caledonia", 166.20, -22.55, 166.55, -22.25),
    ("new_caledonia_lagoon", 164.20, -20.10, 164.55, -19.80),
    ("fiji_suva", 178.35, -18.25, 178.65, -17.95),
    ("philippines_palawan", 119.90, 11.40, 120.20, 11.70),
    ("indonesia_wakatobi", 123.45, -5.65, 123.75, -5.35),
    ("seychelles_mahe", 55.35, -4.75, 55.65, -4.45),
    ("gulf_mannar", 79.05, 9.15, 79.35, 9.45),
    ("zanzibar", 39.25, -6.35, 39.55, -6.05),
]

LAYERS = {BENTHIC_LAYER: BENTHIC_MAP, GEOMORPHIC_LAYER: GEOMORPHIC_MAP}


def _wfs_url(layer: str, bbox: tuple[float, float, float, float]) -> str:
    minx, miny, maxx, maxy = bbox
    return (
        f"{WFS_BASE}?service=wfs&version=2.0.0&request=GetFeature"
        f"&typeNames=coral-atlas:{layer}&outputFormat=application/json"
        f"&srsName=EPSG:4326&count=1000000"
        f"&bbox={minx},{miny},{maxx},{maxy},EPSG:4326"
    )


def _download_all() -> None:
    """Download benthic + geomorphic WFS GeoJSON per region to raw/ (atomic, idempotent)."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    for region in REGIONS:
        name, minx, miny, maxx, maxy = region
        for layer in LAYERS:
            dst = raw / f"{name}__{layer}.geojson"
            if dst.exists():
                continue
            url = _wfs_url(layer, (minx, miny, maxx, maxy))
            print(f"downloading {name} {layer} ...", flush=True)
            download.download_http(
                url,
                dst,
                headers={"User-Agent": "Mozilla/5.0 (olmoearth-pretrain)"},
                timeout=600,
            )
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Allen Coral Atlas (ASU / Planet / UQ), https://allencoralatlas.org/ .\n"
            "Open WFS: https://allencoralatlas.org/geoserver/ows (CC-BY-4.0, Fees NONE).\n"
            "Layers coral-atlas:benthic_data_verbose + coral-atlas:geomorphic_data_verbose\n"
            "(EPSG:4326 MultiPolygons, attrs class_name/area_sqkm). Bounded regional sample\n"
            f"of {len(REGIONS)} reef regions; count=1000000 per region-layer bbox request.\n"
        )


def _reproject_to_pixels(geom: Any, transformer: Transformer) -> Any:
    """Reproject a WGS84 shapely geom into UTM 10 m *pixel* coords (px=utm_x/10, py=utm_y/-10)."""

    def fn(coords: np.ndarray) -> np.ndarray:
        x, y = transformer.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([np.asarray(x) / 10.0, np.asarray(y) / -10.0])

    return shapely.transform(geom, fn)


def _build_region_tiles(region: tuple, layer: str) -> list[dict[str, Any]]:
    """Load one region-layer GeoJSON, reproject to UTM pixels, group into 64x64 tiles.

    Returns candidate tile records: {epsg, bounds, layer, region, source_id,
    classes_present, shapes} where shapes is a list of (class_id, wkb_bytes) in pixel space.
    """
    name, minx, miny, maxx, maxy = region
    cmap = LAYERS[layer]
    path = io.raw_dir(SLUG) / f"{name}__{layer}.geojson"
    with path.open() as f:
        data = json.load(f)
    feats = data.get("features", [])
    if not feats:
        return []

    clon, clat = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    proj = io.utm_projection_for_lonlat(clon, clat)
    epsg = int(proj.crs.to_epsg())
    transformer = Transformer.from_crs(
        "EPSG:4326", proj.crs.to_string(), always_xy=True
    )

    # cell (cx,cy) -> list of (class_id, pixel_geom)
    cells: dict[tuple[int, int], list[tuple[int, Any]]] = {}
    for feat in feats:
        cn = feat["properties"].get("class_name")
        cid = cmap.get(cn)
        if cid is None:
            continue
        geom = feat.get("geometry")
        if not geom:
            continue
        try:
            g = shapely.geometry.shape(geom)
        except Exception:
            continue
        if g.is_empty:
            continue
        if not g.is_valid:
            g = g.buffer(0)
            if g.is_empty:
                continue
        pg = _reproject_to_pixels(g, transformer)
        if pg.is_empty:
            continue
        gminx, gminy, gmaxx, gmaxy = pg.bounds
        cx0, cx1 = math.floor(gminx / TILE), math.floor((gmaxx - 1e-9) / TILE)
        cy0, cy1 = math.floor(gminy / TILE), math.floor((gmaxy - 1e-9) / TILE)
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                box = shapely.box(
                    cx * TILE, cy * TILE, cx * TILE + TILE, cy * TILE + TILE
                )
                if not pg.intersects(box):
                    continue
                # Clip to the tile so we don't store/ship a giant polygon per cell it spans.
                clip = pg.intersection(box)
                if clip.is_empty:
                    continue
                cells.setdefault((cx, cy), []).append((cid, clip))

    records: list[dict[str, Any]] = []
    for (cx, cy), items in cells.items():
        shapes = [(cid, shapely.to_wkb(pg)) for cid, pg in items]
        present = sorted({cid for cid, _ in items})
        records.append(
            {
                "epsg": epsg,
                "bounds": [cx * TILE, cy * TILE, cx * TILE + TILE, cy * TILE + TILE],
                "layer": "benthic" if layer == BENTHIC_LAYER else "geomorphic",
                "region": name,
                "source_id": f"{name}:{layer}:{cx}:{cy}",
                "classes_present": present,
                "shapes": shapes,
            }
        )
    return records


def _write_tile(rec: dict[str, Any]) -> str | None:
    """Rasterize one candidate tile (fill=nodata; positive-only) and write tif + json."""
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        # Idempotent skip: recover the actual classes from the written sidecar so a re-run
        # still tallies correct metadata counts.
        jpath = io.locations_dir(SLUG) / f"{sample_id}.json"
        try:
            with jpath.open() as f:
                present = json.load(f).get("classes_present", [])
            return "|".join(str(int(c)) for c in present)
        except Exception:
            return ""
    proj = Projection(CRS.from_epsg(rec["epsg"]), io.RESOLUTION, -io.RESOLUTION)
    bounds = tuple(rec["bounds"])
    # Rasterize larger polygons first so tiny classes drawn last win overlaps (rare).
    shapes = [(shapely.from_wkb(wkb), cid) for cid, wkb in rec["shapes"]]
    shapes.sort(key=lambda s: s[0].area, reverse=True)
    raster_in = [(g, cid) for g, cid in shapes]
    arr = rasterize_shapes(
        raster_in, bounds, fill=NODATA, dtype="uint8", all_touched=True
    )[0]
    present = sorted(int(v) for v in np.unique(arr) if v != NODATA)
    if not present:
        return ""  # nothing landed (all polygons clipped out); skip
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(YEAR),
        source_id=rec["source_id"],
        classes_present=present,
    )
    return "|".join(str(c) for c in present)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    # 1. Download bounded regional WFS GeoJSON.
    _download_all()
    io.check_disk()

    # 2. Build candidate tiles per (region, layer) in parallel.
    jobs = [dict(region=r, layer=layer) for r in REGIONS for layer in LAYERS]
    candidates: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for recs in star_imap_unordered(p, _build_region_tiles, jobs):
            candidates.extend(recs)
    print(f"candidate tiles: {len(candidates)}", flush=True)
    cand_cls: Counter = Counter()
    for r in candidates:
        for c in r["classes_present"]:
            cand_cls[c] += 1
    print(f"candidate class->tile counts: {dict(sorted(cand_cls.items()))}", flush=True)

    # 3. Tiles-per-class balanced selection (rarest first), <=1000/class, <=25k total.
    selected = select_tiles_per_class(
        candidates, classes_key="classes_present", per_class=PER_CLASS
    )
    # Deterministic sample-id order.
    selected.sort(key=lambda r: (r["layer"], r["region"], tuple(r["bounds"])))
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected tiles: {len(selected)}", flush=True)

    # 4. Rasterize + write selected tiles in parallel.
    io.check_disk()
    class_counts: Counter = Counter()
    layer_counts: Counter = Counter()
    written = 0
    with multiprocessing.Pool(args.workers) as p:
        for present in star_imap_unordered(
            p, _write_tile, [dict(rec=r) for r in selected]
        ):
            if present:
                written += 1
                for c in present.split("|"):
                    class_counts[int(c)] += 1
    for r in selected:
        layer_counts[r["layer"]] += 1

    # 5. Metadata.
    id_to_name = {v: k for k, v in {**BENTHIC_MAP, **GEOMORPHIC_MAP}.items()}
    classes = [
        {"id": cid, "name": id_to_name[cid], "description": CLASS_DESCRIPTIONS[cid]}
        for cid in range(len(id_to_name))
    ]
    region_counts = dict(sorted(Counter(r["region"] for r in selected).items()))
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Allen Coral Atlas (ASU / Planet / Univ. of Queensland)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://allencoralatlas.org/",
                "have_locally": False,
                "annotation_method": (
                    "Planet Dove mosaic classification (2018-2021) trained/validated with "
                    "field photo-quadrats + contextual editing; open WFS "
                    "(coral-atlas:benthic_data_verbose / geomorphic_data_verbose)"
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes,
            "nodata_value": NODATA,
            "num_samples": written,
            "class_tile_counts": {
                id_to_name[c]: class_counts[c] for c in sorted(class_counts)
            },
            "layer_tile_counts": dict(layer_counts),
            "region_tile_counts": region_counts,
            "tile_size": TILE,
            "time_range": [f"{YEAR}-01-01", f"{YEAR + 1}-01-01"],
            "notes": (
                "Global ACA reef maps via open GeoServer WFS (CC-BY-4.0); bounded regional "
                f"sample of {len(REGIONS)} reef regions across all major reef provinces. "
                "Unified legend: benthic cover ids 0-5, geomorphic zones ids 6-16. Benthic "
                "and geomorphic are orthogonal segmentations of the same pixels, so each "
                "tile is rasterized from ONE family only (classes_present says which); they "
                "share one dataset legend. Positive-only reef maps: non-reef/unmapped pixels "
                "= nodata 255 (no background class; assembly supplies negatives). Kept the "
                "product's full top-level legend (resolvable at 10 m); no finer sub-classes "
                "attempted. Rare classes (Microalgal Mats, Patch Reef, Terrestrial Reef "
                "Flat) may fall short of 1000 tiles. Time range = static 1-year window "
                f"({YEAR}); maps are a 2018-2021 persistent-habitat composite."
            ),
        },
    )
    print(f"written tiles: {written}", flush=True)
    print(f"class->tile counts: {dict(sorted(class_counts.items()))}", flush=True)
    print(f"layer counts: {dict(layer_counts)}", flush=True)
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=written
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
