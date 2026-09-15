"""Process Cal-FF (California CAFOs) into animal-type-labeled facility-footprint tiles.

Source: reglab/cal-ff on the Hugging Face Hub (public, CC0-1.0). Cal-FF is a
human-validated, near-complete census of Concentrated Animal Feeding Operations (CAFOs)
in California, compiled with satellite imagery + computer vision + human-in-the-loop
validation (Magesh et al., accepted at Nature Scientific Data, 2025). We download only the
label file ``facilities.geojson`` (2,121 facilities); pretraining supplies its own imagery.

Each facility is a **MultiPolygon building footprint** in WGS84 (lon/lat) with an
``animal_types`` list (e.g. ``["dairy","cattle"]``, ``["poultry"]``) and construction/
destruction date annotations. Georeferencing is exact (WGS84 polygons), so labels place
cleanly on the S2 grid.

Task: **positive-only polygon segmentation** (label_type: polygons), with a **unified
animal-type class scheme** derived from the ``animal_types`` tags:

    0 = cattle         (beef / feedlot cattle CAFO; "cattle" without "dairy")
    1 = poultry        (poultry CAFO)
    2 = dairy_cattle   (dairy operation; any facility tagged "dairy")
    3 = swine          (swine CAFO)
    4 = unknown        (validated CAFO footprint, animal type not determined)
    5 = sheep
    6 = goats
    255 = nodata / ignore (all pixels outside a CAFO footprint)

Per-facility class = priority(dairy > poultry > swine > cattle > sheep > goats > unknown)
over its ``animal_types`` tags (so a "dairy, cattle" facility is dairy_cattle, a plain
"cattle" facility is beef cattle). The CAFO building footprints ARE the "infrastructure
footprints" mentioned in the manifest; there is no separate per-feature infrastructure
attribute in the release, so each footprint is labeled solely by its facility animal type.

Positive-only / no-background (spec section 5): non-footprint pixels are left as nodata
(255); we do NOT fabricate synthetic negatives. The pretraining assembly step supplies
negatives by sampling locations from other datasets.

Rasterization: one <=64x64 UTM 10 m tile centered on each selected facility's centroid.
ALL facility footprints intersecting the tile are rasterized to their own animal-type class
id (all_touched=True so small barns survive); a tile therefore counts toward every class it
contains. Facilities whose footprint exceeds a 640 m tile (~13% of the set, up to ~2.3 km)
are captured as a central all-CAFO window -- still a valid positive patch.

Sampling (spec section 5): tiles-per-class balanced via balance_by_class keyed on the center
facility's class, up to 1000 tiles/class (25k total cap; never approached here). The
dominant "cattle" class (1,578 facilities) is truncated to 1,000; all other classes are kept
in full. Rare classes (goats=1, sheep=3) are retained per spec section 5.

Time range: CAFO buildings/lagoons are persistent structures. Every facility is present in
the 2016-2017 reference imagery the dataset was built from (all destruction dates are 2018+;
construction upper-bounds are almost all <=1998, latest 2017). We anchor a static 1-year
window on 2017 (change_time=null); this is a presence/state label, not a change label.

Run: ``python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.cal_ff_california_cafos``
Idempotent: existing ``locations/{id}.tif`` are skipped.
"""

import argparse
import json
import multiprocessing
from collections import Counter
from typing import Any

import numpy as np
import shapely
import tqdm
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import (
    download,
    io,
    manifest,
    rasterize,
    sampling,
)

SLUG = "cal_ff_california_cafos"
NAME = "Cal-FF (California CAFOs)"
REPO_ID = "reglab/cal-ff"
URL = "https://huggingface.co/datasets/reglab/cal-ff"
GEOJSON = "facilities.geojson"

TILE = io.MAX_TILE  # 64 px @ 10 m = 640 m
PER_CLASS = 1000  # spec section 5 (25k total cap enforced by balance_by_class)
YEAR = 2017  # static 1-year window; all facilities present in 2016-2017 imagery
SEED = 42

# Unified animal-type class scheme (ids 0.. in descending facility frequency).
CLASSES = [
    (
        "cattle",
        "Beef / feedlot cattle concentrated animal feeding operation (CAFO): a facility "
        "tagged 'cattle' but not 'dairy'. Building/pen/corral footprint hand-validated "
        "from satellite imagery (Cal-FF).",
    ),
    (
        "poultry",
        "Poultry CAFO (chicken/turkey/egg operation): long barn/house footprints, "
        "hand-validated from satellite imagery (Cal-FF).",
    ),
    (
        "dairy_cattle",
        "Dairy cattle operation: any facility tagged 'dairy' (typically 'dairy, cattle'). "
        "Barns, freestalls, and often a manure lagoon; footprint hand-validated from "
        "satellite imagery (Cal-FF).",
    ),
    (
        "swine",
        "Swine (hog/pig) CAFO footprint, hand-validated from satellite imagery (Cal-FF).",
    ),
    (
        "unknown",
        "Validated CAFO/animal-feeding-operation footprint whose specific animal type "
        "could not be determined by annotators.",
    ),
    ("sheep", "Sheep feeding-operation footprint (Cal-FF)."),
    ("goats", "Goat feeding-operation footprint (Cal-FF)."),
]
CLASS_ID = {name: i for i, (name, _desc) in enumerate(CLASSES)}


def classify(animal_types: list[str] | None) -> str:
    """Map a facility's animal_types tags to one unified class name (priority order)."""
    t = {str(x).strip().lower() for x in (animal_types or [])}
    if "dairy" in t:
        return "dairy_cattle"
    if "poultry" in t:
        return "poultry"
    if "swine" in t:
        return "swine"
    if "cattle" in t:
        return "cattle"
    if "sheep" in t:
        return "sheep"
    if "goat" in t or "goats" in t:
        return "goats"
    return "unknown"


def _download() -> str:
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    out = download.hf_download(REPO_ID, GEOJSON, raw)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Cal-FF (California CAFOs), reglab/cal-ff on the Hugging Face Hub, CC0-1.0.\n"
            f"{URL}\n"
            "Magesh, Rothbacher, Comess, Maneri, Rodolfa, Tartof, Casey, Nachman, Ho, "
            "'Cal-FF: A Comprehensive Dataset of Factory Farms in California Compiled Using "
            "Computer Vision and Human Validation' (accepted, Nature Scientific Data 2025).\n"
            "Label-only download of facilities.geojson (2,121 CAFO building-footprint "
            "MultiPolygons in WGS84, with animal_types + construction/destruction dates). "
            "Imagery is supplied by pretraining, not downloaded here.\n"
        )
    return str(out)


def _load_facilities() -> list[dict[str, Any]]:
    """Parse facilities.geojson into per-facility records with class + WGS84 geom."""
    path = io.raw_dir(SLUG) / GEOJSON
    fc = json.load(path.open())
    facs: list[dict[str, Any]] = []
    for feat in fc["features"]:
        p = feat["properties"]
        try:
            geom = shapely.geometry.shape(feat["geometry"])
        except Exception:
            continue
        if geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
            if geom.is_empty:
                continue
        cls = classify(p.get("animal_types"))
        # Tile center = a guaranteed-interior representative point of the footprint (the
        # recorded facility lat/lon is occasionally offset a few hundred metres from the
        # digitized geometry, which would center a tile off the footprint). The centroid
        # is used when it falls inside the (multi)polygon, else representative_point().
        c = geom.centroid
        if not geom.contains(c):
            c = geom.representative_point()
        lon, lat = float(c.x), float(c.y)
        facs.append(
            {
                "fid": str(p.get("facility_id") or p.get("id")),
                "lon": lon,
                "lat": lat,
                "cls": cls,
                "class_id": CLASS_ID[cls],
                "geom": geom,
            }
        )
    return facs


def _tile_geoms(
    tree: shapely.STRtree, facs: list[dict[str, Any]], lon: float, lat: float
) -> dict[str, Any]:
    """Build a 64x64 tile centered on (lon, lat); gather intersecting (geom, class_id)."""
    proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    tile_box = shapely.box(*bounds)
    tile_wgs84 = STGeometry(proj, tile_box, None).to_projection(WGS84_PROJECTION).shp
    hits = tree.query(tile_wgs84)
    shapes: list[tuple[bytes, int]] = []
    for i in np.atleast_1d(hits).tolist():
        g = facs[i]["geom"]
        if g.intersects(tile_wgs84):
            shapes.append((shapely.to_wkb(g), facs[i]["class_id"]))
    return {"crs": proj.crs.to_string(), "bounds": list(bounds), "shapes": shapes}


def _write_sample(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return "skip"
    proj = Projection(CRS.from_string(rec["crs"]), io.RESOLUTION, -io.RESOLUTION)
    bounds = tuple(rec["bounds"])
    shapes: list[tuple[Any, int]] = []
    for wkb, class_id in rec["shapes"]:
        g = shapely.from_wkb(wkb)
        gp = rasterize.geom_to_pixels(g, WGS84_PROJECTION, proj)
        if not gp.is_empty:
            shapes.append((gp, class_id))
    if shapes:
        arr = rasterize.rasterize_shapes(
            shapes, bounds, fill=io.CLASS_NODATA, dtype="uint8", all_touched=True
        )
    else:
        w, h = bounds[2] - bounds[0], bounds[3] - bounds[1]
        arr = np.full((1, h, w), io.CLASS_NODATA, dtype=np.uint8)
    present = sorted(int(v) for v in np.unique(arr) if v != io.CLASS_NODATA)
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(YEAR),
        change_time=None,
        source_id=rec["source_id"],
        classes_present=present,
    )
    return "written" if present else "empty"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap number of tiles")
    args = ap.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    _download()
    io.check_disk()

    facs = _load_facilities()
    print(f"loaded {len(facs)} facilities", flush=True)
    avail = Counter(f["cls"] for f in facs)
    print("available per class:", dict(avail), flush=True)

    # Tiles-per-class balanced selection (center facility's class), <=1000/class.
    selected = sampling.balance_by_class(
        facs, key="cls", per_class=PER_CLASS, seed=SEED
    )
    if args.limit > 0:
        selected = selected[: args.limit]
    sel_counts = Counter(f["cls"] for f in selected)
    print(f"selected {len(selected)} facilities:", dict(sel_counts), flush=True)

    geoms = [f["geom"] for f in facs]
    tree = shapely.STRtree(geoms)

    recs: list[dict[str, Any]] = []
    for f in selected:
        t = _tile_geoms(tree, facs, f["lon"], f["lat"])
        t["source_id"] = f"cal_ff/facility_id={f['fid']}/{f['cls']}"
        recs.append(t)
    # Deterministic sample_id ordering independent of selection order.
    recs.sort(key=lambda r: (r["crs"], r["bounds"][0], r["bounds"][1], r["source_id"]))
    for idx, r in enumerate(recs):
        r["sample_id"] = f"{idx:06d}"

    io.check_disk()
    results: Counter = Counter()
    class_tile_counts: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_sample, [dict(rec=r) for r in recs]),
            total=len(recs),
            desc="write tiles",
        ):
            results[res] += 1
    print("write results:", dict(results), flush=True)

    # Class tile counts: a tile counts toward every class present in it.
    for r in recs:
        cids = {cid for _wkb, cid in r["shapes"]}
        for cid in cids:
            class_tile_counts[cid] += 1

    n_written = len(list(io.locations_dir(SLUG).glob("*.tif")))
    io.check_disk()

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Hugging Face (reglab/cal-ff) / Nature Scientific Data",
            "license": "CC0-1.0",
            "provenance": {
                "url": URL,
                "have_locally": False,
                "annotation_method": "computer-vision detection on satellite imagery + "
                "human-in-the-loop validation and animal-type labeling",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_written,
            "class_tile_counts": {
                name: class_tile_counts[i] for i, (name, _d) in enumerate(CLASSES)
            },
            "available_facilities_per_class": dict(avail),
            "tile_size": TILE,
            "window_rule": f"static 1-year window anchored on {YEAR}; change_time=null",
            "notes": (
                "Positive-only animal-type polygon segmentation of California CAFO building "
                "footprints (Cal-FF, 2,121 hand-validated MultiPolygons). 64x64 uint8 tiles "
                "in local UTM at 10 m; class id = facility animal type "
                "(0 cattle / 1 poultry / 2 dairy_cattle / 3 swine / 4 unknown / 5 sheep / "
                "6 goats), 255 = nodata for all non-footprint pixels (NO synthetic "
                "negatives per spec section 5; assembly adds negatives from other datasets). "
                "Per-facility class = priority(dairy>poultry>swine>cattle>sheep>goats>"
                "unknown) over the animal_types tags. One tile per selected facility, "
                "centered on the facility centroid; ALL facility footprints intersecting a "
                "tile are burned in with their own class (all_touched=True), so a tile "
                "counts toward every class present. Facilities larger than a 640 m tile "
                "(~13%, up to ~2.3 km) are captured as a central all-CAFO window. "
                "Tiles-per-class balanced (balance_by_class, key=center-facility class) at "
                f"up to {PER_CLASS}/class: cattle truncated 1578->1000; all other classes "
                "kept in full (rare classes sheep=3, goats=1 retained per spec section 5). "
                f"Time range = static 1-year window anchored on {YEAR} (persistent "
                "structures; all facilities present in the 2016-2017 reference imagery: "
                "destruction dates all 2018+, construction upper-bounds almost all <=1998). "
                "change_time=null (presence/state, not change). The CAFO footprints are the "
                "manifest's 'infrastructure footprints'; no separate per-feature "
                "infrastructure attribute exists in the release."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=n_written
    )
    print(
        "class tile counts:",
        {CLASSES[i][0]: class_tile_counts[i] for i in sorted(class_tile_counts)},
        flush=True,
    )
    print(f"done: {n_written} samples on disk", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
