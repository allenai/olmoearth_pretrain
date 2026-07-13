"""Process FLOGA (Greek wildfire burnt-area mapping) into open-set-segmentation labels.

Source: FLOGA (Sdraka et al. 2024, IEEE JSTARS, doi:10.1109/JSTARS.2024.3381737),
https://github.com/Orion-AI-Lab/FLOGA . FLOGA is an ML-ready Sentinel-2 + MODIS dataset
of Greek wildfire events over 2017-2021 with expert (Hellenic Fire Service) burnt-area
ground truth. The full ML-ready product (v2 GeoTIFFs on HuggingFace
`orion-ai-lab/FLOGA-GeoTIFFs`) is ~130 GB of pre/post imagery + masks — but pretraining
supplies its own imagery, so we only need the *labels*. We therefore use the companion
label-only release `Orion-AI-Lab/FLOGA-annotations` (a few MB of shapefiles): per-event
burnt-area polygons in EPSG:4326 with the wildfire **ignition/end dates** and the pre/post
Sentinel-2 image used.

We use the **v2** polygons (`polygons/v2/fb_{year}_final_images_4326.shp`, 2017-2021),
344 unique wildfire events (deduplicated by `ID`; the multiple rows per ID are alternate
Sentinel-1 pairs sharing one geometry).

Class scheme (dense per-pixel CLASSIFICATION, matching the manifest's 2 classes; ids
follow the fire-dataset convention, cf. cabuar_california_burned_areas):
    id 0 = unburned   (land not inside this event's burnt-area polygon)
    id 1 = burned     (inside the Hellenic Fire Service burnt-area polygon for the event)
    255  = nodata/ignore  (pixels inside a *different* same-year event's burnt-area
           polygon — FLOGA's own "value 2 = burnt in other events"; excluded so we never
           mislabel a neighbouring fire's scar as unburned)

Processing (label_type = dense_raster from polygons): each event polygon is reprojected to
its local UTM zone at 10 m, its bounding box (padded by one tile) is tiled into 64x64
patches, and each patch is rasterized (other same-year events first as 255, then this
event as 1, fill 0). Sampling is tiles-per-class balanced (spec 5): a tile counts toward
every class present, rarer class filled first, up to PER_CLASS tiles/class under the 25k cap.

Time range: burnt area is a change/event label with a **day-precise ignition date**
(`Start date`), so `change_time` is set to the ignition date and `time_range` is a
360-day window centered on it (spec 5). The post-fire Sentinel-2 acquisition (recorded in
the shapefile `S2_e`) lands a few weeks after ignition, well inside the window.

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.floga
"""

import argparse
import multiprocessing
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import shapely
import shapely.ops
import shapely.wkb
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.mp import star_imap_unordered
from shapely.geometry import box
from shapely.validation import make_valid

from olmoearth_pretrain.open_set_segmentation_data import (
    download,
    io,
    manifest,
    rasterize,
)

SLUG = "floga"
NAME = "FLOGA"

RAW = io.raw_dir(SLUG)
YEARS = [2017, 2018, 2019, 2020, 2021]
ANN_REPO = "Orion-AI-Lab/FLOGA-annotations"
ANN_URL = "https://raw.githubusercontent.com/Orion-AI-Lab/FLOGA-annotations/main"
SHP_EXTS = ["shp", "shx", "dbf", "prj", "cpg"]

TILE = 64  # output tile edge (px) at 10 m => 640 m
PAD_TILES = 1  # pad the event bbox by this many tiles for unburned context
PER_CLASS = 1000
MIN_CLASS_PX = 16  # a tile counts toward a class only with >= this many px

UNBURNED, BURNED = 0, 1
CLASSES = [
    (
        "unburned",
        "Land within the wildfire event's footprint that was not burnt: Sentinel-2 pixel "
        "outside the Hellenic Fire Service burnt-area polygon for the event (observed, "
        "non-burnt).",
    ),
    (
        "burned",
        "Wildfire burnt area: pixel inside the expert-annotated (Hellenic Fire Service) "
        "burnt-area polygon for the Greek wildfire event, at the post-fire Sentinel-2 "
        "acquisition.",
    ),
]


# --------------------------------------------------------------------------- raw download
def _ensure_raw() -> None:
    """Download the v2 annotation shapefiles (label-only) into raw_dir; write SOURCE.txt."""
    RAW.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        for ext in SHP_EXTS:
            fname = f"fb_{year}_final_images_4326.{ext}"
            download.download_http(f"{ANN_URL}/polygons/v2/{fname}", RAW / fname)
    (RAW / "SOURCE.txt").write_text(
        "FLOGA burnt-area labels (v2 polygons) from GitHub Orion-AI-Lab/FLOGA-annotations,\n"
        "polygons/v2/fb_{2017..2021}_final_images_4326.shp (EPSG:4326).\n"
        "Fields: ID, Year, 'Start date' (ignition), 'End date', S2_s/S2_e (pre/post S2),\n"
        "MOD_s/MOD_e, S1_* (candidate SAR). One geometry per event ID (rows repeat per S1 pair).\n"
        "The full ML-ready imagery (v2 GeoTIFFs, ~130 GB) at HuggingFace\n"
        "orion-ai-lab/FLOGA-GeoTIFFs is NOT downloaded: pretraining supplies its own imagery,\n"
        "so only the burnt-area label polygons + event dates are needed.\n"
    )


def _load_events() -> list[dict[str, Any]]:
    """Return deduplicated events: {id, year, start(datetime), wkb(4326 geom bytes)}."""
    import geopandas as gpd
    import pandas as pd

    frames = []
    for year in YEARS:
        fp = RAW / f"fb_{year}_final_images_4326.shp"
        frames.append(gpd.read_file(fp.path))
    g = pd.concat(frames, ignore_index=True)
    g = g.drop_duplicates(subset=["ID"]).reset_index(drop=True)

    events: list[dict[str, Any]] = []
    for _, row in g.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            geom = make_valid(geom)
        # keep only polygonal parts
        geom = (
            shapely.ops.unary_union([p for p in _polys(geom)])
            if not geom.is_empty
            else geom
        )
        if geom.is_empty:
            continue
        start = datetime.strptime(str(row["Start date"]), "%Y-%m-%d").replace(
            tzinfo=UTC
        )
        events.append(
            {
                "id": str(row["ID"]),
                "year": int(row["Year"]),
                "start": start,
                "wkb": geom.wkb,
            }
        )
    return events


def _polys(geom: Any) -> list[Any]:
    """Flatten a geometry to its Polygon components."""
    gt = geom.geom_type
    if gt == "Polygon":
        return [geom]
    if gt in ("MultiPolygon", "GeometryCollection"):
        out = []
        for p in geom.geoms:
            out.extend(_polys(p))
        return out
    return []  # lines/points contribute no area


def _event_window(center: datetime) -> tuple[datetime, datetime]:
    """360-day UTC window centered on the ignition date (<= 1 year, spec 3/5)."""
    return (center - timedelta(days=180), center + timedelta(days=180))


def _pixel_geom(wkb: bytes) -> tuple[Projection, Any]:
    """Reproject a WGS84 polygon to local-UTM 10 m *pixel* coordinates."""
    geom = shapely.wkb.loads(wkb)
    c = geom.centroid
    proj = io.utm_projection_for_lonlat(c.x, c.y)
    pg = STGeometry(WGS84_PROJECTION, geom, None).to_projection(proj).shp
    if not pg.is_valid:
        pg = pg.buffer(0)
    return proj, pg


def _tile_grid(pg: Any) -> tuple[int, int, int, int]:
    """Padded, 64-snapped pixel bbox (col0, row0, col1, row1) covering a pixel geom."""
    import math

    minx, miny, maxx, maxy = pg.bounds
    c0 = math.floor(minx / TILE) * TILE - PAD_TILES * TILE
    r0 = math.floor(miny / TILE) * TILE - PAD_TILES * TILE
    c1 = math.ceil(maxx / TILE) * TILE + PAD_TILES * TILE
    r1 = math.ceil(maxy / TILE) * TILE + PAD_TILES * TILE
    return c0, r0, c1, r1


# --------------------------------------------------------------------------- scan phase
def _scan_event(event: dict[str, Any]) -> list[dict[str, Any]]:
    """One candidate record per 64x64 tile of the event's (padded) bbox with a class present.

    Rasterizes this event's polygon once over the full padded bbox, then block-counts to
    determine classes present per tile (fast; no per-tile shapely intersection).
    """
    proj, pg = _pixel_geom(event["wkb"])
    c0, r0, c1, r1 = _tile_grid(pg)
    W, H = c1 - c0, r1 - r0
    if W <= 0 or H <= 0:
        return []
    arr = rasterize.rasterize_shapes([(pg, BURNED)], (c0, r0, c1, r1), fill=UNBURNED)[0]
    epsg = proj.crs.to_epsg()
    recs: list[dict[str, Any]] = []
    total = TILE * TILE
    ncols, nrows = W // TILE, H // TILE
    for ti in range(nrows):
        for tj in range(ncols):
            block = arr[ti * TILE : (ti + 1) * TILE, tj * TILE : (tj + 1) * TILE]
            nb = int((block == BURNED).sum())
            present = []
            if total - nb >= MIN_CLASS_PX:
                present.append(UNBURNED)
            if nb >= MIN_CLASS_PX:
                present.append(BURNED)
            if not present:
                continue
            recs.append(
                {
                    "id": event["id"],
                    "epsg": epsg,
                    "col": c0 + tj * TILE,
                    "row": r0 + ti * TILE,
                    "classes_present": present,
                }
            )
    return recs


# --------------------------------------------------------------------------- write phase
def _write_event(
    event: dict[str, Any],
    others_wkb: list[bytes],
    tiles: list[dict[str, Any]],
) -> None:
    """Rasterize and write all selected tiles of one event (idempotent)."""
    remaining = [
        t
        for t in tiles
        if not (io.locations_dir(SLUG) / f"{t['sample_id']}.tif").exists()
    ]
    if not remaining:
        return
    proj, pg = _pixel_geom(event["wkb"])
    # Reproject same-year neighbouring events into the same pixel space (mask as nodata).
    other_pg = []
    for w in others_wkb:
        geom = shapely.wkb.loads(w)
        og = STGeometry(WGS84_PROJECTION, geom, None).to_projection(proj).shp
        if not og.is_valid:
            og = og.buffer(0)
        if not og.is_empty:
            other_pg.append(og)
    change_time = event["start"]
    tr = _event_window(change_time)
    for t in remaining:
        col, row = t["col"], t["row"]
        bounds = (col, row, col + TILE, row + TILE)
        tile_box = box(*bounds)
        shapes: list[tuple[Any, int]] = []
        for og in other_pg:
            if og.intersects(tile_box):
                shapes.append((og, io.CLASS_NODATA))
        shapes.append((pg, BURNED))
        arr = rasterize.rasterize_shapes(shapes, bounds, fill=UNBURNED)
        io.write_label_geotiff(
            SLUG, t["sample_id"], arr, proj, bounds, nodata=io.CLASS_NODATA
        )
        present = sorted(int(v) for v in np.unique(arr) if v != io.CLASS_NODATA)
        io.write_sample_json(
            SLUG,
            t["sample_id"],
            proj,
            bounds,
            tr,
            change_time=change_time,
            source_id=f"{event['id']}_r{row}_c{col}",
            classes_present=present,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    _ensure_raw()
    events = _load_events()
    print(f"{len(events)} unique wildfire events (v2 polygons, 2017-2021)")

    # Same-year neighbour lookup (bbox-intersection) for nodata masking of other events.
    import shapely as _sh

    by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        e["_geom4326"] = _sh.wkb.loads(e["wkb"])
        by_year[e["year"]].append(e)
    others: dict[str, list[bytes]] = {}
    for year, evs in by_year.items():
        boxes = [_sh.geometry.box(*e["_geom4326"].bounds) for e in evs]
        for i, e in enumerate(evs):
            neigh = [
                evs[j]["wkb"]
                for j in range(len(evs))
                if j != i and boxes[i].intersects(boxes[j])
            ]
            others[e["id"]] = neigh
    for e in events:
        del e["_geom4326"]

    print("Scanning events into 64x64 tiles...")
    scan_args = [{"event": e} for e in events]
    all_recs: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan_event, scan_args), total=len(scan_args)
        ):
            all_recs.extend(recs)
    print(f"  {len(all_recs)} candidate tiles")

    from olmoearth_pretrain.open_set_segmentation_data import sampling

    selected = sampling.select_tiles_per_class(
        all_recs, classes_key="classes_present", per_class=PER_CLASS
    )
    selected.sort(key=lambda r: (r["id"], r["row"], r["col"]))
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(
        f"  selected {len(selected)} tiles (tiles-per-class balanced, <= {PER_CLASS}/class)"
    )

    tile_class_counts = {name: 0 for name, _ in CLASSES}
    for r in selected:
        for c in r["classes_present"]:
            tile_class_counts[CLASSES[c][0]] += 1
    print("candidate-tiles-per-class in selection:", tile_class_counts)

    ev_by_id = {e["id"]: e for e in events}
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in selected:
        by_event[r["id"]].append(r)

    io.check_disk()
    print(f"Writing tiles for {len(by_event)} events...")
    write_args = [
        {"event": ev_by_id[eid], "others_wkb": others.get(eid, []), "tiles": ts}
        for eid, ts in by_event.items()
    ]
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_event, write_args), total=len(write_args)
        ):
            pass

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "FLOGA (GitHub Orion-AI-Lab/FLOGA-annotations, v2 burnt-area polygons)",
            "license": "CC-BY-4.0 / MIT (open, research)",
            "provenance": {
                "url": "https://github.com/Orion-AI-Lab/FLOGA",
                "annotations_url": "https://github.com/Orion-AI-Lab/FLOGA-annotations",
                "have_locally": False,
                "annotation_method": "manual (Hellenic Fire Service experts)",
                "citation": "Sdraka et al. 2024, IEEE JSTARS, doi:10.1109/JSTARS.2024.3381737",
                "files": "polygons/v2/fb_{2017..2021}_final_images_4326.shp",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "tile_class_counts": tile_class_counts,
            "notes": (
                "FLOGA Greek wildfire burnt-area masks (Hellenic Fire Service), from the "
                "label-only v2 annotation polygons (EPSG:4326); the ~130 GB ML-ready "
                "imagery was NOT downloaded (pretraining supplies imagery). 344 unique "
                "events over 2017-2021. Each event polygon is reprojected to local UTM at "
                "10 m and tiled into 64x64 patches (bbox padded 1 tile); rasterized to "
                "0 unburned / 1 burned, with other same-year events' polygons set to 255 "
                "(FLOGA's 'burnt in other events' -> ignore). Tiles-per-class balanced "
                "(<=1000/class), burned filled first. Burn is an event label: "
                "change_time = ignition ('Start date'), time_range = 360-day window "
                "centered on it; the post-fire S2 acquisition falls a few weeks later."
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
