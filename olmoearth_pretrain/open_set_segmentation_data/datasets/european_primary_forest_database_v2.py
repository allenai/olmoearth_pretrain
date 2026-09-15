"""Process the European Primary Forest Database (EPFD) v2.0 into primary/old-growth
forest label patches (positive-only segmentation), classed by EEA European Forest Type.

Source: Sabatini, F.M., Bluhm, H., Kun, Z. et al. "European primary forest database
v2.0." Scientific Data 8, 220 (2021). https://doi.org/10.1038/s41597-021-00988-7. Data
openly available (CC-BY-4.0) on Figshare (https://doi.org/10.6084/m9.figshare.13194095):
one 112 MB zip ``EPFDv2.0_DatabaseOA.zip`` holding an ESRI *personal* geodatabase
``EPFD_v2.0.mdb``. The harmonized open-access feature classes are
``EU_PrimaryForests_Polygons_OA_v20`` (18,411 polygon patches) and
``EU_PrimaryForests_Points_OA_v20`` (299 point locations, "approximate centre" of a patch
with no digitized boundary). The DB harmonizes 48 regional-to-continental primary/
old-growth forest datasets across ~35 European countries.

Reading the .mdb: GDAL here lacks the PGeo driver, so we read the tables with the
``mdbtools`` ``mdb-json`` CLI (base64-encodes the binary ESRI SHAPE column) and decode the
ESRI shape-binary geometry ourselves (Point=type 1, Polygon=type 5; all WGS84 EPSG:4326).
The parse is materialized ONCE into a reproducible GeoPackage
(``raw/{slug}/parsed/epfd_oa.gpkg``, layers ``polygons``/``points``); subsequent runs read
that GPKG and no longer need mdbtools.

Class scheme (label_type points/polygons -> single foreground family with forest-type
sub-classes): we use the DB's ``FOREST_TYPE1`` attribute = the EEA European Forest Type
(EEA Technical Report 9/2006), derived from the map of Potential Vegetation types for
Europe. Codes 1-13 are the 13 named categories (plantations excluded upstream); code 0 =
no EEA type assigned. We keep the code as the class id (0..13, contiguous), so the 14
classes span the broadleaf (beech/oak: 4,5,6,7,8,9,12), coniferous (boreal/alpine/
Mediterranean: 1,3,10) and azonal/mixed (2,11,13) groups. Every one is a *primary/
old-growth* forest of that type. Positive-only: outside a patch = 255 nodata (per spec 5,
no synthetic negatives; assembly supplies negatives from other datasets).

Representation:
- Polygons -> one <=64x64 UTM 10 m tile per polygon, centered on the polygon's interior
  representative point, the polygon (with holes) rasterized to its FOREST_TYPE1 class id
  (all_touched=True so tiny patches survive), rest = 255. Large patches (>640 m) are
  captured as a central all-forest window.
- Points (approximate patch centres, no footprint) -> a small uniform-class tile sized
  from FOREST_EXTENT_MEASURED (ha) when available (side px = sqrt(area)/10 m, clamped
  [3,32]; default 8 when unknown), the whole tile = the FOREST_TYPE1 class id. Points
  without a FOREST_TYPE1 get class 0.

Sampling: tiles-per-class balanced, up to 1000 tiles/class, 25k hard cap
(balance_by_class). Rare EEA types (broadleaved evergreen, Mediterranean conifer) are kept
in full per spec 5.

Time range: primary/old-growth forest is a persistent, static land cover; the DB was
compiled for v2.0 in 2020 and its primary status verified with Landsat time series through
2018. Each sample gets the static 1-year Sentinel-era window 2020 (within the manifest's
2016-2021 range). change_time = None.

Run: ``python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.european_primary_forest_database_v2``
Idempotent: the GPKG intermediate and existing ``locations/{id}.tif`` are skipped.
"""

import argparse
import base64
import json
import math
import multiprocessing
import os
import struct
import subprocess
from collections import Counter
from typing import Any

import numpy as np
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest, sampling
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)

SLUG = "european_primary_forest_database_v2"
NAME = "European Primary Forest Database v2"
RAW = str(io.raw_dir(SLUG))
MDB = os.path.join(RAW, "extracted", "EPFD_v2.0.mdb")
GPKG = os.path.join(RAW, "parsed", "epfd_oa.gpkg")

POLY_LAYER = "EU_PrimaryForests_Polygons_OA_v20"
POINT_LAYER = "EU_PrimaryForests_Points_OA_v20"

TILE = 64
PER_CLASS = 1000
YEAR = (
    2020  # static persistent land cover; v2.0 compiled 2020, status verified thru 2018
)
POINT_DEFAULT_PX = 8  # default point patch side when no measured extent
POINT_MIN_PX = 3
POINT_MAX_PX = 32

# EEA European Forest Type categories (EEA Tech. Report 9/2006), as used by EPFD
# FOREST_TYPE1. Code 0 = no EEA type assigned. Codes 1-13 = the 13 named categories.
CLASSES = [
    (
        0,
        "Unclassified forest type",
        "Primary/old-growth forest patch with no EEA European Forest Type assigned by the "
        "potential-vegetation cross-link (residual/other).",
    ),
    (
        1,
        "Boreal forest",
        "EEA European Forest Type 1: boreal forest (Scots pine / Norway spruce / birch), "
        "Fennoscandia and NW Russia.",
    ),
    (
        2,
        "Hemiboreal & nemoral coniferous / mixed forest",
        "EEA type 2: hemiboreal forest and nemoral coniferous and mixed broadleaved-"
        "coniferous forest.",
    ),
    (
        3,
        "Alpine coniferous forest",
        "EEA type 3: alpine (montane/subalpine) coniferous forest, dominated by Norway "
        "spruce / silver fir / larch / stone pine.",
    ),
    (
        4,
        "Acidophilous oak & oak-birch forest",
        "EEA type 4: acidophilous oakwood and oak-birch forest.",
    ),
    (
        5,
        "Mesophytic deciduous forest",
        "EEA type 5: mesophytic deciduous forest (mixed broadleaved on richer soils).",
    ),
    (
        6,
        "Lowland-submontane beech forest",
        "EEA type 6: lowland to submountainous beech forest (Fagus sylvatica).",
    ),
    (
        7,
        "Mountainous beech forest",
        "EEA type 7: mountainous beech forest (Fagus sylvatica, often with silver fir).",
    ),
    (
        8,
        "Thermophilous deciduous forest",
        "EEA type 8: thermophilous deciduous forest (downy/Turkey oak, hornbeam, chestnut).",
    ),
    (
        9,
        "Broadleaved evergreen forest",
        "EEA type 9: broadleaved evergreen (Mediterranean sclerophyllous) forest — cork/holm "
        "oak, laurel.",
    ),
    (
        10,
        "Mediterranean/Anatolian/Macaronesian coniferous forest",
        "EEA type 10: coniferous forests of the Mediterranean, Anatolian and Macaronesian "
        "regions (black/Bosnian/Macedonian pine, junipers).",
    ),
    (
        11,
        "Mire & swamp forest",
        "EEA type 11: mire and swamp forest (wet coniferous/broadleaved on peat).",
    ),
    (
        12,
        "Floodplain forest",
        "EEA type 12: floodplain (riparian) forest — willow/poplar/alder/ash-elm-oak.",
    ),
    (
        13,
        "Non-riverine alder, birch or aspen forest",
        "EEA type 13: non-riverine alder, birch or aspen forest (azonal pioneer stands).",
    ),
]
CLASS_IDS = {c for c, _n, _d in CLASSES}


# --------------------------------------------------------------------------- mdb parse


def _decode_point(b: bytes):
    import shapely

    (x, y) = struct.unpack("<dd", b[4:20])
    return shapely.Point(x, y)


def _decode_polygon(b: bytes):
    """Decode an ESRI shape-binary Polygon (type 5) into a shapely (Multi)Polygon.

    Rings are separated by orientation (ESRI: clockwise=exterior, ccw=hole); holes are
    assigned to their containing exterior. Returns None for degenerate geometry.
    """
    import shapely
    from shapely.geometry import LinearRing, MultiPolygon, Polygon

    nparts, npts = struct.unpack("<ii", b[36:44])
    off = 44
    parts = struct.unpack(f"<{nparts}i", b[off : off + 4 * nparts])
    off += 4 * nparts
    coords = struct.unpack(f"<{2 * npts}d", b[off : off + 16 * npts])
    bounds = list(parts) + [npts]
    exteriors: list[list] = []
    holes: list[list] = []
    for i in range(nparts):
        s, e = bounds[i], bounds[i + 1]
        ring = [(coords[2 * j], coords[2 * j + 1]) for j in range(s, e)]
        if len(ring) < 4:
            continue
        try:
            ccw = LinearRing(ring).is_ccw
        except Exception:
            continue
        (holes if ccw else exteriors).append(ring)
    if not exteriors:
        exteriors = [
            [
                (coords[2 * j], coords[2 * j + 1])
                for j in range(bounds[i], bounds[i + 1])
            ]
            for i in range(nparts)
            if bounds[i + 1] - bounds[i] >= 4
        ]
        holes = []
    if not exteriors:
        return None
    ext_polys = [Polygon(e) for e in exteriors]
    assigned: list[list] = [[] for _ in ext_polys]
    if holes:
        for h in holes:
            try:
                rep = Polygon(h).representative_point()
            except Exception:
                continue
            for i, ep in enumerate(ext_polys):
                try:
                    if ep.contains(rep):
                        assigned[i].append(h)
                        break
                except Exception:
                    continue
    polys = [Polygon(exteriors[i], assigned[i]) for i in range(len(ext_polys))]
    geom = polys[0] if len(polys) == 1 else MultiPolygon(polys)
    if not geom.is_valid:
        geom = shapely.make_valid(geom)
    return geom


def _iter_mdb(table: str):
    """Yield parsed dict rows (attrs + shapely geometry under 'geometry') from a table."""
    proc = subprocess.Popen(["mdb-json", MDB, table], stdout=subprocess.PIPE)
    assert proc.stdout is not None
    for line in proc.stdout:
        try:
            row = json.loads(line)
        except Exception:
            continue
        sh = row.get("SHAPE")
        if not isinstance(sh, dict) or "$binary" not in sh:
            continue
        b = base64.b64decode(sh["$binary"])
        st = struct.unpack("<i", b[:4])[0]
        if st == 1:
            geom = _decode_point(b)
        elif st == 5:
            geom = _decode_polygon(b)
        else:
            continue
        if geom is None or geom.is_empty:
            continue
        row["geometry"] = geom
        yield row
    proc.wait()


def build_gpkg() -> None:
    """Parse the two OA feature classes from the .mdb into a GeoPackage (idempotent)."""
    if os.path.exists(GPKG):
        print(f"parsed GPKG already exists: {GPKG}")
        return
    if subprocess.run(["which", "mdb-json"], capture_output=True).returncode != 0:
        raise RuntimeError(
            "mdb-json (mdbtools) not found and no parsed GPKG present. Install with "
            "`sudo apt-get install -y mdbtools` to (re)build raw/parsed/epfd_oa.gpkg."
        )
    import geopandas as gpd

    os.makedirs(os.path.dirname(GPKG), exist_ok=True)
    tmp = GPKG + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)

    # Polygons
    prows = list(_iter_mdb(POLY_LAYER))
    print(f"parsed {len(prows)} polygons")
    gpoly = gpd.GeoDataFrame(
        {
            "objectid": [r.get("OBJECTID") for r in prows],
            "forest_type1": [r.get("FOREST_TYPE1") for r in prows],
            "forest_type2": [r.get("FOREST_TYPE2") for r in prows],
            "id_dataset": [r.get("ID_Dataset") for r in prows],
            "area_ha": [r.get("Area_ha") for r in prows],
            "geometry": [r["geometry"] for r in prows],
        },
        crs="EPSG:4326",
    )
    gpoly.to_file(tmp, layer="polygons", driver="GPKG")

    # Points
    qrows = list(_iter_mdb(POINT_LAYER))
    print(f"parsed {len(qrows)} points")
    gpt = gpd.GeoDataFrame(
        {
            "objectid": [r.get("OBJECTID") for r in qrows],
            "forest_type1": [r.get("FOREST_TYPE1") for r in qrows],
            "extent_measured": [r.get("FOREST_EXTENT_MEASURED") for r in qrows],
            "id_dataset": [r.get("ID_Dataset") for r in qrows],
            "geometry": [r["geometry"] for r in qrows],
        },
        crs="EPSG:4326",
    )
    gpt.to_file(tmp, layer="points", driver="GPKG")
    os.rename(tmp, GPKG)
    print(f"wrote {GPKG}")


# --------------------------------------------------------------------------- write


def _class_id(ft: Any) -> int:
    """Map a FOREST_TYPE1 value to a class id (0..13); None/unknown -> 0."""
    try:
        v = int(ft)
    except (TypeError, ValueError):
        return 0
    return v if v in CLASS_IDS else 0


def _point_size_px(extent_ha: Any) -> int:
    try:
        ha = float(extent_ha)
    except (TypeError, ValueError):
        ha = 0.0
    if ha and ha > 0:
        side = int(round(math.sqrt(ha * 10000.0) / io.RESOLUTION))
        return max(POINT_MIN_PX, min(POINT_MAX_PX, side))
    return POINT_DEFAULT_PX


def _write_one(rec: dict[str, Any]) -> str | None:
    import shapely
    from shapely.geometry import box

    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return "skip"

    cid = rec["class_id"]
    lon, lat = rec["lon"], rec["lat"]

    if rec["kind"] == "point":
        size = rec["size"]
        proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
        bounds = io.centered_bounds(col, row, size, size)
        label = np.full((size, size), cid, dtype=np.uint8)
    else:
        proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
        bounds = io.centered_bounds(col, row, TILE, TILE)
        geom = shapely.from_wkb(rec["wkb"])
        # Clip to a generous lon/lat window around the tile for speed on huge polygons.
        mlat = (2 * TILE * io.RESOLUTION) / 111320.0
        mlon = mlat / max(math.cos(math.radians(lat)), 0.1)
        ll_box = box(lon - mlon, lat - mlat, lon + mlon, lat + mlat)
        try:
            geom = geom.intersection(ll_box)
        except Exception:
            pass
        if geom.is_empty:
            label = np.full((TILE, TILE), io.CLASS_NODATA, dtype=np.uint8)
        else:
            px = geom_to_pixels(geom, WGS84_PROJECTION, proj)
            if px.is_empty:
                label = np.full((TILE, TILE), io.CLASS_NODATA, dtype=np.uint8)
            else:
                label = rasterize_shapes(
                    [(px, cid)],
                    bounds,
                    fill=io.CLASS_NODATA,
                    dtype="uint8",
                    all_touched=True,
                )[0]

    present = sorted(int(v) for v in np.unique(label) if v != io.CLASS_NODATA)
    io.write_label_geotiff(SLUG, sample_id, label, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(YEAR),
        source_id=rec["source_id"],
        classes_present=present,
    )
    return "ok" if present else "empty"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap tiles per kind")
    args = ap.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "European Primary Forest Database (EPFD) v2.0. Sabatini et al., Sci Data 8, "
            "220 (2021), https://doi.org/10.1038/s41597-021-00988-7. Open-access data "
            "(CC-BY-4.0) on Figshare https://doi.org/10.6084/m9.figshare.13194095 : file "
            "EPFDv2.0_DatabaseOA.zip (https://ndownloader.figshare.com/files/29091789), "
            "containing ESRI personal geodatabase EPFD_v2.0.mdb. Harmonized OA feature "
            "classes: EU_PrimaryForests_Polygons_OA_v20 (18,411), "
            "EU_PrimaryForests_Points_OA_v20 (299). Read via mdbtools mdb-json + custom "
            "ESRI shape-binary decoder into parsed/epfd_oa.gpkg (EPSG:4326).\n"
        )

    build_gpkg()

    import geopandas as gpd
    import shapely

    gpoly = gpd.read_file(GPKG, layer="polygons")
    gpt = gpd.read_file(GPKG, layer="points")
    print(f"loaded {len(gpoly)} polygons, {len(gpt)} points")

    io.check_disk()

    records: list[dict[str, Any]] = []

    # Polygon records: placement = interior representative point.
    plim = len(gpoly) if args.limit <= 0 else min(len(gpoly), args.limit)
    for i in range(plim):
        geom = gpoly.geometry.iloc[i]
        if geom is None or geom.is_empty:
            continue
        try:
            rep = geom.representative_point()
        except Exception:
            rep = geom.centroid
        records.append(
            {
                "kind": "poly",
                "class_id": _class_id(gpoly["forest_type1"].iloc[i]),
                "lon": float(rep.x),
                "lat": float(rep.y),
                "wkb": shapely.to_wkb(geom),
                "source_id": f"poly:{gpoly['objectid'].iloc[i]}:{gpoly['id_dataset'].iloc[i]}",
            }
        )

    # Point records: small uniform-class tile.
    qlim = len(gpt) if args.limit <= 0 else min(len(gpt), args.limit)
    for i in range(qlim):
        geom = gpt.geometry.iloc[i]
        if geom is None or geom.is_empty:
            continue
        records.append(
            {
                "kind": "point",
                "class_id": _class_id(gpt["forest_type1"].iloc[i]),
                "lon": float(geom.x),
                "lat": float(geom.y),
                "size": _point_size_px(gpt["extent_measured"].iloc[i]),
                "source_id": f"point:{gpt['objectid'].iloc[i]}:{gpt['id_dataset'].iloc[i]}",
            }
        )

    print(f"total candidate records: {len(records)}")
    raw_counts = Counter(r["class_id"] for r in records)
    print("candidates per class:", dict(sorted(raw_counts.items())))

    selected = sampling.balance_by_class(records, "class_id", per_class=PER_CLASS)
    for sid, r in enumerate(selected):
        r["sample_id"] = f"{sid:06d}"
    print(f"selected {len(selected)} tiles (<= {PER_CLASS}/class, 25k cap)")

    io.check_disk()

    counts: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write tiles",
        ):
            if res is not None:
                counts[res] += 1
    print("write results:", dict(counts))

    # Class counts among selected (by class id).
    sel_counts = Counter(r["class_id"] for r in selected)
    n_written = len(list(io.locations_dir(SLUG).glob("*.tif")))

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Sci Data (Sabatini et al. 2021) / Figshare",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://www.nature.com/articles/s41597-021-00988-7",
                "have_locally": False,
                "annotation_method": "field + expert compilation (48 harmonized "
                "regional-to-continental primary/old-growth forest datasets)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [{"id": c, "name": n, "description": d} for c, n, d in CLASSES],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_written,
            "class_counts": {str(c): sel_counts.get(c, 0) for c, _n, _d in CLASSES},
            "notes": (
                "Primary/old-growth forest, positive-only segmentation, classed by EEA "
                "European Forest Type (FOREST_TYPE1; 0=unclassified, 1-13 = the 13 named "
                "EEA categories). 18,411 polygon patches -> one <=64x64 UTM 10 m tile each "
                "(rasterized with holes, all_touched=True, centered on interior point; "
                "large patches captured as a central all-forest window); 299 point patches "
                "(approximate centres, no footprint) -> small uniform-class tiles sized "
                "from FOREST_EXTENT_MEASURED (ha; default 8 px, clamped [3,32] px). Outside "
                "a patch = 255 nodata (no synthetic negatives per spec 5). Tiles-per-class "
                "balanced <=1000/class (rare EEA types kept in full). Static persistent "
                "land cover -> 1-year window anchored on 2020 (v2.0 compilation; primary "
                "status verified with Landsat through 2018). Forest types were derived from "
                "a potential-natural-vegetation map so are somewhat coarse relative to what "
                "S2/S1/Landsat observe; broadleaf vs coniferous distinction is observable, "
                "finer biogeographic types are noisier. Three non-open-access source "
                "datasets (IDs 17/34/48) are excluded upstream by the OA release."
            ),
        },
    )
    print(f"selected per class: {dict(sorted(sel_counts.items()))}")
    print(f"total tif on disk: {n_written}")
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=n_written
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
