"""Process the USGS "Salars of the Lithium Triangle" geodatabase into a unified
salt-flat / lithium-occurrence segmentation+detection label bank.

Source: Mihalasky, Briggs, Baker, Jaskula, Cheriyan & DeLoach-Overton (2020),
"Lithium Occurrences and Processing Facilities of Argentina, and Salars of the Lithium
Triangle, Central South America", U.S. Geological Survey data release,
doi:10.5066/P9RLUH4F (public domain). ScienceBase item 5e90cd8f82ce172707edfc74. The
attached file geodatabase ``Li_Triangle_ARG_MRP_NMIC.gdb`` (delivered as a 7z) holds:

    * ``Salars_Li_Triangle_MRP_NMIC``      -- 186 salar (salt-flat / laguna) MultiPolygon
      outlines across Argentina/Chile/Bolivia/Peru, hand-digitized from 2018-2019 imagery
      (area 0.43 - 12,078 km^2). THE dominant observable landform.
    * ``Arg_Occurrences_MRP_NMIC``         -- 124 lithium occurrence points, split by
      ``Deptype`` into 106 Salar (brine) + 18 Pegmatite occurrences.
    * ``Arg_Facilities_MRP_NMIC``          -- 10 lithium processing-facility points.
    * ``Salar_Centroids_...`` / ``MRP_NMIC_Refs`` -- centroids (redundant with polygons)
      and a bibliography table; not used.

This is a multi-target (polygons + points) source, so per spec Sec.5 the targets are
combined into ONE unified class scheme rather than split into separate datasets:

    0 = background                  (terrain that is neither salt flat nor a marked point)
    1 = salar                       (salt-flat / laguna polygon footprint)
    2 = li_brine_occurrence         (documented Li brine occurrence point, Deptype=Salar)
    3 = li_pegmatite_occurrence     (documented Li pegmatite occurrence point)
    4 = processing_facility         (Li processing / extraction facility point)

255 = nodata/ignore (the tunable detection buffer ring around each point).

Two kinds of uint8 64x64 (640 m) tiles in local UTM at 10 m/pixel are emitted:
  * salar tiles: a salar polygon rasterized into 64x64 windows (class 1 vs background 0).
    Large salars are gridded into non-overlapping 64x64 windows, of which up to
    ``MAX_TILES_PER_SALAR`` intersecting windows are sampled so every salar contributes
    without huge salars (e.g. Uyuni) swamping the set.
  * point tiles: a 64x64 context tile centered on each occurrence/facility point. Any
    salar polygons overlapping the tile are rasterized as class 1 (real context -- most
    brine occurrences sit on a salt flat), then the point gets the tunable detection
    encoding: a ``buffer_size`` (>=10 px) nodata ring (coords are not pixel-exact) with a
    ``positive_size`` square of the point's class at the center.

Balancing (spec Sec.5): tiles-per-class balanced, up to ``PER_CLASS`` (1000) tiles per
class, 25k total cap. The point classes (10-106 tiles each) are all kept; salar tiles are
capped at 1000. No synthetic negatives are fabricated (Sec.5): non-object pixels are real
background/salar or the nodata ring.

Time (spec Sec.5): salars are persistent landforms and the outlines were digitized from
2018-2019 imagery -> a static representative 1-year Sentinel-era window ``REP_YEAR``
(2018); ``change_time`` is null. All labels are post-2016.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.salars_of_the_lithium_triangle_usgs
Idempotent: existing locations/{id}.tif are skipped.
"""

import argparse
import multiprocessing
import random
from collections import Counter
from typing import Any

import numpy as np
import shapely
import shapely.wkb
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import (
    download,
    io,
    manifest,
    sampling,
)

SLUG = "salars_of_the_lithium_triangle_usgs"
NAME = "Salars of the Lithium Triangle (USGS)"

SB_ITEM = "5e90cd8f82ce172707edfc74"
DOI = "https://doi.org/10.5066/P9RLUH4F"
GDB_7Z_URL = (
    "https://www.sciencebase.gov/catalog/file/get/"
    f"{SB_ITEM}?f=__disk__f3%2F05%2F2a%2Ff3052adc421218ebc09149087234463c8eba50c5"
)
UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122 Safari/537.36"
)

GDB_7Z = io.raw_dir(SLUG) / "Li_Triangle_ARG_MRP_NMIC.gdb.7z"
EXTRACT_DIR = io.raw_dir(SLUG) / "extracted"
GDB_PATH = EXTRACT_DIR / "Li_Triangle_ARG_MRP_NMIC.gdb"

L_SALAR_POLY = "Salars_Li_Triangle_MRP_NMIC"
L_OCC = "Arg_Occurrences_MRP_NMIC"
L_FAC = "Arg_Facilities_MRP_NMIC"

TILE = 64
REP_YEAR = 2018  # salar outlines digitized from 2018-2019 imagery; static landform
BUFFER_SIZE = 10  # nodata ring around each point (coords not pixel-exact; spec Sec.4)
POSITIVE_SIZE = 1  # point marks a location, not a resolved object extent
MAX_TILES_PER_SALAR = 16  # cap candidate windows/salar so huge salars don't dominate
PER_CLASS = 1000
MAX_SAMPLES = sampling.MAX_SAMPLES_PER_DATASET  # 25000

BG, SALAR, BRINE, PEGMATITE, FACILITY = 0, 1, 2, 3, 4
CLASSES = [
    (
        "background",
        "Terrain within the 640 m context tile that is neither a mapped salt flat nor a "
        "marked lithium occurrence/facility point (observed non-target context).",
    ),
    (
        "salar",
        "Salar (endorheic salt flat) or laguna: a persistent evaporite/brine-bearing "
        "closed-basin salt pan of the central Andean Lithium Triangle. Outlines were "
        "manually digitized by USGS from 2018-2019 satellite imagery; footprints range "
        "from ~0.4 to ~12,000 km^2 (Salar de Uyuni).",
    ),
    (
        "li_brine_occurrence",
        "Documented lithium BRINE occurrence (Deptype='Salar'): a point where Li-bearing "
        "subsurface brine has been reported within/beside a salar. Marks a geologic "
        "occurrence location; the brine itself is subsurface, so the observable context "
        "is the salt-flat surface it sits on.",
    ),
    (
        "li_pegmatite_occurrence",
        "Documented lithium PEGMATITE occurrence (Deptype='Pegmatite'): a hard-rock "
        "Li-bearing pegmatite occurrence point, generally outside the salt flats.",
    ),
    (
        "processing_facility",
        "Lithium processing / extraction facility (evaporation-pond and plant complexes) "
        "operating on or beside a salar; point marks the facility location.",
    ),
]


# --------------------------------------------------------------------------- download


def download_gdb() -> str:
    """Download + extract the file geodatabase. Returns the .gdb path."""
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    with (io.raw_dir(SLUG) / "SOURCE.txt").open("w") as f:
        f.write(
            "Salars of the Lithium Triangle (USGS), public domain.\n"
            f"DOI: {DOI}\nScienceBase item: {SB_ITEM}\n"
            f"Geodatabase (7z): {GDB_7Z_URL}\n"
            "Feature classes used: Salars_Li_Triangle_MRP_NMIC (186 salar polygons), "
            "Arg_Occurrences_MRP_NMIC (124 Li occurrence points, Deptype Salar/Pegmatite), "
            "Arg_Facilities_MRP_NMIC (10 processing-facility points).\n"
            "Not used: Salar_Centroids_* (redundant with polygons), MRP_NMIC_Refs "
            "(bibliography table).\n"
        )

    if GDB_PATH.exists() and any(GDB_PATH.iterdir()):
        print(f"  [skip] gdb present: {GDB_PATH}")
        return str(GDB_PATH)

    if not GDB_7Z.exists() or GDB_7Z.stat().st_size < 100_000:
        print(f"  downloading {GDB_7Z_URL}")
        download.download_http(
            GDB_7Z_URL, GDB_7Z, skip_existing=False, headers={"User-Agent": UA}
        )

    import py7zr

    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(GDB_7Z.path, "r") as z:
        z.extractall(path=EXTRACT_DIR.path)
    if not (GDB_PATH.exists() and any(GDB_PATH.iterdir())):
        raise RuntimeError(f"no geodatabase after extracting {GDB_7Z}")
    print(f"  extracted geodatabase: {GDB_PATH}")
    return str(GDB_PATH)


# --------------------------------------------------------------------------- candidates


def _salar_candidates(feat: dict[str, Any]) -> list[dict[str, Any]]:
    """Candidate salar tiles for one polygon (bounds + clipped pixel geom)."""
    import math

    from shapely.geometry import box
    from shapely.prepared import prep

    from olmoearth_pretrain.open_set_segmentation_data.rasterize import geom_to_pixels

    geom = shapely.wkb.loads(feat["wkb"])
    if geom.is_empty:
        return []
    c = geom.centroid
    proj = io.utm_projection_for_lonlat(float(c.x), float(c.y))
    px = geom_to_pixels(geom, WGS84_PROJECTION, proj)
    if px.is_empty:
        return []
    minx, miny, maxx, maxy = px.bounds
    w, h = maxx - minx, maxy - miny
    crs = proj.crs.to_string()
    base = {"crs": crs, "kind": "salar", "source_id": feat["source_id"]}
    out: list[dict[str, Any]] = []

    if w <= TILE and h <= TILE:
        col = round((minx + maxx) / 2.0)
        row = round((miny + maxy) / 2.0)
        b = io.centered_bounds(col, row, TILE, TILE)
        clip = px.intersection(box(*b))
        clip = clip if not clip.is_empty else px
        out.append({**base, "bounds": b, "clip_wkb": shapely.wkb.dumps(clip)})
        return out

    # Large salar: grid the bbox into non-overlapping 64x64 windows, keep intersecting.
    x0, y0 = math.floor(minx), math.floor(miny)
    cells = []
    x = x0
    while x < maxx:
        y = y0
        while y < maxy:
            cells.append((x, y, x + TILE, y + TILE))
            y += TILE
        x += TILE
    prepared = prep(px)
    inter = [b for b in cells if prepared.intersects(box(*b))]
    rng = random.Random(feat["idx"])
    rng.shuffle(inter)
    for b in inter[:MAX_TILES_PER_SALAR]:
        clip = px.intersection(box(*b))
        if clip.is_empty:
            continue
        out.append({**base, "bounds": b, "clip_wkb": shapely.wkb.dumps(clip)})
    return out


def _point_candidate(
    lon: float, lat: float, cls: int, source_id: str, salars_wgs84: list[Any]
) -> dict[str, Any]:
    """Build one point context tile: salar-context clips + point pixel + class."""
    from shapely.geometry import box

    from olmoearth_pretrain.open_set_segmentation_data.rasterize import geom_to_pixels

    proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    prow = row - bounds[1]
    pcol = col - bounds[0]
    tile_box = box(*bounds)
    salar_clips: list[bytes] = []
    for g in salars_wgs84:
        gpx = geom_to_pixels(g, WGS84_PROJECTION, proj)
        if gpx.is_empty:
            continue
        clip = gpx.intersection(tile_box)
        if not clip.is_empty:
            salar_clips.append(shapely.wkb.dumps(clip))
    balance = [cls] + ([SALAR] if salar_clips else [])
    return {
        "crs": proj.crs.to_string(),
        "kind": "point",
        "bounds": bounds,
        "cls": cls,
        "ppix": (int(prow), int(pcol)),
        "salar_clips": salar_clips,
        "source_id": source_id,
        "balance_classes": balance,
    }


# --------------------------------------------------------------------------- writing


def _write_one(rec: dict[str, Any]) -> str | None:
    from rasterio.crs import CRS

    from olmoearth_pretrain.open_set_segmentation_data.rasterize import rasterize_shapes

    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return None

    proj = Projection(CRS.from_string(rec["crs"]), io.RESOLUTION, -io.RESOLUTION)
    bounds = tuple(rec["bounds"])

    if rec["kind"] == "salar":
        clip = shapely.wkb.loads(rec["clip_wkb"])
        arr = rasterize_shapes(
            [(clip, SALAR)], bounds, fill=BG, dtype="uint8", all_touched=True
        )[0]
    else:  # point tile: salar context, then detection encoding for the point
        shapes = [(shapely.wkb.loads(w), SALAR) for w in rec["salar_clips"]]
        if shapes:
            arr = rasterize_shapes(
                shapes, bounds, fill=BG, dtype="uint8", all_touched=True
            )[0]
        else:
            arr = np.full((TILE, TILE), BG, dtype=np.uint8)
        prow, pcol = rec["ppix"]
        r0 = max(0, prow - POSITIVE_SIZE // 2 - BUFFER_SIZE)
        r1 = min(TILE, prow + POSITIVE_SIZE // 2 + BUFFER_SIZE + 1)
        c0 = max(0, pcol - POSITIVE_SIZE // 2 - BUFFER_SIZE)
        c1 = min(TILE, pcol + POSITIVE_SIZE // 2 + BUFFER_SIZE + 1)
        arr[r0:r1, c0:c1] = io.CLASS_NODATA
        pr0 = max(0, prow - POSITIVE_SIZE // 2)
        pr1 = min(TILE, prow + POSITIVE_SIZE // 2 + 1)
        pc0 = max(0, pcol - POSITIVE_SIZE // 2)
        pc1 = min(TILE, pcol + POSITIVE_SIZE // 2 + 1)
        arr[pr0:pr1, pc0:pc1] = rec["cls"]

    present = sorted(int(v) for v in np.unique(arr) if int(v) != io.CLASS_NODATA)
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(REP_YEAR),
        change_time=None,
        source_id=rec["source_id"],
        classes_present=present,
    )
    return rec["kind"]


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    io.check_disk()
    gdb = download_gdb()
    io.check_disk()

    import geopandas as gpd

    sal = gpd.read_file(gdb, layer=L_SALAR_POLY).to_crs("EPSG:4326")
    occ = gpd.read_file(gdb, layer=L_OCC).to_crs("EPSG:4326")
    fac = gpd.read_file(gdb, layer=L_FAC).to_crs("EPSG:4326")
    print(f"salars={len(sal)} occurrences={len(occ)} facilities={len(fac)}")

    salars_wgs84 = [g for g in sal.geometry if g is not None and not g.is_empty]
    # bbox per salar for cheap point<->salar prefilter
    salar_bboxes = [g.bounds for g in salars_wgs84]

    # ---- salar candidate tiles (parallel)
    salar_feats = [
        {
            "idx": i,
            "wkb": shapely.wkb.dumps(g),
            "source_id": f"salar:{sal.iloc[i].get('Name')}:row={i}",
        }
        for i, g in enumerate(sal.geometry)
        if g is not None and not g.is_empty
    ]
    salar_cands: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for cands in tqdm.tqdm(
            star_imap_unordered(
                p, _salar_candidates, [dict(feat=f) for f in salar_feats]
            ),
            total=len(salar_feats),
            desc="salar candidates",
        ):
            salar_cands.extend(cands)
    for r in salar_cands:
        r["balance_classes"] = [SALAR]
    print(f"{len(salar_cands)} salar candidate tiles from {len(salar_feats)} salars")

    # ---- point candidate tiles (serial; only 134 points)
    point_cands: list[dict[str, Any]] = []

    def _nearby_salars(lon: float, lat: float, pad: float = 0.02) -> list[Any]:
        out = []
        for g, (minx, miny, maxx, maxy) in zip(salars_wgs84, salar_bboxes):
            if (
                lon >= minx - pad
                and lon <= maxx + pad
                and lat >= miny - pad
                and lat <= maxy + pad
            ):
                out.append(g)
        return out

    for _, r in occ.iterrows():
        g = r.geometry
        if g is None or g.is_empty:
            continue
        lon, lat = float(g.x), float(g.y)
        cls = BRINE if str(r.get("Deptype")).strip().lower() == "salar" else PEGMATITE
        sid = f"occ:{r.get('Deptype')}:ID={r.get('ID')}:{r.get('SalPegName')}"
        point_cands.append(
            _point_candidate(lon, lat, cls, sid, _nearby_salars(lon, lat))
        )
    for _, r in fac.iterrows():
        g = r.geometry
        if g is None or g.is_empty:
            continue
        lon, lat = float(g.x), float(g.y)
        sid = f"fac:ID={r.get('ID')}:{r.get('Salar_Name')}"
        point_cands.append(
            _point_candidate(lon, lat, FACILITY, sid, _nearby_salars(lon, lat))
        )
    pc = Counter(c["cls"] for c in point_cands)
    print(
        f"{len(point_cands)} point tiles "
        f"(brine={pc[BRINE]} pegmatite={pc[PEGMATITE]} facility={pc[FACILITY]})"
    )

    # ---- unified tiles-per-class balancing (Sec.5): points all kept, salar capped 1000
    all_cands = salar_cands + point_cands
    selected = sampling.balance_tiles_by_class(
        all_cands,
        classes_key="balance_classes",
        per_class=PER_CLASS,
        total_cap=MAX_SAMPLES,
    )
    for j, r in enumerate(selected):
        r["sample_id"] = f"{j:06d}"
    print(f"selected {len(selected)} tiles (cap {MAX_SAMPLES})")

    # ---- write tiles (parallel)
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

    # ---- class pixel/tile stats from the selected records for the summary
    tile_class_counts: Counter = Counter()
    for r in selected:
        for c in set(r["balance_classes"]):
            tile_class_counts[c] += 1

    n_written = len(list(io.locations_dir(SLUG).glob("*.tif")))
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "USGS ScienceBase",
            "license": "public domain",
            "provenance": {
                "url": DOI,
                "sciencebase_item": SB_ITEM,
                "have_locally": False,
                "annotation_method": "manual digitization from 2018-2019 imagery (salar polygons); compiled occurrence/facility points",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": idx, "name": name, "description": desc}
                for idx, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_written,
            "tile_size": TILE,
            "rep_year": REP_YEAR,
            "detection_encoding": {
                "positive_size": POSITIVE_SIZE,
                "buffer_size": BUFFER_SIZE,
                "tile_size": TILE,
            },
            "tiles_per_class": {
                CLASSES[c][0]: tile_class_counts.get(c, 0)
                for c in (SALAR, BRINE, PEGMATITE, FACILITY)
            },
            "tile_kind_counts": {
                "salar_tiles": counts.get("salar", 0),
                "point_tiles": counts.get("point", 0),
            },
            "notes": (
                "Unified multi-target (polygons + points) label bank for the central "
                "Andean Lithium Triangle. 64x64 uint8 tiles in local UTM at 10 m. Classes: "
                "0=background, 1=salar (salt-flat polygon), 2=li_brine_occurrence, "
                "3=li_pegmatite_occurrence, 4=processing_facility; 255=nodata (detection "
                "buffer ring). Salar polygons rasterized into <=64x64 windows (large salars "
                f"gridded, <={MAX_TILES_PER_SALAR} windows/salar sampled); occurrence/"
                "facility points encoded as a positive center + >=10 px nodata ring within "
                "a 64x64 context tile that also rasterizes overlapping salar polygons as "
                "context. Static landform -> representative 1-year window (REP_YEAR="
                f"{REP_YEAR}); salar outlines digitized from 2018-2019 imagery; change_time "
                "null; all labels post-2016. Tiles-per-class balanced: point classes fully "
                f"kept, salar capped at {PER_CLASS}. No synthetic negatives (Sec.5). Brine/"
                "pegmatite occurrence points mark subsurface/hard-rock geologic occurrences "
                "not themselves resolvable at 10-30 m; retained per Sec.5 (downstream drops "
                "too-sparse classes) with the salt-flat surface as observable context."
            ),
        },
    )
    print("tile kind counts:", dict(counts))
    print(
        "tiles-per-class:",
        {CLASSES[c][0]: tile_class_counts.get(c, 0) for c in range(5)},
    )
    print("total tif on disk:", n_written)
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=n_written
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
