"""Process the GLiM (Global Lithological Map) vector database into surface-lithology
segmentation label tiles (label_type: polygons; family: geology).

Source: Hartmann, J. & Moosdorf, N. (2012), "The new global lithological map database
GLiM: A representation of rock properties at the Earth surface", G-Cubed 13, Q12004,
doi:10.1029/2012GC004370. The 0.5-degree gridded raster is archived at PANGAEA
(doi:10.1594/PANGAEA.788537) but is far too coarse (~55 km cells) for 10 m tiles; the
value is the **original vector GIS database** (LiMW_GIS 2015.gdb), an ESRI file
geodatabase of 1,235,259 lithology polygons distributed by CCGM / Univ. Hamburg under
CC-BY, downloadable from the authors' Dropbox link referenced on
https://www.geo.uni-hamburg.de/en/geologie/forschung/aquatische-geochemie/glim.html and
https://www.ccgm.org/en/product/lithological-map-of-the-world/ . No credential required.

The GLiM lithological classification has three hierarchical levels; level 1 (field ``xx``)
has 16 classes. We use level 1. Class ``nd`` (No Data) is dropped (it is not a lithology).
Class ``wb`` (Water Bodies) and ``ig`` (Ice and Glaciers) are retained as legitimate,
10-30 m-observable surface types. That leaves **15 classes** (ids 0-14, assigned in
descending global polygon frequency; see CLASSES).

CAUTION (coarseness): GLiM's average source scale is ~1:3,750,000 -- a very
generalized product. Surface lithology influences terrain, soils and vegetation and is
partially inferable from S2/S1/Landsat at 10-30 m, but the map itself is coarse. Per spec
5 (large derived product), we therefore **sample bounded tiles from spatially-homogeneous
regions** rather than trying to trace polygon boundaries precisely:

  * We keep only polygons with equal-area footprint >= MIN_AREA_M2 (2 km^2), large enough
    to fully (or nearly) contain a 640 m tile -- every one of the 15 kept classes still
    has >= 1000 such polygons.
  * Each selected polygon seeds ONE 64x64 (640 m) tile in a local UTM projection at 10
    m/pixel, centered on the polygon's interior representative point (guaranteed inside
    the polygon, even for L-shaped / multipart polygons).
  * The seed polygon is rasterized into the tile with its class id; pixels OUTSIDE the
    seed polygon are set to 255 (nodata / ignore), NOT a fabricated "background" class --
    on a lithology map every land pixel is *some* rock type, and we deliberately do not
    resolve the neighboring lithology at this coarse scale. This is the positive-only /
    foreground-mask pattern (spec 5); downstream assembly supplies negatives from other
    datasets. Most tiles are near-uniform single-class (a homogeneous lithology patch);
    the ignore border only appears where a tile straddles a polygon edge.
  * Candidates are dropped if the seed class covers < MIN_COVERAGE of the tile, so kept
    tiles are genuinely homogeneous.

Selection: class-balanced by the seed polygon's lithology (spec 5), up to PER_CLASS=1000
tiles per class, under the 25,000 cap. Rare classes (ig, ev) keep all they can.

Time: lithology is a STATIC label with no per-polygon date -> a representative Sentinel-era
1-year window (REP_YEAR); change_time is null.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.glim_global_lithological_map
Idempotent: the raw GDB is skipped if already extracted; existing locations/{id}.tif are
skipped on re-run.
"""

import argparse
import glob
import multiprocessing
import random
import zipfile
from collections import Counter
from typing import Any

import numpy as np
import shapely
import shapely.geometry
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

SLUG = "glim_global_lithological_map"
NAME = "GLiM (Global Lithological Map)"

# CC-BY vector GDB (LiMW_GIS 2015.gdb) referenced by CCGM / Univ. Hamburg. Dropbox is the
# authors' distribution endpoint (dl=1 forces the file rather than the HTML preview).
GDB_ZIP_URL = "https://www.dropbox.com/s/9vuowtebp9f1iud/LiMW_GIS%202015.gdb.zip?dl=1"
PAPER_DOI = "https://doi.org/10.1029/2012GC004370"
PANGAEA_DOI = (
    "https://doi.org/10.1594/PANGAEA.788537"  # coarse 0.5deg raster (not used)
)
GLIM_PAGE = "https://www.geo.uni-hamburg.de/en/geologie/forschung/aquatische-geochemie/glim.html"

GDB_ZIP = io.raw_dir(SLUG) / "LiMW_GIS_2015.gdb.zip"
GDB_DIR = io.raw_dir(SLUG) / "extracted"
LAYER = "GLiM_export"

TILE = 64
REP_YEAR = (
    2020  # representative Sentinel-era year (lithology is static / time-invariant)
)
MIN_AREA_M2 = 2_000_000.0  # 2 km^2: large enough to (nearly) contain a 640 m tile
CAND_PER_CLASS = 2000  # over-sample candidates per class before coverage filter
PER_CLASS = 1000
MIN_COVERAGE = 0.5  # seed class must cover >= this fraction of the tile
MAX_SAMPLES = sampling.MAX_SAMPLES_PER_DATASET  # 25000
NODATA = io.CLASS_NODATA  # 255

# (code, name, description). Ordered by descending global polygon frequency -> class id.
# 'nd' (No Data) is intentionally excluded.
CLASSES: list[tuple[str, str, str]] = [
    (
        "su",
        "unconsolidated_sediments",
        "Unconsolidated (mostly Quaternary) sediments: alluvium, colluvium, aeolian, glacial "
        "and coastal deposits, and soils; poorly to non-lithified clastic material.",
    ),
    (
        "ss",
        "siliciclastic_sedimentary_rocks",
        "Consolidated siliciclastic sedimentary rocks: sandstone, shale, siltstone, mudstone "
        "and conglomerate.",
    ),
    (
        "sc",
        "carbonate_sedimentary_rocks",
        "Carbonate sedimentary rocks: limestone, dolostone and other carbonate-dominated rocks.",
    ),
    (
        "sm",
        "mixed_sedimentary_rocks",
        "Mixed sedimentary rocks: sequences of interbedded / unspecified clastic and carbonate "
        "sedimentary rocks.",
    ),
    (
        "pa",
        "acid_plutonic_rocks",
        "Acid (felsic) plutonic rocks: granite, granodiorite and related coarse-grained "
        "intrusive rocks.",
    ),
    (
        "mt",
        "metamorphics",
        "Metamorphic rocks: gneiss, schist, quartzite, marble, slate and other regional / "
        "contact metamorphic rocks.",
    ),
    (
        "vb",
        "basic_volcanic_rocks",
        "Basic (mafic) volcanic rocks: basalt and related lavas, including large flood-basalt "
        "provinces.",
    ),
    (
        "va",
        "acid_volcanic_rocks",
        "Acid (felsic) volcanic rocks: rhyolite, dacite and related extrusive rocks.",
    ),
    (
        "vi",
        "intermediate_volcanic_rocks",
        "Intermediate volcanic rocks: andesite and related extrusive rocks.",
    ),
    (
        "wb",
        "water_bodies",
        "Inland water bodies mapped as polygons in the source geology (large lakes, "
        "reservoirs, wide rivers).",
    ),
    (
        "pb",
        "basic_plutonic_rocks",
        "Basic (mafic) plutonic rocks: gabbro and related coarse-grained mafic intrusives.",
    ),
    (
        "pi",
        "intermediate_plutonic_rocks",
        "Intermediate plutonic rocks: diorite and related coarse-grained intrusives.",
    ),
    (
        "py",
        "pyroclastics",
        "Pyroclastic / volcaniclastic deposits: tuff, ignimbrite, ash and other fragmental "
        "volcanic materials.",
    ),
    (
        "ev",
        "evaporites",
        "Evaporites: halite, gypsum, anhydrite and other salts precipitated from evaporating "
        "water.",
    ),
    (
        "ig",
        "ice_and_glaciers",
        "Permanent ice and glaciers (perennial ice cover mapped as a surface unit).",
    ),
]
CODE_TO_ID = {code: i for i, (code, _n, _d) in enumerate(CLASSES)}
KEEP_CODES = set(CODE_TO_ID)


# --------------------------------------------------------------------------- download


def download_gdb() -> str:
    """Download + extract the GLiM vector GDB. Returns the .gdb directory path (idempotent)."""
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    with (io.raw_dir(SLUG) / "SOURCE.txt").open("w") as f:
        f.write(
            "GLiM - Global Lithological Map (vector GIS database, LiMW_GIS 2015.gdb).\n"
            f"Paper DOI: {PAPER_DOI}\n"
            f"Project page: {GLIM_PAGE}\n"
            f"Vector GDB (CC-BY, CCGM/Univ. Hamburg distribution): {GDB_ZIP_URL}\n"
            f"Coarse 0.5deg raster (NOT used, too coarse): {PANGAEA_DOI}\n"
            "License: Creative Commons Attribution (CC-BY). 1,235,259 lithology polygons; "
            "level-1 field 'xx' (16 classes). No credential required.\n"
        )
    existing = glob.glob(str(GDB_DIR / "*.gdb"))
    if existing:
        print(f"  [skip] extracted GDB present: {existing[0]}")
        return existing[0]
    if not GDB_ZIP.exists() or GDB_ZIP.stat().st_size < 1_000_000:
        print(f"  downloading {GDB_ZIP_URL}")
        download.download_http(GDB_ZIP_URL, GDB_ZIP, skip_existing=False)
    if not zipfile.is_zipfile(str(GDB_ZIP)):
        raise RuntimeError(
            f"TRANSIENT: {GDB_ZIP} is not a valid zip (Dropbox delivery issue?). "
            f"Retry later: curl -L '{GDB_ZIP_URL}' -o LiMW_GIS_2015.gdb.zip"
        )
    GDB_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {GDB_ZIP.name} -> {GDB_DIR}")
    with zipfile.ZipFile(str(GDB_ZIP)) as z:
        z.extractall(str(GDB_DIR))
    found = glob.glob(str(GDB_DIR / "*.gdb"))
    if not found:
        raise RuntimeError(f"no .gdb found after extracting {GDB_ZIP}")
    return found[0]


# --------------------------------------------------------------------------- tiling


def _candidate_task(rec: dict[str, Any]) -> dict[str, Any] | None:
    """Build one homogeneous candidate tile from a seed polygon (WGS84 WKB).

    Centers a 64x64 tile on the polygon's interior representative point, rasterizes the
    seed polygon (clipped to a small local window then to the tile) and keeps the tile
    only if the seed class covers >= MIN_COVERAGE. Returns a candidate record (crs, bounds,
    clip pixel-geom WKB, seed_class, coverage, source_id) or None.
    """
    from shapely.geometry import box

    from olmoearth_pretrain.open_set_segmentation_data.rasterize import geom_to_pixels

    geom = shapely.wkb.loads(rec["wkb"])
    if geom.is_empty:
        return None
    rp = geom.representative_point()
    lon, lat = float(rp.x), float(rp.y)
    proj = io.utm_projection_for_lonlat(lon, lat)
    _, col, row = io.lonlat_to_utm_pixel(lon, lat, proj)
    bounds = io.centered_bounds(col, row, TILE, TILE)

    # Clip the (possibly huge) polygon to a small WGS84 window around the seed point before
    # reprojecting, so per-candidate reprojection stays cheap regardless of polygon size.
    d = 0.05  # deg (~5.5 km at equator), comfortably larger than the 640 m tile
    local = shapely.clip_by_rect(geom, lon - d, lat - d, lon + d, lat + d)
    if local.is_empty:
        return None
    px = geom_to_pixels(local, WGS84_PROJECTION, proj)
    if px.is_empty or not px.is_valid:
        px = px.buffer(0)
    if px.is_empty:
        return None
    clip = px.intersection(box(*bounds))
    if clip.is_empty:
        return None
    coverage = float(clip.area) / float(TILE * TILE)
    if coverage < MIN_COVERAGE:
        return None
    return {
        "crs": proj.crs.to_string(),
        "bounds": list(bounds),
        "clip_wkb": shapely.wkb.dumps(clip),
        "seed_class": rec["class_id"],
        "coverage": round(coverage, 4),
        "source_id": rec["source_id"],
    }


def _write_one(rec: dict[str, Any]) -> tuple[int, bool] | None:
    from rasterio.crs import CRS

    from olmoearth_pretrain.open_set_segmentation_data.rasterize import rasterize_shapes

    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return None

    proj = Projection(CRS.from_string(rec["crs"]), io.RESOLUTION, -io.RESOLUTION)
    bounds = tuple(rec["bounds"])
    clip = shapely.wkb.loads(rec["clip_wkb"])
    cls = rec["seed_class"]
    # Rasterize seed lithology; outside-polygon pixels -> 255 (nodata/ignore), not a class.
    label = rasterize_shapes(
        [(clip, cls)], bounds, fill=NODATA, dtype="uint8", all_touched=True
    )[0]
    present = sorted(int(v) for v in np.unique(label) if int(v) != NODATA)
    time_range = io.year_range(REP_YEAR)
    io.write_label_geotiff(SLUG, sample_id, label, proj, bounds, nodata=NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        time_range,
        change_time=None,
        source_id=rec["source_id"],
        classes_present=present,
    )
    has_ignore = bool((label == NODATA).any())
    return cls, has_ignore


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--per_class", type=int, default=PER_CLASS)
    ap.add_argument("--cand_per_class", type=int, default=CAND_PER_CLASS)
    ap.add_argument("--min_area_m2", type=float, default=MIN_AREA_M2)
    args = ap.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    gdb_path = download_gdb()
    io.check_disk()

    import pyogrio

    # Read only large, non-'nd' polygons (still ~728k). CRS is ESRI:54012 Eckert IV
    # (equal-area, metres), so Shape_Area is a reliable m^2 footprint.
    print("reading GLiM polygons (area >= %.0f m^2, xx != 'nd') ..." % args.min_area_m2)
    where = f"xx <> 'nd' AND Shape_Area >= {int(args.min_area_m2)}"
    gdf = pyogrio.read_dataframe(
        gdb_path, layer=LAYER, columns=["xx", "IDENTITY_", "Shape_Area"], where=where
    )
    gdf = gdf[gdf["xx"].isin(KEEP_CODES)].reset_index(drop=True)
    print(f"  {len(gdf)} candidate polygons")
    print("  by class:", {k: int(v) for k, v in gdf["xx"].value_counts().items()})

    # Sample up to cand_per_class polygons per class (deterministic), then reproject only
    # those to WGS84 (avoids reprojecting all ~728k geometries).
    rng = random.Random(42)
    sel_idx: list[int] = []
    by_class_idx: dict[str, list[int]] = {}
    for code in KEEP_CODES:
        idxs = gdf.index[gdf["xx"] == code].tolist()
        by_class_idx[code] = idxs
        rng.shuffle(idxs)
        sel_idx.extend(idxs[: args.cand_per_class])
    sub = gdf.iloc[sorted(set(sel_idx))].copy()
    print(
        f"  sampled {len(sub)} polygons for candidate generation; reprojecting to WGS84"
    )
    sub = sub.to_crs(4326)

    cand_recs = [
        dict(
            rec={
                "wkb": shapely.wkb.dumps(row.geometry),
                "class_id": CODE_TO_ID[row.xx],
                "source_id": f"{row.IDENTITY_}:{row.xx}",
            }
        )
        for row in sub.itertuples()
        if row.geometry is not None and not row.geometry.is_empty
    ]
    print(f"  {len(cand_recs)} candidate tasks")

    io.check_disk()
    candidates: list[dict[str, Any]] = []
    cov_by_class: dict[int, list[float]] = {}
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _candidate_task, cand_recs),
            total=len(cand_recs),
            desc="candidates",
        ):
            if res is not None:
                candidates.append(res)
                cov_by_class.setdefault(res["seed_class"], []).append(res["coverage"])
    print(f"{len(candidates)} candidate tiles (coverage >= {MIN_COVERAGE})")

    # Class-balanced selection by seed lithology (spec 5): up to per_class per class.
    selected = sampling.balance_by_class(
        candidates, "seed_class", per_class=args.per_class, total_cap=MAX_SAMPLES
    )
    for j, r in enumerate(selected):
        r["sample_id"] = f"{j:06d}"
    print(
        f"selected {len(selected)} tiles (<= {args.per_class}/class, cap {MAX_SAMPLES})"
    )

    io.check_disk()
    written_by_class: Counter = Counter()
    ignore_tiles = 0
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write tiles",
        ):
            if res is not None:
                cls, has_ignore = res
                written_by_class[cls] += 1
                ignore_tiles += int(has_ignore)

    n_written = len(list(io.locations_dir(SLUG).glob("*.tif")))
    sel_counts = Counter(r["seed_class"] for r in selected)
    cov_summary = {
        CLASSES[c][1]: {
            "mean": round(float(np.mean(v)), 3),
            "min": round(float(np.min(v)), 3),
            "n": len(v),
        }
        for c, v in sorted(cov_by_class.items())
    }

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "GLiM / CCGM / Univ. Hamburg (Hartmann & Moosdorf 2012)",
            "license": "CC-BY",
            "provenance": {
                "url": GLIM_PAGE,
                "paper_doi": PAPER_DOI,
                "vector_gdb": GDB_ZIP_URL,
                "pangaea_raster_doi_not_used": PANGAEA_DOI,
                "have_locally": False,
                "annotation_method": "compilation of 92 regional geological maps into a global lithological vector map (level-1 = 16 classes)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "code": code, "description": desc}
                for i, (code, name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": NODATA,
            "num_samples": n_written,
            "min_polygon_area_m2": args.min_area_m2,
            "min_coverage": MIN_COVERAGE,
            "n_source_polygons_total": 1235259,
            "dropped_classes": ["nd (No Data): not a lithology"],
            "selected_tiles_per_class": {
                CLASSES[c][1]: sel_counts.get(c, 0) for c in range(len(CLASSES))
            },
            "written_tiles_per_class": {
                CLASSES[c][1]: written_by_class.get(c, 0) for c in range(len(CLASSES))
            },
            "coverage_by_class": cov_summary,
            "tiles_with_ignore_border": ignore_tiles,
            "rep_year": REP_YEAR,
            "notes": (
                "Surface-lithology segmentation from the GLiM vector database "
                "(LiMW_GIS 2015.gdb, 1,235,259 polygons, level-1 field 'xx'). 64x64 uint8 "
                "tiles in local UTM at 10 m. 15 classes (ids 0-14, descending global "
                "polygon frequency); 'nd' (No Data) dropped. Each tile seeds on one "
                f">= {int(args.min_area_m2)} m^2 polygon and rasterizes the seed lithology "
                "at its interior representative point; pixels outside the seed polygon are "
                "255 (nodata/ignore), NOT a fabricated background class -- every land pixel "
                "is some rock type and neighbours are intentionally not resolved at this "
                "coarse scale (positive-only foreground mask, spec 5). Tiles kept only if "
                f"the seed class covers >= {MIN_COVERAGE} of the tile, so tiles are "
                "spatially homogeneous. CAUTION: GLiM is a coarse ~1:3,750,000 generalized "
                "product; lithology is only partially inferable at 10-30 m via its "
                "influence on terrain/soil/vegetation -- boundaries are approximate. Static "
                f"label -> representative 1-year window (REP_YEAR={REP_YEAR}); change_time "
                "null. Class-balanced by seed lithology up to 1000/class (cap 25000)."
            ),
        },
    )
    print(
        "selected tiles per class:",
        {CLASSES[c][1]: sel_counts.get(c, 0) for c in range(len(CLASSES))},
    )
    print("total tif on disk:", n_written, "| tiles with ignore border:", ignore_tiles)
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=n_written
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
