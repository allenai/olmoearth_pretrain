"""Process the USGS State Geologic Map Compilation (SGMC) vector geodatabase into
surface generalized-lithology segmentation label tiles (label_type: polygons; family:
geology).

Source: Horton, J.D., San Juan, C.A., and Stoeser, D.B. (2017), "The State Geologic Map
Compilation (SGMC) geodatabase of the conterminous United States", USGS Data Series 1052,
doi:10.3133/ds1052 ; data release doi:10.5066/F7WH2N65 . Distributed by the USGS as a
public-domain ESRI file geodatabase (USGS_SGMC_Geodatabase.zip, ~416 MB) from ScienceBase
(item 5888bf4fe4b05ccb964bab9d) and referenced at https://mrdata.usgs.gov/geology/state/ .
No credential required.

The SGMC merges 48 conterminous-US state geologic maps into a single seamless polygon
feature class (SGMC_Geology, 313,732 MultiPolygons) in USA Contiguous Albers Equal-Area
Conic (ESRI:102039, metres -> Shape_Area is a reliable m^2 footprint). Each polygon
carries a curated **GENERALIZED_LITH** field: a standardized generalized-lithology category
(33 distinct values), which is exactly the "generalized lithology categories" this dataset
advertises. We use GENERALIZED_LITH directly as the per-pixel class -- no CSV join needed.
(The geodatabase also carries geologic age via the Age table; a single-band per-pixel label
can only encode one attribute, and surface lithology is the more directly observable-at-
10-30 m property, so age is not used -- see summary.)

Two of the 33 GENERALIZED_LITH values are not lithologies and are dropped: 'Unknown'
(26 polygons) and 'Dam' (7 polygons, an anthropogenic structure). The remaining **31
classes** are kept (ids 0-30, assigned in descending global polygon frequency; see CLASSES),
including natural non-rock surface types 'Water' and 'Ice' (both observable at 10-30 m,
retained as legitimate surface units as in the GLiM sibling script).

CAUTION (coarseness): SGMC is a compilation of state maps whose source scales are broadly
~1:500,000 -- a generalized product. Surface lithology influences terrain, soils and
vegetation and is partially inferable from S2/S1/Landsat at 10-30 m, but the map itself is
coarse and boundaries are approximate. Per spec 5 (large derived product) and mirroring the
GLiM approach, we therefore **sample bounded homogeneous tiles from large polygons** rather
than tracing polygon boundaries precisely:

  * We keep only polygons with equal-area footprint >= MIN_AREA_M2 (1 km^2, ~2.4x a 640 m
    tile), large enough to (nearly) contain a tile.
  * Each selected polygon seeds ONE 64x64 (640 m) tile in a local UTM projection at 10
    m/pixel, centered on the polygon's interior representative point (guaranteed inside the
    polygon, even for L-shaped / multipart polygons).
  * The seed polygon is rasterized into the tile with its class id; pixels OUTSIDE the seed
    polygon are set to 255 (nodata / ignore), NOT a fabricated "background" class -- on a
    geology map every land pixel is *some* rock type and we deliberately do not resolve the
    neighbouring lithology at this coarse scale (positive-only foreground mask, spec 5).
    Downstream assembly supplies negatives from other datasets.
  * Candidates are dropped if the seed class covers < MIN_COVERAGE of the tile, so kept
    tiles are genuinely homogeneous.

Selection: class-balanced by the seed polygon's GENERALIZED_LITH (spec 5), up to
PER_CLASS=1000 tiles per class, under the 25,000 cap. Rare classes keep all they can.

Time: lithology is a STATIC label with no per-polygon date -> a representative Sentinel-era
1-year window (REP_YEAR); change_time is null.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.usgs_state_geologic_map_compilation_sgmc
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

from olmoearth_pretrain.open_set_segmentation_data import io, manifest, sampling

SLUG = "usgs_state_geologic_map_compilation_sgmc"
NAME = "USGS State Geologic Map Compilation (SGMC)"

# Public-domain ESRI file geodatabase from USGS ScienceBase (Horton et al. 2017).
SB_ITEM = "5888bf4fe4b05ccb964bab9d"
GDB_ZIP_URL = (
    "https://www.sciencebase.gov/catalog/file/get/5888bf4fe4b05ccb964bab9d?"
    "f=__disk__24%2Ff6%2Fe1%2F24f6e139c181fd9fe43df2aaf7f50b1c5b3b6297"
)
CSV_ZIP_URL = (
    "https://www.sciencebase.gov/catalog/file/get/5888bf4fe4b05ccb964bab9d?"
    "f=__disk__01%2F72%2F86%2F01728693d80f18886230b67bdc78786215c5142c"
)
DOI = "https://doi.org/10.5066/F7WH2N65"
REPORT_DOI = "https://doi.org/10.3133/ds1052"
MRDATA_PAGE = "https://mrdata.usgs.gov/geology/state/"

GDB_ZIP = io.raw_dir(SLUG) / "USGS_SGMC_Geodatabase.zip"
GDB_DIR = io.raw_dir(SLUG) / "extracted"
LAYER = "SGMC_Geology"
LITH_FIELD = "GENERALIZED_LITH"

TILE = 64
REP_YEAR = (
    2020  # representative Sentinel-era year (lithology is static / time-invariant)
)
MIN_AREA_M2 = (
    1_000_000.0  # 1 km^2 (~2.4x a 640 m tile): large enough to (nearly) contain a tile
)
CAND_PER_CLASS = 2000  # over-sample candidate polygons per class before coverage filter
PER_CLASS = 1000
MIN_COVERAGE = 0.5  # seed class must cover >= this fraction of the tile
MAX_SAMPLES = sampling.MAX_SAMPLES_PER_DATASET  # 25000
NODATA = io.CLASS_NODATA  # 255
N_SOURCE_POLYGONS = 313732

# GENERALIZED_LITH values that are NOT lithologies -> dropped.
DROP_LITH = {"Unknown", "Dam"}

# (GENERALIZED_LITH value, snake_case name, description). Ordered by descending global
# polygon frequency in SGMC_Geology -> class id (0..30).
CLASSES: list[tuple[str, str, str]] = [
    (
        "Sedimentary, clastic",
        "sedimentary_clastic",
        "Clastic sedimentary rocks: sandstone, siltstone, shale, mudstone and conglomerate "
        "(consolidated detrital sediments).",
    ),
    (
        "Unconsolidated, undifferentiated",
        "unconsolidated_undifferentiated",
        "Unconsolidated (mostly Quaternary) surficial deposits: alluvium, colluvium, glacial, "
        "aeolian and coastal sediments; undifferentiated.",
    ),
    (
        "Sedimentary, undifferentiated",
        "sedimentary_undifferentiated",
        "Sedimentary rocks of undifferentiated or mixed clastic/carbonate composition.",
    ),
    (
        "Sedimentary, carbonate",
        "sedimentary_carbonate",
        "Carbonate sedimentary rocks: limestone, dolostone and other carbonate-dominated rocks.",
    ),
    (
        "Igneous, volcanic",
        "igneous_volcanic",
        "Volcanic (extrusive) igneous rocks: basalt, andesite, rhyolite, tuff and related lavas "
        "and flows.",
    ),
    (
        "Igneous, intrusive",
        "igneous_intrusive",
        "Intrusive (plutonic) igneous rocks: granite, granodiorite, diorite, gabbro and related "
        "coarse-grained intrusives.",
    ),
    (
        "Water",
        "water",
        "Inland water bodies mapped as polygons in the source geology (large lakes, reservoirs, "
        "wide rivers).",
    ),
    (
        "Metamorphic, undifferentiated",
        "metamorphic_undifferentiated",
        "Metamorphic rocks of undifferentiated type or mixed metamorphic composition.",
    ),
    (
        "Metamorphic, sedimentary clastic",
        "metamorphic_sedimentary_clastic",
        "Metamorphosed clastic sedimentary rocks (metasedimentary): metasandstone, metapelite, "
        "phyllite and related.",
    ),
    (
        "Metamorphic, gneiss",
        "metamorphic_gneiss",
        "Gneiss and other high-grade banded regional metamorphic rocks.",
    ),
    (
        "Metamorphic and Sedimentary, undifferentiated",
        "metamorphic_and_sedimentary_undifferentiated",
        "Undifferentiated mixtures / interlayered sequences of metamorphic and sedimentary rocks.",
    ),
    (
        "Igneous and Sedimentary, undifferentiated",
        "igneous_and_sedimentary_undifferentiated",
        "Undifferentiated mixtures / interlayered sequences of igneous and sedimentary rocks.",
    ),
    (
        "Unconsolidated and Sedimentary, undifferentiated",
        "unconsolidated_and_sedimentary_undifferentiated",
        "Undifferentiated mixtures of unconsolidated surficial deposits and consolidated "
        "sedimentary rocks.",
    ),
    (
        "Sedimentary, chemical",
        "sedimentary_chemical",
        "Chemical sedimentary rocks (excluding evaporites and iron formation): chert and other "
        "chemically precipitated sediments.",
    ),
    (
        "Igneous, undifferentiated",
        "igneous_undifferentiated",
        "Igneous rocks of undifferentiated intrusive/extrusive character.",
    ),
    (
        "Metamorphic, schist",
        "metamorphic_schist",
        "Schist: strongly foliated medium-grade regional metamorphic rock.",
    ),
    (
        "Igneous and Metamorphic, undifferentiated",
        "igneous_and_metamorphic_undifferentiated",
        "Undifferentiated mixtures / interlayered sequences of igneous and metamorphic rocks.",
    ),
    (
        "Metamorphic, volcanic",
        "metamorphic_volcanic",
        "Metamorphosed volcanic rocks (metavolcanics): greenstone, metabasalt and related.",
    ),
    (
        "Metamorphic, amphibolite",
        "metamorphic_amphibolite",
        "Amphibolite: mafic medium- to high-grade metamorphic rock.",
    ),
    (
        "Metamorphic, carbonate",
        "metamorphic_carbonate",
        "Metamorphosed carbonate rocks: marble and related.",
    ),
    (
        "Metamorphic, intrusive",
        "metamorphic_intrusive",
        "Metamorphosed intrusive igneous rocks (meta-plutonic): orthogneiss and related.",
    ),
    (
        "Metamorphic, serpentinite",
        "metamorphic_serpentinite",
        "Serpentinite and related ultramafic metamorphic rocks.",
    ),
    (
        "Metamorphic, sedimentary",
        "metamorphic_sedimentary",
        "Metamorphosed sedimentary rocks (metasedimentary), undifferentiated clastic/carbonate.",
    ),
    (
        "Metamorphic, other",
        "metamorphic_other",
        "Other / miscellaneous metamorphic rock types not covered by the specific classes.",
    ),
    (
        "Tectonite, undifferentiated",
        "tectonite_undifferentiated",
        "Tectonites: strongly deformed fault-zone rocks (mylonite, cataclasite), "
        "undifferentiated.",
    ),
    (
        "Sedimentary, iron formation, undifferentiated",
        "sedimentary_iron_formation_undifferentiated",
        "Iron formation: banded iron-rich chemical sedimentary rocks, undifferentiated.",
    ),
    (
        "Melange",
        "melange",
        "Melange: chaotic tectonic mixture of heterogeneous blocks in a sheared matrix.",
    ),
    (
        "Sedimentary, evaporite",
        "sedimentary_evaporite",
        "Evaporites: gypsum, anhydrite, halite and other salts precipitated from evaporating "
        "water.",
    ),
    (
        "Metamorphic, granulite",
        "metamorphic_granulite",
        "Granulite: high-grade granoblastic metamorphic rock.",
    ),
    (
        "Metamorphic, igneous",
        "metamorphic_igneous",
        "Metamorphosed igneous rocks (meta-igneous), undifferentiated.",
    ),
    ("Ice", "ice", "Permanent ice / perennial snowfields mapped as a surface unit."),
]
LITH_TO_ID = {lith: i for i, (lith, _n, _d) in enumerate(CLASSES)}
KEEP_LITH = set(LITH_TO_ID)


# --------------------------------------------------------------------------- download


def download_gdb() -> str:
    """Download + extract the SGMC geodatabase. Returns the .gdb directory path (idempotent)."""
    from olmoearth_pretrain.open_set_segmentation_data import download

    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    with (io.raw_dir(SLUG) / "SOURCE.txt").open("w") as f:
        f.write(
            "USGS State Geologic Map Compilation (SGMC), Horton et al. 2017.\n"
            f"Report DOI: {REPORT_DOI}\n"
            f"Data release DOI: {DOI}\n"
            f"ScienceBase item: https://www.sciencebase.gov/catalog/item/{SB_ITEM}\n"
            f"mrdata page: {MRDATA_PAGE}\n"
            f"Geodatabase zip (~416 MB): {GDB_ZIP_URL}\n"
            f"CSV attribute tables (~1.5 MB): {CSV_ZIP_URL}\n"
            "License: public domain (US Government work). Polygon feature class "
            "'SGMC_Geology' (313,732 polygons) carries curated 'GENERALIZED_LITH' "
            "generalized-lithology categories used here. No credential required.\n"
        )
    existing = glob.glob(str(GDB_DIR / "**" / "*.gdb"), recursive=True)
    if existing:
        print(f"  [skip] extracted GDB present: {existing[0]}")
        return existing[0]
    if not GDB_ZIP.exists() or GDB_ZIP.stat().st_size < 1_000_000:
        print(f"  downloading {GDB_ZIP_URL}")
        download.download_http(GDB_ZIP_URL, GDB_ZIP, skip_existing=False)
    if not zipfile.is_zipfile(str(GDB_ZIP)):
        raise RuntimeError(
            f"TRANSIENT: {GDB_ZIP} is not a valid zip (ScienceBase delivery issue?). "
            f"Retry later: curl -L '{GDB_ZIP_URL}' -o USGS_SGMC_Geodatabase.zip"
        )
    GDB_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {GDB_ZIP.name} -> {GDB_DIR}")
    with zipfile.ZipFile(str(GDB_ZIP)) as z:
        z.extractall(str(GDB_DIR))
    found = glob.glob(str(GDB_DIR / "**" / "*.gdb"), recursive=True)
    if not found:
        raise RuntimeError(f"no .gdb found after extracting {GDB_ZIP}")
    return found[0]


# --------------------------------------------------------------------------- tiling


def _candidate_task(rec: dict[str, Any]) -> dict[str, Any] | None:
    """Build one homogeneous candidate tile from a seed polygon (WGS84 WKB)."""
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

    # Read only large polygons (Shape_Area in m^2, Albers equal-area). Keep GENERALIZED_LITH,
    # UNIT_LINK, STATE for the class + provenance.
    print(
        "reading SGMC_Geology polygons (Shape_Area >= %.0f m^2) ..." % args.min_area_m2
    )
    where = f"Shape_Area >= {int(args.min_area_m2)}"
    gdf = pyogrio.read_dataframe(
        gdb_path,
        layer=LAYER,
        columns=[LITH_FIELD, "UNIT_LINK", "STATE", "Shape_Area"],
        where=where,
    )
    gdf = gdf[gdf[LITH_FIELD].isin(KEEP_LITH)].reset_index(drop=True)
    print(
        f"  {len(gdf)} candidate polygons (kept lithologies, dropped {sorted(DROP_LITH)})"
    )
    print("  by class:", {k: int(v) for k, v in gdf[LITH_FIELD].value_counts().items()})

    # Sample up to cand_per_class polygons per class (deterministic), then reproject only
    # those to WGS84 (avoids reprojecting all geometries).
    rng = random.Random(42)
    sel_idx: list[int] = []
    for lith in KEEP_LITH:
        idxs = gdf.index[gdf[LITH_FIELD] == lith].tolist()
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
                "class_id": LITH_TO_ID[getattr(row, LITH_FIELD)],
                "source_id": f"{row.STATE}:{row.UNIT_LINK}",
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
            "source": "USGS State Geologic Map Compilation (Horton et al. 2017, DS 1052)",
            "license": "public domain",
            "provenance": {
                "url": MRDATA_PAGE,
                "data_release_doi": DOI,
                "report_doi": REPORT_DOI,
                "sciencebase_item": SB_ITEM,
                "geodatabase_zip": GDB_ZIP_URL,
                "have_locally": False,
                "annotation_method": "compilation of 48 conterminous-US state geologic maps into a seamless polygon feature class with standardized GENERALIZED_LITH generalized-lithology categories",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "code": lith, "description": desc}
                for i, (lith, name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": NODATA,
            "num_samples": n_written,
            "min_polygon_area_m2": args.min_area_m2,
            "min_coverage": MIN_COVERAGE,
            "n_source_polygons_total": N_SOURCE_POLYGONS,
            "dropped_classes": [
                "Unknown (26 polygons): not a lithology",
                "Dam (7 polygons): anthropogenic structure, not a lithology",
            ],
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
                "Surface generalized-lithology segmentation from the USGS SGMC vector "
                "geodatabase (SGMC_Geology, 313,732 polygons; per-polygon curated "
                "GENERALIZED_LITH field). 64x64 uint8 tiles in local UTM at 10 m. 31 "
                "classes (ids 0-30, descending global polygon frequency); GENERALIZED_LITH "
                "'Unknown' and 'Dam' dropped as non-lithology; natural surface types 'Water' "
                "and 'Ice' retained. Each tile seeds on one >= "
                f"{int(args.min_area_m2)} m^2 polygon and rasterizes the seed lithology at "
                "its interior representative point; pixels outside the seed polygon are 255 "
                "(nodata/ignore), NOT a fabricated background class -- every land pixel is "
                "some rock type and neighbours are intentionally not resolved at this coarse "
                "scale (positive-only foreground mask, spec 5). Tiles kept only if the seed "
                f"class covers >= {MIN_COVERAGE} of the tile, so tiles are spatially "
                "homogeneous. CAUTION: SGMC is a generalized ~1:500,000-scale compilation; "
                "lithology is only partially inferable at 10-30 m via its influence on "
                "terrain/soil/vegetation -- boundaries are approximate. Geologic age (also "
                "in the geodatabase) is not encoded: a single-band per-pixel label holds one "
                "attribute and lithology is the more directly observable surface property. "
                f"Static label -> representative 1-year window (REP_YEAR={REP_YEAR}); "
                "change_time null. Class-balanced by seed lithology up to 1000/class "
                "(cap 25000). Region: conterminous US only."
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
