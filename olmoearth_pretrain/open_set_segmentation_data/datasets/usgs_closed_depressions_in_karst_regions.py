"""Process the USGS Closed Depressions in Karst Regions inventory into sinkhole /
closed-depression segmentation label tiles.

Source: Jones, Doctor, Wood, Falgout & Rapstine (2021), "Closed depression density in
karst regions of the conterminous United States: features and grid data", USGS
ScienceBase, doi:10.5066/P9EV2I12 (public domain). ScienceBase item
60f79cb0d34e9143a4ba4f4e. Closed depressions were extracted by automated algorithms from
the 1/3 arc-second (~10 m) National Elevation Dataset (NED/3DEP), the DEM first
hydro-conditioned (breaching digital dams at road/stream crossings), then restricted to
karst-prone geologic units and screened against developed land / open water / wetlands /
glacial-alluvial cover. Multiple attached files ship with the item; we use the vector
depression footprints:

    * ``karst_depression_polys_conus.zip`` -> ``karst_depression_polys_conus.shp`` :
      individual closed-depression (sinkhole) polygons across the conterminous US. THIS
      is the observable phenomenon and the target of this dataset.

Files NOT used and why:
    * ``sink_density_1km_conus`` / ``sink_density_classified_1km_conus`` /
      ``sink_density_classified_polys_1km_conus`` -- the manifest "sink-density classes".
      These are a DERIVED 1 km aggregate: the *count/density of depressions per square
      km*, a regional landscape statistic that is NOT a per-pixel land feature observable
      in a single 10-30 m S2/S1/Landsat patch (100x coarser than our 10 m grid, and
      density is not a thing a pixel "looks like"). We therefore drop the density-class
      layer and keep only the directly-observable depression footprints. (Judgment call;
      recorded in the summary.)
    * ``USGS_karst_depression_density_conus.gdb.zip`` -- the same content packaged as a
      file geodatabase; redundant with the shapefile.

Task: **binary closed-depression segmentation** (label_type: polygons):

    0 = background          (terrain outside a mapped closed depression -- genuine,
                             observed non-depression context around the sinkhole; not a
                             fabricated negative)
    1 = closed_depression   (a mapped karst closed depression / sinkhole footprint)

Nodata 255 is declared but unused (every pixel in the context tile is observed).

Observability (spec §8): depressions were derived from a 10 m DEM, so many are tiny --
below what 10-30 m optical/SAR imagery can resolve. We keep only depressions with mapped
area >= ``MIN_AREA_M2`` (default 900 m^2, ~= one Landsat 30 m pixel / a 3x3 S2 block) and
drop smaller ones as unresolvable; the full area distribution and the kept/dropped counts
are logged and recorded in metadata. ``all_touched=True`` rasterization guarantees a kept
depression contributes at least one positive pixel.

Tiling: each kept depression is reprojected to a local UTM projection at 10 m/pixel and
centered in a 64x64 (640 m) context tile (inside polygon -> 1, outside -> 0). A depression
whose footprint exceeds 64 px on an axis (rare for sinkholes) is gridded into
non-overlapping 64x64 windows, keeping intersecting windows (up to MAX_TILES_PER_FEATURE).
Selection is round-robin across depressions (every depression contributes >=1 tile before
extras are added), capped at 25,000 tiles total (spec §5).

Time: closed depressions are static topographic features (persistent across the Sentinel
era). There is no per-feature date, so a representative 1-year window (``REP_YEAR``) is
assigned; ``change_time`` is null.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.usgs_closed_depressions_in_karst_regions
Idempotent: existing locations/{id}.tif are skipped.
"""

import argparse
import glob
import math
import multiprocessing
import random
import zipfile
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

SLUG = "usgs_closed_depressions_in_karst_regions"
NAME = "USGS Closed Depressions in Karst Regions"

SB_ITEM = "60f79cb0d34e9143a4ba4f4e"
DOI = "https://doi.org/10.5066/P9EV2I12"
# ScienceBase attached-file download endpoint for karst_depression_polys_conus.zip.
POLY_ZIP_URL = (
    "https://www.sciencebase.gov/catalog/file/get/"
    f"{SB_ITEM}?f=__disk__0a%2F43%2F6a%2F0a436a5eeeaec4c07eb7f5b229c406d6b1d0b8a7"
)
UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122 Safari/537.36"
)

POLY_ZIP = io.raw_dir(SLUG) / "karst_depression_polys_conus.zip"
SHP_DIR = io.raw_dir(SLUG) / "karst_depression_polys_conus"

TILE = 64
REP_YEAR = 2020  # representative Sentinel-era year (features are static; item pub 2021)
MIN_AREA_M2 = (
    900.0  # observability floor (~1 Landsat 30 m pixel); tune per distribution
)
MAX_TILES_PER_FEATURE = 16
MAX_SAMPLES = sampling.MAX_SAMPLES_PER_DATASET  # 25000
AREA_CRS = "EPSG:5070"  # CONUS Albers equal-area (meters) for robust area computation

BG, DEPRESSION = 0, 1
CLASSES = [
    (
        "background",
        "Terrain outside a mapped closed depression: the genuine, observed non-depression "
        "context surrounding a sinkhole within the 640 m tile. The USGS inventory delimits "
        "each depression's footprint, so out-of-polygon pixels are real negatives (no "
        "synthetic far negatives are added).",
    ),
    (
        "closed_depression",
        "A karst closed depression / sinkhole footprint, extracted by automated algorithms "
        "from the 1/3 arc-second (~10 m) National Elevation Dataset (hydro-conditioned to "
        "breach digital dams), restricted to karst-prone geology and screened against "
        "developed land, open water, wetlands, and glacial/alluvial cover. Includes some "
        "false positives (DEM artifacts) and some non-karst depressions (see source notes).",
    ),
]


# --------------------------------------------------------------------------- download


def _transient_msg(detail: str) -> str:
    return (
        f"TRANSIENT: {detail}. The ScienceBase /catalog/file/get/ delivery endpoint was "
        "returning HTTP 404 for ALL attached files (across items) while the catalog and "
        "metadata API were healthy -- a source-side outage, not a permanent gate. No "
        "credentials are required. Retry later: `python3 -m olmoearth_pretrain."
        "open_set_segmentation_data.datasets.usgs_closed_depressions_in_karst_regions`. "
        f"Manual check: curl -A '<browser UA>' '{POLY_ZIP_URL}'."
    )


def download_polygons() -> str:
    """Download + extract the depression polygon shapefile. Returns the .shp path.

    Raises RuntimeError (a TRANSIENT failure) if the ScienceBase file-delivery endpoint
    does not return a valid zip. As of the initial run the ``/catalog/file/get/`` endpoint
    was returning HTTP 404 for every attached file (across items), while the catalog and
    metadata API were healthy -- a source-side outage. Re-run this script once the endpoint
    recovers; no credentials are required.
    """
    io.raw_dir(SLUG).mkdir(parents=True, exist_ok=True)
    with (io.raw_dir(SLUG) / "SOURCE.txt").open("w") as f:
        f.write(
            "USGS Closed Depressions in Karst Regions (public domain).\n"
            f"DOI: {DOI}\nScienceBase item: {SB_ITEM}\n"
            f"Polygon shapefile: {POLY_ZIP_URL}\n"
            "Used file: karst_depression_polys_conus.zip (depression footprints).\n"
            "Density-class layers (sink_density*_1km) intentionally not used: 1 km "
            "aggregate density is not observable per-pixel at 10-30 m.\n"
        )

    existing = glob.glob(str(SHP_DIR / "*.shp"))
    if existing:
        print(f"  [skip] shapefile present: {existing[0]}")
        return existing[0]

    if not POLY_ZIP.exists() or POLY_ZIP.stat().st_size < 100_000:
        print(f"  downloading {POLY_ZIP_URL}")
        try:
            download.download_http(
                POLY_ZIP_URL, POLY_ZIP, skip_existing=False, headers={"User-Agent": UA}
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(_transient_msg(f"download failed: {e!r}")) from e
    # Validate it is really a zip (endpoint returns an HTML 404 page when down).
    if not zipfile.is_zipfile(str(POLY_ZIP)):
        head = b""
        try:
            with POLY_ZIP.open("rb") as fh:
                head = fh.read(64)
        except Exception:  # noqa: BLE001
            pass
        try:
            POLY_ZIP.unlink()
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(
            _transient_msg(f"endpoint did not return a zip (got {head!r}...)")
        )

    SHP_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(POLY_ZIP)) as z:
        z.extractall(str(SHP_DIR))
    found = glob.glob(str(SHP_DIR / "**" / "*.shp"), recursive=True)
    if not found:
        raise RuntimeError(f"no .shp found after extracting {POLY_ZIP}")
    print(f"  extracted shapefile: {found[0]}")
    return found[0]


# --------------------------------------------------------------------------- tiling


def _feature_candidates(feat: dict[str, Any]) -> list[dict[str, Any]]:
    """Return candidate tile records for one depression (bounds + clipped pixel geom)."""
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
    base = {"crs": crs, "source_id": feat["source_id"]}
    out: list[dict[str, Any]] = []

    if w <= TILE and h <= TILE:
        col = round((minx + maxx) / 2.0)
        row = round((miny + maxy) / 2.0)
        b = io.centered_bounds(col, row, TILE, TILE)
        clip = px.intersection(box(*b))
        # all_touched raster below guarantees a positive even for sub-pixel footprints;
        # keep the tile as long as the polygon overlaps it at all.
        if not clip.is_empty:
            out.append({**base, "bounds": b, "clip_wkb": shapely.wkb.dumps(clip)})
        elif not px.is_empty:
            out.append({**base, "bounds": b, "clip_wkb": shapely.wkb.dumps(px)})
        return out

    # Large depression (rare): grid the bbox into non-overlapping 64x64 windows.
    x0, y0 = math.floor(minx), math.floor(miny)
    cells = []
    x = x0
    while x < maxx:
        y = y0
        while y < maxy:
            cells.append((x, y, x + TILE, y + TILE))
            y += TILE
        x += TILE
    rng = random.Random(feat["idx"])
    rng.shuffle(cells)
    prepared = prep(px)
    for b in cells:
        if len(out) >= MAX_TILES_PER_FEATURE:
            break
        bx = box(*b)
        if not prepared.intersects(bx):
            continue
        clip = px.intersection(bx)
        if clip.is_empty:
            continue
        out.append({**base, "bounds": b, "clip_wkb": shapely.wkb.dumps(clip)})
    return out


def _write_one(rec: dict[str, Any]) -> str | None:
    from rasterio.crs import CRS

    from olmoearth_pretrain.open_set_segmentation_data.rasterize import rasterize_shapes

    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return None

    proj = Projection(CRS.from_string(rec["crs"]), io.RESOLUTION, -io.RESOLUTION)
    bounds = tuple(rec["bounds"])
    clip = shapely.wkb.loads(rec["clip_wkb"])
    label = rasterize_shapes(
        [(clip, DEPRESSION)], bounds, fill=BG, dtype="uint8", all_touched=True
    )[0]

    time_range = io.year_range(REP_YEAR)
    present = sorted(int(v) for v in np.unique(label))
    io.write_label_geotiff(SLUG, sample_id, label, proj, bounds, nodata=io.CLASS_NODATA)
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
    return "with_bg" if BG in present else "depression_only"


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--min_area_m2", type=float, default=MIN_AREA_M2)
    args = ap.parse_args()

    io.check_disk()
    shp_path = download_polygons()
    io.check_disk()

    # ---- load polygons (geopandas), compute area in equal-area CRS, filter by area
    import geopandas as gpd

    gdf = gpd.read_file(shp_path)
    n_total = len(gdf)
    print(f"{n_total} depression polygons in source")
    if gdf.crs is None:
        raise RuntimeError("shapefile has no CRS (.prj); cannot georeference")

    areas = gdf.to_crs(AREA_CRS).geometry.area.to_numpy()
    qs = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    pct = {f"p{q}": float(np.percentile(areas, q)) for q in qs}
    print("area(m^2) distribution:", {k: round(v, 1) for k, v in pct.items()})

    keep_mask = areas >= args.min_area_m2
    n_keep = int(keep_mask.sum())
    print(
        f"kept {n_keep}/{n_total} depressions with area >= {args.min_area_m2} m^2 "
        f"(dropped {n_total - n_keep} as unresolvable at 10-30 m)"
    )

    gdf_ll = gdf.to_crs("EPSG:4326")
    feats: list[dict[str, Any]] = []
    for i, (keep, geom) in enumerate(zip(keep_mask, gdf_ll.geometry)):
        if not keep or geom is None or geom.is_empty:
            continue
        feats.append(
            {
                "idx": i,
                "wkb": shapely.wkb.dumps(geom),
                "source_id": f"row={i}:area_m2={round(float(areas[i]), 1)}",
            }
        )
    print(f"{len(feats)} features after filtering")

    # ---- per-feature candidate tiles (parallel)
    io.check_disk()
    per_feat: list[list[dict[str, Any]]] = []
    with multiprocessing.Pool(args.workers) as p:
        for cands in tqdm.tqdm(
            star_imap_unordered(
                p, _feature_candidates, [dict(feat=fr) for fr in feats]
            ),
            total=len(feats),
            desc="candidates",
        ):
            if cands:
                per_feat.append(cands)
    total_cand = sum(len(c) for c in per_feat)
    print(f"{total_cand} candidate tiles across {len(per_feat)} depressions")

    # ---- round-robin selection across depressions, capped at MAX_SAMPLES
    rng = random.Random(42)
    for lst in per_feat:
        rng.shuffle(lst)
    rng.shuffle(per_feat)
    selected: list[dict[str, Any]] = []
    i = 0
    active = [lst for lst in per_feat if lst]
    while active and len(selected) < MAX_SAMPLES:
        lst = active[i % len(active)]
        selected.append(lst.pop())
        i += 1
        if i % len(active) == 0:
            active = [lst for lst in active if lst]
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
                "annotation_method": "automated closed-depression extraction from 10 m NED/3DEP DEM",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": idx, "name": name, "description": desc}
                for idx, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_written,
            "area_filter_m2": args.min_area_m2,
            "area_distribution_m2": pct,
            "n_source_polygons": n_total,
            "n_kept_after_area_filter": n_keep,
            "tile_counts": {
                "tiles_with_background": counts.get("with_bg", 0),
                "depression_only_tiles": counts.get("depression_only", 0),
            },
            "rep_year": REP_YEAR,
            "notes": (
                "Binary closed-depression (sinkhole) segmentation from the USGS CONUS "
                "karst closed-depression inventory (karst_depression_polys_conus.shp). "
                "64x64 uint8 tiles in local UTM at 10 m; 0=background, 1=closed_depression "
                "(255 nodata declared, unused). all_touched rasterization. Static features "
                f"-> representative 1-year window (REP_YEAR={REP_YEAR}); change_time null. "
                f"Observability: kept depressions with area >= {args.min_area_m2} m^2, "
                "dropped smaller ones as unresolvable at 10-30 m. Manifest 'sink-density "
                "classes' (1 km aggregate density layers) intentionally excluded: a 1 km "
                "density statistic is not observable per-pixel in 10-30 m imagery. Source "
                "includes some false positives (DEM artifacts) and non-karst depressions. "
                "Round-robin selection across depressions capped at 25000."
            ),
        },
    )
    print("tile counts:", dict(counts))
    print("total tif on disk:", n_written)
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=n_written
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
