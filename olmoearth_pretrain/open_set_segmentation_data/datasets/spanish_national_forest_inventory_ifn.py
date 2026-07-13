"""Spanish National Forest Inventory (IFN) -> dominant-tree-species segmentation tiles.

Manifest entry: name "Spanish National Forest Inventory (IFN)", family tree_species,
label_type "points/polygons", region Spain, classes "~59 dominant tree species",
time_range 2016-2019, source "MITECO / NFI Downloader", url descargaifn.gsic.uva.es.

PRODUCT CHOICE (spec triage, two candidate IFN products):
  (a) IFN field PLOTS (points): each plot has a dominant species measured in the field, but
      the public NFI plot coordinates are deliberately degraded (rounded to ~1 km for
      conservation of the permanent plots). At ~1 km precision a plot is NOT reliably
      observable at 10 m, so plot-level *species* points are a REJECT-level observability
      problem (spec 2, "coordinate-fuzzed points like FIA ~1 mi").
  (b) Mapa Forestal de Espana (MFE) POLYGONS: the MFE is the official forest cartography of
      Spain and is the *cartographic base of the IFN* itself (the MFE25 edition is the base
      map of the 4th National Forest Inventory, IFN4). Each polygon (tesela) carries the
      dominant tree species (up to 3, with occupancy) photointerpreted at 1:25,000 with
      field checking. These polygons ARE observable at 10 m and rasterize to a
      forest-type / dominant-species class map. This is the stronger, usable product.

We therefore use **(b) MFE25 (IFN4 base) dominant-species polygons**, exactly as the task
spec recommends ("PREFER MFE polygons ... homogeneous tiles at interior points, like the
GLiM lithology approach").

ACCESS: the per-province MFE25 shapefile downloads on mapama.gob.es are gated behind a
Google reCAPTCHA (not scriptable). The identical MFE25/IFN4 data is served, without any
credential or captcha, by MITECO's public OGC API - Features endpoint, collection
``biodiversidad:MFE`` ("LC.Mapa Forestal de Espana 1:25.000 (MFE25), Base Cartografica del
Cuarto Inventario Forestal Nacional (IFN4)"). We page that endpoint with a CQL2 filter and
startIndex paging. Geometries come back as WGS84 (CRS84) polygons; we reproject each to a
local UTM projection at 10 m.

METHOD (GLiM-style homogeneous tiles):
  * Candidate polygons: server-side CQL2 filter ``especie1 <> 'sin datos' AND
    superficie_ha > 40 AND o1 >= 70`` -> large (>40 ha, big enough to contain a 640 m tile),
    single-dominant-species (species-1 occupies >= 70% of the canopy) forest teselas.
    ~81,700 such polygons nationwide (spanning all of Spain's biogeographic regions).
  * Each candidate seeds ONE 64x64 (640 m) tile in local UTM at 10 m, centered on the
    polygon's interior representative point (guaranteed inside, even for concave/multipart
    polygons). The seed polygon is rasterized with its dominant-species class id; pixels
    OUTSIDE the seed polygon are 255 (nodata/ignore), NOT a fabricated background class
    (positive-only foreground mask, spec 5). Candidates are kept only if the seed species
    covers >= MIN_COVERAGE of the tile, so tiles are spatially homogeneous.
  * Classes = distinct ``especie1`` (dominant species) among the coverage-passing
    candidates, ids assigned by descending frequency (uint8, 255=nodata). ~50-90 species
    (well under the 254 cap; nothing dropped). Class-balanced by dominant species up to
    PER_CLASS=1000 tiles/class, under the 25,000 cap.

TIME: forest type / dominant species is a static, persistent label; the MFE25/IFN4 mapping
spans multiple years (~2007-2018). Per spec 5 (static labels) we anchor a representative
1-year Sentinel-era window (REP_YEAR=2018, within the manifest's 2016-2019 range).
change_time is null.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.spanish_national_forest_inventory_ifn
Idempotent: downloaded API pages are cached and skipped; existing locations/{id}.tif are
skipped on re-run.
"""

import argparse
import json
import multiprocessing
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterator
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

SLUG = "spanish_national_forest_inventory_ifn"
NAME = "Spanish National Forest Inventory (IFN)"

# MITECO public OGC API - Features (no credential / no captcha).
OGC_BASE = (
    "https://wmts.mapama.gob.es/sig-api/ogc/features/v1/"
    "collections/biodiversidad:MFE/items"
)
OGC_COLLECTION_TITLE = (
    "LC.Mapa Forestal de Espana 1:25.000 (MFE25), "
    "Base Cartografica del Cuarto Inventario Forestal Nacional (IFN4)"
)
INFO_URL = (
    "https://www.miteco.gob.es/es/biodiversidad/servicios/banco-datos-naturaleza/"
    "informacion-disponible/mfe25_descargas_ccaa.html"
)
NFI_DOWNLOADER_URL = "https://descargaifn.gsic.uva.es/"

# CQL2 filter selecting large, single-dominant-species forest teselas.
CQL_FILTER = "especie1<>'sin datos' AND superficie_ha>40 AND o1>=70"

TILE = 64
PAGE = 1000
REP_YEAR = 2018  # representative Sentinel-era year (forest type is static)
MIN_COVERAGE = 0.5  # seed species must cover >= this fraction of the tile
PER_CLASS = 1000
MAX_CLASSES = 254  # uint8 cap (0..253, 255=nodata)
MAX_SAMPLES = sampling.MAX_SAMPLES_PER_DATASET  # 25000
NODATA = io.CLASS_NODATA  # 255
CLIP_DEG = 0.02  # ~2.2 km WGS84 window to clip each polygon before reprojection

PAGES_DIR = io.raw_dir(SLUG) / "pages"


# --------------------------------------------------------------------------- download


def _page_url(start_index: int, skip_geometry: bool = False) -> str:
    params = {
        "f": "json",
        "limit": str(PAGE),
        "filter-lang": "cql2-text",
        "filter": CQL_FILTER,
        "sortby": "-superficie_ha",
        "startIndex": str(start_index),
        "properties": "objectid,especie1,o1,superficie_ha,prov_nom,regbio,poligon_origen",
    }
    if skip_geometry:
        params["skipGeometry"] = "true"
    return OGC_BASE + "?" + urllib.parse.urlencode(params)


def _count_matched() -> int:
    url = _page_url(0, skip_geometry=True).replace(f"limit={PAGE}", "limit=1")
    with urllib.request.urlopen(url, timeout=180) as r:
        d = json.load(r)
    return int(d["numberMatched"])


def _download_page(start_index: int) -> str:
    """Download one page of features (with geometry) to a cached JSON file (idempotent)."""
    dst = PAGES_DIR / f"page_{start_index:07d}.json"
    if dst.exists():
        try:
            with dst.open() as f:
                json.load(f)
            return str(dst)
        except Exception:
            dst.unlink()  # corrupt cache -> re-fetch
    url = _page_url(start_index, skip_geometry=False)
    last_err: Exception | None = None
    for _ in range(4):
        try:
            download.download_http(url, dst, skip_existing=False, timeout=300)
            with dst.open() as f:
                json.load(f)  # validate
            return str(dst)
        except Exception as e:  # transient server / truncation
            last_err = e
            if dst.exists():
                dst.unlink()
    raise RuntimeError(
        f"TRANSIENT: failed to fetch page startIndex={start_index}: {last_err}"
    )


def download_pages(workers: int) -> tuple[list[str], int]:
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    with (io.raw_dir(SLUG) / "SOURCE.txt").open("w") as f:
        f.write(
            "Spanish National Forest Inventory (IFN) - dominant tree species.\n"
            "Product used: Mapa Forestal de Espana 1:25.000 (MFE25), the cartographic base "
            "of the 4th National Forest Inventory (IFN4).\n"
            f"OGC API - Features collection biodiversidad:MFE ({OGC_COLLECTION_TITLE}).\n"
            f"Endpoint: {OGC_BASE}\n"
            f"CQL2 filter: {CQL_FILTER}\n"
            f"Info page: {INFO_URL}\n"
            f"NFI Downloader (plot product, not used - fuzzed coords): {NFI_DOWNLOADER_URL}\n"
            "License: Spanish government open data (MITECO). No credential required.\n"
            "Note: per-province MFE25 shapefile downloads on mapama.gob.es are gated behind "
            "a Google reCAPTCHA; the OGC API serves the identical data without captcha.\n"
        )
    n = _count_matched()
    starts = list(range(0, n, PAGE))
    print(f"MFE candidates matched: {n} -> {len(starts)} pages")
    with multiprocessing.Pool(workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(
                p, _download_page, [dict(start_index=s) for s in starts]
            ),
            total=len(starts),
            desc="download pages",
        ):
            pass
    paths = sorted(str(x) for x in PAGES_DIR.glob("page_*.json"))
    return paths, n


# --------------------------------------------------------------------------- candidates


def _iter_feature_args(page_paths: list[str]) -> Iterator[dict[str, Any]]:
    for pp in page_paths:
        with open(pp) as f:
            d = json.load(f)
        for feat in d.get("features", []):
            if feat.get("geometry") is None:
                continue
            yield dict(feat=feat)


def _candidate_task(feat: dict[str, Any]) -> dict[str, Any] | None:
    """Build one homogeneous candidate tile from a seed MFE polygon (WGS84)."""
    from shapely.geometry import box

    from olmoearth_pretrain.open_set_segmentation_data.rasterize import geom_to_pixels

    props = feat["properties"]
    especie = (props.get("especie1") or "").strip()
    if not especie or especie == "sin datos":
        return None
    geom = shapely.geometry.shape(feat["geometry"])
    if geom.is_empty:
        return None
    if not geom.is_valid:
        geom = geom.buffer(0)
        if geom.is_empty:
            return None
    rp = geom.representative_point()
    lon, lat = float(rp.x), float(rp.y)
    proj = io.utm_projection_for_lonlat(lon, lat)
    _, col, row = io.lonlat_to_utm_pixel(lon, lat, proj)
    bounds = io.centered_bounds(col, row, TILE, TILE)

    # Clip to a small WGS84 window around the seed point before reprojecting (cheap).
    d = CLIP_DEG
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
        "especie1": especie,
        "coverage": round(coverage, 4),
        "source_id": str(props.get("poligon_origen") or props.get("objectid")),
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
    cls = rec["class_id"]
    label = rasterize_shapes(
        [(clip, cls)], bounds, fill=NODATA, dtype="uint8", all_touched=True
    )[0]
    present = sorted(int(v) for v in np.unique(label) if int(v) != NODATA)
    io.write_label_geotiff(SLUG, sample_id, label, proj, bounds, nodata=NODATA)
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
    return cls, bool((label == NODATA).any())


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--dl_workers", type=int, default=12)
    ap.add_argument("--per_class", type=int, default=PER_CLASS)
    args = ap.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    page_paths, n_matched = download_pages(args.dl_workers)
    io.check_disk()

    # Candidate tiles (parallel; stream features from cached pages -> bounded memory).
    candidates: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _candidate_task, _iter_feature_args(page_paths)),
            total=n_matched,
            desc="candidates",
        ):
            if res is not None:
                candidates.append(res)
    print(f"{len(candidates)} homogeneous candidate tiles (coverage >= {MIN_COVERAGE})")

    # Class scheme: distinct especie1 among candidates, ids by descending frequency.
    freq = Counter(c["especie1"] for c in candidates)
    ordered = [name for name, _ in freq.most_common()]
    dropped_species: list[str] = []
    if len(ordered) > MAX_CLASSES:
        dropped_species = ordered[MAX_CLASSES:]
        ordered = ordered[:MAX_CLASSES]
    name_to_id = {name: i for i, name in enumerate(ordered)}
    keep = set(name_to_id)
    candidates = [c for c in candidates if c["especie1"] in keep]
    for c in candidates:
        c["class_id"] = name_to_id[c["especie1"]]
    print(
        f"{len(name_to_id)} species classes; dropped {len(dropped_species)} rare over cap"
    )

    # Class-balanced selection by dominant species (spec 5).
    selected = sampling.balance_by_class(
        candidates, "class_id", per_class=args.per_class, total_cap=MAX_SAMPLES
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
    sel_counts = Counter(r["class_id"] for r in selected)
    classes_meta = [
        {"id": i, "name": name, "description": None}
        for name, i in sorted(name_to_id.items(), key=lambda kv: kv[1])
    ]

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "MITECO - Mapa Forestal de Espana 1:25.000 (MFE25 / IFN4 base)",
            "license": "Spanish government open data (MITECO)",
            "provenance": {
                "url": INFO_URL,
                "ogc_api": OGC_BASE,
                "ogc_collection": "biodiversidad:MFE",
                "ogc_collection_title": OGC_COLLECTION_TITLE,
                "cql2_filter": CQL_FILTER,
                "nfi_downloader_plot_product_not_used": NFI_DOWNLOADER_URL,
                "have_locally": False,
                "annotation_method": (
                    "photointerpretation at 1:25,000 with field checking; MFE25 is the "
                    "cartographic base of the 4th Spanish National Forest Inventory (IFN4)"
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes_meta,
            "nodata_value": NODATA,
            "num_samples": n_written,
            "product_choice": (
                "MFE25 dominant-species polygons (IFN4 base). The IFN field-plot product "
                "(descargaifn.gsic.uva.es) was NOT used: its public plot coordinates are "
                "degraded to ~1 km, so species are not observable at 10 m."
            ),
            "tile_size": TILE,
            "min_coverage": MIN_COVERAGE,
            "n_candidate_polygons_matched": n_matched,
            "n_homogeneous_candidates": len(candidates),
            "dropped_species_over_254_cap": dropped_species,
            "selected_tiles_per_class": {
                classes_meta[c]["name"]: sel_counts.get(c, 0)
                for c in range(len(classes_meta))
            },
            "written_tiles_per_class": {
                classes_meta[c]["name"]: written_by_class.get(c, 0)
                for c in range(len(classes_meta))
            },
            "tiles_with_ignore_border": ignore_tiles,
            "rep_year": REP_YEAR,
            "notes": (
                "Dominant-tree-species segmentation from MFE25 (IFN4 base) polygons via "
                "MITECO's public OGC API - Features (collection biodiversidad:MFE). 64x64 "
                "uint8 tiles in local UTM at 10 m. Each tile seeds on one large (>40 ha), "
                "single-dominant-species (o1 >= 70) forest tesela at its interior "
                "representative point; pixels outside the seed polygon are 255 "
                "(nodata/ignore), NOT a background class (positive-only foreground mask, "
                f"spec 5). Tiles kept only if the seed species covers >= {MIN_COVERAGE} of "
                "the tile (homogeneous). Classes = distinct especie1 (dominant species), "
                "ids by descending frequency. Static/persistent label -> representative "
                f"1-year window (REP_YEAR={REP_YEAR}); change_time null. Class-balanced by "
                f"dominant species up to {args.per_class}/class (cap {MAX_SAMPLES}). "
                "Per-province MFE25 shapefiles are behind a reCAPTCHA; the OGC API serves "
                "the identical data without a credential."
            ),
        },
    )
    print(
        "selected tiles per class:",
        {
            classes_meta[c]["name"]: sel_counts.get(c, 0)
            for c in range(len(classes_meta))
        },
    )
    print("total tif on disk:", n_written, "| tiles with ignore border:", ignore_tiles)
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=n_written
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
