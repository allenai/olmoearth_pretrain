"""Process GRID3 Settlement Extents into 3-class settlement-type label tiles.

Source: GRID3 (Geo-Referenced Infrastructure and Demographic Data for Development;
CIESIN/Columbia University, Novel-T, WorldPop, UNFPA, Flowminder), Settlement Extents
v3.0 / v3.1. Distributed per country as OGC GeoPackages on the Humanitarian Data Exchange
(https://data.humdata.org/organization/grid3), license CC-BY-SA-4.0 (no account needed).

Each country's settlement-extents layer is a set of settlement polygons, derived by
aggregating open building-footprint data (Google/Microsoft/OSM) onto a 3-arc-second
(~100 m) grid, delineating settled contours, and classifying each resulting polygon by
building count / built-up area into three settlement types (codebook field ``type``):

    Built-up Area (BUA)        >= 40 ha with >= 13 buildings/ha (urban, street grid)
    Small Settlement Area (SSA) >= 50 buildings, not a BUA (semi-urban / peri-urban)
    Hamlet                     up to 49 buildings (rural, low-density)

This is a LARGE regional product (>15M settlements across 50 Sub-Saharan countries), so we
do BOUNDED sampling (spec section 5): we download the settlement-extents GeoPackages for a
representative set of 6 countries spanning West / East / Central / Southern Africa and the
Sahel, and draw a class-balanced sample from them. We do NOT attempt continental coverage.

    NGA (Nigeria, v3.1)    West Africa
    SEN (Senegal, v3.0)    West Africa / Sahel
    KEN (Kenya, v3.0)      East Africa
    TZA (Tanzania, v3.0)   East Africa
    COD (DR Congo, v3.1)   Central Africa
    ZMB (Zambia, v3.0)     Southern Africa

Class scheme (positive-only; settlement types are foreground land cover — non-settlement
is NOT a fabricated background class, it is nodata/ignore per spec section 5):

    0 = built-up area          (BUA)
    1 = small settlement area  (SSA)
    2 = hamlet
    255 = nodata / ignore      (all pixels outside any settlement polygon)

Rasterization (label_type: polygons): each selected polygon -> one 64x64 UTM 10 m tile
(640 m) centered on the polygon (centroid, or an interior representative point for concave
shapes). EVERY settlement polygon intersecting the tile bbox is burned in with its own type
id (all_touched=True so tiny hamlets survive at 10 m); all other pixels are 255. BUAs are
typically larger than a 640 m tile, so a BUA tile captures a central all-built-up window;
hamlets are small, so a hamlet tile is a few positive pixels in a mostly-nodata patch (the
pretraining assembly step supplies negatives from other datasets).

Sampling: candidate placement points are pooled across the 6 countries (capped per
country/class to bound memory) and selected class-balanced at up to 1000 tiles per class
(spec section 5), so BUAs (globally rare: ~3.5k across these 6 countries) reach parity with
the abundant hamlets/SSAs. Tiles are counted by the centered polygon's type.

Time range: settlement extents are a persistent land-use footprint. The v3.x product was
derived in 2024 from building footprints spanning ~2016-2023 (Google 2023, Microsoft
2014-2023, WSF 2016, GHSL 2018), and settlements persist across the Sentinel era. We assign
each tile a 1-year static-label window on 2021 (a representative year within the manifest's
2016-2021 span; settlements are persistent so any Sentinel-era window shows them). No
change_time (this is presence/type classification, not a dated change event).

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.grid3_settlement_extents
Idempotent: existing locations/{id}.tif are skipped; raw zips are downloaded+extracted once.
"""

import argparse
import glob
import multiprocessing
import os
import random
import zipfile
import zlib
from collections import Counter
from typing import Any

import numpy as np
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import (
    download,
    io,
    manifest,
    sampling,
)
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)

SLUG = "grid3_settlement_extents"
NAME = "GRID3 Settlement Extents"

TILE = 64  # 64 px * 10 m = 640 m tile.
PER_CLASS = 1000  # spec section 5: up to 1000 locations per class.
CAP_PER_COUNTRY_CLASS = (
    3000  # bound the candidate pool per country/class before balancing.
)
REP_YEAR = 2021  # representative 1-year static window (persistent land use).
QUERY_MARGIN_DEG_BASE = (
    2 * TILE * io.RESOLUTION / 111320.0
)  # ~2x tile in latitude degrees.
SEED = 42

# HDX settlement-extents GeoPackage zips (CC-BY-SA-4.0). name -> (iso3, url).
COUNTRIES = {
    "grid3_nga_settlement_extents_v3_1_gpkg.zip": (
        "NGA",
        "https://data.humdata.org/dataset/af838671-b9a6-4ae9-8ed5-eea750b05597/resource/"
        "0a22d6fc-7f1f-4f50-aead-09ef7be0455d/download/grid3_nga_settlement_extents_v3_1_gpkg.zip",
    ),
    "grid3_sen_settlement_extents_v3_0.zip": (
        "SEN",
        "https://data.humdata.org/dataset/51abf6bd-c20c-4da0-997f-1896ddfd7e52/resource/"
        "7321a6fe-ec39-47ea-bc7f-1278ce259504/download/grid3_sen_settlement_extents_v3_0.zip",
    ),
    "grid3_ken_settlement_extents_v3_0.zip": (
        "KEN",
        "https://data.humdata.org/dataset/f5714c49-51ce-40a8-9081-83e4d71eb787/resource/"
        "affb8c36-0db4-41f1-97b8-aab47027667e/download/grid3_ken_settlement_extents_v3_0.zip",
    ),
    "grid3_tza_settlement_extents_v3_0.zip": (
        "TZA",
        "https://data.humdata.org/dataset/7416c345-1023-4c0e-9663-4fa6b0825972/resource/"
        "de458d92-eef7-4e0a-8cfb-3d1461b48dd7/download/grid3_tza_settlement_extents_v3_0.zip",
    ),
    "grid3_cod_settlement_extents_v3_1_gpkg.zip": (
        "COD",
        "https://data.humdata.org/dataset/335743dd-f27f-4382-9586-c5bfdf281aa0/resource/"
        "cd6afdc2-fbd1-43cb-bc92-d1c7b3aed0cc/download/grid3_cod_settlement_extents_v3_1_gpkg.zip",
    ),
    "grid3_zmb_settlement_extents_v3_0.zip": (
        "ZMB",
        "https://data.humdata.org/dataset/ad39df0c-4e3e-49ec-8ef8-adf75a3dde09/resource/"
        "6f1dc3c1-0356-4509-86d1-46b6f0c2dd27/download/grid3_zmb_settlement_extents_v3_0.zip",
    ),
}

# Codebook ``type`` string -> class id.
TYPE_TO_ID = {
    "built-up area": 0,
    "small settlement area": 1,
    "hamlet": 2,
}
CLASSES = [
    (
        "built-up area",
        "Built-up area (BUA): an area of urbanization with moderately-to-densely-spaced "
        "buildings and a visible grid of streets and blocks; >= 40 ha (400,000 m2) with a "
        "building density of >= 13 buildings/ha. GRID3 settlement-extent polygon classified "
        "by building density and area.",
    ),
    (
        "small settlement area",
        "Small settlement area (SSA): an assemblage of >= 50 buildings not classified as a "
        "BUA; typically semi-urban / peri-urban settlement, sometimes from urban sprawl. "
        "GRID3 settlement-extent polygon.",
    ),
    (
        "hamlet",
        "Hamlet: a settlement with a building count of up to 49; typically rural, "
        "low-density and isolated (often hard-to-reach). GRID3 settlement-extent polygon.",
    ),
]


def _type_to_id(s: Any) -> int | None:
    if s is None:
        return None
    return TYPE_TO_ID.get(str(s).strip().lower())


def _gpkg_path(iso3: str) -> str | None:
    hits = sorted(
        glob.glob(
            os.path.join(str(io.raw_dir(SLUG)), f"*{iso3.lower()}*", f"*{iso3}*.gpkg")
        )
    )
    return hits[0] if hits else None


def download_and_extract() -> None:
    """Download the 6 country extents zips and extract their GeoPackages (idempotent)."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    for fname, (iso3, url) in COUNTRIES.items():
        if _gpkg_path(iso3):
            continue
        zip_path = raw / fname
        print(f"downloading {iso3} ...")
        download.download_http(url, zip_path)
        dst = raw / fname[:-4]
        dst.mkdir(parents=True, exist_ok=True)
        print(f"extracting {fname} ...")
        with zipfile.ZipFile(str(zip_path)) as zf:
            zf.extractall(str(dst))
        if not _gpkg_path(iso3):
            raise RuntimeError(f"no .gpkg extracted for {iso3}")


# --------------------------------------------------------------------------- scan


def scan_country(iso3: str) -> list[dict[str, Any]]:
    """Read a country's polygons; return class-balanced-capped placement candidates.

    Each candidate: {iso3, path, lon, lat, stype, src_id}. Placement point is the polygon
    centroid, or an interior representative point when the centroid falls outside a concave
    polygon. Per (country, class) we keep at most CAP_PER_COUNTRY_CLASS candidates (random,
    seeded) to bound the pool; rare BUAs are kept in full.
    """
    import shapely

    path = _gpkg_path(iso3)
    import pyogrio

    gdf = pyogrio.read_dataframe(path, columns=["type"], read_geometry=True)
    geoms = gdf.geometry.values
    types = gdf["type"].values

    cent = shapely.centroid(geoms)
    inside = shapely.contains(geoms, cent)
    lon = np.array([c.x for c in cent], dtype="float64")
    lat = np.array([c.y for c in cent], dtype="float64")
    # Fix placement for the few polygons whose centroid is outside the shape.
    for i in np.nonzero(~inside)[0]:
        try:
            rp = shapely.force_2d(geoms[i]).representative_point()
            lon[i], lat[i] = rp.x, rp.y
        except Exception:
            pass

    by_class: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for i in range(len(geoms)):
        cid = _type_to_id(types[i])
        if cid is None or not np.isfinite(lon[i]) or not np.isfinite(lat[i]):
            continue
        by_class[cid].append(i)

    rng = random.Random(zlib.crc32(iso3.encode()) ^ SEED)
    out: list[dict[str, Any]] = []
    for cid, idxs in by_class.items():
        if len(idxs) > CAP_PER_COUNTRY_CLASS:
            idxs = rng.sample(idxs, CAP_PER_COUNTRY_CLASS)
        for i in idxs:
            out.append(
                {
                    "iso3": iso3,
                    "path": path,
                    "lon": float(lon[i]),
                    "lat": float(lat[i]),
                    "stype": cid,
                    "src_id": f"{iso3}:{cid}:{int(i)}",
                }
            )
    return out


# --------------------------------------------------------------------------- write


def _rasterize_tile(rec: dict[str, Any]):
    import pyogrio
    import shapely

    lon, lat = rec["lon"], rec["lat"]
    proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
    bounds = io.centered_bounds(col, row, TILE, TILE)

    mlat = QUERY_MARGIN_DEG_BASE
    mlon = mlat / max(np.cos(np.radians(lat)), 0.1)
    sub = pyogrio.read_dataframe(
        rec["path"],
        columns=["type"],
        read_geometry=True,
        bbox=(lon - mlon, lat - mlat, lon + mlon, lat + mlat),
    )

    shapes: list[tuple[Any, int]] = []
    for geom, tval in zip(sub.geometry.values, sub["type"].values):
        cid = _type_to_id(tval)
        if cid is None or geom is None or geom.is_empty:
            continue
        px = geom_to_pixels(shapely.force_2d(geom), WGS84_PROJECTION, proj)
        if px.is_empty:
            continue
        shapes.append((px, cid))

    if shapes:
        label = rasterize_shapes(
            shapes, bounds, fill=io.CLASS_NODATA, dtype="uint8", all_touched=True
        )[0]
    else:
        label = np.full((TILE, TILE), io.CLASS_NODATA, dtype=np.uint8)
    return proj, bounds, label


def _write_one(rec: dict[str, Any]) -> str | None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return "skip"

    proj, bounds, label = _rasterize_tile(rec)
    present = sorted(int(v) for v in np.unique(label) if v != io.CLASS_NODATA)
    if not present:
        return "empty"

    io.write_label_geotiff(SLUG, sample_id, label, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(REP_YEAR),
        source_id=rec["src_id"],
        classes_present=present,
    )
    return f"class{rec['stype']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--per-class", type=int, default=PER_CLASS)
    args = ap.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    download_and_extract()
    isos = [iso for (iso, _u) in COUNTRIES.values()]
    print(f"scanning {len(isos)} countries: {isos}")

    with multiprocessing.Pool(min(args.workers, len(isos))) as p:
        parts = list(
            tqdm.tqdm(
                star_imap_unordered(p, scan_country, [dict(iso3=i) for i in isos]),
                total=len(isos),
                desc="scan",
            )
        )
    candidates: list[dict[str, Any]] = [r for part in parts for r in part]
    pool_counts = Counter(r["stype"] for r in candidates)
    print(f"candidate pool: {len(candidates)} " + str(dict(pool_counts)))

    io.check_disk()

    selected = sampling.balance_by_class(
        candidates, key="stype", per_class=args.per_class, seed=SEED
    )
    for j, rec in enumerate(sorted(selected, key=lambda r: (r["stype"], r["src_id"]))):
        rec["sample_id"] = f"{j:06d}"
    sel_counts = Counter(r["stype"] for r in selected)
    print(f"selected {len(selected)} tiles " + str(dict(sel_counts)))

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
    print("write results:", dict(counts))
    print("total tif on disk:", n_written)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "GRID3 (CIESIN/Columbia University) Settlement Extents v3.0/v3.1",
            "license": "CC-BY-SA-4.0",
            "provenance": {
                "url": "https://data.humdata.org/organization/grid3",
                "have_locally": False,
                "annotation_method": (
                    "model-derived: open building footprints (Google/Microsoft/OSM) "
                    "aggregated on a ~100 m grid, settled contours delineated, polygons "
                    "classified into BUA/SSA/hamlet by building density and area"
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_written,
            "notes": (
                "3-class settlement-type polygon segmentation from GRID3 Settlement Extents "
                "v3.0/v3.1 (CC-BY-SA-4.0, HDX). Bounded sample of 6 Sub-Saharan countries "
                "(NGA, SEN, KEN, TZA, COD, ZMB) spanning West/East/Central/Southern Africa "
                "and the Sahel; NOT continental coverage. 64x64 uint8 tiles in local UTM at "
                "10 m; class 0=built-up area, 1=small settlement area, 2=hamlet, 255=nodata "
                "(all non-settlement pixels — positive-only per spec section 5, NO synthetic "
                "negatives; assembly adds negatives from other datasets). One tile per "
                "selected polygon, centered on it; every settlement polygon intersecting the "
                "tile is burned in with its type id (all_touched=True). Class-balanced at up "
                "to 1000 tiles/class (BUAs are globally rare). Time range = 1-year window on "
                "2021 (persistent land use; product derived 2024 from ~2016-2023 building "
                "footprints)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=n_written
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
