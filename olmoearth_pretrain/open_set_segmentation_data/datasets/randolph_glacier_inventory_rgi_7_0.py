"""Process the Randolph Glacier Inventory (RGI 7.0) glacier product into open-set
segmentation label patches.

Source: NSIDC nsidc-0770 v7 (RGI 7.0 Consortium 2023, doi:10.5067/f6jmovy5navz),
coordinated with GLIMS. We use the **glacier product (G)** -- individual, manually
delineated glacier outline polygons. It is distributed as one zipped ESRI shapefile per
first-order region (19 regions, ~274k glaciers globally), each in EPSG:4326, downloaded
over HTTP from the NSIDC DAAC (NASA Earthdata / URS OAuth; credentials from
.env -> ~/.netrc).

Task: **binary per-pixel segmentation, glacier vs background.**
  0 background  -- non-glacier terrain (rock, snow-free ground, water, vegetation)
  1 glacier     -- inside an RGI glacier outline
The glacier outline is a true boundary against *observable* non-glacier terrain, so this
is a genuine two-class segmentation (not a positive-only presence mask). Each tile is
rasterized with ALL glacier polygons intersecting it (via a spatial index), so adjacent
glaciers in the same tile are correctly labeled -- not just the one the tile is centered
on.

Why not terminus type: the manifest notes "glacier (with terminus-type attributes)", but
in RGI 7.0 the `term_type` attribute is "not assigned" (code 9) for 99.4% of glaciers
(only 1561 of 274531 carry a real terminus code), so it cannot support a class scheme.
We record `term_type` per sample (source_id) for provenance instead.

Sampling (bounded regional, spec 5): glaciers >= 0.1 km^2 (drops sub-resolution slivers
and improves temporal stability) are sampled round-robin across all 19 regions for
geographic diversity, up to 1000 glacier-centered tiles (the per-class cap; background
co-occurs in most tiles). Each glacier's centroid (cenlon/cenlat) is the tile center; the
64x64 UTM 10 m window spans 640 m so small glaciers show their boundary against background
while large glaciers yield glacier-filled tiles.

Time range (spec 5, static/persistent label): RGI 7.0 is the nominal-2000 inventory --
outline source dates are ~99.9% pre-2016 (mean year 2001). Glacier extent is a slowly
changing, persistent feature, so per the task ("static extent -> representative
Sentinel-era 1-year window") every sample is assigned a uniform 1-year window in the
Sentinel era (2020). The original outline source date (RGI2000 acquisition year) is
recorded per sample in source_id. Caveat: glaciers -- especially small ones -- have
retreated somewhat since ~2000, so a 2020 image may show a modestly smaller glacier than
the RGI2000 outline; the >= 0.1 km^2 floor limits (but does not eliminate) this mismatch.
"""

import argparse
import multiprocessing
import os
import random
import warnings
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import tqdm
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import download, io
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)

warnings.filterwarnings("ignore")

SLUG = "randolph_glacier_inventory_rgi_7_0"
NAME = "Randolph Glacier Inventory (RGI 7.0)"
RAW = io.raw_dir(SLUG)

NSIDC_BASE = (
    "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/"
    "regional_files/RGI2000-v7.0-G"
)

REGIONS = [
    "01_alaska",
    "02_western_canada_usa",
    "03_arctic_canada_north",
    "04_arctic_canada_south",
    "05_greenland_periphery",
    "06_iceland",
    "07_svalbard_jan_mayen",
    "08_scandinavia",
    "09_russian_arctic",
    "10_north_asia",
    "11_central_europe",
    "12_caucasus_middle_east",
    "13_central_asia",
    "14_south_asia_west",
    "15_south_asia_east",
    "16_low_latitudes",
    "17_southern_andes",
    "18_new_zealand",
    "19_subantarctic_antarctic_islands",
]

GLACIER_CLASS = 1
BACKGROUND_CLASS = 0
PER_CLASS = 1000
TILE = 64
YEAR = 2020
MIN_AREA_KM2 = 0.1  # drop sub-resolution slivers; improves temporal stability

CLASSES = [
    (
        0,
        "background",
        "Non-glacier terrain (bedrock, snow-free ground, seasonal snow, water, "
        "vegetation) outside every RGI glacier outline.",
    ),
    (
        1,
        "glacier",
        "Land ice inside a manually delineated RGI 7.0 glacier outline (glacier product "
        "G; RGI2000 nominal-2000 extent, coordinated with GLIMS).",
    ),
]
ID_TO_NAME = {cid: n for cid, n, _d in CLASSES}


def _region_stem(region: str) -> str:
    return f"RGI2000-v7.0-G-{region}"


def _shp_path(region: str) -> str:
    return (RAW / region / f"{_region_stem(region)}.shp").path


def _attr_csv_path(region: str) -> str:
    return (RAW / region / f"{_region_stem(region)}-attributes.csv").path


# --------------------------------------------------------------------------- download


def download_region(region: str) -> None:
    """Download + extract one regional G shapefile zip (idempotent)."""
    stem = _region_stem(region)
    if os.path.exists(_shp_path(region)):
        return
    zip_dst = RAW / f"{stem}.zip"
    url = f"{NSIDC_BASE}/{stem}.zip"
    download.download_earthdata(url, zip_dst)
    download.extract_zip(zip_dst, RAW / region)


# --------------------------------------------------------------------------- scan


def _scan_one(region: str) -> list[dict[str, Any]]:
    """Read one region's attributes CSV -> lightweight per-glacier candidate records."""
    import pandas as pd

    path = _attr_csv_path(region)
    if not os.path.exists(path):
        return []
    df = pd.read_csv(
        path,
        usecols=["rgi_id", "cenlon", "cenlat", "area_km2", "term_type", "src_date"],
    )
    df = df[df["area_km2"] >= MIN_AREA_KM2]
    recs: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        lon, lat = float(row.cenlon), float(row.cenlat)
        if not (np.isfinite(lon) and np.isfinite(lat)):
            continue
        recs.append(
            {
                "region": region,
                "rgi_id": str(row.rgi_id),
                "lon": lon,
                "lat": lat,
                "area_km2": float(row.area_km2),
                "term_type": int(row.term_type),
                "src_date": str(row.src_date),
            }
        )
    return recs


def scan() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with multiprocessing.Pool(min(len(REGIONS), 19)) as p:
        for recs in tqdm.tqdm(
            star_imap_unordered(p, _scan_one, [dict(region=r) for r in REGIONS]),
            total=len(REGIONS),
            desc="scan",
        ):
            out.extend(recs)
    return out


# ---------------------------------------------------------------------- selection


def sample_round_robin(
    cands: list[dict[str, Any]], target: int, seed: int = 42
) -> list[dict[str, Any]]:
    """Round-robin across regions to maximize geographic diversity, up to ``target``."""
    by_reg: dict[str, list] = defaultdict(list)
    for c in cands:
        by_reg[c["region"]].append(c)
    rng = random.Random(seed)
    for v in by_reg.values():
        rng.shuffle(v)
    regs = sorted(by_reg)
    out: list[dict[str, Any]] = []
    idx = {r: 0 for r in regs}
    while len(out) < target:
        progressed = False
        for r in regs:
            if idx[r] < len(by_reg[r]):
                out.append(by_reg[r][idx[r]])
                idx[r] += 1
                progressed = True
                if len(out) >= target:
                    break
        if not progressed:
            break
    return out


# --------------------------------------------------------------------------- write


def _write_region(
    region: str, recs: list[dict[str, Any]]
) -> list[tuple[str, list[int]]]:
    """Rasterize + write all selected samples for one region.

    For each sample tile (centered on a glacier centroid) we rasterize every glacier
    polygon in the region that intersects the tile footprint, so the binary mask is
    correct even where glaciers are dense. Returns (sample_id, classes_present).
    """
    import geopandas as gpd
    import shapely
    from rasterio.crs import CRS as RioCRS

    gdf = gpd.read_file(_shp_path(region))
    src_proj = Projection(RioCRS.from_wkt(gdf.crs.to_wkt()), 1, 1)
    geom_by_id = dict(zip(gdf["rgi_id"], gdf.geometry))
    sindex = gdf.sindex

    written: list[tuple[str, list[int]]] = []
    for rec in recs:
        sample_id = rec["sample_id"]
        tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
        if tif.exists():
            written.append((sample_id, rec.get("classes_present", [GLACIER_CLASS])))
            continue

        center_geom = geom_by_id.get(rec["rgi_id"])
        if center_geom is None or center_geom.is_empty:
            continue

        lon, lat = rec["lon"], rec["lat"]
        proj, col, row = io.lonlat_to_utm_pixel(lon, lat)
        bounds = io.centered_bounds(col, row, TILE, TILE)

        # Query neighbor glaciers within a generous lon/lat box around the center
        # (half-tile is 320 m; use ~2 km so all overlapping polygons are captured).
        dlat = 0.02
        dlon = dlat / max(0.05, np.cos(np.radians(lat)))
        box = shapely.box(lon - dlon, lat - dlat, lon + dlon, lat + dlat)
        hits = sindex.query(box, predicate="intersects")

        shapes: list[tuple[Any, int]] = []
        seen: set[int] = set()
        # Always rasterize the center glacier itself.
        shapes.append((geom_to_pixels(center_geom, src_proj, proj), GLACIER_CLASS))
        for j in hits:
            j = int(j)
            g = gdf.geometry.iloc[j]
            if g is None or g.is_empty or gdf["rgi_id"].iloc[j] == rec["rgi_id"]:
                continue
            if j in seen:
                continue
            seen.add(j)
            shapes.append((geom_to_pixels(g, src_proj, proj), GLACIER_CLASS))

        arr = rasterize_shapes(
            shapes, bounds, fill=BACKGROUND_CLASS, dtype="uint8", all_touched=False
        )
        if not np.any(arr == GLACIER_CLASS):
            continue  # degenerate (tiny polygon fell between pixel centers)

        classes_present = sorted(int(v) for v in np.unique(arr))
        io.write_label_geotiff(
            SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA
        )
        src_id = (
            f"{rec['rgi_id']}@src_date={rec['src_date']};term_type={rec['term_type']}"
        )
        io.write_sample_json(
            SLUG,
            sample_id,
            proj,
            bounds,
            io.year_range(YEAR),
            source_id=src_id,
            classes_present=classes_present,
        )
        written.append((sample_id, classes_present))
    return written


# ---------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--target", type=int, default=PER_CLASS)
    args = parser.parse_args()

    io.check_disk()
    RAW.mkdir(parents=True, exist_ok=True)

    # 1. Download all 19 regional G shapefiles.
    print(f"downloading {len(REGIONS)} RGI 7.0 regional glacier shapefiles ...")
    with multiprocessing.Pool(min(len(REGIONS), 19)) as p:
        list(
            tqdm.tqdm(
                star_imap_unordered(
                    p, download_region, [dict(region=r) for r in REGIONS]
                ),
                total=len(REGIONS),
                desc="download",
            )
        )
    with (RAW / "SOURCE.txt").open("w") as f:
        f.write(
            "RGI 7.0 glacier product (G), NSIDC nsidc-0770 v7 "
            "(doi:10.5067/f6jmovy5navz).\n"
            f"Per-region shapefiles from {NSIDC_BASE}/RGI2000-v7.0-G-<region>.zip\n"
            "(NASA Earthdata / URS OAuth; ~/.netrc).\n"
        )

    io.check_disk()

    # 2. Scan candidates and sample round-robin across regions.
    recs = scan()
    print(
        f"scanned {len(recs)} glaciers (area >= {MIN_AREA_KM2} km^2) across "
        f"{len({r['region'] for r in recs})} regions"
    )
    selected = sample_round_robin(recs, args.target)
    by_reg = Counter(r["region"] for r in selected)
    print(
        f"selected {len(selected)} glacier-centered tiles; per-region: {dict(by_reg)}"
    )

    # Deterministic ordering -> stable sample ids (idempotent reruns).
    selected.sort(key=lambda r: r["rgi_id"])
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"

    io.check_disk()

    # 3. Rasterize + write, parallel over regions.
    by_region: dict[str, list] = defaultdict(list)
    for r in selected:
        by_region[r["region"]].append(r)
    jobs = [dict(region=reg, recs=rs) for reg, rs in by_region.items()]

    written: list[tuple[str, list[int]]] = []
    with multiprocessing.Pool(min(args.workers, len(jobs))) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_region, jobs), total=len(jobs), desc="write"
        ):
            written.extend(res)

    # Class (pixel-presence) tile counts.
    tile_counts = Counter()
    for _sid, classes in written:
        for c in classes:
            tile_counts[c] += 1
    print(f"wrote {len(written)} tiles")
    for cid, name, _d in CLASSES:
        print(f"  class {cid} ({name}): present in {tile_counts.get(cid, 0)} tiles")

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "NSIDC / GLIMS (RGI 7.0)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://nsidc.org/data/nsidc-0770/versions/7",
                "have_locally": False,
                "annotation_method": "manual delineation / photointerpretation "
                "(RGI 7.0 Consortium 2023, coordinated with GLIMS)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": cid, "name": name, "description": desc}
                for cid, name, desc in CLASSES
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(written),
            "tile_class_counts": {
                ID_TO_NAME[cid]: tile_counts.get(cid, 0) for cid, *_ in CLASSES
            },
            "regional_tile_counts": dict(by_reg),
            "notes": (
                "Binary glacier/background 64x64 UTM 10 m segmentation tiles from the RGI "
                "7.0 glacier product (G). Each tile is centered on a glacier centroid and "
                "rasterizes ALL glacier polygons intersecting the tile (background = 0, "
                "glacier = 1). Round-robin sampled across all 19 RGI regions; glaciers "
                f">= {MIN_AREA_KM2} km^2 only. Uniform {YEAR} 1-year window (RGI 7.0 is the "
                "nominal-2000 inventory; outlines are ~99.9% pre-2016 but glacier extent is "
                "treated as a persistent/static label per the task -> Sentinel-era window). "
                "term_type is 'not assigned' for 99.4% of glaciers so it is not used as a "
                "class; recorded per sample in source_id. nodata=255 is declared but unused "
                "(every pixel is class 0 or 1)."
            ),
        },
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
