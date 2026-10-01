"""Process the US National Wetlands Inventory (NWI) into wetland-type segmentation tiles.

Source: US Fish & Wildlife Service, National Wetlands Inventory -- the authoritative
national wetland polygon layer for the United States, produced by photointerpretation and
classified with the full Cowardin hierarchy. Public domain. Data are distributed as
per-state File Geodatabase downloads (no credentials) at:

    https://documentst.ecosphere.fws.gov/wetlands/data/State-Downloads/{ST}_geodatabase_wetlands.zip

The national layer is enormous (hundreds of millions of polygons across all states), so per
spec  5 (bounded sampling of large coverage products) we download only a **bounded, diverse
set of states** and sample tiles from them -- NOT all of CONUS. States used (chosen to cover
every Cowardin system and multiple biogeographic settings):

    LA  Louisiana  -- Gulf coast: Marine, Estuarine, Riverine, Lacustrine, Palustrine
    FL  Florida    -- Everglades + coasts: Marine, Estuarine, freshwater forested/emergent
    ND  North Dakota -- Prairie Pothole Region: Palustrine emergent, ponds, lakes
    NC  North Carolina -- Atlantic coastal plain: estuarine, riverine, forested swamps

Each state GDB has a ``{ST}_Wetlands`` polygon layer (EPSG:5070 CONUS Albers) with fields
``ATTRIBUTE`` (the raw Cowardin code, e.g. PEM1C, E2EM1P, R2UBH) and ``WETLAND_TYPE`` (NWI's
own simplified legend derived from the Cowardin code). We use ``WETLAND_TYPE`` as a
manageable, semantically-clean class scheme (8 classes):

    0 Freshwater Emergent Wetland
    1 Freshwater Forested/Shrub Wetland
    2 Riverine
    3 Freshwater Pond
    4 Other                                  (misc. NWI legend catch-all, e.g. farmed / other)
    5 Estuarine and Marine Wetland
    6 Estuarine and Marine Deepwater
    7 Lake

This is a **positive-only, multi-class wetland segmentation** (spec  4 polygons /  5): NWI
maps only wetland/deepwater features, so pixels outside every polygon are left as
**nodata/ignore (255)** rather than a fabricated "upland" background -- the pretraining
assembly step supplies negatives from other datasets. (We deliberately do not assert upland
everywhere unmapped, since NWI's minimum mapping unit can omit small features.)

Tiling: we sample 64x64 windows (640 m, local UTM at 10 m/pixel). Candidate windows are
seeded from polygons of every class (so rare classes -- Lake, estuarine/marine -- get
coverage), snapped to a 64-px grid (deduplicating nearby seeds into shared tiles). Every NWI
polygon intersecting a window is rasterized into it (value = WETLAND_TYPE class id;
outside-polygon = 255), yielding dense multi-class tiles. Tiles are then selected
**tiles-per-class balanced** (rarest class first, up to 1000 tiles/class, 25k total cap).

Time range: wetlands are persistent/static features, so per spec  5 each sample gets a
representative 1-year Sentinel-era window (2020), change_time = None.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.us_national_wetlands_inventory_nwi
Idempotent: existing locations/{id}.tif are skipped.
"""

import argparse
import multiprocessing
import random
from collections import Counter
from typing import Any

import numpy as np
import shapely.wkb
import tqdm
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.mp import star_imap_unordered
from shapely.strtree import STRtree

from olmoearth_pretrain.open_set_segmentation_data import io, manifest, sampling

SLUG = "us_national_wetlands_inventory_nwi"
NAME = "US National Wetlands Inventory (NWI)"

BASE_URL = (
    "https://documentst.ecosphere.fws.gov/wetlands/data/State-Downloads/"
    "{ST}_geodatabase_wetlands.zip"
)
LANDING = "https://www.fws.gov/program/national-wetlands-inventory/data-download"

STATES = ["LA", "FL", "ND", "NC"]

TILE = 64
SEED_PER_CLASS = 1500  # candidate-window seeds per class per state
PER_CLASS = 1000  # target selected tiles per class
MAX_SAMPLES = sampling.MAX_SAMPLES_PER_DATASET  # 25000
REP_YEAR = 2020  # representative Sentinel-era year (static labels)
NODATA = io.CLASS_NODATA  # 255; outside-polygon / ignore

# NWI WETLAND_TYPE -> class id (fixed, ordered by national frequency in the sampled states).
CLASS_ORDER = [
    (
        "Freshwater Emergent Wetland",
        "Palustrine emergent wetland: herbaceous marsh / wet meadow (Cowardin PEM). "
        "Persistent or non-persistent emergent vegetation.",
    ),
    (
        "Freshwater Forested/Shrub Wetland",
        "Palustrine forested or scrub-shrub wetland: wooded swamps and shrub bogs "
        "(Cowardin PFO / PSS).",
    ),
    (
        "Riverine",
        "Riverine system: wetlands and deepwater habitats within a river/stream channel "
        "(Cowardin R1-R5), e.g. streambed, unconsolidated bottom.",
    ),
    (
        "Freshwater Pond",
        "Freshwater ponds: small palustrine open-water bodies -- unconsolidated bottom / "
        "aquatic bed (Cowardin PUB / PAB / PUS ponds).",
    ),
    (
        "Other",
        "NWI simplified-legend catch-all for freshwater features not in the other classes "
        "(e.g. farmed wetlands and miscellaneous palustrine unconsolidated types). Common in "
        "the Prairie Pothole Region.",
    ),
    (
        "Estuarine and Marine Wetland",
        "Intertidal estuarine/marine wetland: salt & brackish marsh, tidal flats, mangrove, "
        "reef (Cowardin E2 / M2).",
    ),
    (
        "Estuarine and Marine Deepwater",
        "Subtidal estuarine/marine deepwater: bays, sounds, nearshore ocean below low tide "
        "(Cowardin E1 / M1).",
    ),
    (
        "Lake",
        "Lacustrine system: lakes and reservoirs >= 8 ha or deep (Cowardin L1 / L2).",
    ),
]
WT_TO_ID = {name: i for i, (name, _desc) in enumerate(CLASS_ORDER)}

PROJ_5070 = Projection(
    CRS.from_epsg(5070), 1, 1
)  # geometry coords == metres in EPSG:5070


# --------------------------------------------------------------------------- download


def download_states() -> None:
    """Download + extract each state's wetlands geodatabase to raw/{slug}/ (idempotent)."""
    from olmoearth_pretrain.open_set_segmentation_data import download

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "US Fish & Wildlife Service National Wetlands Inventory (public domain).\n"
            f"Landing: {LANDING}\n"
            f"Per-state File Geodatabase downloads: {BASE_URL}\n"
            f"States downloaded (bounded sampling, spec  5): {', '.join(STATES)}\n"
        )
    for st in STATES:
        gdb = raw / f"{st}_geodatabase_wetlands.gdb"
        if gdb.exists():
            print(f"  [skip] {st} extracted")
            continue
        io.check_disk()
        zpath = raw / f"{st}.zip"
        print(f"  downloading {st} ...")
        download.download_http(BASE_URL.format(ST=st), zpath)
        print(f"  extracting {st} ...")
        download.extract_zip(zpath, raw, skip_existing=False)
    io.check_disk()


# --------------------------------------------------------------------------- load / candidates


def _load_state(st: str) -> tuple[list[Any], np.ndarray]:
    """Load a state's wetland polygons -> (list[shapely geom in 5070], class_id array).

    Uses pyogrio for a fast vectorized read of the File Geodatabase (per-feature fiona
    iteration over the ~0.6-2M polygons is prohibitively slow).
    """
    import pyogrio

    gdb = str(io.raw_dir(SLUG) / f"{st}_geodatabase_wetlands.gdb")
    gdf = pyogrio.read_dataframe(gdb, layer=f"{st}_Wetlands", columns=["WETLAND_TYPE"])
    cids = gdf["WETLAND_TYPE"].map(WT_TO_ID)
    keep = cids.notna() & gdf.geometry.notna() & ~gdf.geometry.is_empty
    gdf = gdf[keep]
    geoms = list(gdf.geometry.values)
    return geoms, cids[keep].to_numpy(dtype=np.uint8)


def _candidate_windows(
    st: str, geoms: list[Any], cids: np.ndarray, seed: int
) -> list[dict[str, Any]]:
    """Seed 64x64 candidate windows from polygons of every class and attach the polygons
    (as WKB) that actually intersect each window, plus classes_present.

    Seed placement is fully vectorized: polygon centroids (EPSG:5070) are batch-reprojected
    to lon/lat and then to their local UTM zone with cached pyproj transformers, so the
    ~0.6-2M-polygon states are handled in seconds (per-geometry representative_point() +
    per-point reprojection was the dominant cost otherwise).
    """
    import shapely as _sh
    from pyproj import Transformer

    tree = STRtree(geoms)
    rng = random.Random(seed)

    # Vectorized centroids in 5070 metres for every polygon.
    geoms_arr = np.asarray(geoms, dtype=object)
    cent = _sh.centroid(geoms_arr)
    cx = _sh.get_x(cent)
    cy = _sh.get_y(cent)

    # Seed polygon indices: up to SEED_PER_CLASS per class (rare classes fully covered).
    by_class: dict[int, list[int]] = {}
    for i, c in enumerate(cids):
        by_class.setdefault(int(c), []).append(i)
    seed_idx: list[int] = []
    for c, idxs in by_class.items():
        rng.shuffle(idxs)
        seed_idx.extend(idxs[:SEED_PER_CLASS])
    seed_idx_arr = np.asarray(seed_idx, dtype=np.int64)

    # 5070 -> lon/lat (batch), then per-UTM-zone lon/lat -> UTM metres (batch).
    tf_ll = Transformer.from_crs(5070, 4326, always_xy=True)
    lon, lat = tf_ll.transform(cx[seed_idx_arr], cy[seed_idx_arr])
    zone = np.floor((lon + 180.0) / 6.0).astype(int) + 1
    epsg = np.where(lat >= 0, 32600 + zone, 32700 + zone)

    windows: dict[tuple[str, int, int], dict[str, Any]] = {}
    for e in np.unique(epsg):
        mask = epsg == e
        tf_utm = Transformer.from_crs(4326, int(e), always_xy=True)
        mx, my = tf_utm.transform(lon[mask], lat[mask])
        # UTM metre -> projection pixel (x_res=10, y_res=-10), then snap to a 64-px grid.
        col = np.floor(mx / io.RESOLUTION).astype(np.int64)
        row = np.floor(my / -io.RESOLUTION).astype(np.int64)
        x0 = (np.floor(col / TILE) * TILE).astype(np.int64)
        y0 = (np.floor(row / TILE) * TILE).astype(np.int64)
        crs = f"EPSG:{int(e)}"
        for xi, yi in zip(x0.tolist(), y0.tolist()):
            key = (crs, xi, yi)
            if key not in windows:
                windows[key] = {
                    "crs": crs,
                    "bounds": (xi, yi, xi + TILE, yi + TILE),
                    "state": st,
                }

    # For each window: reproject its box to 5070, query the tree, keep truly-intersecting
    # polygons, record classes_present + the polygons as WKB for writing.
    from shapely.geometry import box

    out: list[dict[str, Any]] = []
    for w in windows.values():
        utm = Projection(CRS.from_string(w["crs"]), io.RESOLUTION, -io.RESOLUTION)
        win_utm = box(*w["bounds"])
        win5070 = STGeometry(utm, win_utm, None).to_projection(PROJ_5070).shp
        # Vectorized GEOS intersects test (C-level) -> indices of truly-intersecting polys.
        hit = tree.query(win5070, predicate="intersects")
        if len(hit) == 0:
            continue
        shapes: list[tuple[bytes, int]] = [
            (shapely.wkb.dumps(geoms[int(j)]), int(cids[int(j)])) for j in hit
        ]
        present = {int(cids[int(j)]) for j in hit}
        out.append(
            {
                "crs": w["crs"],
                "bounds": w["bounds"],
                "state": st,
                "shapes_wkb": shapes,
                "classes_present": sorted(present),
            }
        )
    return out


# --------------------------------------------------------------------------- write


def _write_one(rec: dict[str, Any]) -> list[int] | None:
    from shapely.geometry import box

    from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
        geom_to_pixels,
        rasterize_shapes,
    )

    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return None

    proj = Projection(CRS.from_string(rec["crs"]), io.RESOLUTION, -io.RESOLUTION)
    bounds = tuple(rec["bounds"])
    win = box(*bounds)
    shapes_px: list[tuple[Any, int]] = []
    for wkb, cid in rec["shapes_wkb"]:
        g = shapely.wkb.loads(wkb)
        px = geom_to_pixels(g, PROJ_5070, proj)
        if px.is_empty:
            continue
        clip = px.intersection(win)
        if clip.is_empty or clip.area <= 0:
            continue
        shapes_px.append((clip, cid))
    if not shapes_px:
        return None
    # NWI polygons form a planar (non-overlapping) coverage, so paint order rarely matters;
    # sort by class id so the higher-id (rarer) class wins any incidental shared edge pixel.
    shapes_px.sort(key=lambda s: s[1])
    label = rasterize_shapes(shapes_px, bounds, fill=NODATA, dtype="uint8")[0]

    present = sorted(int(v) for v in np.unique(label) if v != NODATA)
    if not present:
        return None
    time_range = io.year_range(REP_YEAR)
    io.write_label_geotiff(SLUG, sample_id, label, proj, bounds, nodata=NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        time_range,
        change_time=None,
        source_id=f"{rec['state']}:{bounds[0]}_{bounds[1]}",
        classes_present=present,
    )
    return present


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    io.check_disk()
    download_states()

    # ---- Phase A: per-state load + candidate windows (sequential per state to bound memory)
    records: list[dict[str, Any]] = []
    for si, st in enumerate(STATES):
        io.check_disk()
        print(f"[{st}] loading polygons ...")
        geoms, cids = _load_state(st)
        print(f"[{st}] {len(geoms)} polygons; building candidate windows ...")
        recs = _candidate_windows(st, geoms, cids, seed=42 + si)
        print(f"[{st}] {len(recs)} candidate windows")
        records.extend(recs)
        del geoms, cids
    print(f"total candidate windows: {len(records)}")
    cand_class = Counter()
    for r in records:
        for c in r["classes_present"]:
            cand_class[c] += 1
    print("candidate tiles per class:", dict(sorted(cand_class.items())))

    # ---- Phase B: tiles-per-class balanced selection
    selected = sampling.select_tiles_per_class(
        records,
        classes_key="classes_present",
        per_class=PER_CLASS,
        total_cap=MAX_SAMPLES,
        seed=42,
    )
    for j, r in enumerate(selected):
        r["sample_id"] = f"{j:06d}"
    print(f"selected {len(selected)} tiles (cap {MAX_SAMPLES})")

    # ---- Phase C: write tiles (parallel)
    io.check_disk()
    class_counts: Counter = Counter()
    state_counts: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res, rec in tqdm.tqdm(
            zip(
                star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
                selected,
            ),
            total=len(selected),
            desc="write tiles",
        ):
            if res is not None:
                for c in res:
                    class_counts[c] += 1
                state_counts[rec["state"]] += 1

    n_written = len(list(io.locations_dir(SLUG).glob("*.tif")))

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "US Fish & Wildlife Service, National Wetlands Inventory",
            "license": "public domain",
            "provenance": {
                "url": LANDING,
                "download_pattern": BASE_URL,
                "states": STATES,
                "have_locally": False,
                "annotation_method": "photointerpretation (Cowardin classification)",
                "class_field": "WETLAND_TYPE",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASS_ORDER)
            ],
            "nodata_value": NODATA,
            "num_samples": n_written,
            "class_tile_counts": {str(k): v for k, v in sorted(class_counts.items())},
            "state_tile_counts": dict(sorted(state_counts.items())),
            "is_change_dataset": False,
            "notes": (
                "Positive-only multi-class wetland-type segmentation from USFWS NWI. 64x64 "
                "uint8 tiles, local UTM at 10 m; class ids 0-7 are NWI WETLAND_TYPE "
                "categories, 255 = nodata/ignore (pixels outside every mapped wetland "
                "polygon -- no fabricated upland background; assembly supplies negatives). "
                f"Bounded state sampling (spec  5): states {', '.join(STATES)} chosen to "
                "cover all Cowardin systems (Marine/Estuarine/Riverine/Lacustrine/"
                "Palustrine) across Gulf, Atlantic, and Prairie-Pothole settings; NOT all "
                "of CONUS. Candidate 64x64 windows seeded from polygons of every class "
                "(rare classes prioritized) and snapped to a 64-px grid; every intersecting "
                "NWI polygon rasterized in. Tiles-per-class balanced (<=1000/class, 25k "
                "cap). Wetlands are static features -> representative 1-year window "
                f"({REP_YEAR}); change_time=None."
            ),
        },
    )
    print("class tile counts:", dict(sorted(class_counts.items())))
    print("state tile counts:", dict(sorted(state_counts.items())))
    print("total tif on disk:", n_written)

    manifest.write_registry_entry(
        SLUG,
        "completed",
        task_type="classification",
        num_samples=n_written,
        notes=f"NWI wetland-type segmentation; states {','.join(STATES)}; 8 classes.",
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
