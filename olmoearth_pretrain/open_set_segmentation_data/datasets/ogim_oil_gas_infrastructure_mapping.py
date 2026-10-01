"""Process the OGIM (Oil & Gas Infrastructure Mapping) database into unified
oil/gas-infrastructure detection/segmentation label tiles.

Source: "Oil and Gas Infrastructure Mapping (OGIM) database", v2.5.1, Environmental
Defense Fund (EDF) / MethaneSAT LLC, Zenodo (https://zenodo.org/records/13259749,
doi:10.5281/zenodo.13259749), CC-BY-4.0. Described in Omara et al., ESSD 2023
(https://doi.org/10.5194/essd-15-3761-2023). A single ~3 GB GeoPackage
(``OGIM_v2.5.1.gpkg``) with ~6.7M curated, integrated oil & gas infrastructure features
(official government + industry + academic sources), all in EPSG:4326. We download only
the label GeoPackage -- NO imagery; pretraining supplies its own S2/S1/Landsat.

Relevant layers -> unified class scheme (spec section 5: mixed points + lines are combined
into ONE dataset with one class map). Point layers carry LONGITUDE/LATITUDE attributes
(verified identical to geometry); the pipeline layer is LineStrings.

    0 = background   (land/water with no mapped O&G infrastructure)
    1 = well         (Oil_and_Natural_Gas_Wells; 4.52M points)
    2 = platform     (Offshore_Platforms; 9.8k points)
    3 = facility     (Natural_Gas_Compressor_Stations + Gathering_and_Processing +
                      LNG_Facilities; ~23k points -- "compressor/processing/LNG facilities")
    4 = refinery     (Crude_Oil_Refineries; 686 points)
    5 = pipeline     (Oil_Natural_Gas_Pipelines; 1.90M LineStrings)
    255 = nodata/ignore (detection buffer rings around imprecise point locations)

Task type: object DETECTION / presence segmentation encoded as per-pixel classes
(spec section 4). Point features mark presence; their exact pixel is uncertain, so each is
a ``positive_size=1`` positive ringed by a ``buffer_size=10`` px nodata band (21x21 ignore)
inside a 64x64 (640 m) UTM 10 m context tile, the rest background. Pipelines are precise
line geometry, rasterized (buffered to a ~30 m ribbon) as class 5. Every tile is labeled
with ALL OGIM features that fall inside it (all classes), so the class map is genuinely
unified rather than one-target-per-tile.

Individual small wellheads are near/below 10 m resolution (spec section 8 flags this): the
detection label is best read as "well SITE present within this ~200 m ignore region" --
onshore well pads / clustered well fields produce visible surface disturbance (cleared
pads, tanks, access roads) at 10 m, and the thick nodata buffer already absorbs positional
imprecision. Kept with this caveat documented; downstream assembly filters classes that
prove unusable.

Time / change handling (spec section 5). Infrastructure is PERSISTENT, not a dated change
event, so change_time is null. SRC_DATE is the source-publication/update date (ISO
YYYY-MM-DD), not an event date; we anchor each tile's 1-year window on its SRC_DATE year,
clamped to the Sentinel era: year = SRC_DATE year if 2016<=year<=2024 else 2020 (a
representative recent year -- the structure is persistent and observable across the Sentinel
era regardless of when its source record was published; ~81% of wells are dated 2024 and
almost all foreground features are 2016+). No change labels are emitted (SRC_DATE is only a
source date, coarser than the ~1-2 month change-timing bar).

Sampling (spec section 5): up to 1000 anchor tiles per foreground class (PER_CLASS),
tiles-per-class balanced, plus up to 1000 background NEGATIVE tiles (locations >~2 km from
any point feature and with no pipeline in the tile bbox -> guaranteed all-background). To
avoid a pile of near-identical overlapping tiles in dense basins, anchors are grid-deduped
(one per ~2 km cell) before random sampling. Total well under the 25k cap.

Run (idempotent; skips already-written tiles):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.ogim_oil_gas_infrastructure_mapping
"""

import argparse
import math
import multiprocessing
import random
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.mp import star_imap_unordered
from scipy.spatial import cKDTree

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.download import download_zenodo
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)

SLUG = "ogim_oil_gas_infrastructure_mapping"
NAME = "OGIM (Oil & Gas Infrastructure Mapping)"
ZENODO_URL = "https://zenodo.org/records/13259749"
ZENODO_RECORD = "13259749"
DOI = "10.5281/zenodo.13259749"
GPKG_FILE = "OGIM_v2.5.1.gpkg"

# Class ids (unified scheme).
BG, WELL, PLATFORM, FACILITY, REFINERY, PIPELINE = 0, 1, 2, 3, 4, 5

# Point layers -> class id. Facility class merges the three facility-type layers.
POINT_LAYERS: list[tuple[str, int]] = [
    ("Oil_and_Natural_Gas_Wells", WELL),
    ("Offshore_Platforms", PLATFORM),
    ("Natural_Gas_Compressor_Stations", FACILITY),
    ("Gathering_and_Processing", FACILITY),
    ("LNG_Facilities", FACILITY),
    ("Crude_Oil_Refineries", REFINERY),
]
PIPELINE_LAYER = "Oil_Natural_Gas_Pipelines"

CLASSES = [
    (
        BG,
        "background",
        "Land or water surface with no mapped OGIM oil/gas infrastructure. In each 64x64 "
        "context tile, all pixels that are neither a mapped feature nor a detection ignore "
        "buffer are background; plus dedicated negative tiles placed >~2 km from any feature.",
    ),
    (
        WELL,
        "well",
        "Oil or natural gas well (OGIM Oil_and_Natural_Gas_Wells; CATEGORY 'OIL AND NATURAL "
        "GAS WELLS'). Point location of a well/wellhead. Individual wellheads are near/below "
        "10 m resolution; label denotes a well SITE present within the ~200 m detection ignore "
        "region (onshore well pads / clustered fields are visible at 10 m).",
    ),
    (
        PLATFORM,
        "offshore_platform",
        "Offshore oil/gas production platform (OGIM Offshore_Platforms).",
    ),
    (
        FACILITY,
        "facility",
        "Gas compressor station, gathering & processing facility, or LNG facility (OGIM "
        "Natural_Gas_Compressor_Stations + Gathering_and_Processing + LNG_Facilities) -- the "
        "manifest 'compressor/processing/LNG facilities' class.",
    ),
    (REFINERY, "refinery", "Crude oil refinery (OGIM Crude_Oil_Refineries)."),
    (
        PIPELINE,
        "pipeline",
        "Oil or natural gas pipeline (OGIM Oil_Natural_Gas_Pipelines), rasterized from the "
        "line geometry and buffered to a ~30 m ribbon so it is resolvable at 10-30 m.",
    ),
]
CLASS_NAME = {cid: name for cid, name, _ in CLASSES}

# Tile / encoding parameters (spec section 4).
TILE = 64
POS_SIZE = 1
BUFFER = 10
PIPE_HALF_PX = 1.5  # half-width in 10 m px -> ~30 m ribbon

PER_CLASS = 1000
N_NEG = 1000
SEED = 42

GRID_DEG = 0.02  # anchor dedup cell (~2 km) to spread tiles, reduce overlap
DEDUP_MAX_KEEP = 40000  # early-stop dedup once this many cells kept (>> PER_CLASS)
NEI_RADIUS_DEG = 0.007  # neighbor point query radius (> tile half-diagonal ~452 m)
MAX_NEI = 1000  # cap burned point neighbors per tile (>= tile pixel count anyway)
NEG_OFFSET_M = (10000.0, 50000.0)  # negative offset from a real feature
NEG_CLEAR_DEG = 0.02  # ~2 km min clearance from any point feature

_GPKG_PATH: str | None = None  # set in main / passed to workers


# --------------------------------------------------------------------------- load helpers


def _clamp_years(src: pd.Series) -> np.ndarray:
    """Vectorized SRC_DATE -> Sentinel-era year (2016-2024 else 2020)."""
    y = pd.to_numeric(src.astype(str).str[:4], errors="coerce")
    y = y.where((y >= 2016) & (y <= 2024))
    return y.fillna(2020).astype(int).to_numpy()


def _grid_dedup_idx(
    lon: np.ndarray, lat: np.ndarray, seed: int, max_keep: int
) -> list[int]:
    """Keep one random index per ~GRID_DEG cell (spreads anchors, reduces tile overlap).

    Shuffles indices then keeps the first per cell, so the retained representative is random
    and cells span the globe. Early-stops at ``max_keep`` kept cells (a later random sample
    of PER_CLASS from these is still spatially spread).
    """
    idx = list(range(len(lon)))
    random.Random(seed).shuffle(idx)
    ci = np.round(lon / GRID_DEG).astype(np.int64)
    cj = np.round(lat / GRID_DEG).astype(np.int64)
    seen: set[tuple[int, int]] = set()
    keep: list[int] = []
    for i in idx:
        cell = (int(ci[i]), int(cj[i]))
        if cell in seen:
            continue
        seen.add(cell)
        keep.append(i)
        if len(keep) >= max_keep:
            break
    return keep


def _load_point_arrays() -> tuple[
    dict[int, dict[str, np.ndarray]], np.ndarray, np.ndarray
]:
    """Load all point features (vectorized).

    Returns (class_arrays, all_coords[N,2], all_cids[N]).
    class_arrays[cid] = {"lon","lat","year","srcid"} numpy arrays for that class
    (facility merges its three sublayers). all_coords/all_cids cover EVERY point feature.
    """
    import pyogrio

    by_class: dict[int, dict[str, list[np.ndarray]]] = {}
    all_lon: list[np.ndarray] = []
    all_lat: list[np.ndarray] = []
    all_cid: list[np.ndarray] = []
    for layer, cid in POINT_LAYERS:
        df = pyogrio.read_dataframe(
            _GPKG_PATH,
            layer=layer,
            columns=["LONGITUDE", "LATITUDE", "SRC_DATE", "OGIM_ID"],
            read_geometry=False,
        )
        lon = df["LONGITUDE"].to_numpy(dtype="float64")
        lat = df["LATITUDE"].to_numpy(dtype="float64")
        ok = (
            np.isfinite(lon)
            & np.isfinite(lat)
            & (np.abs(lat) <= 89.9)
            & (np.abs(lon) <= 180.0)
        )
        lon = lon[ok]
        lat = lat[ok]
        year = _clamp_years(df["SRC_DATE"])[ok]
        ogid = df["OGIM_ID"].to_numpy()[ok]
        srcid = np.array([f"{layer}/{o}" for o in ogid], dtype=object)
        d = by_class.setdefault(cid, {"lon": [], "lat": [], "year": [], "srcid": []})
        d["lon"].append(lon)
        d["lat"].append(lat)
        d["year"].append(year)
        d["srcid"].append(srcid)
        all_lon.append(lon)
        all_lat.append(lat)
        all_cid.append(np.full(lon.shape, cid, dtype="int16"))
        print(f"  {layer}: {len(lon)} points (class {cid})", flush=True)
    class_arrays = {
        cid: {k: np.concatenate(v) for k, v in d.items()} for cid, d in by_class.items()
    }
    all_coords = np.column_stack([np.concatenate(all_lon), np.concatenate(all_lat)])
    all_cids = np.concatenate(all_cid)
    return class_arrays, all_coords, all_cids


def _load_pipeline_anchor_arrays() -> dict[str, np.ndarray]:
    """Read all pipelines; return arrays of representative-point lon/lat/year/srcid."""
    import pyogrio

    df = pyogrio.read_dataframe(
        _GPKG_PATH,
        layer=PIPELINE_LAYER,
        columns=["SRC_DATE", "OGIM_ID"],
        read_geometry=True,
    )
    rp = df.geometry.representative_point()
    lon = rp.x.to_numpy()
    lat = rp.y.to_numpy()
    ok = (
        np.isfinite(lon)
        & np.isfinite(lat)
        & (np.abs(lat) <= 89.9)
        & (np.abs(lon) <= 180.0)
    )
    year = _clamp_years(df["SRC_DATE"])[ok]
    ogid = df["OGIM_ID"].to_numpy()[ok]
    srcid = np.array([f"{PIPELINE_LAYER}/{o}" for o in ogid], dtype=object)
    print(f"  {PIPELINE_LAYER}: {int(ok.sum())} lines (class {PIPELINE})", flush=True)
    return {"lon": lon[ok], "lat": lat[ok], "year": year, "srcid": srcid}


def _anchors_from_arrays(arr: dict[str, np.ndarray], cid: int) -> list[dict[str, Any]]:
    """Grid-dedup an array-of-points class and return up to PER_CLASS anchor dicts."""
    keep = _grid_dedup_idx(
        arr["lon"], arr["lat"], seed=SEED + cid, max_keep=DEDUP_MAX_KEEP
    )
    random.Random(SEED + 100 + cid).shuffle(keep)
    keep = keep[:PER_CLASS]
    return [
        {
            "lon": float(arr["lon"][i]),
            "lat": float(arr["lat"][i]),
            "year": int(arr["year"][i]),
            "source_id": str(arr["srcid"][i]),
            "cid": cid,
            "read_pipes": True,
        }
        for i in keep
    ]


# --------------------------------------------------------------------------- write


def _init_worker(gpkg_path: str) -> None:
    global _GPKG_PATH
    _GPKG_PATH = gpkg_path


def _tile_lonlat_bbox(lon: float, lat: float) -> tuple[float, float, float, float]:
    dlat = (TILE * io.RESOLUTION * 0.75) / 111320.0
    dlon = dlat / max(0.15, math.cos(math.radians(lat)))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def _read_pipelines_in_bbox(bbox: tuple[float, float, float, float]) -> list[Any]:
    import pyogrio

    df = pyogrio.read_dataframe(
        _GPKG_PATH,
        layer=PIPELINE_LAYER,
        columns=["OGIM_ID"],
        read_geometry=True,
        bbox=bbox,
    )
    return [g for g in df.geometry.values if g is not None and not g.is_empty]


def _write_tile(rec: dict[str, Any]) -> list[int] | None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return None

    lon, lat = rec["lon"], rec["lat"]
    proj = io.utm_projection_for_lonlat(lon, lat)
    _, col, row = io.lonlat_to_utm_pixel(lon, lat, proj)
    bounds = io.centered_bounds(col, row, TILE, TILE)
    x_min, y_min = bounds[0], bounds[1]

    arr = np.zeros((TILE, TILE), dtype=np.uint8)  # background

    # Point neighbors (all classes) -> tile-local (lr, lc, cid).
    local_pts: list[tuple[int, int, int]] = []
    for nlon, nlat, ncid in rec["neighbors"]:
        _, c, r = io.lonlat_to_utm_pixel(float(nlon), float(nlat), proj)
        lc, lr = c - x_min, r - y_min
        if 0 <= lc < TILE and 0 <= lr < TILE:
            local_pts.append((lr, lc, int(ncid)))

    # 1) nodata buffer rings around every point.
    for lr, lc, _cid in local_pts:
        r0 = max(0, lr - POS_SIZE // 2 - BUFFER)
        r1 = min(TILE, lr + POS_SIZE // 2 + BUFFER + 1)
        c0 = max(0, lc - POS_SIZE // 2 - BUFFER)
        c1 = min(TILE, lc + POS_SIZE // 2 + BUFFER + 1)
        arr[r0:r1, c0:c1] = io.CLASS_NODATA

    # 2) pipelines (precise geometry) -> class 5, overriding buffer nodata.
    if rec["read_pipes"]:
        geoms = _read_pipelines_in_bbox(_tile_lonlat_bbox(lon, lat))
        shapes = []
        for g in geoms:
            gp = geom_to_pixels(g, WGS84_PROJECTION, proj).buffer(PIPE_HALF_PX)
            if not gp.is_empty:
                shapes.append((gp, PIPELINE))
        if shapes:
            pipe = rasterize_shapes(
                shapes, bounds, fill=0, dtype="uint8", all_touched=True
            )[0]
            arr[pipe == PIPELINE] = PIPELINE

    # 3) positive centers (win over pipeline/buffer).
    for lr, lc, cid in local_pts:
        r0 = max(0, lr - POS_SIZE // 2)
        r1 = min(TILE, lr + POS_SIZE // 2 + 1)
        c0 = max(0, lc - POS_SIZE // 2)
        c1 = min(TILE, lc + POS_SIZE // 2 + 1)
        arr[r0:r1, c0:c1] = cid

    present = sorted(int(v) for v in np.unique(arr) if v != io.CLASS_NODATA)
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        change_time=None,
        source_id=rec["source_id"],
        classes_present=present,
    )
    return present


# --------------------------------------------------------------------------- main


def main() -> None:
    global _GPKG_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    gpkg = raw / GPKG_FILE
    if not gpkg.exists():
        print("downloading OGIM GeoPackage from Zenodo ...", flush=True)
        download_zenodo(ZENODO_RECORD, raw, filenames=[GPKG_FILE])
    _GPKG_PATH = str(gpkg)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Oil and Gas Infrastructure Mapping (OGIM) database v2.5.1 (EDF / MethaneSAT).\n"
            f"{ZENODO_URL}\ndoi:{DOI}  license CC-BY-4.0\n"
            f"File: {GPKG_FILE} (~3 GB GeoPackage, EPSG:4326, ~6.7M features). "
            "Also fetched the schema + data-source-references PDFs. NO imagery downloaded.\n"
            "Layers used -> classes: Oil_and_Natural_Gas_Wells=well; Offshore_Platforms="
            "platform; Natural_Gas_Compressor_Stations+Gathering_and_Processing+LNG_Facilities"
            "=facility; Crude_Oil_Refineries=refinery; Oil_Natural_Gas_Pipelines=pipeline.\n"
        )

    print("reading point layers ...", flush=True)
    class_arrays, all_coords, all_cids = _load_point_arrays()
    io.check_disk()

    print("reading pipeline anchors ...", flush=True)
    pipe_arr = _load_pipeline_anchor_arrays()

    # Combined KDTree over all point features for neighbor burn-in + negative clearance.
    print("building KDTree over all point features ...", flush=True)
    tree = cKDTree(all_coords)

    # Build anchors: grid-dedup then random-sample up to PER_CLASS per class.
    anchors: list[dict[str, Any]] = []
    anchor_counts: dict[int, int] = {}
    for cid in (WELL, PLATFORM, FACILITY, REFINERY):
        sel = _anchors_from_arrays(class_arrays[cid], cid)
        anchors.extend(sel)
        anchor_counts[cid] = len(sel)
        print(
            f"  class {cid} ({CLASS_NAME[cid]}): {len(class_arrays[cid]['lon'])} -> {len(sel)} anchors",
            flush=True,
        )
    sel_pipe = _anchors_from_arrays(pipe_arr, PIPELINE)
    anchors.extend(sel_pipe)
    anchor_counts[PIPELINE] = len(sel_pipe)
    print(
        f"  class {PIPELINE} (pipeline): {len(pipe_arr['lon'])} -> {len(sel_pipe)} anchors",
        flush=True,
    )

    # Negatives: offset from a random point feature, require clearance from ALL point
    # features (KDTree) and no pipeline in the tile bbox -> guaranteed background.
    rng = random.Random(SEED)
    n_pts = len(all_coords)
    negatives: list[dict[str, Any]] = []
    attempts = 0
    while len(negatives) < N_NEG and attempts < N_NEG * 200:
        attempts += 1
        base = all_coords[rng.randrange(n_pts)]
        dist = rng.uniform(*NEG_OFFSET_M)
        bearing = rng.uniform(0, 2 * math.pi)
        dlat = (dist * math.cos(bearing)) / 111320.0
        dlon = (dist * math.sin(bearing)) / (
            111320.0 * max(0.15, math.cos(math.radians(base[1])))
        )
        nlon, nlat = base[0] + dlon, base[1] + dlat
        if not (-180 <= nlon <= 180 and -85 <= nlat <= 85):
            continue
        d, _ = tree.query([nlon, nlat], k=1)
        if d < NEG_CLEAR_DEG:
            continue
        negatives.append(
            {
                "lon": float(nlon),
                "lat": float(nlat),
                "year": 2020,
                "cid": BG,
                "read_pipes": True,
                "source_id": "negative",
            }
        )
    print(f"  negatives: {len(negatives)} (background-only tiles)", flush=True)

    # Precompute cross-class point neighbors for every tile (small per-tile lists).
    all_recs = anchors + negatives
    for r in all_recs:
        idxs = tree.query_ball_point([r["lon"], r["lat"]], r=NEI_RADIUS_DEG)
        if len(idxs) > MAX_NEI:
            idxs = rng.sample(idxs, MAX_NEI)
        r["neighbors"] = [
            (float(all_coords[j, 0]), float(all_coords[j, 1]), int(all_cids[j]))
            for j in idxs
        ]
    for i, r in enumerate(all_recs):
        r["sample_id"] = f"{i:06d}"
    print(f"total tiles to write: {len(all_recs)}", flush=True)

    io.check_disk()
    tiles_per_class: Counter = Counter()
    with multiprocessing.Pool(
        args.workers, initializer=_init_worker, initargs=(str(gpkg),)
    ) as p:
        for present in tqdm.tqdm(
            star_imap_unordered(p, _write_tile, [dict(rec=r) for r in all_recs]),
            total=len(all_recs),
            desc="write tiles",
        ):
            if present is not None:
                for c in present:
                    tiles_per_class[c] += 1
    io.check_disk()

    n_written = len(list(io.locations_dir(SLUG).glob("*.tif")))
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",  # detection/presence encoded as per-pixel classes
            "source": "Zenodo / ESSD (EDF, MethaneSAT)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": ZENODO_URL,
                "doi": DOI,
                "have_locally": False,
                "annotation_method": "curated integration of official/industry/academic sources",
                "file": GPKG_FILE,
                "version": "v2.5.1",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": cid, "name": name, "description": desc}
                for cid, name, desc in CLASSES
            ],
            "nodata_value": io.CLASS_NODATA,
            "detection_encoding": {
                "tile_size": TILE,
                "positive_size": POS_SIZE,
                "buffer_size": BUFFER,
                "pipeline_half_width_px": PIPE_HALF_PX,
            },
            "num_samples": n_written,
            "anchor_tiles_per_class": {
                CLASS_NAME[c]: anchor_counts.get(c, 0) for c in anchor_counts
            },
            "negative_tiles": len(negatives),
            "tiles_containing_class": {
                CLASS_NAME.get(c, str(c)): tiles_per_class.get(c, 0)
                for c in sorted(tiles_per_class)
            },
            "notes": (
                "OGIM v2.5.1 (EDF/MethaneSAT) unified oil & gas infrastructure "
                "detection/segmentation. Mixed points + lines combined into ONE class map "
                "(spec 5): 0 background, 1 well, 2 offshore_platform, 3 facility "
                "(compressor+gathering/processing+LNG), 4 refinery, 5 pipeline; 255 nodata. "
                "64x64 UTM 10 m tiles. Point features: 1 px positive + 10 px nodata buffer "
                "(21x21 ignore); pipelines rasterized to a ~30 m ribbon (class 5). Each tile "
                "labeled with ALL OGIM features inside it (cross-class neighbors burned in), "
                "so the map is unified. Individual wellheads are near/below 10 m; the well "
                "label denotes a well SITE within the ~200 m ignore region (pads/fields "
                "visible at 10 m) -- kept with caveat, downstream may filter. Persistent "
                "structures: change_time=null; 1-year window anchored on SRC_DATE year "
                "clamped to 2016-2024 (else 2020; SRC_DATE is a source-publication date, not "
                "an event date, so no dated change labels). Up to 1000 anchor tiles/class "
                "(grid-deduped to ~2 km to reduce overlap) + up to 1000 background negatives "
                "(>~2 km from any feature, empty pipeline bbox). Well under the 25k cap."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=n_written
    )
    print("anchor tiles per class:", anchor_counts, flush=True)
    print("tiles containing each class:", dict(tiles_per_class), flush=True)
    print(f"done: {n_written} tiles on disk", flush=True)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
