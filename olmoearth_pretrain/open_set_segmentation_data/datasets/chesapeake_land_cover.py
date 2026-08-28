"""Process the Chesapeake Bay 1 m Land Cover (2022 Edition, 2017/18) into open-set-seg tiles.

Source: USGS ScienceBase "Chesapeake Bay Land Use and Land Cover (LULC) Database, 2022
Edition" (DOI 10.5066/P981GV1L; Chesapeake Bay Program / Chesapeake Conservancy / USGS /
UVM SAL). The database ships one-meter **13-class Land Cover (LC)** and 54-class LULC
rasters for two epochs, 2013/14 and 2017/18. We use the **2017/18 LC** product (post-2016,
Sentinel era) at 1 m, distributed as per-state single-band GeoTIFFs in USA Contiguous
Albers Equal Area Conic (ESRI:102039), nodata 255.

Why this and not the LILA "Chesapeake Land Cover" (CVPR 2019) tiles the manifest URL points
at: that ML-ready release is a 6-class product derived from **2013/2014** NAIP -- entirely
pre-2016, outside the Sentinel era (spec rejects all-pre-2016 labels). The manifest's own
time_range [2016, 2022] and its "13/54-class LULC variants" note refer to THIS newer USGS
2022-Edition database, which has a fully post-2016 (2017/18) epoch and the 13-class LC. So
we process the 2017/18 13-class LC instead; see the summary.

Recipe (spec 4, dense_raster / VHR-native): the 1 m categorical LC is reprojected to a
local UTM grid at 10 m/pixel with **MODE** (majority) resampling -- never bilinear -- via a
rasterio WarpedVRT, and tiled into 64x64 patches. Source values 1..12 -> output ids 0..11;
254 (Aberdeen Proving Ground, an unmapped U.S. Army facility) and 255 (nodata) -> 255. Each
state raster is reprojected to the UTM zone of its centroid; candidate tiles whose true UTM
zone differs from that zone are dropped, so every written tile is in its correct local UTM
(this also naturally focuses sampling on the Chesapeake -- eastern -- portion of each state).
Tiles are kept only when >=50% of pixels are labeled. Selection is tiles-per-class balanced
(each tile counts toward every class present), rare classes first, up to 1000 tiles/class
and <=25k total. Time range = a 1-year window on the state's LC epoch year (2017 or 2018).

Reproduce:
    python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.chesapeake_land_cover
"""

import argparse
import math
import multiprocessing
import os
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import rasterio
import shapely
import tqdm
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest

SLUG = "chesapeake_land_cover"
NAME = "Chesapeake Land Cover"
DOI = "https://doi.org/10.5066/P981GV1L"
SB_ITEM = "633302d8d34e900e86c61f81"
SCIENCEBASE_DIR = os.path.join(io.raw_dir(SLUG).path, "sciencebase")

TILE = io.MAX_TILE  # 64
TARGET_RES = 10.0
LABELED_FRAC_MIN = 0.5  # keep a tile only if >=50% of its pixels are labeled
MAX_ATTEMPTS_PER_STATE = 12000  # random grid positions probed per state
PER_CLASS = 1000
SEED = 42

# Per-state 2017/18 LC GeoTIFF (unzipped basename) + LC epoch year (NAIP acquisition year of
# the 2017/18 product for that state, from the data dictionary Table 5 file naming).
STATES: dict[str, tuple[str, int]] = {
    "de": ("de_lc_2018_2022-Edition.tif", 2018),
    "md": ("md_lc_2018_2022-Edition.tif", 2018),
    "va": ("va_lc_2018_2022-Edition.tif", 2018),
    "wv": ("wv_lc_2018_2022-Edition.tif", 2018),
    "ny": ("ny_lc_2017_2022-Edition.tif", 2017),
    "pa": ("pa_lc_2017_2022-Edition.tif", 2017),
    "dc": ("dc_lc_2017_2022-Edition.tif", 2017),
}

# Output classes: (source raster value, name, description). Index = 0-based output id.
# From the 2022-Edition data dictionary "Land Cover (13) Classes" + class table (Table 6).
CLASSES: list[tuple[int, str, str]] = [
    (
        1,
        "Water",
        "All areas of open water, incl. ponds, rivers, lakes, farm ponds, storm-water "
        "retention structures, and boats not attached to docks. MMU 25 m^2.",
    ),
    (
        2,
        "Emergent Wetlands",
        "Low vegetation along marine/estuarine regions with a saturated-ground appearance, "
        "adjacent to major waterways; for VA tidal zones also includes low vegetation, woody "
        "vegetation and barren overlapping NOAA C-CAP wetlands within 1 ft of tidal water. "
        "MMU 225 m^2.",
    ),
    (
        3,
        "Tree Canopy",
        "Deciduous and evergreen woody vegetation >~3 m tall, of natural succession or human "
        "planting; stand-alone, clumped, or interlocking individuals. MMU 9 m^2.",
    ),
    (
        4,
        "Scrub/Shrub",
        "Heterogeneous deciduous/evergreen woody vegetation of variable height: patchy shrubs "
        "and young or stunted trees interspersed with grasses and lower vegetation. MMU 225 m^2.",
    ),
    (
        5,
        "Low Vegetation",
        "Plant material <~3 m tall, incl. lawns, tilled fields, nursery plantings, recently "
        "cut forest-management areas, and natural ground cover. MMU 9 m^2.",
    ),
    (
        6,
        "Barren",
        "Areas void of vegetation of natural earthen material, incl. beaches, mud flats, and "
        "bare ground in construction sites. MMU 25 m^2.",
    ),
    (
        7,
        "Impervious Structures",
        "Human-constructed impervious structures >~2 m tall (houses, malls, electrical "
        "towers). MMU 9 m^2.",
    ),
    (
        8,
        "Other Impervious",
        "Human-constructed non-permeable surfaces <~2 m tall (sidewalks, parking lots, "
        "driveways, some private roads). MMU 9 m^2.",
    ),
    (
        9,
        "Impervious Roads",
        "Impervious surfaces used and maintained for transportation, from local planimetric "
        "and road-network data. MMU 9 m^2.",
    ),
    (
        10,
        "Tree Canopy Over Structures",
        "Tree/forest cover overhanging impervious structures (independently-mapped tree "
        "canopy superimposed on structures). MMU 9 m^2.",
    ),
    (
        11,
        "Tree Canopy Over Other Impervious",
        "Tree/forest cover overhanging other impervious surfaces. MMU 9 m^2.",
    ),
    (
        12,
        "Tree Canopy Over Impervious Roads",
        "Tree/forest cover overhanging impervious roads. MMU 9 m^2.",
    ),
]
NUM_CLASSES = len(CLASSES)  # 12
SRCVAL_TO_ID = {srcval: i for i, (srcval, _n, _d) in enumerate(CLASSES)}
_REMAP = np.full(256, io.CLASS_NODATA, dtype=np.uint8)
for _srcval, _id in [(sv, i) for i, (sv, _n, _d) in enumerate(CLASSES)]:
    _REMAP[_srcval] = _id


def _state_tif(state: str) -> str:
    return os.path.join(SCIENCEBASE_DIR, f"{state}_lc", STATES[state][0])


def _vrt_params(state: str) -> dict[str, Any]:
    """Compute the aligned UTM 10 m grid (crs + snapped transform + size) for a state."""
    with rasterio.open(_state_tif(state)) as ds:
        lon0, lat0, lon1, lat1 = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
        clon, clat = (lon0 + lon1) / 2, (lat0 + lat1) / 2
        proj = get_utm_ups_projection(clon, clat, TARGET_RES, -TARGET_RES)
        utm_crs = proj.crs
        ux0, uy0, ux1, uy1 = transform_bounds(ds.crs, utm_crs, *ds.bounds)
    # Snap origin to a multiple of 10 (aligns to the rslearn Projection pixel grid).
    x0 = math.floor(ux0 / TARGET_RES) * TARGET_RES
    y0 = math.ceil(uy1 / TARGET_RES) * TARGET_RES
    width = int(math.ceil((ux1 - x0) / TARGET_RES))
    height = int(math.ceil((y0 - uy0) / TARGET_RES))
    return {
        "utm_crs": utm_crs.to_string(),
        "x0": x0,
        "y0": y0,
        "width": width,
        "height": height,
        "ncol": width // TILE,
        "nrow": height // TILE,
    }


def _scan_chunk(
    state: str, vp: dict[str, Any], positions: list[tuple[int, int]]
) -> list[dict[str, Any]]:
    """Read a chunk of tile positions from a state's UTM VRT; keep labeled, in-zone tiles."""
    utm_crs = CRS.from_string(vp["utm_crs"])
    dst_transform = Affine(TARGET_RES, 0, vp["x0"], 0, -TARGET_RES, vp["y0"])
    col_base = int(round(vp["x0"] / TARGET_RES))
    row_base = int(round(-vp["y0"] / TARGET_RES))
    proj = Projection(utm_crs, TARGET_RES, -TARGET_RES)
    out: list[dict[str, Any]] = []
    min_labeled = int(LABELED_FRAC_MIN * TILE * TILE)
    with (
        rasterio.open(_state_tif(state)) as ds,
        WarpedVRT(
            ds,
            crs=utm_crs,
            transform=dst_transform,
            width=vp["width"],
            height=vp["height"],
            resampling=Resampling.mode,
            src_nodata=255,
            nodata=255,
        ) as vrt,
    ):
        for r, c in positions:
            arr = vrt.read(1, window=Window(c * TILE, r * TILE, TILE, TILE))
            if arr.shape != (TILE, TILE):
                continue
            out_arr = _REMAP[arr]
            labeled = int((out_arr != io.CLASS_NODATA).sum())
            if labeled < min_labeled:
                continue
            col0 = col_base + c * TILE
            row0 = row_base + r * TILE
            # Verify the tile's true UTM zone matches this state VRT zone; drop if not.
            # rslearn Projection coords are pixel coords (col, row).
            ll = STGeometry(
                proj, shapely.Point(col0 + TILE / 2, row0 + TILE / 2), None
            ).to_projection(WGS84_PROJECTION)
            true_crs = get_utm_ups_projection(
                ll.shp.x, ll.shp.y, TARGET_RES, -TARGET_RES
            ).crs
            if true_crs != utm_crs:
                continue
            present = sorted(int(v) for v in np.unique(out_arr) if v != io.CLASS_NODATA)
            if not present:
                continue
            out.append(
                {
                    "state": state,
                    "crs": vp["utm_crs"],
                    "bounds": (col0, row0, col0 + TILE, row0 + TILE),
                    "classes_present": present,
                    "array": out_arr,
                    "source_id": f"{state}_r{row0}_c{col0}",
                    "year": STATES[state][1],
                }
            )
    return out


def _greedy_balance(cands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tiles-per-class balanced selection: rare classes first, <=PER_CLASS per class,
    <=25k total. A selected tile counts toward every class present in it.
    """
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(cands):
        for cid in r["classes_present"]:
            by_class[cid].append(i)
    rng = np.random.default_rng(SEED)
    for cid in by_class:
        rng.shuffle(by_class[cid])
    counts: Counter = Counter()
    chosen: set[int] = set()
    # Rarest class (fewest candidates) first so scarce classes get filled before the pool
    # is exhausted by common ones.
    for cid in sorted(by_class, key=lambda k: len(by_class[k])):
        for idx in by_class[cid]:
            if counts[cid] >= PER_CLASS:
                break
            if idx in chosen:
                continue
            if len(chosen) >= 25000:
                break
            chosen.add(idx)
            for c2 in cands[idx]["classes_present"]:
                counts[c2] += 1
        if len(chosen) >= 25000:
            break
    return [cands[i] for i in sorted(chosen)]


def _write_one(rec: dict[str, Any]) -> tuple[str, list[int]]:
    sid = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sid}.tif").exists():
        return sid, rec["classes_present"]
    proj = Projection(CRS.from_string(rec["crs"]), TARGET_RES, -TARGET_RES)
    bounds = tuple(rec["bounds"])
    io.write_label_geotiff(
        SLUG, sid, rec["array"], proj, bounds, nodata=io.CLASS_NODATA
    )
    io.write_sample_json(
        SLUG,
        sid,
        proj,
        bounds,
        io.year_range(rec["year"]),
        source_id=rec["source_id"],
        classes_present=rec["classes_present"],
    )
    return sid, rec["classes_present"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    states = [s for s in STATES if os.path.exists(_state_tif(s))]
    missing = [s for s in STATES if s not in states]
    if missing:
        print(
            f"WARNING: missing unzipped rasters for states {missing}; proceeding with {states}"
        )
    if not states:
        raise FileNotFoundError(
            f"No state LC rasters under {SCIENCEBASE_DIR}. Download + unzip the 2017/18 "
            f"LC zips from ScienceBase item {SB_ITEM} ({DOI})."
        )

    # ---- Phase 1: probe random tile positions per state, in parallel ------------------
    tasks: list[dict[str, Any]] = []
    for si, state in enumerate(states):
        vp = _vrt_params(state)
        rng = np.random.default_rng(SEED * 1000 + si)
        n = vp["ncol"] * vp["nrow"]
        k = min(MAX_ATTEMPTS_PER_STATE, n)
        flat = rng.choice(n, size=k, replace=False)
        positions = [(int(f // vp["ncol"]), int(f % vp["ncol"])) for f in flat]
        # split into chunks for the pool
        for i in range(0, len(positions), 400):
            tasks.append(dict(state=state, vp=vp, positions=positions[i : i + 400]))
    print(
        f"scanning {len(states)} states, {sum(len(t['positions']) for t in tasks)} probes "
        f"in {len(tasks)} chunks"
    )

    cands: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as pool:
        for recs in tqdm.tqdm(
            star_imap_unordered(pool, _scan_chunk, tasks), total=len(tasks)
        ):
            cands.extend(recs)
    print(f"candidate tiles (>=50% labeled, in-zone): {len(cands)}")
    avail = Counter()
    for r in cands:
        for cid in r["classes_present"]:
            avail[cid] += 1
    print("candidate tiles per class:")
    for i, (_sv, name, _d) in enumerate(CLASSES):
        print(f"  {i:>2} {name:34} {avail.get(i, 0)}")

    # ---- Phase 2: tiles-per-class balanced selection ---------------------------------
    selected = _greedy_balance(cands)
    selected.sort(key=lambda r: r["source_id"])
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} tiles (<= {PER_CLASS}/class, 25k cap)")

    # ---- Phase 3: write patches in parallel ------------------------------------------
    tile_counts = Counter()
    with multiprocessing.Pool(args.workers) as pool:
        done = 0
        for _sid, present in tqdm.tqdm(
            star_imap_unordered(pool, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            for cid in present:
                tile_counts[cid] += 1
            done += 1
            if done % 2000 == 0:
                io.check_disk()

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "USGS ScienceBase / Chesapeake Bay Program",
            "license": "public domain (U.S. Government work; USGS data release)",
            "provenance": {
                "url": DOI,
                "sciencebase_item": SB_ITEM,
                "have_locally": False,
                "annotation_method": (
                    "1 m land cover from eCognition supervised classification of NAIP + lidar "
                    "with manual QA/photointerpretation (Chesapeake Conservancy / UVM SAL / USGS)"
                ),
                "product": "Land Cover (LC), 13-class, 2017/18 epoch, 2022 Edition",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc, "source_value": srcval}
                for i, (srcval, name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_tile_counts": {
                CLASSES[i][1]: int(tile_counts.get(i, 0)) for i in range(NUM_CLASSES)
            },
            "states": states,
            "notes": (
                "1 m 13-class Land Cover (2017/18 epoch) reprojected from ESRI:102039 (Albers) to "
                "local UTM at 10 m with MODE resampling and tiled into 64x64 patches. Source "
                "values 1..12 -> ids 0..11; 254 (Aberdeen Proving Ground, unmapped) and 255 -> "
                "nodata 255. Kept tiles with >=50% labeled pixels; each state reprojected to its "
                "centroid UTM zone with out-of-zone tiles dropped (focuses on the Chesapeake/"
                "eastern watershed). Tiles-per-class balanced, <=1000 tiles/class, 25k cap. "
                "Time range = 1-year window on each state's LC epoch year (2017 or 2018). "
                "Low-confidence at 10 m (thin/overlap classes rarely survive majority "
                "resampling): Impervious Roads, Tree Canopy Over Structures / Other Impervious / "
                "Impervious Roads -- kept per spec, downstream filters too-small classes. "
                "NOTE: the LILA CVPR-2019 'Chesapeake Land Cover' 6-class tiles (manifest URL) are "
                "2013/14 (all pre-2016) and were NOT used; this uses the post-2016 USGS 2022 "
                "Edition 13-class LC instead."
            ),
        },
    )
    print("tile counts per class:")
    for i, (_sv, name, _d) in enumerate(CLASSES):
        print(f"  {i:>2} {name:34} {tile_counts.get(i, 0)}")

    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
