"""Process ARCC10-IM (Abandoned & Reclaimed Cropland, Inner Mongolia) into label patches.

Source: figshare 25687278 (Wuyun et al., Sci Data), "A 10-meter annual cropland activity
map and dataset of abandonment and reclaimed cropland". The dataset bundles, over Inner
Mongolia, China, for study years 2016-2023:

  * ARCC10-IM-ACA  -- 8 annual 10 m cropland-ACTIVITY maps (one GeoTIFF/year), values
                      {1: inactive cropland, 2: active cropland}, 0 = nodata/non-cropland;
                      plus per-year reference sample points ({type 0: inactive, 1: active}).
  * ARCC10-IM_AC   -- abandoned-cropland mask (1 = abandoned).
  * ARCC10-IM_RC   -- reclaimed-cropland mask (1 = reclaimed).
  * ARCC10-IM-CLU  -- cumulative 2016-2023 land-use ({1: continuously abandoned,
                      2: unstable, 3: continuously active}).

CHANGE-TIMING DECISION (spec 5 / 8): the AC / RC / CLU layers encode cropland
ABANDONMENT / RECLAMATION *transitions* derived by a multi-year sliding-window temporal
segmentation. Those events are only resolved to a year-of-change (or a multi-year span),
never to ~1-2 months, so per spec 5 they are NOT usable as dated change labels and we do
NOT force a change_time. Instead we use the ANNUAL cropland-ACTIVITY maps (ARCC10-IM-ACA):
each pixel's per-YEAR state (active vs inactive cropland) is a persistent static class over
that year's 1-year window (change_time=null). This is exactly the "recast as a persistent
per-year state" path the spec allows, and keeps every label post-2016.

So this is a two-class dense_raster classification (like rapeseedmap10):

    id 0 = inactive cropland   (fallow / abandoned / bare in that year; source value 1)
    id 1 = active cropland      (cultivated in that year; source value 2)

The annual maps are large regional derived-product rasters (321788 x 177294 px, EPSG:4326
~10 m), so we do BOUNDED-TILE dense sampling (spec 5): scan every annual raster in
64x64 native blocks, keep high-cropland-coverage blocks (>= MIN_VALID_FRAC mapped cropland
so the tile is not dominated by non-cropland ignore), reproject each selected block to its
local UTM zone at 10 m (nearest resampling; categorical), and write a 64x64 label patch.
Tiles-per-class balanced (spec 5): inactive is the rarer class and is prioritized. Each
tile is tagged with a 1-year time range on its labeled year; change_time=null.

Run (from repo root):
    python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.\
arcc10_im_abandoned_reclaimed_cropland_inner_mongolia
"""

import argparse
import glob
import multiprocessing
import os
from collections import Counter
from typing import Any

import numpy as np
import rasterio
import rasterio.windows as rw
import tqdm
from affine import Affine
from rasterio.warp import Resampling, reproject
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, sampling

SLUG = "arcc10_im_abandoned_reclaimed_cropland_inner_mongolia"

# Source raster encoding.
VAL_INACTIVE = 1
VAL_ACTIVE = 2

# Output class ids (must start at 0; inactive is the rarer class, kept as id 0).
CLASSES = [
    (
        "inactive cropland",
        "Cropland parcel that was NOT actively cultivated in the labeled year "
        "(fallow / abandoned / bare). Source ARCC10-IM annual cropland-activity value 1. "
        "A per-year persistent state, not a dated change event.",
    ),
    (
        "active cropland",
        "Cropland parcel under active cultivation in the labeled year (has a growing-season "
        "crop signal in the Sentinel-1/2 time series). Source ARCC10-IM annual value 2.",
    ),
]
INACTIVE_ID = 0
ACTIVE_ID = 1

# Sampling parameters.
BLOCK = 64  # native-pixel block == output tile size (64 px * 10 m = 640 m).
PER_CLASS = 1000
MIN_VALID_FRAC = (
    0.40  # block must be >=40% mapped cropland (few non-cropland/ignore px).
)
MIN_CLASS_FRAC = (
    0.05  # a class counts as present in a tile at >=5% (>=205 px) coverage.
)
# Per-scan-chunk reservoir caps (bound memory; plenty to balance ~1000/class from).
CAP_INACT_PER_CHUNK = 12
CAP_ACT_PER_CHUNK = 3
CHUNK = 3840  # block-aligned 2D scan window (mult of 64 and native 128 tiling).
REPROJ_MARGIN = 130  # native-px margin around a block for reprojection source.
SEED = 42

YEARS = list(range(2016, 2024))


def _list_source_tifs() -> list[tuple[int, str]]:
    """Return [(year, path)] for the 8 annual cropland-activity maps."""
    base = io.raw_dir(SLUG) / "ACA" / "ARCC10-IM_ACA"
    out = []
    for year in YEARS:
        matches = glob.glob(
            str(base / f"ARCC10-IM_{year}" / f"classified_cropland_{year}_final*.tif")
        )
        if matches:
            out.append((year, matches[0]))
    return out


def _classes_present(n_inact: int, n_act: int) -> list[int]:
    thr = MIN_CLASS_FRAC * BLOCK * BLOCK
    present = []
    if n_inact >= thr:
        present.append(INACTIVE_ID)
    if n_act >= thr:
        present.append(ACTIVE_ID)
    return present


def scan_chunk(year: int, path: str, row0: int, col0: int) -> list[dict[str, Any]]:
    """Scan one block-aligned 2D chunk; return per-block candidate records."""
    import random
    import zlib

    rng = random.Random(zlib.crc32(f"{year}_{row0}_{col0}".encode()))
    inact: list[dict[str, Any]] = []
    act: list[dict[str, Any]] = []
    n_inact_seen = 0
    n_act_seen = 0
    thr_valid = MIN_VALID_FRAC * BLOCK * BLOCK

    with rasterio.open(path) as ds:
        W, H = ds.width, ds.height
        cw = min(CHUNK, W - col0)
        ch = min(CHUNK, H - row0)
        nbx = cw // BLOCK
        nby = ch // BLOCK
        if nbx == 0 or nby == 0:
            return []
        arr = ds.read(1, window=rw.Window(col0, row0, nbx * BLOCK, nby * BLOCK))
        # Quick reject: no mapped cropland anywhere in the chunk.
        valid_mask = (arr == VAL_INACTIVE) | (arr == VAL_ACTIVE)
        if not valid_mask.any():
            return []
        # (nby, BLOCK, nbx, BLOCK) -> (nby, nbx, BLOCK*BLOCK)
        blocks = arr.reshape(nby, BLOCK, nbx, BLOCK).transpose(0, 2, 1, 3)
        blocks = blocks.reshape(nby, nbx, BLOCK * BLOCK)
        n_inact_arr = (blocks == VAL_INACTIVE).sum(axis=2)
        n_act_arr = (blocks == VAL_ACTIVE).sum(axis=2)
        n_valid_arr = n_inact_arr + n_act_arr
        for bi in range(nby):
            for bj in range(nbx):
                nv = int(n_valid_arr[bi, bj])
                if nv < thr_valid:
                    continue
                ni = int(n_inact_arr[bi, bj])
                na = int(n_act_arr[bi, bj])
                present = _classes_present(ni, na)
                if not present:
                    continue
                row_c = row0 + bi * BLOCK + BLOCK // 2
                col_c = col0 + bj * BLOCK + BLOCK // 2
                lon, lat = ds.xy(row_c, col_c)
                rec = {
                    "src": path,
                    "col": col_c,
                    "row": row_c,
                    "lon": float(lon),
                    "lat": float(lat),
                    "year": year,
                    "classes_present": present,
                }
                if INACTIVE_ID in present:
                    n_inact_seen += 1
                    if len(inact) < CAP_INACT_PER_CHUNK:
                        inact.append(rec)
                    else:
                        k = rng.randint(0, n_inact_seen - 1)
                        if k < CAP_INACT_PER_CHUNK:
                            inact[k] = rec
                else:  # active-only
                    n_act_seen += 1
                    if len(act) < CAP_ACT_PER_CHUNK:
                        act.append(rec)
                    else:
                        k = rng.randint(0, n_act_seen - 1)
                        if k < CAP_ACT_PER_CHUNK:
                            act[k] = rec
    return inact + act


def _write_one(rec: dict[str, Any]) -> None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return

    proj, col, row = io.lonlat_to_utm_pixel(rec["lon"], rec["lat"])
    bounds = io.centered_bounds(col, row, BLOCK, BLOCK)
    dst_transform = Affine(
        proj.x_resolution,
        0,
        bounds[0] * proj.x_resolution,
        0,
        proj.y_resolution,
        bounds[1] * proj.y_resolution,
    )

    m = REPROJ_MARGIN
    with rasterio.open(rec["src"]) as ds:
        c0 = max(0, rec["col"] - m)
        r0 = max(0, rec["row"] - m)
        c1 = min(ds.width, rec["col"] + m)
        r1 = min(ds.height, rec["row"] + m)
        win = rw.Window(c0, r0, c1 - c0, r1 - r0)
        src_arr = ds.read(1, window=win)
        src_transform = ds.window_transform(win)
        src_crs = ds.crs

    raw = np.zeros((BLOCK, BLOCK), dtype=np.uint8)  # 0 == source nodata
    reproject(
        source=src_arr,
        destination=raw,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=proj.crs,
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0,
    )
    out = np.full((BLOCK, BLOCK), io.CLASS_NODATA, dtype=np.uint8)
    out[raw == VAL_INACTIVE] = INACTIVE_ID
    out[raw == VAL_ACTIVE] = ACTIVE_ID
    present = sorted(int(v) for v in np.unique(out) if v != io.CLASS_NODATA)
    if not present:
        return  # degenerate reprojection (all nodata); skip

    io.write_label_geotiff(SLUG, sample_id, out, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(rec["year"]),
        change_time=None,
        source_id=f"{os.path.basename(rec['src'])}:{rec['col']}_{rec['row']}",
        classes_present=present,
    )


def _chunk_tasks(tifs: list[tuple[int, str]]) -> list[dict[str, Any]]:
    tasks = []
    for year, path in tifs:
        with rasterio.open(path) as ds:
            W, H = ds.width, ds.height
        for row0 in range(0, H - BLOCK + 1, CHUNK):
            for col0 in range(0, W - BLOCK + 1, CHUNK):
                tasks.append(dict(year=year, path=path, row0=row0, col0=col0))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()

    tifs = _list_source_tifs()
    print(f"{len(tifs)} annual source rasters: years {[y for y, _ in tifs]}")
    if len(tifs) != len(YEARS):
        raise RuntimeError(f"expected {len(YEARS)} annual rasters, found {len(tifs)}")

    tasks = _chunk_tasks(tifs)
    print(f"{len(tasks)} scan chunks")

    # Scan phase.
    with multiprocessing.Pool(args.workers) as p:
        results = list(
            tqdm.tqdm(
                star_imap_unordered(p, scan_chunk, tasks),
                total=len(tasks),
                desc="scan",
            )
        )
    candidates = [r for sub in results for r in sub]
    n_inact = sum(1 for r in candidates if INACTIVE_ID in r["classes_present"])
    n_act = sum(1 for r in candidates if ACTIVE_ID in r["classes_present"])
    print(
        f"candidates: {len(candidates)} "
        f"(containing inactive={n_inact}, containing active={n_act})"
    )

    io.check_disk()

    # Tiles-per-class balanced selection (rarest class prioritized; 25k cap enforced).
    selected = sampling.balance_tiles_by_class(
        candidates, classes_key="classes_present", per_class=PER_CLASS, seed=SEED
    )
    for i, r in enumerate(
        sorted(selected, key=lambda r: (r["year"], r["src"], r["row"], r["col"]))
    ):
        r["sample_id"] = f"{i:06d}"
    print(f"selected {len(selected)} tiles")

    # Write phase.
    with multiprocessing.Pool(args.workers) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write",
        ):
            pass

    # Count actually-written samples and per-class tile counts.
    written = sorted(glob.glob(str(io.locations_dir(SLUG) / "*.tif")))
    sel_by_id = {r["sample_id"]: r for r in selected}
    class_tiles: Counter = Counter()
    per_year: Counter = Counter()
    n_written = 0
    for t in written:
        sid = os.path.basename(t)[:-4]
        r = sel_by_id.get(sid)
        if r is None:
            continue
        n_written += 1
        per_year[r["year"]] += 1
        for c in r["classes_present"]:
            class_tiles[c] += 1

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "ARCC10-IM (Abandoned & Reclaimed Cropland, Inner Mongolia)",
            "task_type": "classification",
            "source": "figshare 25687278 (Wuyun et al., Sci Data)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://figshare.com/articles/dataset/_b_A_10-meter_annual_cropland_activity_map_and_dataset_of_abandonment_and_reclaimed_cropland_b_/25687278",
                "have_locally": False,
                "annotation_method": "ML classification of Sentinel-1/2 time series (multi-feature stacking); reference sample points per year",
            },
            "sensors_relevant": ["sentinel2", "sentinel1"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": n_written,
            "class_tile_counts": {
                CLASSES[c][0]: class_tiles.get(c, 0) for c in (INACTIVE_ID, ACTIVE_ID)
            },
            "per_year_counts": {str(y): per_year.get(y, 0) for y in YEARS},
            "notes": (
                "Bounded-tile dense_raster sampling from the ARCC10-IM annual cropland-"
                "activity maps (8 years, 2016-2023, EPSG:4326 ~10 m). 64x64 tiles "
                "reprojected to local UTM at 10 m (nearest resampling). Two classes: "
                "0=inactive cropland, 1=active cropland (source values 1/2 remapped; "
                "0=non-cropland -> 255 ignore). Tiles require >=40% mapped cropland; a class "
                "counts as present at >=5% coverage; tiles-per-class balanced (inactive is "
                "the rarer class, prioritized). Each tile has a 1-year time range on its "
                "labeled year, change_time=null: the per-year active/inactive state is a "
                "persistent static class, NOT a dated event. The dataset's ABANDONMENT / "
                "RECLAMATION (AC/RC/CLU) transition layers were intentionally NOT used as "
                "change labels because their change dates are only year/multi-year resolved "
                "(coarser than the spec's ~1-2 month change-timing requirement)."
            ),
        },
    )
    print(f"wrote metadata; num_samples={n_written} class_tiles={dict(class_tiles)}")
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
