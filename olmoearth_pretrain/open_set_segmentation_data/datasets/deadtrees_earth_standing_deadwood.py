"""deadtrees.earth standing-deadwood -> fractional-cover regression label patches.

Source: deadtrees.earth (Univ. Freiburg / Wageningen; Mosig, Schiefer, Kattenborn et al.,
Remote Sensing of Environment 2025, doi:10.1016/j.rse.2025.115027). An open-access global
database of centimeter-scale drone/aerial orthophotos with expert-delineated *standing
deadwood* polygons. Each manual label set is tied to one orthophoto and to an
Area-Of-Interest (AOI) multipolygon: inside the AOI, area not delineated as deadwood is
known to be alive tree / non-tree, so the AOI defines the *observed* extent (outside the
AOI is unobserved). Data are public/CC BY and are read from the platform's authentication-
free self-hosted Supabase REST API (https://supabase.deadtrees.earth/rest/v1), the same
public endpoint the website uses. We download ONLY the labels (deadwood polygons + AOI +
per-orthophoto acquisition date); the cm-scale RGB orthomosaics themselves are NOT needed
because pretraining supplies its own Sentinel imagery.

RESOLUTION / OBSERVABILITY DECISION (the crux for this dataset)
--------------------------------------------------------------
Native labels are centimeter-scale; an *individual* standing-dead tree is sub-pixel at 10 m
Sentinel resolution. Encoding presence/absence of individual dead trees at 10 m would be
dishonest. Instead we aggregate the cm-scale deadwood mask into the honest 10 m signal that
the manifest itself calls out ("Deadwood fractional cover predictable at 10 m") and that
deadtrees.earth's own satellite models regress: **fractional standing-deadwood cover per
10 m pixel** = (deadwood area) / (observed area) within each 10 m cell, restricted to the
labeled AOI. This is a REGRESSION target in [0, 1]. Aggregation: rasterize the WGS84
deadwood polygons and the AOI multipolygon onto a fine 0.5 m sub-grid in local UTM, clip
deadwood to the AOI, then average each 20x20 sub-block down to one 10 m pixel. A 10 m pixel
is kept only if >= 50% of its area lies inside the AOI (else nodata); its value is the
fraction of that observed area covered by deadwood.

Only the manual **expert** delineations (label_source = visual_interpretation) are used --
the ~12k SegFormer auto-prediction label sets on the platform are a derived ML product and
are excluded to keep this a high-confidence reference bank (SOP: prefer reference over
derived maps). Datasets are further restricted to public, CC BY, non-archived records with
a complete acquisition date in the Sentinel era (>= 2016). AOIs are small drone footprints
(median ~230 m), so most datasets yield a single tile sized to their footprint; larger
airborne AOIs are tiled into a <=64x64 grid.

Output: single-band float32 GeoTIFFs, local UTM, 10 m/pixel, <=64x64, values = deadwood
fraction 0..1, nodata -99999 (io.REGRESSION_NODATA). change_time=null (single-date state);
one <=1-year time range per tile anchored on the orthophoto acquisition year.

Run:  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.deadtrees_earth_standing_deadwood
Idempotent: existing raw label files and output .tif tiles are skipped.
"""

import argparse
import json
import math
import multiprocessing
import urllib.parse
import urllib.request
from typing import Any

import numpy as np
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.mp import star_imap_unordered
from shapely.geometry import shape

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)

SLUG = "deadtrees_earth_standing_deadwood"
NAME = "deadtrees.earth (standing deadwood)"
URL = "https://deadtrees.earth"
DOI = "https://doi.org/10.1016/j.rse.2025.115027"

SUPABASE = "https://supabase.deadtrees.earth/rest/v1"
ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.ewogICJyb2xlIjogImFub24iLAogICJpc3MiOiAic3Vw"
    "YWJhc2UiLAogICJpYXQiOiAxNzQwODcwMDAwLAogICJleHAiOiAxODk4NjM2NDAwCn0."
    "A3HdTofLNcrRrtDDbDAP9kRBobxXqnUKB6IYHvM6da4"
)
HDR = {"apikey": ANON_KEY, "Authorization": f"Bearer {ANON_KEY}"}

SUB = 20  # 0.5 m sub-pixels per 10 m cell (10 / 20)
TILE = 64  # hard cap tile size (px) => 640 m
MIN_OBS_PIXELS = 25  # keep a tile only if >= 25 observed (in-AOI) 10 m pixels (0.25 ha)
MIN_CELL_OBS_FRAC = (
    0.5  # a 10 m pixel is "observed" if >=50% of its area is inside the AOI
)
REGRESSION_TOTAL = 5000  # regression cap per spec (well above expected count here)
MIN_YEAR = 2016
SEED = 42


# ---------------------------------------------------------------------------
# Supabase REST (authentication-free public read; same endpoint the site uses)
# ---------------------------------------------------------------------------
def _rest_all(
    path: str, params: dict[str, str], page: int = 1000, order: str = "id"
) -> list[dict[str, Any]]:
    # PostgREST range-paging is only stable with an explicit ORDER BY; without it
    # multi-page queries silently skip/repeat rows (observed dropping ~25% of the
    # dataset view). Always page over a deterministic order.
    out: list[dict[str, Any]] = []
    off = 0
    params = {**params, "order": order}
    while True:
        url = SUPABASE + path + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url,
            headers={**HDR, "Range-Unit": "items", "Range": f"{off}-{off + page - 1}"},
        )
        with urllib.request.urlopen(req, timeout=180) as r:
            rows = json.loads(r.read())
        out.extend(rows)
        if len(rows) < page:
            break
        off += page
    return out


def labels_dir():
    return io.raw_dir(SLUG) / "labels"


def gather_labels() -> list[dict[str, Any]]:
    """Qualifying manual deadwood label sets joined with public dataset metadata."""
    labels = _rest_all(
        "/v2_labels",
        {
            "label_data": "eq.deadwood",
            "label_type": "eq.semantic_segmentation",
            "label_source": "eq.visual_interpretation",
            "is_active": "eq.true",
            "select": "id,dataset_id,aoi_id,label_quality",
        },
    )
    ds = _rest_all(
        "/v2_full_dataset_view_public",
        {
            "select": "id,license,platform,aquisition_year,aquisition_month,"
            "aquisition_day,data_access,archived"
        },
    )
    by_id = {d["id"]: d for d in ds}
    out: list[dict[str, Any]] = []
    for lab in labels:
        d = by_id.get(lab["dataset_id"])
        if not d or d.get("data_access") != "public" or d.get("archived"):
            continue
        if lab.get("aoi_id") is None:
            continue
        y, m, day = (
            d.get("aquisition_year"),
            d.get("aquisition_month"),
            d.get("aquisition_day"),
        )
        if not (y and m and day) or y < MIN_YEAR:
            continue
        out.append(
            {
                "label_id": lab["id"],
                "dataset_id": lab["dataset_id"],
                "aoi_id": lab["aoi_id"],
                "label_quality": lab.get("label_quality"),
                "year": int(y),
                "license": d.get("license"),
                "platform": d.get("platform"),
            }
        )
    out.sort(key=lambda r: r["label_id"])
    return out


def _download_one(rec: dict[str, Any]) -> dict[str, Any]:
    """Fetch AOI geometry + deadwood polygons for one label; write raw json (atomic)."""
    label_id = rec["label_id"]
    dst = labels_dir() / f"{label_id}.json"
    if dst.exists():
        return {"label_id": label_id, "skipped": True}
    aoi_rows = _rest_all(
        "/v2_aois", {"id": f"eq.{rec['aoi_id']}", "select": "geometry,is_whole_image"}
    )
    aoi_geom = aoi_rows[0].get("geometry") if aoi_rows else None
    dw = _rest_all(
        "/v2_deadwood_geometries",
        {"label_id": f"eq.{label_id}", "is_deleted": "eq.false", "select": "geometry"},
    )
    polys = [g["geometry"] for g in dw if g.get("geometry")]
    obj = {**rec, "aoi": aoi_geom, "deadwood": polys}
    labels_dir().mkdir(parents=True, exist_ok=True)
    tmp = labels_dir() / f"{label_id}.json.tmp"
    with tmp.open("w") as f:
        json.dump(obj, f)
    tmp.rename(dst)
    return {"label_id": label_id, "n_polys": len(polys)}


def download_all(records: list[dict[str, Any]], workers: int) -> None:
    labels_dir().mkdir(parents=True, exist_ok=True)
    jobs = [dict(rec=r) for r in records]
    with multiprocessing.Pool(min(workers, 32)) as p:
        for _ in tqdm.tqdm(
            star_imap_unordered(p, _download_one, jobs),
            total=len(jobs),
            desc="download",
        ):
            pass


# ---------------------------------------------------------------------------
# Tiling / fractional-cover computation
# ---------------------------------------------------------------------------
def _aoi_pixel_box(aoi_geom, proj10):
    px = STGeometry(WGS84_PROJECTION, aoi_geom, None).to_projection(proj10).shp
    minx, miny, maxx, maxy = px.bounds
    return (
        int(math.floor(minx)),
        int(math.floor(miny)),
        int(math.ceil(maxx)),
        int(math.ceil(maxy)),
    )


def _observed_pixels(aoi_geom, projfine, bounds10) -> int:
    """Count 10 m pixels in a tile whose area is >= MIN_CELL_OBS_FRAC inside the AOI."""
    x0, y0, x1, y1 = bounds10
    W, H = x1 - x0, y1 - y0
    fine_bounds = (x0 * SUB, y0 * SUB, x1 * SUB, y1 * SUB)
    aoi_fine = geom_to_pixels(aoi_geom, WGS84_PROJECTION, projfine)
    aoi_sub = rasterize_shapes([(aoi_fine, 1)], fine_bounds, fill=0, dtype="uint8")[0]
    aoi_blk = aoi_sub.reshape(H, SUB, W, SUB).sum(axis=(1, 3)).astype(np.float32)
    return int((aoi_blk >= (MIN_CELL_OBS_FRAC * SUB * SUB)).sum())


def plan_label(rec: dict[str, Any]) -> dict[str, Any]:
    """Return candidate tile bounds (10 m px) for one label's AOI box.

    Tiles with fewer than MIN_OBS_PIXELS observed (in-AOI) 10 m pixels are dropped here
    (observed area depends only on the AOI, not on deadwood), so downstream sample ids are
    contiguous and num_samples is exact.
    """
    label_id = rec["label_id"]
    with (labels_dir() / f"{label_id}.json").open() as f:
        obj = json.load(f)
    if not obj.get("aoi"):
        return {"label_id": label_id, "tiles": []}
    aoi_geom = shape(obj["aoi"])
    c = aoi_geom.centroid
    proj10 = get_utm_ups_projection(c.x, c.y, io.RESOLUTION, -io.RESOLUTION)
    projfine = get_utm_ups_projection(
        c.x, c.y, io.RESOLUTION / SUB, -io.RESOLUTION / SUB
    )
    col0, row0, col1, row1 = _aoi_pixel_box(aoi_geom, proj10)
    w, h = col1 - col0, row1 - row0
    if w <= 0 or h <= 0:
        return {"label_id": label_id, "tiles": []}
    tiles = []
    for r in range(row0, row1, TILE):
        for cc in range(col0, col1, TILE):
            tw = min(TILE, col1 - cc)
            th = min(TILE, row1 - r)
            bounds = (cc, r, cc + tw, r + th)
            if _observed_pixels(aoi_geom, projfine, bounds) >= MIN_OBS_PIXELS:
                tiles.append(bounds)
    return {
        "label_id": label_id,
        "lon": float(c.x),
        "lat": float(c.y),
        "crs": proj10.crs.to_string(),
        "tiles": tiles,
    }


def _fraction_for_bounds(aoi_geom, polys, projfine, bounds10):
    """Deadwood fraction (H,W) float32 for a 10 m tile; nodata where < MIN_CELL_OBS_FRAC in AOI."""
    x0, y0, x1, y1 = bounds10
    W, H = x1 - x0, y1 - y0
    fine_bounds = (x0 * SUB, y0 * SUB, x1 * SUB, y1 * SUB)
    aoi_fine = geom_to_pixels(aoi_geom, WGS84_PROJECTION, projfine)
    aoi_sub = rasterize_shapes([(aoi_fine, 1)], fine_bounds, fill=0, dtype="uint8")[0]
    if polys:
        shapes = [(geom_to_pixels(p, WGS84_PROJECTION, projfine), 1) for p in polys]
        dead_sub = rasterize_shapes(shapes, fine_bounds, fill=0, dtype="uint8")[0]
        dead_sub = dead_sub & aoi_sub
    else:
        dead_sub = np.zeros_like(aoi_sub)
    aoi_blk = aoi_sub.reshape(H, SUB, W, SUB).sum(axis=(1, 3)).astype(np.float32)
    dead_blk = dead_sub.reshape(H, SUB, W, SUB).sum(axis=(1, 3)).astype(np.float32)
    frac = np.full((H, W), float(io.REGRESSION_NODATA), dtype=np.float32)
    obs = aoi_blk >= (MIN_CELL_OBS_FRAC * SUB * SUB)
    frac[obs] = np.clip(dead_blk[obs] / aoi_blk[obs], 0.0, 1.0)
    return frac, int(obs.sum())


def write_label_tiles(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Write all assigned tiles for one label; return per-tile stats."""
    label_id = job["label_id"]
    with (labels_dir() / f"{label_id}.json").open() as f:
        obj = json.load(f)
    aoi_geom = shape(obj["aoi"])
    polys = [shape(p) for p in obj.get("deadwood", [])]
    c = aoi_geom.centroid
    projfine = get_utm_ups_projection(
        c.x, c.y, io.RESOLUTION / SUB, -io.RESOLUTION / SUB
    )
    proj10 = get_utm_ups_projection(c.x, c.y, io.RESOLUTION, -io.RESOLUTION)
    year = job["year"]
    tr = io.year_range(year)
    src_id = f"dataset_{job['dataset_id']}_label_{label_id}"

    stats = []
    for sample_id, bounds in job["assign"]:
        tif_path = io.locations_dir(SLUG) / f"{sample_id}.tif"
        if tif_path.exists():
            import rasterio

            with rasterio.open(tif_path.path) as ds:
                ev = ds.read(1)
            good = ev[ev != io.REGRESSION_NODATA]
            stats.append(
                {
                    "sample_id": sample_id,
                    "n_obs": int(good.size),
                    "mean": float(good.mean()) if good.size else 0.0,
                    "max": float(good.max()) if good.size else 0.0,
                    "n_pos": int((good > 0).sum()),
                }
            )
            continue
        frac, n_obs = _fraction_for_bounds(aoi_geom, polys, projfine, bounds)
        io.write_label_geotiff(
            SLUG, sample_id, frac, proj10, bounds, nodata=io.REGRESSION_NODATA
        )
        io.write_sample_json(
            SLUG, sample_id, proj10, bounds, tr, change_time=None, source_id=src_id
        )
        good = frac[frac != io.REGRESSION_NODATA]
        stats.append(
            {
                "sample_id": sample_id,
                "n_obs": int(good.size),
                "mean": float(good.mean()) if good.size else 0.0,
                "max": float(good.max()) if good.size else 0.0,
                "n_pos": int((good > 0).sum()),
            }
        )
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    records = gather_labels()
    print(f"qualifying manual deadwood label sets: {len(records)}")
    if not records:
        manifest.write_registry_entry(
            SLUG,
            "temporary_failure",
            notes="deadtrees.earth Supabase returned no qualifying manual deadwood labels",
        )
        raise SystemExit("no qualifying labels")

    download_all(records, args.workers)
    io.check_disk()

    # Plan candidate tiles per label (parallel; cheap AOI-only pass).
    with multiprocessing.Pool(min(args.workers, 32)) as p:
        plans = list(
            tqdm.tqdm(
                star_imap_unordered(p, plan_label, [dict(rec=r) for r in records]),
                total=len(records),
                desc="plan",
            )
        )
    plans_by_id = {pl["label_id"]: pl for pl in plans}
    rec_by_id = {r["label_id"]: r for r in records}

    # Flatten in deterministic (label_id, tile) order and assign running sample ids.
    flat: list[tuple[int, tuple[int, int, int, int]]] = []
    for label_id in sorted(plans_by_id):
        for t in plans_by_id[label_id]["tiles"]:
            flat.append((label_id, tuple(t)))
    print(f"candidate tiles across labels: {len(flat)}")

    # Regression cap (deterministic subsample if ever exceeded; not expected here).
    if len(flat) > REGRESSION_TOTAL:
        import random

        rng = random.Random(SEED)
        rng.shuffle(flat)
        flat = sorted(flat[:REGRESSION_TOTAL], key=lambda x: (x[0], x[1]))
        print(f"capped to {len(flat)} tiles (regression cap)")

    assign_by_label: dict[int, list] = {}
    for i, (label_id, bounds) in enumerate(flat):
        assign_by_label.setdefault(label_id, []).append((f"{i:06d}", bounds))

    io.locations_dir(SLUG).mkdir(parents=True, exist_ok=True)
    jobs = [
        dict(
            job=dict(
                label_id=label_id,
                dataset_id=rec_by_id[label_id]["dataset_id"],
                year=rec_by_id[label_id]["year"],
                assign=assign,
            )
        )
        for label_id, assign in assign_by_label.items()
    ]
    all_stats: list[dict[str, Any]] = []
    with multiprocessing.Pool(args.workers) as p:
        for s in tqdm.tqdm(
            star_imap_unordered(p, write_label_tiles, jobs),
            total=len(jobs),
            desc="write",
        ):
            all_stats.extend(s)

    # Drop tiles that ended up with too little observed area (should be rare); count kept.
    kept = [s for s in all_stats if s["n_obs"] >= MIN_OBS_PIXELS]
    n_written = len(all_stats)
    means = (
        np.array([s["mean"] for s in kept], dtype=np.float64)
        if kept
        else np.array([0.0])
    )
    pix_max = max((s["max"] for s in kept), default=0.0)
    pos_tile_frac = float(np.mean([s["n_pos"] > 0 for s in kept])) if kept else 0.0

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "regression",
            "source": "deadtrees.earth (Univ. Freiburg / Wageningen)",
            "license": "CC BY 4.0",
            "provenance": {
                "url": URL,
                "doi": DOI,
                "have_locally": False,
                "annotation_method": "manual expert delineation on cm-scale aerial imagery",
                "access": "public Supabase REST (supabase.deadtrees.earth), no credential",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "regression": {
                "name": "standing_deadwood_fractional_cover",
                "description": (
                    "Fraction (0-1) of each 10 m pixel's observed area covered by standing "
                    "deadwood, aggregated from expert-delineated centimeter-scale deadwood "
                    "polygons on deadtrees.earth drone/aerial orthophotos. Deadwood polygons "
                    "and the labeling Area-Of-Interest (AOI) are rasterized on a 0.5 m "
                    "sub-grid in local UTM; deadwood is clipped to the AOI and averaged to "
                    "10 m. Within the AOI, area not delineated as deadwood is known alive/"
                    "non-tree (fraction contribution 0); outside the AOI is unobserved "
                    "(nodata). A 10 m pixel is kept only if >=50% of its area lies in the AOI."
                ),
                "unit": "fraction (0-1)",
                "dtype": "float32",
                "value_range": [0.0, round(float(pix_max), 4)],
                "nodata_value": io.REGRESSION_NODATA,
            },
            "num_samples": n_written,
            "notes": (
                "Manual expert (visual_interpretation) semantic-segmentation deadwood labels "
                "only; the platform's ~12k SegFormer auto-prediction label sets were excluded "
                "to keep this a high-confidence reference bank. Restricted to public, CC BY, "
                "non-archived datasets with a complete acquisition date >=2016. VHR->10 m "
                "fractional-cover aggregation (0.5 m sub-grid, 20x20 block mean). AOIs are "
                "small drone footprints (median ~230 m), so most datasets contribute a single "
                "footprint-sized tile; larger airborne AOIs are tiled to <=64x64. change_time"
                "=null (single-date state); 1-year window anchored on the orthophoto "
                "acquisition year. Deadwood fractional cover is heavily zero-inflated (most "
                f"10 m pixels are alive/background); {round(100 * pos_tile_frac, 1)}% of tiles "
                "contain some deadwood."
            ),
        },
    )

    print(f"tiles written: {n_written}; kept (>= {MIN_OBS_PIXELS} obs px): {len(kept)}")
    print(
        f"per-tile mean-fraction: min {means.min():.4f} med {np.median(means):.4f} max {means.max():.4f}"
    )
    print(
        f"max per-pixel fraction: {pix_max:.4f}; tiles with any deadwood: {round(100 * pos_tile_frac, 1)}%"
    )
    print(f"num_samples={n_written} task_type=regression")
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
