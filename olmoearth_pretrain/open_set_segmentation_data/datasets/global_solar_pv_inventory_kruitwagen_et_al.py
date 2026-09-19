"""Process the Global Solar PV Inventory (Kruitwagen et al., 2021) into label patches.

Source: Kruitwagen, L. et al. "A global inventory of photovoltaic solar energy generating
units", Nature 598, 604-610 (2021). Dataset on Zenodo record 5005868 (CC-BY-4.0):

  https://zenodo.org/api/records/5005868/files/predicted_set.geojson/content   (68,661 polygons)
  https://zenodo.org/api/records/5005868/files/test_polygons.geojson/content   (manual test set)

The inventory maps utility-scale photovoltaic (PV) generating units globally, detected from
a 2016-2018 Sentinel-2 composite + SPOT 6/7, with a manually photointerpreted test set. We
use ``predicted_set.geojson`` (the FULL inventory) because — unlike the ``test_polygons``
file — it carries per-feature ``install_date`` and ``capacity_mw``, which we need to assign
time ranges. (test_polygons has geometry only: aoi/id, no dates/capacity.)

This is a single-foreground-class **polygon** dataset. Each PV polygon is rasterized into a
footprint-sized (<=64x64) local-UTM 10 m tile:
  0 = background   (non-PV land inside the tile; genuine surrounding land, spatially
                    meaningful — same convention as global_renewables_watch / olmoearth_solar_farm)
  1 = solar_pv     (photovoltaic generating-unit footprint)
  255 = nodata     (not used here; no ignore pixels for polygons)

Because a footprint-sized tile centered on the polygon contains real surrounding land, its
background pixels are legitimate negatives (NOT fabricated) — so no separate negative tiles
are emitted (spec section 5). Large farms (>640 m, ~3.4% of polygons) yield an all-solar
64x64 tile centered on the polygon.

Time range: solar farms are persistent once built, and every polygon in the inventory is
present in the 2018 detection snapshot. ``install_date`` is either "YYYY-MM-..." (a concrete
2016/2017/2018 commissioning month), "<2016-..." (built before 2016), or empty (unknown).
We assign a 1-year window in which the farm is FULLY present:
  install year 2016 -> 2017 window;  2017 -> 2018;  2018 -> 2019;
  "<2016" or unknown -> 2018 window (representative Sentinel-era snapshot; farm present).
All windows are post-2016, so no polygon is dropped on the pre-2016 rule.

Sampling: derived-product (model) labels, so we prefer the higher-confidence detections
(confidence A/B) and cap at 1000 solar_pv tiles, stratified across install-year buckets
{2016, 2017, 2018, pre-2016, unknown} for temporal + geographic diversity (spec section 5).

Run (idempotent):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_solar_pv_inventory_kruitwagen_et_al
"""

import argparse
import multiprocessing
import re
from collections import Counter
from typing import Any

import fiona
import numpy as np
import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.download import download_http
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "global_solar_pv_inventory_kruitwagen_et_al"
NAME = "Global Solar PV Inventory (Kruitwagen et al.)"
ZENODO = "https://zenodo.org/api/records/5005868/files"
PRED_FILE = "predicted_set.geojson"
TEST_FILE = "test_polygons.geojson"

CID_BACKGROUND = 0
CID_SOLAR = 1
CLASSES = [
    {
        "id": CID_BACKGROUND,
        "name": "background",
        "description": "Non-PV land surrounding the generating unit within the tile "
        "(any other land cover). Genuine negative pixels, not fabricated.",
    },
    {
        "id": CID_SOLAR,
        "name": "solar_pv",
        "description": "Utility-scale photovoltaic solar generating-unit footprint "
        "(panel array / plant boundary), from Kruitwagen et al. 2021 detections on a "
        "2016-2018 Sentinel-2 composite + SPOT 6/7, rasterized at 10 m.",
    },
]

# Prefer the higher-confidence detections (derived product, spec section 4/5).
KEEP_CONFIDENCE = {"A", "B"}
PER_CLASS = 1000
N_BUCKETS = 5  # 2016, 2017, 2018, pre2016, unknown
PER_BUCKET = PER_CLASS // N_BUCKETS  # 200 -> 1000 total
SEED = 42

MAX_SOLAR_TILE = io.MAX_TILE  # 64 px @ 10 m = 640 m


def _install_year(raw: str) -> int | None:
    """Return the concrete install YEAR (2016/2017/2018) or None for '<2016'/empty/other."""
    raw = (raw or "").strip()
    if not raw or raw.startswith("<"):
        return None
    m = re.match(r"(\d{4})", raw)
    if not m:
        return None
    y = int(m.group(1))
    return y if y in (2016, 2017, 2018) else None


def _year_bucket(install_year: int | None, raw: str) -> str:
    """Bucket for stratified sampling (temporal diversity)."""
    if install_year is not None:
        return str(install_year)
    return "pre2016" if (raw or "").strip().startswith("<") else "unknown"


def _time_range(install_year: int | None):
    """1-year window in which the farm is fully present (see module docstring)."""
    if install_year in (2016, 2017, 2018):
        return io.year_range(install_year + 1)  # first full year after commissioning
    return io.year_range(2018)  # <2016 / unknown: representative Sentinel-era snapshot


def read_polygons() -> list[dict[str, Any]]:
    """Read PV polygons (confidence A/B) into lightweight records."""
    path = io.raw_dir(SLUG) / PRED_FILE
    recs: list[dict[str, Any]] = []
    with fiona.open(path.path) as src:
        for feat in src:
            p = feat["properties"]
            if p.get("confidence") not in KEEP_CONFIDENCE:
                continue
            geom = shapely.geometry.shape(feat["geometry"])
            if geom.is_empty or not geom.is_valid:
                geom = geom.buffer(0)
                if geom.is_empty or not geom.is_valid:
                    continue
            c = geom.centroid
            raw_date = p.get("install_date") or ""
            iy = _install_year(raw_date)
            recs.append(
                {
                    "lon": float(c.x),
                    "lat": float(c.y),
                    "geom_wkb": shapely.to_wkb(geom),
                    "install_year": iy,
                    "year_bucket": _year_bucket(iy, raw_date),
                    "capacity_mw": p.get("capacity_mw"),
                    "confidence": p.get("confidence"),
                    "source_id": f"unique_id/{p.get('unique_id')}",
                }
            )
    return recs


def _write_solar(rec: dict[str, Any]) -> str:
    sample_id = rec["sample_id"]
    tif = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif.exists():
        return "skip"
    proj = io.utm_projection_for_lonlat(rec["lon"], rec["lat"])
    geom = shapely.from_wkb(rec["geom_wkb"])
    pix = geom_to_pixels(geom, WGS84_PROJECTION, proj)
    minx, miny, maxx, maxy = pix.bounds
    cx = int(round((minx + maxx) / 2))
    cy = int(round((miny + maxy) / 2))
    w = min(MAX_SOLAR_TILE, max(1, int(np.ceil(maxx - minx))))
    h = min(MAX_SOLAR_TILE, max(1, int(np.ceil(maxy - miny))))
    bounds = io.centered_bounds(cx, cy, w, h)
    arr = rasterize_shapes(
        [(pix, CID_SOLAR)], bounds, fill=CID_BACKGROUND, dtype="uint8", all_touched=True
    )
    io.write_label_geotiff(SLUG, sample_id, arr, proj, bounds, nodata=io.CLASS_NODATA)
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        _time_range(rec["install_year"]),
        source_id=rec["source_id"],
        classes_present=sorted(set(np.unique(arr).tolist()) - {io.CLASS_NODATA}),
    )
    return "solar"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    io.check_disk()

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    print("downloading source geojson ...", flush=True)
    download_http(f"{ZENODO}/{PRED_FILE}/content", raw / PRED_FILE)
    download_http(f"{ZENODO}/{TEST_FILE}/content", raw / TEST_FILE)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Global Solar PV Inventory - Kruitwagen et al., Nature 598 (2021).\n"
            "Zenodo record 5005868 (CC-BY-4.0).\n"
            f"{ZENODO}/{PRED_FILE}/content  (full predicted inventory: install_date + capacity_mw)\n"
            f"{ZENODO}/{TEST_FILE}/content  (manual test polygons: geometry only)\n"
            "We use predicted_set.geojson (dates+capacity needed for time ranges).\n"
        )

    io.check_disk()
    print("reading PV polygons (confidence A/B) ...", flush=True)
    recs = read_polygons()
    bkt = Counter(r["year_bucket"] for r in recs)
    print(f"  {len(recs)} A/B polygons; buckets: {dict(bkt)}", flush=True)

    # Stratify across install-year buckets for temporal + geographic diversity, cap 1000.
    selected = balance_by_class(
        recs, "year_bucket", per_class=PER_BUCKET, seed=SEED, total_cap=None
    )[:PER_CLASS]
    sel_bkt = Counter(r["year_bucket"] for r in selected)
    print(f"selected {len(selected)} tiles; buckets: {dict(sel_bkt)}", flush=True)

    selected.sort(key=lambda r: r["source_id"])
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"

    io.check_disk()
    results: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_solar, [dict(rec=r) for r in selected]),
            total=len(selected),
        ):
            results[res] += 1
    print("write results:", dict(results), flush=True)

    io.check_disk()
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "classification",
            "source": "Zenodo (Kruitwagen et al., Nature 2021)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://doi.org/10.5281/zenodo.5005868",
                "have_locally": False,
                "annotation_method": "model-derived (Sentinel-2 2016-2018 + SPOT 6/7); "
                "photointerpreted test set",
                "file_used": PRED_FILE,
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": CLASSES,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {
                "solar_pv_tiles": len(selected),
                "confidence_kept": sorted(KEEP_CONFIDENCE),
            },
            "sampling": {
                "per_class": PER_CLASS,
                "stratified_by": "install_year_bucket",
                "bucket_counts": dict(sel_bkt),
            },
            "notes": (
                "Single-foreground-class PV polygon dataset from Kruitwagen et al. 2021 "
                "global inventory (predicted_set.geojson, 68,661 polygons; we use the "
                "53,876 confidence A/B ones). Each polygon rasterized (all_touched) into a "
                "footprint-sized <=64x64 local-UTM 10 m tile: 1=solar_pv, 0=background "
                "(real surrounding land). No fabricated negatives. Large farms (>640 m) "
                "give an all-solar 64x64 center tile. Time range = 1-year window where the "
                "farm is fully present: install year Y in {2016,2017,2018} -> Y+1 window; "
                "'<2016'/unknown -> 2018 window (all present in the 2018 detection snapshot; "
                "all windows post-2016). Capped at 1000 tiles, stratified across install-year "
                "buckets {2016,2017,2018,pre2016,unknown} for temporal + geographic diversity. "
                "Derived product; prefer higher-confidence (A/B) detections."
            ),
        },
    )
    print(f"done: {len(selected)} tiles")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
