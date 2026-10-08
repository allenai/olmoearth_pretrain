"""Process the ETH Zurich PRS Sentinel-1 Lake Ice Detection dataset.

Source: prs-eth/sentinel_lakeice (https://github.com/prs-eth/sentinel_lakeice), the code
and ground-truth release accompanying Tom et al., "Lake Ice Detection from Sentinel-1 SAR
with Deep Learning" (ISPRS Annals, 2020). MIT license (labels). Sentinel-1 SAR lake-ice /
water semantic segmentation over four Swiss Alpine lakes across two winters (2016-17,
2017-18).

What the source provides that we use (all inside the GitHub repo, small text/vector files):
  * data/gt/{2016_17,2017_18}/{sihl,sils,silvaplana,stmoritz}.txt -- per-lake, per-DAY
    whole-lake state labels from daily webcam observation (semi-automated ground truth).
    Each day the whole lake is assigned one state code:
        s  = snow on ice, lake frozen ~90-100%      -> FROZEN (ice)
        i  = ice,         lake frozen ~90-100%       -> FROZEN (ice)
        w  = water,       lake ~90-100% open water   -> NON-FROZEN (water)
        ms/mi/mw (60-90% partial), c (cloud/fog),
        u (unclear), n (no data), and any composite  -> EXCLUDED (ambiguous)
    Only the three unambiguous ("clean") codes {s, i, w} are used, matching the paper's
    use of clearly-frozen / clearly-open days. The whole-lake state is propagated to every
    pixel inside the lake polygon (this is exactly how the paper builds its per-pixel GT).
  * data/shapes/UTM32N.shp -- lake-outline polygons (EPSG:32632). The four labelled lakes:
        stmoritz   -> "Lej da San Murezzan" (0.75 km2)
        silvaplana -> "Lej da Silvaplauna"  (2.66 km2)
        sils       -> "Lej da Segl"         (4.09 km2)
        sihl       -> "Sihlsee"            (10.49 km2)

The Sentinel-1 SAR rasters themselves are NOT needed here: OlmoEarth pretraining supplies
Sentinel-1 imagery independently and pairs it with these labels by geography + time. (The
authors' polybox link that hosted the S1 rasters is dead / 404 as of processing, but that
only held imagery, not labels, so it does not block us.)

Task: **dense_raster, binary classification** (label_type: dense_raster):
    0 = frozen (ice)          (lake surface frozen: ice or snow-on-ice)
    1 = non-frozen (water)    (open water)
  255 = nodata/ignore         (pixels outside the lake polygon)

Each sample is a <=64x64, 10 m, UTM (EPSG:32632) tile covering part of one lake on one
clean-state day: pixels inside the lake polygon carry that day's class, pixels outside are
255. Lakes larger than a 640 m tile are covered by a fixed grid of 64x64 tiles (only tiles
with >=5% lake coverage are kept). A given lake-day contributes all of its kept tiles.

Sampling (spec 5, tiles-per-class balanced, <=1000/class): the four lakes are balanced by
allocating ~250 samples per (lake, class); the number of clean days drawn per lake is
ceil(250 / n_tiles_for_lake), evenly spaced across that lake's sorted clean-day list, and
every kept tile of a selected day is emitted. A final balance_by_class(per_class=1000) caps
each class at 1000 while preserving the cross-lake / cross-date spread. Result ~2000 tiles.

Time range (spec 5): lake-ice presence is a specific-date / seasonal STATE, so each sample
gets a TIGHT 1-day window [obs_day 00:00 UTC, obs_day+1 00:00 UTC) anchored on the webcam
observation date, with change_time=null (a per-date state, not a dated change event). The
webcam observation date is used as the source acquisition date because the exact per-scene
S1 acquisition timestamps were only in the (now-unavailable) polybox raster share;
downstream assembly pairs the label with whatever S1 scene falls in the window (the frozen /
open state persists across neighbouring days, so a 1-day anchor is temporally coherent).

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.sentinel_1_lake_ice_detection
Idempotent: existing locations/{id}.tif are skipped.
"""

import argparse
import math
import multiprocessing
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io, manifest, sampling
from olmoearth_pretrain.open_set_segmentation_data.rasterize import (
    geom_to_pixels,
    rasterize_shapes,
)

SLUG = "sentinel_1_lake_ice_detection"
NAME = "Sentinel-1 Lake Ice Detection"

RAW_BASE = "https://raw.githubusercontent.com/prs-eth/sentinel_lakeice/master"
WINTERS = {"2016_17": 2016, "2017_18": 2017}  # start year of the winter
LAKES = ["sihl", "sils", "silvaplana", "stmoritz"]
# gt-file lake name -> polygon "name" attribute in data/shapes/UTM32N.shp
LAKE_POLY_NAME = {
    "stmoritz": "Lej da San Murezzan",
    "silvaplana": "Lej da Silvaplauna",
    "sils": "Lej da Segl",
    "sihl": "Sihlsee",
}

FROZEN, WATER, NODATA = 0, 1, 255
STATE_TO_CLASS = {"s": FROZEN, "i": FROZEN, "w": WATER}  # clean states only

TILE = 64
MIN_LAKE_FRAC = 0.05  # keep a tile only if >=5% of it is inside the lake polygon
PER_LAKE_PER_CLASS = 250  # balance target before the final per-class cap
PER_CLASS_CAP = 1000

CLASSES = [
    {
        "id": FROZEN,
        "name": "frozen (ice)",
        "description": (
            "Lake surface frozen (bare ice or snow-on-ice), ~90-100% frozen per daily "
            "webcam observation; the state is propagated to all pixels inside the lake "
            "polygon."
        ),
    },
    {
        "id": WATER,
        "name": "non-frozen (water)",
        "description": (
            "Open (non-frozen) lake water, ~90-100% unfrozen per daily webcam "
            "observation; propagated to all pixels inside the lake polygon."
        ),
    },
]

# Populated in main() before the Pool is created; inherited by workers via fork.
# lake -> {"proj": Projection, "tiles": [{"bounds": (x0,y0,x1,y1), "mask": np.bool_(64,64)}]}
LAKE_TILES: dict[str, dict[str, Any]] = {}


# --------------------------------------------------------------------------- download
def download_raw() -> None:
    """Fetch the small gt text files and lake shapefiles into raw/{slug}/ (idempotent)."""
    import urllib.request

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    files = []
    for w in WINTERS:
        for lk in LAKES:
            files.append(f"data/gt/{w}/{lk}.txt")
    for base in ("UTM31N", "UTM32N"):
        for ext in ("shp", "dbf", "prj", "shx"):
            files.append(f"data/shapes/{base}.{ext}")
    for rel in files:
        dst = raw / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.parent / (dst.name + ".tmp")
        urllib.request.urlretrieve(f"{RAW_BASE}/{rel}", str(tmp))
        tmp.rename(dst)
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "ETH Zurich PRS Sentinel-1 Lake Ice Detection.\n"
            "Repo: https://github.com/prs-eth/sentinel_lakeice (MIT, labels).\n"
            "Paper: Tom et al., 'Lake Ice Detection from Sentinel-1 SAR with Deep "
            "Learning', ISPRS Annals 2020.\n"
            "Used: data/gt/{2016_17,2017_18}/*.txt (per-day whole-lake webcam states) "
            "+ data/shapes/UTM32N.* (lake polygons, EPSG:32632).\n"
            "Sentinel-1 rasters (authors' polybox share) are NOT used and are unavailable "
            "(404); pretraining supplies S1 imagery separately.\n"
        )


# --------------------------------------------------------------------------- gt parse
def parse_gt(path: str, start_year: int) -> list[tuple[datetime, int]]:
    """Parse one gt file -> list of (obs_date_utc, class_id) for clean states only."""
    out: list[tuple[datetime, int]] = []
    started = False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith("-9999"):  # marker preceding the data block
                started = True
                continue
            if not started:
                continue
            m = re.match(r"^(\d{2})\.(\d{2})\.?\s+(\S+)", s)
            if not m:
                continue
            dd, mm, code = int(m.group(1)), int(m.group(2)), m.group(3).lower()
            cls = STATE_TO_CLASS.get(code)
            if cls is None:  # partial / cloud / unclear / no-data / composite -> skip
                continue
            year = start_year if mm >= 9 else start_year + 1  # Sep-Dec vs Jan-May
            out.append((datetime(year, mm, dd, tzinfo=UTC), cls))
    return out


def clean_days_by_lake_class() -> dict[str, dict[int, list[datetime]]]:
    """Lake -> class_id -> sorted list of clean observation dates (both winters merged)."""
    raw = io.raw_dir(SLUG)
    result: dict[str, dict[int, list[datetime]]] = {
        lk: defaultdict(list) for lk in LAKES
    }
    for w, sy in WINTERS.items():
        for lk in LAKES:
            for dt, cls in parse_gt(str(raw / "data" / "gt" / w / f"{lk}.txt"), sy):
                result[lk][cls].append(dt)
    for lk in LAKES:
        for cls in result[lk]:
            result[lk][cls].sort()
    return result


# --------------------------------------------------------------------------- tiling
def build_lake_tiles() -> None:
    """Populate LAKE_TILES: per-lake UTM projection + kept 64x64 tile masks."""
    import geopandas as gpd

    raw = io.raw_dir(SLUG)
    gdf = gpd.read_file(str(raw / "data" / "shapes" / "UTM32N.shp"))
    gdf_wgs = gdf.to_crs(4326)
    for lk in LAKES:
        poly_name = LAKE_POLY_NAME[lk]
        idx = gdf.index[gdf["name"] == poly_name][0]
        poly_wgs = gdf_wgs.geometry.loc[idx]
        c = poly_wgs.centroid
        proj = io.utm_projection_for_lonlat(c.x, c.y)
        poly_px = geom_to_pixels(poly_wgs, WGS84_PROJECTION, proj)
        minx, miny, maxx, maxy = poly_px.bounds
        x0 = int(math.floor(minx))
        y0 = int(math.floor(miny))
        ntx = int(math.ceil((maxx - x0) / TILE))
        nty = int(math.ceil((maxy - y0) / TILE))
        tiles = []
        for ty in range(nty):
            for tx in range(ntx):
                bx = x0 + tx * TILE
                by = y0 + ty * TILE
                bounds = (bx, by, bx + TILE, by + TILE)
                arr = rasterize_shapes(
                    [(poly_px, 1)], bounds, fill=0, dtype="uint8", all_touched=False
                )[0]
                mask = arr.astype(bool)
                if mask.mean() >= MIN_LAKE_FRAC:
                    tiles.append({"bounds": bounds, "mask": mask})
        LAKE_TILES[lk] = {"proj": proj, "tiles": tiles}
        print(f"  {lk}: {len(tiles)} kept tiles (proj {proj.crs.to_string()})")


def _even_indices(n_avail: int, n_pick: int) -> list[int]:
    """Evenly-spaced indices into a list of length n_avail (n_pick <= n_avail)."""
    n_pick = min(n_pick, n_avail)
    if n_pick <= 0:
        return []
    if n_pick == 1:
        return [n_avail // 2]
    return sorted({int(round(i)) for i in np.linspace(0, n_avail - 1, n_pick)})


# --------------------------------------------------------------------------- write
def _write_one(rec: dict[str, Any]) -> str | None:
    sample_id = rec["sample_id"]
    if (io.locations_dir(SLUG) / f"{sample_id}.tif").exists():
        return rec["class_name"]
    lk = rec["lake"]
    entry = LAKE_TILES[lk]
    proj = entry["proj"]
    tile = entry["tiles"][rec["tile_idx"]]
    bounds = tile["bounds"]
    mask = tile["mask"]
    cls = rec["class_id"]
    label = np.where(mask, cls, NODATA).astype(np.uint8)
    present = sorted(int(v) for v in np.unique(label) if v != NODATA)
    obs = rec["obs_date"]
    time_range = (obs, obs + timedelta(days=1))
    io.write_label_geotiff(SLUG, sample_id, label, proj, bounds, nodata=NODATA)
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
    return rec["class_name"]


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    download_raw()
    build_lake_tiles()
    days = clean_days_by_lake_class()

    class_name = {FROZEN: "frozen", WATER: "water"}
    # Report availability.
    for lk in LAKES:
        nt = len(LAKE_TILES[lk]["tiles"])
        f_days = len(days[lk].get(FROZEN, []))
        w_days = len(days[lk].get(WATER, []))
        print(
            f"  {lk}: tiles={nt} frozen_days={f_days} water_days={w_days} "
            f"(max frozen={nt * f_days}, water={nt * w_days})"
        )

    # Build candidate records: per (lake, class) pick evenly-spaced days, emit all tiles.
    records: list[dict[str, Any]] = []
    for lk in LAKES:
        tiles = LAKE_TILES[lk]["tiles"]
        nt = len(tiles)
        if nt == 0:
            continue
        for cls in (FROZEN, WATER):
            day_list = days[lk].get(cls, [])
            if not day_list:
                continue
            n_days = max(1, math.ceil(PER_LAKE_PER_CLASS / nt))
            for di in _even_indices(len(day_list), n_days):
                obs = day_list[di]
                for tidx in range(nt):
                    records.append(
                        {
                            "lake": lk,
                            "class_id": cls,
                            "class_name": class_name[cls],
                            "tile_idx": tidx,
                            "obs_date": obs,
                            "source_id": f"{lk}:{obs.date().isoformat()}:tile{tidx:03d}",
                        }
                    )

    pre = Counter(r["class_name"] for r in records)
    print(f"candidate records: {len(records)} {dict(pre)}")

    # Final per-class cap (<=1000/class), preserving cross-lake/date spread.
    records = sampling.balance_by_class(
        records, key="class_id", per_class=PER_CLASS_CAP
    )
    records.sort(key=lambda r: (r["lake"], r["class_id"], r["source_id"]))
    for i, r in enumerate(records):
        r["sample_id"] = f"{i:06d}"
    post = Counter(r["class_name"] for r in records)
    print(f"selected records: {len(records)} {dict(post)}")

    io.check_disk()
    counts: Counter = Counter()
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in records]),
            total=len(records),
            desc="write tiles",
        ):
            if res is not None:
                counts[res] += 1
    print(f"written per class: {dict(counts)}")

    metadata = {
        "dataset": SLUG,
        "name": NAME,
        "task_type": "classification",
        "source": "ETH Zurich PRS (prs-eth/sentinel_lakeice)",
        "license": "MIT (labels)",
        "provenance": {
            "url": "https://github.com/prs-eth/sentinel_lakeice",
            "have_locally": False,
            "annotation_method": (
                "semi-automated: daily whole-lake state from webcam observation "
                "(Tom et al. 2020), propagated to lake-polygon pixels; only "
                "unambiguous frozen (s/i) and open-water (w) days used"
            ),
        },
        "sensors_relevant": ["sentinel1"],
        "classes": CLASSES,
        "nodata_value": NODATA,
        "num_samples": len(records),
        "notes": (
            "Four Swiss Alpine lakes (Sils/Silvaplana/St.Moritz/Sihl), winters 2016-17 & "
            "2017-18, EPSG:32632, 10 m. Dense binary lake-ice(0)/water(1) tiles; pixels "
            "outside the lake polygon are nodata(255). Each sample is one lake on one "
            "clean-state day with a tight 1-day time_range (per-date STATE, "
            "change_time=null). S1 rasters not stored (pretraining supplies S1)."
        ),
    }
    io.write_dataset_metadata(SLUG, metadata)

    manifest.write_registry_entry(
        SLUG,
        "completed",
        task_type="classification",
        num_samples=len(records),
        notes=(
            f"dense_raster binary lake-ice/water; {dict(counts)}; 4 Swiss lakes x 2 "
            "winters; tight 1-day per-date STATE windows."
        ),
    )
    print("DONE")


if __name__ == "__main__":
    main()
