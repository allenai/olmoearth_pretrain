"""Process VIIRS Nighttime Lights (Annual VNL V2) into open-set regression label patches.

Source product: **Annual VIIRS Nighttime Lights V2** (Earth Observation Group / Payne
Institute, Colorado School of Mines) -- global annual cloud-free radiance composites at
~15 arc-second (~500 m) native resolution, in nW/cm^2/sr. The canonical distribution
(https://eogdata.mines.edu/products/vnl/) now sits behind a Keycloak/SSO login gate for
which we have no credential (nothing in .env matches EOG, and the
authorized GEE service-account key referenced by TEST_GEE_SERVICE_ACCOUNT_CREDENTIALS is
absent on this host). We therefore access the *identical* VNL V2 annual product through a
public, ungated CC-BY-4.0 mirror on the Hugging Face Hub:

    Major-TOM/Core-VIIRS-Nighttime-Light
    (2016-2021 = Annual VNL V2.1, 2022-2024 = Annual VNL V2.2; band = annual "median")

The Major TOM packaging is ideal for our sampling contract: the global product is already
diced into ~1056x1056 single-band float32 GeoTIFF patches, **each in a local UTM zone at
exactly 10 m/pixel**, one patch per Major TOM grid cell (globally, evenly distributed).
Every patch is individually addressable in its year shard via an (offset, size) byte range
recorded in INDEX.parquet.

Access mechanics: the year's product is split into 16 spatially-contiguous ~4.5 GB shard
zips. Anonymous per-patch HTTP Range reads against the Hub are aggressively rate-limited
(HTTP 429), so we instead download a curated subset of shards that together span every
inhabited continent (~31 GB, well within the disk budget) via ``huggingface_hub`` (CDN +
automatic 429 backoff), then read each sampled patch by **seeking to its INDEX byte offset
in the local shard** (the stored .tif is uncompressed, so a seek+read yields the exact tif
bytes). All measure/write work is therefore local and fast; only the shard pulls touch the
network.

RESOLUTION CAVEAT (documented in metadata + summary): VNL is natively ~500 m. The Major TOM
mirror has already resampled it onto a 10 m grid, so within any 64x64 (=640 m) tile the label
is a smooth/near-constant upsampled field carrying only ~1-2 native VIIRS pixels of real
information. This is an intentionally coarse regression probe (a settlement / economic-activity
proxy), not a 10 m-native signal.

This is a *regression* dataset: per-pixel continuous night-time radiance. Radiance is an
intensity (resolution-invariant), so it is stored as-is -- no unit conversion. Output:
single-band float32 GeoTIFFs reusing each patch's own local-UTM 10 m grid, cropped to the
center 64x64 window (one sample per sampled Major TOM cell), nodata = io.REGRESSION_NODATA
(-99999). Time range = the composite year (a 1-year window; change_time=null). Candidates
are drawn stratified across settlement levels + continents so the value spread spans dark
rural / ocean floor up to bright urban cores, then bucket-balanced across log-radiance.
"""

import argparse
import io as _io
import math
import multiprocessing
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import tqdm
from rasterio.windows import Window
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered

from olmoearth_pretrain.open_set_segmentation_data import io
from olmoearth_pretrain.open_set_segmentation_data.download import hf_download
from olmoearth_pretrain.open_set_segmentation_data.sampling import (
    bucket_balance_regression,
)

SLUG = "viirs_nighttime_lights_annual_vnl_v2"
NAME = "VIIRS Nighttime Lights (Annual VNL V2)"
URL = "https://eogdata.mines.edu/products/vnl/"
HF_REPO = "Major-TOM/Core-VIIRS-Nighttime-Light"

# The composite year to sample. 2020 is post-2016 (Annual VNL V2.1) and near the middle of
# the manifest range 2016-2024.
YEAR = 2020

# Curated subset of the 16 year shards that together span every inhabited continent
# (see per-shard lon/lat extents in the summary). ~31 GB total; avoids pulling the whole
# ~72 GB year while still giving a global, all-continents sample.
SHARD_IDS = ["001", "005", "006", "007", "008", "013", "015"]


def shard_rel(shard_id: str) -> str:
    return f"{YEAR}/MAJORTOM-VIIRS-NTL_{YEAR}_median_{shard_id}.zip"


SHARDS = [shard_rel(s) for s in SHARD_IDS]

TILE = 64
TOTAL = 5000
N_BUCKETS = 10
SEED = 42

# Candidate-pool composition (before measuring radiance and bucket-balancing to TOTAL).
# Kept well above TOTAL so every log-radiance bucket has enough members after balancing.
N_BRIGHT = 4000  # highest-population land cells -> guarantees the bright urban tail
N_BROAD = 6000  # land cells stratified across human-modification deciles (rural..urban)
N_OCEAN = 1500  # ocean/dark cells -> genuine near-zero noise floor
CAP_PER_COUNTRY = 200  # per-country cap on the bright tail so no one country dominates

OCEAN = "Ocean/Sea/Lakes"


def index_path() -> str:
    return (io.raw_dir(SLUG) / "INDEX.parquet").path


def shard_local_path(shard_rel_path: str) -> str:
    return (io.raw_dir(SLUG) / shard_rel_path).path


# ---------------------------------------------------------------------------
# Downloads: the small patch index + the curated shard subset. Patches are then read by
# seeking to their INDEX byte offset in the local shard (no per-patch network).
# ---------------------------------------------------------------------------
def download_index() -> None:
    io.check_disk()
    dst = io.raw_dir(SLUG) / "INDEX.parquet"
    if dst.exists():
        print(f"[skip] INDEX.parquet present ({dst})")
        return
    print("downloading INDEX.parquet ...")
    hf_download(HF_REPO, "INDEX.parquet", io.raw_dir(SLUG))
    print(f"[got] {dst}")


def download_shards() -> None:
    """Download the curated shard subset via huggingface_hub (CDN + 429 backoff)."""
    from huggingface_hub import hf_hub_download

    for rel in SHARDS:
        io.check_disk()
        dst = io.raw_dir(SLUG) / rel
        if dst.exists():
            print(f"[skip] {rel} present ({dst.stat().st_size / 1e9:.1f} GB)")
            continue
        print(f"downloading {rel} ...")
        hf_hub_download(
            repo_id=HF_REPO,
            filename=rel,
            repo_type="dataset",
            local_dir=io.raw_dir(SLUG).path,
        )
        print(f"[got] {rel} ({dst.stat().st_size / 1e9:.1f} GB)")


# ---------------------------------------------------------------------------
# Candidate selection (stratified, from the index only -- no network)
# ---------------------------------------------------------------------------
def select_candidates() -> list[dict[str, Any]]:
    df = pd.read_parquet(index_path())
    df = df[(df["year"] == YEAR) & (df["shard"].isin(SHARDS))].copy()
    if df.empty:
        raise RuntimeError(f"no patches for year {YEAR} in the downloaded shards")
    df["lon"] = df["bbox"].map(lambda b: float(b["xmin"]))
    df["lat"] = df["bbox"].map(lambda b: float(b["ymin"]))
    pop = df["socio:population"].fillna(0.0)
    hm = df["socio:human_modification"].fillna(0.0)
    df["_pop"] = pop
    df["_hm"] = hm

    land = df[df["admin:country"] != OCEAN]
    ocean = df[df["admin:country"] == OCEAN]

    # Bright urban tail: highest population, capped per country for continental spread.
    bright_sorted = land.sort_values("_pop", ascending=False)
    bright = bright_sorted.groupby("admin:country", sort=False).head(CAP_PER_COUNTRY)
    bright = bright[bright["_pop"] > 0].head(N_BRIGHT)

    # Broad land spread: stratify across human-modification deciles (dark rural -> peri-urban).
    land_rem = land.drop(index=bright.index, errors="ignore").copy()
    try:
        land_rem["_hmb"] = pd.qcut(land_rem["_hm"], 10, labels=False, duplicates="drop")
    except ValueError:
        land_rem["_hmb"] = 0
    per_bucket = max(1, N_BROAD // max(1, land_rem["_hmb"].nunique()))
    broad = land_rem.groupby("_hmb", group_keys=False).apply(
        lambda g: g.sample(n=min(len(g), per_bucket), random_state=SEED)
    )

    # Ocean/dark: near-zero radiance floor (a realistic slice, not oversampled).
    dark = ocean.sample(n=min(len(ocean), N_OCEAN), random_state=SEED)

    cand = pd.concat([bright, broad, dark])
    cand = cand[~cand.index.duplicated(keep="first")]
    recs = [
        {
            "id": r["id"],
            "shard": r["shard"],
            "offset": int(r["offset"]),
            "size": int(r["size"]),
            "lon": r["lon"],
            "lat": r["lat"],
            "country": r["admin:country"],
        }
        for _, r in cand.iterrows()
    ]
    print(
        f"selected {len(recs)} candidate cells for {YEAR} "
        f"(bright={len(bright)}, broad={len(broad)}, ocean={len(dark)})"
    )
    return recs


# ---------------------------------------------------------------------------
# Patch read (local seek to INDEX byte offset) + center-window crop
# ---------------------------------------------------------------------------
def _read_patch_blob(shard: str, offset: int, size: int) -> bytes:
    with open(shard_local_path(shard), "rb") as f:
        f.seek(offset)
        blob = f.read(size)
    if len(blob) != size:
        raise RuntimeError(f"short read {len(blob)}!={size} for {shard}@{offset}")
    return blob


def _crop_center(
    blob: bytes,
) -> tuple[np.ndarray, Projection, tuple[int, int, int, int]]:
    """Return (arr[64,64] float32, projection, rslearn pixel_bounds) for the patch center."""
    with rasterio.open(_io.BytesIO(blob)) as ds:
        h, w = ds.shape
        po = (w - TILE) // 2
        ro = (h - TILE) // 2
        arr = ds.read(1, window=Window(po, ro, TILE, TILE)).astype(np.float32)
        tf = ds.transform
        crs = ds.crs
        left_m = tf.c + po * tf.a  # tf.a = +10
        top_m = tf.f + ro * tf.e  # tf.e = -10
    # rslearn pixel coords under Projection(crs, 10, -10): x_pix = x_m/10, y_pix = y_m/-10.
    col_left = int(round(left_m / 10.0))
    y_top = int(round(top_m / -10.0))
    bounds = (col_left, y_top, col_left + TILE, y_top + TILE)
    proj = Projection(crs, 10, -10)
    # Sanitize: mark non-finite as nodata; radiance is a non-negative intensity.
    bad = ~np.isfinite(arr)
    arr[bad] = io.REGRESSION_NODATA
    return arr, proj, bounds


def _measure_one(rec: dict[str, Any]) -> dict[str, Any] | None:
    try:
        blob = _read_patch_blob(rec["shard"], rec["offset"], rec["size"])
        arr, _proj, _bounds = _crop_center(blob)
    except Exception as e:  # noqa: BLE001
        return {"id": rec["id"], "error": str(e)}
    valid = arr[arr != io.REGRESSION_NODATA]
    if valid.size == 0:
        return None
    return {
        **rec,
        "radiance_mean": float(valid.mean()),
        "radiance_min": float(valid.min()),
        "radiance_max": float(valid.max()),
    }


def _write_one(rec: dict[str, Any]) -> dict[str, Any] | None:
    sample_id = rec["sample_id"]
    tif_path = io.locations_dir(SLUG) / f"{sample_id}.tif"
    if tif_path.exists():
        return None
    try:
        blob = _read_patch_blob(rec["shard"], rec["offset"], rec["size"])
        arr, proj, bounds = _crop_center(blob)
    except Exception as e:  # noqa: BLE001
        return {"sample_id": sample_id, "error": str(e)}
    io.write_label_geotiff(
        SLUG, sample_id, arr, proj, bounds, nodata=io.REGRESSION_NODATA
    )
    io.write_sample_json(
        SLUG,
        sample_id,
        proj,
        bounds,
        io.year_range(YEAR),
        source_id=f"{rec['id']}_{YEAR}",
    )
    valid = arr[arr != io.REGRESSION_NODATA]
    return {
        "sample_id": sample_id,
        "n_valid": int(valid.size),
        "min": float(valid.min()) if valid.size else None,
        "max": float(valid.max()) if valid.size else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    io.check_disk()
    if not args.skip_download:
        download_index()
        download_shards()

    cands = select_candidates()

    # Phase 1: measure representative radiance for each candidate (local seek-reads).
    measured: list[dict[str, Any]] = []
    n_err = 0
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _measure_one, [dict(rec=r) for r in cands]),
            total=len(cands),
            desc="measure",
        ):
            if res is None:
                continue
            if "error" in res:
                n_err += 1
                continue
            measured.append(res)
    print(f"measured {len(measured)} candidates ({n_err} read errors)")
    if not measured:
        raise RuntimeError(
            "no candidate patches could be read (shards missing/corrupt?)"
        )

    # Phase 2: bucket-balance across log10(radiance) so the value distribution is spread
    # (VNL radiance is extremely right-skewed: most cells near the noise floor, a long
    # bright tail). +1 keeps the log finite for the ~0.1-0.4 floor.
    def log_rad(r: dict[str, Any]) -> float:
        return math.log10(max(r["radiance_mean"], 0.0) + 1.0)

    selected, log_edges = bucket_balance_regression(
        measured, log_rad, total=TOTAL, n_buckets=N_BUCKETS
    )
    rad_edges = [round(10.0**e - 1.0, 4) for e in log_edges]
    for i, r in enumerate(selected):
        r["sample_id"] = f"{i:06d}"
    print(
        f"selected {len(selected)} tiles (<= {TOTAL}); radiance bucket edges {rad_edges}"
    )

    io.locations_dir(SLUG).mkdir(parents=True, exist_ok=True)
    io.check_disk()

    # Phase 3: fetch + crop + write the selected tiles (idempotent; skips existing .tif).
    stats: list[dict[str, Any]] = []
    w_err = 0
    with multiprocessing.Pool(args.workers) as p:
        for res in tqdm.tqdm(
            star_imap_unordered(p, _write_one, [dict(rec=r) for r in selected]),
            total=len(selected),
            desc="write",
        ):
            if res is None:
                continue
            if "error" in res:
                w_err += 1
                continue
            stats.append(res)
    print(f"wrote {len(stats)} tiles ({w_err} write errors)")

    # Aggregate stats for metadata + report.
    sel_mean = np.array([r["radiance_mean"] for r in selected], dtype=np.float64)
    country_counts = Counter(r["country"] for r in selected)
    valid_stats = [s for s in stats if s.get("n_valid", 0) > 0]
    pix_min = min((s["min"] for s in valid_stats), default=0.0)
    pix_max = max((s["max"] for s in valid_stats), default=0.0)
    num_samples = len(selected)

    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "regression",
            "source": "Earth Observation Group (Colorado School of Mines)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": URL,
                "have_locally": False,
                "annotation_method": (
                    "sensor/model-derived VIIRS DNB annual median composite (VNL V2); "
                    f"accessed via public HF mirror {HF_REPO} (Major TOM grid, "
                    "range-request per patch)"
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "regression": {
                "name": "nighttime_radiance",
                "description": (
                    "VIIRS Day/Night Band annual median cloud-free radiance (Annual VNL V2; "
                    f"{YEAR} = V2.1). A standard proxy for human settlement / electrification / "
                    "economic activity. NATIVE RESOLUTION IS ~500 m (15 arc-sec): the values "
                    "here were resampled to a 10 m grid by the Major TOM mirror, so a 64x64 "
                    "(640 m) tile is a smooth upsampled field carrying only ~1-2 native VIIRS "
                    "pixels of real information -- a deliberately coarse regression probe. Dark "
                    "areas sit at the sensor noise floor (~0.1-0.4), not exactly zero, because "
                    "this is the (un-masked) median product."
                ),
                "unit": "nW/cm^2/sr",
                "dtype": "float32",
                "value_range": [round(pix_min, 4), round(pix_max, 4)],
                "nodata_value": io.REGRESSION_NODATA,
                "buckets": rad_edges,
                "native_resolution_m": 500,
            },
            "num_samples": num_samples,
            "country_counts": dict(sorted(country_counts.items())),
            "notes": (
                f"Annual VNL V2 median composite, year {YEAR}. Bounded-tile sampling of Major "
                "TOM grid cells (no global coverage): one center 64x64 window per sampled cell, "
                "reusing the mirror's local-UTM 10 m grid directly (no reprojection). Candidates "
                "stratified across population/human-modification and continents, then "
                "bucket-balanced across log10(radiance) deciles. Time range = the composite year; "
                "change_time=null. "
                f"selected-tile mean-radiance percentiles: p50={np.percentile(sel_mean, 50):.2f}, "
                f"p90={np.percentile(sel_mean, 90):.2f}, p99={np.percentile(sel_mean, 99):.2f}, "
                f"max={sel_mean.max():.2f} nW/cm^2/sr."
            ),
        },
    )

    hist_edges = [0, 0.5, 1, 2, 5, 10, 25, 50, 100, 500, np.inf]
    hist, _ = np.histogram(sel_mean, bins=hist_edges)
    print("selected-tile mean-radiance histogram (nW/cm^2/sr):")
    for lo, hi, c in zip(hist_edges[:-1], hist_edges[1:], hist):
        print(f"  [{lo:>6}, {hi:>6}) : {c}")
    print(
        f"per-pixel value range across tiles: [{pix_min:.3f}, {pix_max:.3f}] nW/cm^2/sr"
    )
    print(
        f"top countries: {dict(sorted(country_counts.items(), key=lambda kv: -kv[1])[:12])}"
    )
    print(f"num_samples={num_samples} task_type=regression")
    print("done")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
