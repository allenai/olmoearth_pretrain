"""Stage 4: check a built shard and project the cost of the full corpus.

1. Values: for ``--samples`` random windows, a 48-hour slice of six bands
   (instantaneous, accumulated and both evaporation components) is re-read
   from the store through xarray and de-accumulated hour by hour; it must
   match the shard exactly.
2. Physics: per band, the range, mean and no-data share over the shard, and
   mean daily transpiration and bare-soil evaporation per Koppen class
   (deserts should have almost no transpiration, rainforests the most).
3. Cost: the compressed size of a sample of store chunks and the shard's build
   time give the projected download and job time for every shard of the plan.

Writes ``progress/shard_XXX.validate.json``.

Usage::

    python -m olmoearth_pretrain.dataset_creation.era5_hourly.validate --shard 0
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pandas as pd
import xarray as xr

from .common import (
    BANDS,
    DEFAULT_ROOT,
    HOURLY_URL,
    NODATA,
    WINDOW_HOURS,
    edh_storage_options,
    open_store,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHECK_BANDS = ("t2m", "tp", "ssrd", "evavt", "evabs", "stl4")
SLICE_HOURS = 48


def check_values(
    shard: np.ndarray, index: pd.DataFrame, samples: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Compare random slices of the shard with an independent read of the store.

    Args:
        shard: The shard (memory-mapped).
        index: The shard's index rows, in row order.
        samples: Windows to check.
        rng: Random generator.

    Returns:
        Max absolute difference per band and the number of slices compared.
    """
    ds = xr.open_dataset(
        HOURLY_URL, engine="zarr", chunks=None, storage_options=edh_storage_options()
    )
    by_name = {b.name: (k, b) for k, b in enumerate(BANDS)}
    worst = {name: 0.0 for name in CHECK_BANDS}
    rows = rng.choice(len(index), size=min(samples, len(index)), replace=False)
    for row in rows.tolist():
        w = index.iloc[row]
        k0 = int(rng.integers(0, WINDOW_HOURS - SLICE_HOURS))
        h0 = int(w["first_hour"]) + k0
        for name in CHECK_BANDS:
            b, band = by_name[name]
            raw = ds[band.source].isel(
                valid_time=slice(h0 - 1, h0 + SLICE_HOURS),
                latitude=int(w["ilat"]), longitude=int(w["ilon"]),
            ).values.astype(np.float32)  # fmt: skip
            expected = np.empty(SLICE_HOURS, np.float32)
            for i in range(SLICE_HOURS):
                if not band.accumulated:
                    expected[i] = raw[i + 1]
                elif (h0 + i) % 24 == 1:
                    expected[i] = raw[i + 1]
                else:
                    expected[i] = raw[i + 1] - raw[i]
            expected = np.where(np.isfinite(expected), expected, NODATA)
            got = np.asarray(shard[row, k0 : k0 + SLICE_HOURS, b])
            worst[name] = max(worst[name], float(np.abs(got - expected).max()))
    logger.info(
        "values: max |shard - store| per band over %d windows: %s", len(rows), worst
    )
    return dict(
        windows=len(rows),
        max_abs_diff=worst,
        exact=all(v == 0.0 for v in worst.values()),
    )


def check_physics(
    shard: np.ndarray,
    index: pd.DataFrame,
    rng: np.random.Generator,
    max_windows: int = 2000,
) -> dict[str, Any]:
    """Band ranges and the evaporation split by climate class.

    Args:
        shard: The shard (memory-mapped).
        index: The shard's index rows, in row order.
        rng: Random generator.
        max_windows: Windows read.

    Returns:
        Per-band stats and per-Koppen-class daily evaporation components (mm).
    """
    rows = np.sort(
        rng.choice(len(index), size=min(max_windows, len(index)), replace=False)
    )
    data = np.asarray(shard[rows])
    bands = {}
    for k, band in enumerate(BANDS):
        v = data[:, :, k]
        valid = v != NODATA
        vv = v[valid]
        bands[band.name] = dict(
            min=float(vv.min()) if vv.size else None, max=float(vv.max()) if vv.size else None,
            mean=float(vv.mean()) if vv.size else None, nodata_share=float(1 - valid.mean()),
        )  # fmt: skip
    k_vt = [b.name for b in BANDS].index("evavt")
    k_bs = [b.name for b in BANDS].index("evabs")
    # Evaporation is negative in ERA5-Land; report mm per day as positive.
    frame = pd.DataFrame(dict(
        koppen=index["koppen_class"].to_numpy()[rows],
        transpiration_mm_day=-data[:, :, k_vt].sum(axis=1) / 365 * 1000,
        bare_soil_mm_day=-data[:, :, k_bs].sum(axis=1) / 365 * 1000,
    ))  # fmt: skip
    by_class = frame.groupby("koppen").agg(["mean", "count"]).round(3)
    by_class = by_class[by_class[("transpiration_mm_day", "count")] >= 10]
    logger.info("evaporation split by Koppen class (mm/day):\n%s", by_class.to_string())
    for name, s in bands.items():
        logger.info("  %-6s min %.4g max %.4g mean %.4g nodata %.2g", name, s["min"], s["max"],
                    s["mean"], s["nodata_share"])  # fmt: skip
    return dict(
        windows=len(rows), bands=bands,
        evaporation_by_koppen={
            c: dict(transpiration_mm_day=float(r[("transpiration_mm_day", "mean")]),
                    bare_soil_mm_day=float(r[("bare_soil_mm_day", "mean")]),
                    windows=int(r[("transpiration_mm_day", "count")]))
            for c, r in by_class.iterrows()
        },
    )  # fmt: skip


def check_cost(
    root: Path, shard: int, rng: np.random.Generator, per_band: int = 2
) -> dict[str, Any]:
    """Compressed chunk size and the projected download and job time.

    Args:
        root: Output root.
        shard: Shard number.
        rng: Random generator.
        per_band: Chunks fetched per band for the size sample.

    Returns:
        Sample sizes and projections.
    """
    plan = json.loads((root / "shards.json").read_text())["shards"]
    tiles = plan[shard]["tiles"]
    group = open_store()
    fs = fsspec.filesystem("https", **edh_storage_options())
    sizes, seconds = [], []
    for band in BANDS:
        array = group[band.source]
        for _ in range(per_band):
            tile = tiles[int(rng.integers(len(tiles)))]
            tc = tile["time_chunks"][int(rng.integers(len(tile["time_chunks"])))]
            coords = (tc, tile["tile_lat"], tile["tile_lon"])
            key = array.metadata.encode_chunk_key(coords)
            started = time.time()
            sizes.append(len(fs.cat_file(f"{HOURLY_URL}/{band.source}/{key}")))
            seconds.append(time.time() - started)
    mean_mb = float(np.mean(sizes)) / 1e6
    chunks_all = sum(len(t["time_chunks"]) for s in plan for t in s["tiles"]) * len(
        BANDS
    )
    progress = json.loads((root / "progress" / f"shard_{shard:03d}.json").read_text())
    chunk_rate = progress["chunks"] / max(progress["seconds_all_tiles"], 1.0)
    result = dict(
        sampled_chunks=len(sizes), mean_chunk_mb=round(mean_mb, 2),
        single_chunk_seconds_median=round(float(np.median(seconds)), 2),
        shard_chunks=progress["chunks"], shard_hours=round(progress["seconds_all_tiles"] / 3600, 2),
        shard_mb_per_s=round(chunk_rate * mean_mb, 1),
        all_shards=len(plan), all_chunks=chunks_all,
        projected_download_tb=round(chunks_all * mean_mb / 1e6, 2),
        projected_job_hours_one_at_a_time=round(chunks_all / chunk_rate / 3600, 1),
    )  # fmt: skip
    logger.info("cost: %s", result)
    return result


def run(root: Path, shard: int, samples: int, seed: int) -> None:
    """Validate one built shard.

    Args:
        root: Output root.
        shard: Shard number.
        samples: Windows for the value check.
        seed: Sampling seed.
    """
    rng = np.random.default_rng(seed)
    index = pd.read_parquet(root / "index.parquet")
    index = index[index["shard"] == shard].sort_values("row").reset_index(drop=True)
    data = np.load(root / "shards" / f"shard_{shard:03d}.npy", mmap_mode="r")
    assert data.shape == (len(index), WINDOW_HOURS, len(BANDS)), data.shape
    result = dict(
        shard=shard,
        values=check_values(data, index, samples, rng),
        physics=check_physics(data, index, rng),
        cost=check_cost(root, shard, rng),
    )
    out = root / "progress" / f"shard_{shard:03d}.validate.json"
    out.write_text(json.dumps(result, indent=2))
    logger.info("wrote %s (values exact: %s)", out, result["values"]["exact"])


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(Path(args.root), args.shard, args.samples, args.seed)


if __name__ == "__main__":
    main()
