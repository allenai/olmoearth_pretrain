"""Stage 2: stream ERA5-Land daily chunks and reduce them to per-region means.

Reads the ERA5-Land daily UTC Zarr on EarthDataHub through rslearn's
``ERA5LandDailyUTCv1`` (same source and chunking as the pretraining ingest),
one ``(time, lat, lon)`` chunk at a time, multiplies it by the sparse
region x cell weight matrices from ``build_weights`` and writes one ``.npz``
partial per chunk. The raw grid is never written to disk.

Per partial and weight variant (``crop``, ``area``): regions whose cells all
lie inside the chunk get their final weighted mean ``(R, T, C)``; regions that
straddle a chunk boundary get partial sums and weight totals that
``write_windows`` merges. No-data cells (NaN in the Zarr) are excluded from the
mean by summing weights over valid cells only, per band and day.

Resumable (existing partials are skipped) and shardable over spatial chunks
(``--shard i --num-shards n``).

Usage::

    EARTHDATAHUB_TOKEN=... python -m olmoearth_pretrain.dataset_creation.cybench.aggregate_era5 \
        --meta-dir .../meta --out-dir .../agg --start 1998-09-01 --end 2025-07-01 --workers 16
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .common import (
    DEFAULT_ROOT,
    ERA5L_BANDS,
    NODATA,
    i_to_lat,
    j_to_lon360,
    partial_name,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

T = TypeVar("T")
VARIANTS = ("crop", "area")
GRID_TOLERANCE_DEG = 0.011
FETCH_RETRIES = 6


# --- Zarr access ------------------------------------------------------------
class Era5Zarr:
    """Thin handle on the ERA5-Land Zarr: coordinates, chunk sizes, chunk fetch."""

    def __init__(self) -> None:
        """Open the store through rslearn (reuses its auth and chunk handling)."""
        from rslearn.data_sources.earthdatahub import ERA5LandDailyUTCv1

        if not os.environ.get("EARTHDATAHUB_TOKEN"):
            raise RuntimeError("EARTHDATAHUB_TOKEN is not set")
        source = ERA5LandDailyUTCv1(band_names=list(ERA5L_BANDS))
        self.ds = source._get_dataset()  # noqa: SLF001
        self.time_cs, self.lat_cs, self.lon_cs = source._chunk_sizes  # noqa: SLF001
        self.times = self.ds["valid_time"].values
        self.lats = self.ds["latitude"].values
        self.lons = self.ds["longitude"].values

    def time_chunks(self, start: datetime, end: datetime) -> range:
        """Time-chunk indices overlapping ``[start, end)``."""
        days = self.times.astype("datetime64[D]")
        first = max(
            int(
                np.searchsorted(
                    days, np.datetime64(start.replace(tzinfo=None), "D"), "right"
                )
            )
            - 1,
            0,
        )
        last = int(
            np.searchsorted(days, np.datetime64(end.replace(tzinfo=None), "D"), "left")
        )  # exclusive
        last = min(max(last, first + 1), len(days))
        return range(first // self.time_cs, -(-last // self.time_cs))

    def fetch(
        self, tc: int, latc: int, lonc: int, height: int, width: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load one chunk -> (days since epoch ``(T,)``, values ``(C, T, H*W)`` float32)."""
        t_sl = slice(tc * self.time_cs, min((tc + 1) * self.time_cs, len(self.times)))
        lat_sl = slice(latc * self.lat_cs, latc * self.lat_cs + height)
        lon_sl = slice(lonc * self.lon_cs, lonc * self.lon_cs + width)
        subset = _retry(
            lambda: self.ds[list(ERA5L_BANDS)]
            .isel(valid_time=t_sl, latitude=lat_sl, longitude=lon_sl)
            .load(),
            what=partial_name(tc, latc, lonc),
        )
        bands = [
            subset[b]
            .transpose("valid_time", "latitude", "longitude")
            .values.astype(np.float32)
            for b in ERA5L_BANDS
        ]
        values = np.stack(bands)  # (C, T, H, W)
        dates = subset["valid_time"].values.astype("datetime64[D]").astype(np.int64)
        return dates, values.reshape(values.shape[0], values.shape[1], -1)


def _retry(fn: Callable[[], T], what: str) -> T:
    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - remote store hiccups
            if attempt == FETCH_RETRIES:
                raise
            logger.warning("%s: fetch failed (%s), retry %d", what, e, attempt)
            time.sleep(10 * attempt)
    raise AssertionError("unreachable")


# --- planning ---------------------------------------------------------------
@dataclass
class ChunkPlan:
    """Everything needed to reduce one spatial chunk: which regions, with which weights."""

    latc: int
    lonc: int
    height: int
    width: int
    region_idx: np.ndarray  # rows of regions.parquet present in this chunk
    inside: np.ndarray  # bool per region: all of its cells are in this chunk
    weights: dict[str, sp.csr_matrix]  # variant -> (regions x cells)
    year_span: tuple[int, int] | None  # min/max label year over these regions


@dataclass
class ChunkTask:
    """One (time chunk, spatial chunk) unit of work."""

    tc: int
    plan: ChunkPlan
    out: Path


def nearest_index(
    axis: np.ndarray, targets: np.ndarray, period: float | None = None
) -> np.ndarray:
    """Index of the axis value nearest to each target (axis may be unsorted; optional wrap-around)."""
    order = np.argsort(axis)
    pos = np.clip(np.searchsorted(axis[order], targets), 0, len(axis) - 1)
    prev = np.clip(pos - 1, 0, len(axis) - 1)

    def dist(idx: np.ndarray) -> np.ndarray:
        d = np.abs(axis[order][idx] - targets)
        return np.minimum(d, period - d) if period else d

    best = np.where(dist(pos) <= dist(prev), pos, prev)
    err = dist(best).max()
    if err > GRID_TOLERANCE_DEG:
        raise RuntimeError(
            f"canonical grid does not match the Zarr grid (max error {err:.4f} deg)"
        )
    return order[best].astype(np.int64)


def plan_chunks(
    weights: pd.DataFrame, regions: pd.DataFrame, zarr: Era5Zarr
) -> list[ChunkPlan]:
    """Map weights onto the Zarr grid and build one sparse matrix per spatial chunk."""
    w = weights.copy()
    w["zi"] = nearest_index(zarr.lats, i_to_lat(w["i"].to_numpy()))
    w["zj"] = nearest_index(
        np.mod(zarr.lons, 360.0), j_to_lon360(w["j"].to_numpy()), period=360.0
    )
    w["latc"] = w["zi"] // zarr.lat_cs
    w["lonc"] = w["zj"] // zarr.lon_cs
    w["ridx"] = w["region"].map({r: k for k, r in enumerate(regions["region"])})
    if w["ridx"].isna().any():
        raise RuntimeError("weights reference regions missing from regions.parquet")
    w["ridx"] = w["ridx"].astype(int)
    chunks_per_region = w.groupby("ridx")[["latc", "lonc"]].nunique().max(axis=1)
    single_chunk_regions = set(chunks_per_region[chunks_per_region == 1].index)

    plans = []
    for (latc, lonc), g in w.groupby(["latc", "lonc"]):
        height = min(zarr.lat_cs, len(zarr.lats) - latc * zarr.lat_cs)
        width = min(zarr.lon_cs, len(zarr.lons) - lonc * zarr.lon_cs)
        cell = (g["zi"] - latc * zarr.lat_cs) * width + (g["zj"] - lonc * zarr.lon_cs)
        region_idx = np.sort(g["ridx"].unique())
        row = np.searchsorted(region_idx, g["ridx"].to_numpy())
        mats = {
            v: sp.csr_matrix(
                (g[f"w_{v}"].to_numpy(np.float32), (row, cell.to_numpy())),
                shape=(len(region_idx), height * width),
            )
            for v in VARIANTS
        }
        years = regions.loc[region_idx, ["min_year", "max_year"]]
        years = years[years["min_year"] > 0]
        plans.append(
            ChunkPlan(
                latc=int(latc),
                lonc=int(lonc),
                height=height,
                width=width,
                region_idx=region_idx,
                inside=np.isin(region_idx, list(single_chunk_regions)),
                weights=mats,
                year_span=(int(years["min_year"].min()), int(years["max_year"].max()))
                if len(years)
                else None,
            )  # fmt: skip
        )
    return plans


def plan_tasks(
    plans: list[ChunkPlan], zarr: Era5Zarr, args: argparse.Namespace, out_dir: Path
) -> list[ChunkTask]:
    """Cross spatial chunks with the time chunks they need (optionally limited to label years)."""
    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)
    tasks = []
    for k, plan in enumerate(sorted(plans, key=lambda p: (p.latc, p.lonc))):
        if not args.check_complete and k % args.num_shards != args.shard:
            continue
        s, e = start, end
        if args.limit_to_label_years and plan.year_span:
            s = max(start, datetime(plan.year_span[0] - 2, 1, 1, tzinfo=UTC))
            e = min(end, datetime(plan.year_span[1] + 1, 1, 1, tzinfo=UTC))
        for tc in zarr.time_chunks(s, e):
            tasks.append(
                ChunkTask(
                    tc,
                    plan,
                    out_dir
                    / "partials"
                    / f"{partial_name(tc, plan.latc, plan.lonc)}.npz",
                )
            )
    return tasks


# --- reduction (workers) ----------------------------------------------------
_ZARR: Era5Zarr | None = None  # per-worker handle (xarray objects are not picklable)


def _init_worker() -> None:
    global _ZARR  # noqa: PLW0603
    _ZARR = Era5Zarr()


def reduce_chunk(plan: ChunkPlan, values: np.ndarray) -> dict[str, np.ndarray]:
    """Weighted means (inside regions) and partial sums (edge regions) for one chunk.

    ``values`` is ``(C, T, cells)``; the reduction is one sparse matmul per
    variant: ``(regions x cells) @ (cells x T*C)``.
    """
    c, t, _ = values.shape
    valid = np.isfinite(values) & (values != NODATA)
    flat_values = (
        np.where(valid, values, 0.0)
        .astype(np.float32)
        .transpose(2, 1, 0)
        .reshape(-1, t * c)
    )
    flat_valid = valid.astype(np.float32).transpose(2, 1, 0).reshape(-1, t * c)
    out: dict[str, np.ndarray] = {
        "ridx_in": plan.region_idx[plan.inside],
        "ridx_edge": plan.region_idx[~plan.inside],
    }
    for variant, mat in plan.weights.items():
        sums = (mat @ flat_values).reshape(len(plan.region_idx), t, c)
        wsum = (mat @ flat_valid).reshape(len(plan.region_idx), t, c)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(wsum > 0, sums / wsum, NODATA).astype(np.float32)
        out[f"mean_{variant}"] = mean[plan.inside]
        out[f"sums_{variant}"] = sums[~plan.inside].astype(np.float32)
        out[f"wsum_{variant}"] = wsum[~plan.inside].astype(np.float32)
    return out


def process_chunk(task: ChunkTask) -> tuple[str, float, str]:
    """Worker: fetch, reduce, write one partial. Returns (name, seconds, status)."""
    t0 = time.time()
    if task.out.exists():
        return task.out.stem, 0.0, "exists"
    assert _ZARR is not None
    plan = task.plan
    dates, values = _ZARR.fetch(task.tc, plan.latc, plan.lonc, plan.height, plan.width)
    payload = reduce_chunk(plan, values)
    payload.update(tc=task.tc, latc=plan.latc, lonc=plan.lonc, dates=dates)
    task.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = task.out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, task.out)
    return task.out.stem, time.time() - t0, "ok"


# --- CLI --------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--meta-dir", default=str(DEFAULT_ROOT / "meta"))
    ap.add_argument("--out-dir", default=str(DEFAULT_ROOT / "agg"))
    ap.add_argument(
        "--start", default="1998-09-01", help="first day to aggregate (UTC)"
    )
    ap.add_argument("--end", default="2025-07-01", help="exclusive last day (UTC)")
    ap.add_argument(
        "--limit-to-label-years",
        action="store_true",
        help="per spatial chunk, fetch only [min label year - 2, max label year]",
    )
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument(
        "--max-chunks", type=int, default=None, help="debug: stop after N chunks"
    )
    ap.add_argument(
        "--check-complete",
        action="store_true",
        help="no fetching; exit 0 iff every planned partial exists (all shards)",
    )
    ap.add_argument(
        "--mp-context",
        choices=["spawn", "fork"],
        default="spawn",
        help="'fork' lets tests monkeypatch the Zarr",
    )
    args = ap.parse_args(argv)

    meta_dir, out_dir = Path(args.meta_dir), Path(args.out_dir)
    weights = pd.read_parquet(meta_dir / "weights.parquet")
    regions = pd.read_parquet(meta_dir / "regions.parquet")
    zarr = Era5Zarr()
    logger.info(
        "%d weight rows, %d regions | zarr: %d days (%s .. %s), %d lat x %d lon, chunks (%d, %d, %d)",
        len(weights), len(regions), len(zarr.times), str(zarr.times[0])[:10], str(zarr.times[-1])[:10],
        len(zarr.lats), len(zarr.lons), zarr.time_cs, zarr.lat_cs, zarr.lon_cs,
    )  # fmt: skip

    plans = plan_chunks(weights, regions, zarr)
    tasks = plan_tasks(plans, zarr, args, out_dir)
    todo = [t for t in tasks if not t.out.exists()]
    logger.info(
        "%d spatial chunks; shard %d/%d: %d tasks, %d to do",
        len(plans),
        args.shard,
        args.num_shards,
        len(tasks),
        len(todo),
    )
    if args.check_complete:
        raise SystemExit(0 if not todo else 1)
    if args.max_chunks:
        todo = todo[: args.max_chunks]

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_date_axis(zarr, args, out_dir)
    regions.to_parquet(out_dir / "regions.parquet", index=False)

    t_start = time.time()
    with multiprocessing.get_context(args.mp_context).Pool(
        args.workers, initializer=_init_worker
    ) as pool:
        for done, (name, secs, status) in enumerate(
            pool.imap_unordered(process_chunk, todo, chunksize=1), start=1
        ):
            if done % 20 == 0 or status != "ok":
                rate = done / max(time.time() - t_start, 1e-6)
                logger.info(
                    "%d/%d %s (%s, %.1fs) - %.2f chunks/s, ETA %.0f min",
                    done,
                    len(todo),
                    name,
                    status,
                    secs,
                    rate,
                    (len(todo) - done) / max(rate, 1e-6) / 60,
                )
    logger.info("done: %d chunks in %.1f min", len(todo), (time.time() - t_start) / 60)


def _write_date_axis(zarr: Era5Zarr, args: argparse.Namespace, out_dir: Path) -> None:
    """``dates.npy``: the global day axis (days since epoch) that ``write_windows`` indexes into."""
    days = zarr.times.astype("datetime64[D]")
    s = int(np.searchsorted(days, np.datetime64(args.start, "D")))
    e = int(np.searchsorted(days, np.datetime64(args.end, "D")))
    np.save(out_dir / "dates.npy", days[s:e].astype(np.int64))


if __name__ == "__main__":
    main()
