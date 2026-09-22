"""Stage 2: stream ERA5-Land daily chunks and reduce them to per-region means.

Reads the ERA5-Land daily UTC Zarr on EarthDataHub through rslearn's
``ERA5LandDailyUTCv1`` (same source, auth and chunk geometry as the pretraining
ingest) one ``(time, lat, lon)`` chunk at a time, multiplies it by the sparse
region x cell weight matrices from ``build_weights`` and writes one compressed
``.npz`` partial per chunk. The raw grid is never written to disk.

Per partial, for each weight variant (``crop`` and ``area``):

* regions whose cells all lie inside this spatial chunk get their final
  weighted mean ``(R_in, T, C)`` (``NODATA`` where no valid cell);
* regions straddling a chunk boundary get partial weighted sums and weight
  totals ``(R_edge, T, C)`` that ``write_windows`` merges.

No-data (ocean / lake cells, NaN in the Zarr) is excluded from the mean by
carrying the weight total of *valid* cells per band and day.

Resumable: existing partials are skipped. Shardable over spatial chunks with
``--shard i --num-shards n`` so several Beaker jobs can run concurrently.

Usage::

    EARTHDATAHUB_TOKEN=... python -m olmoearth_pretrain.dataset_creation.cybench.aggregate_era5 \
        --meta-dir /weka/.../cybench/meta --out-dir /weka/.../cybench/agg \
        --start 1998-09-01 --end 2025-07-01 --workers 16
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .common import (
    DEFAULT_ROOT,
    ERA5L_BANDS,
    NODATA,
    chunk_index_range,
    fmt_chunk,
    i_to_lat,
    j_to_lon360,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VARIANTS = ("crop", "area")

# Per-worker state (xarray datasets are not picklable).
_CTX: dict[str, Any] = {}


def _open_source() -> tuple[Any, Any, tuple[int, int, int]]:
    """Open the ERA5-Land Zarr via rslearn's data source; returns (src, ds, chunk sizes)."""
    from rslearn.data_sources.earthdatahub import ERA5LandDailyUTCv1

    if not os.environ.get("EARTHDATAHUB_TOKEN"):
        raise RuntimeError("EARTHDATAHUB_TOKEN is not set")
    src = ERA5LandDailyUTCv1(band_names=list(ERA5L_BANDS))
    ds = src._get_dataset()  # noqa: SLF001 — reuse rslearn's auth + chunk handling
    return src, ds, src._chunk_sizes  # noqa: SLF001


def _init_worker() -> None:
    src, ds, cs = _open_source()
    _CTX["src"] = src
    _CTX["ds"] = ds
    _CTX["cs"] = cs
    _CTX["time_vals"] = ds["valid_time"].values


def map_grid_to_zarr(
    i: np.ndarray, j: np.ndarray, lat_vals: np.ndarray, lon_vals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map canonical (i, j) cell indices onto Zarr latitude/longitude indices."""
    lat_c = i_to_lat(i)
    lon_c = j_to_lon360(j)
    lon360 = np.mod(lon_vals, 360.0)
    # Nearest-neighbour via sorted search on each axis (axes may be ascending or
    # descending; sort once and map back).
    lat_order = np.argsort(lat_vals)
    lon_order = np.argsort(lon360)
    zi = lat_order[
        np.clip(np.searchsorted(lat_vals[lat_order], lat_c), 0, len(lat_vals) - 1)
    ]
    zi_alt = lat_order[
        np.clip(np.searchsorted(lat_vals[lat_order], lat_c) - 1, 0, len(lat_vals) - 1)
    ]
    zi = np.where(
        np.abs(lat_vals[zi] - lat_c) <= np.abs(lat_vals[zi_alt] - lat_c), zi, zi_alt
    )
    zj = lon_order[
        np.clip(np.searchsorted(lon360[lon_order], lon_c), 0, len(lon360) - 1)
    ]
    zj_alt = lon_order[
        np.clip(np.searchsorted(lon360[lon_order], lon_c) - 1, 0, len(lon360) - 1)
    ]
    dj = np.minimum(np.abs(lon360[zj] - lon_c), 360 - np.abs(lon360[zj] - lon_c))
    dj_alt = np.minimum(
        np.abs(lon360[zj_alt] - lon_c), 360 - np.abs(lon360[zj_alt] - lon_c)
    )
    zj = np.where(dj <= dj_alt, zj, zj_alt)
    err_lat = np.abs(lat_vals[zi] - lat_c).max()
    err_lon = np.minimum(
        np.abs(lon360[zj] - lon_c), 360 - np.abs(lon360[zj] - lon_c)
    ).max()
    if err_lat > 0.011 or err_lon > 0.011:
        raise RuntimeError(
            f"canonical grid does not match the Zarr grid (max err lat {err_lat:.4f}, lon {err_lon:.4f})"
        )
    return zi.astype(np.int64), zj.astype(np.int64)


def build_chunk_plans(
    weights: pd.DataFrame,
    regions: pd.DataFrame,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    cs: tuple[int, int, int],
) -> dict[tuple[int, int], dict[str, Any]]:
    """Group weights by spatial chunk and build sparse matrices per chunk.

    Returns ``{(latc, lonc): plan}`` where plan holds the region index arrays,
    which of them are fully inside the chunk, the CSR matrices for each weight
    variant, and the label year span of the regions in the chunk.
    """
    _, lat_cs, lon_cs = cs
    n_lat, n_lon = len(lat_vals), len(lon_vals)
    zi, zj = map_grid_to_zarr(
        weights["i"].to_numpy(), weights["j"].to_numpy(), lat_vals, lon_vals
    )
    w = weights.copy()
    w["zi"] = zi
    w["zj"] = zj
    w["latc"] = zi // lat_cs
    w["lonc"] = zj // lon_cs
    region_index = {r: k for k, r in enumerate(regions["region"])}
    w["ridx"] = w["region"].map(region_index).astype(int)
    if w["ridx"].isna().any():
        raise RuntimeError("weights reference regions missing from regions.parquet")
    n_chunks_per_region = w.groupby("ridx")[["latc", "lonc"]].nunique().max(axis=1)
    inside = set(n_chunks_per_region[n_chunks_per_region == 1].index)

    plans: dict[tuple[int, int], dict[str, Any]] = {}
    year_span = regions.set_index(regions.index)[["min_year", "max_year"]]
    for (latc, lonc), g in w.groupby(["latc", "lonc"]):
        h = min(lat_cs, n_lat - latc * lat_cs)
        wd = min(lon_cs, n_lon - lonc * lon_cs)
        local = (g["zi"] - latc * lat_cs) * wd + (g["zj"] - lonc * lon_cs)
        ridx = np.sort(g["ridx"].unique())
        row = np.searchsorted(ridx, g["ridx"].to_numpy())
        mats = {
            v: sp.csr_matrix(
                (g[f"w_{v}"].to_numpy(np.float32), (row, local.to_numpy())),
                shape=(len(ridx), h * wd),
            )
            for v in VARIANTS
        }
        yrs = year_span.loc[ridx]
        valid_years = yrs[yrs["min_year"] > 0]
        plans[(int(latc), int(lonc))] = dict(
            ridx=ridx,
            inside=np.array([r in inside for r in ridx]),
            mats=mats,
            h=h,
            w=wd,
            min_year=int(valid_years["min_year"].min()) if len(valid_years) else None,
            max_year=int(valid_years["max_year"].max()) if len(valid_years) else None,
        )
    return plans


def _process_chunk(task: dict[str, Any]) -> tuple[str, float, str]:
    """Worker: fetch one chunk, reduce, write the partial. Returns (name, secs, status)."""
    t0 = time.time()
    out = Path(task["out"])
    if out.exists():
        return out.stem, 0.0, "exists"
    ds = _CTX["ds"]
    time_cs, lat_cs, lon_cs = _CTX["cs"]
    tc, latc, lonc = task["tc"], task["latc"], task["lonc"]
    n_times = len(_CTX["time_vals"])
    t_sl = slice(tc * time_cs, min((tc + 1) * time_cs, n_times))
    lat_sl = slice(latc * lat_cs, latc * lat_cs + task["h"])
    lon_sl = slice(lonc * lon_cs, lonc * lon_cs + task["w"])

    for attempt in range(6):
        try:
            subset = (
                ds[list(ERA5L_BANDS)]
                .isel(valid_time=t_sl, latitude=lat_sl, longitude=lon_sl)
                .load()
            )
            break
        except Exception as e:  # noqa: BLE001 — remote store hiccups; retry with backoff
            if attempt == 5:
                raise
            logger.warning("%s: fetch failed (%s), retry %d", out.stem, e, attempt + 1)
            time.sleep(10 * (attempt + 1))

    arrs = [
        subset[b]
        .transpose("valid_time", "latitude", "longitude")
        .values.astype(np.float32)
        for b in ERA5L_BANDS
    ]
    x = np.stack(arrs, axis=0)  # (C, T, H, W)
    c, t, h, wd = x.shape
    x = x.reshape(c, t, h * wd)
    valid = np.isfinite(x) & (x != NODATA)
    xz = np.where(valid, x, 0.0).astype(np.float32)
    validf = valid.astype(np.float32)
    dates = (
        subset["valid_time"].values.astype("datetime64[D]").astype(np.int64)
    )  # days since epoch

    ridx = task["ridx"]
    inside = task["inside"]
    payload: dict[str, Any] = dict(
        tc=tc,
        latc=latc,
        lonc=lonc,
        dates=dates,
        ridx_in=ridx[inside],
        ridx_edge=ridx[~inside],
    )
    for v, mat in task["mats"].items():
        # (R, HW) @ (HW, T*C) -> (R, T, C)
        sums = (mat @ xz.transpose(2, 1, 0).reshape(h * wd, t * c)).reshape(
            len(ridx), t, c
        )
        wsum = (mat @ validf.transpose(2, 1, 0).reshape(h * wd, t * c)).reshape(
            len(ridx), t, c
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(wsum > 0, sums / wsum, NODATA).astype(np.float32)
        payload[f"mean_{v}"] = mean[inside]
        payload[f"sums_{v}"] = sums[~inside].astype(np.float32)
        payload[f"wsum_{v}"] = wsum[~inside].astype(np.float32)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, out)
    return out.stem, time.time() - t0, "ok"


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
        help="per spatial chunk, only fetch [min_label_year-2, max_label_year] instead of the full range",
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
        help="do not fetch; exit 0 iff every planned partial exists (all shards)",
    )
    ap.add_argument(
        "--mp-context",
        choices=["spawn", "fork"],
        default="spawn",
        help="multiprocessing start method ('fork' lets tests monkeypatch the source)",
    )
    args = ap.parse_args(argv)

    start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)
    meta = Path(args.meta_dir)
    weights = pd.read_parquet(meta / "weights.parquet")
    regions = pd.read_parquet(meta / "regions.parquet")
    logger.info("%d weight rows, %d regions", len(weights), len(regions))

    _, ds, cs = _open_source()
    time_vals = ds["valid_time"].values
    lat_vals = ds["latitude"].values
    lon_vals = ds["longitude"].values
    logger.info(
        "zarr: %d days (%s .. %s), %d lat x %d lon, chunks %s",
        len(time_vals),
        str(time_vals[0])[:10],
        str(time_vals[-1])[:10],
        len(lat_vals),
        len(lon_vals),
        cs,
    )
    plans = build_chunk_plans(weights, regions, lat_vals, lon_vals, cs)
    logger.info("%d spatial chunks", len(plans))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Global date axis for write_windows (days since epoch).
    full_range = chunk_index_range(time_vals, start, end, cs[0])
    tv = time_vals.astype("datetime64[D]")
    s_idx = int(np.searchsorted(tv, np.datetime64(start.replace(tzinfo=None), "D")))
    e_idx = int(np.searchsorted(tv, np.datetime64(end.replace(tzinfo=None), "D")))
    np.save(out_dir / "dates.npy", tv[s_idx:e_idx].astype(np.int64))
    regions.to_parquet(out_dir / "regions.parquet", index=False)

    tasks: list[dict[str, Any]] = []
    for k, ((latc, lonc), plan) in enumerate(sorted(plans.items())):
        if not args.check_complete and k % args.num_shards != args.shard:
            continue
        tcs = full_range
        if args.limit_to_label_years and plan["min_year"] is not None:
            s = max(start, datetime(plan["min_year"] - 2, 1, 1, tzinfo=UTC))
            e = min(end, datetime(plan["max_year"] + 1, 1, 1, tzinfo=UTC))
            tcs = chunk_index_range(time_vals, s, e, cs[0])
        for tc in tcs:
            tasks.append(
                dict(
                    tc=tc,
                    latc=latc,
                    lonc=lonc,
                    h=plan["h"],
                    w=plan["w"],
                    ridx=plan["ridx"],
                    inside=plan["inside"],
                    mats=plan["mats"],
                    out=str(out_dir / "partials" / f"{fmt_chunk(tc, latc, lonc)}.npz"),
                )
            )
    todo = [t for t in tasks if not Path(t["out"]).exists()]
    if args.check_complete:
        logger.info("%d planned partials, %d missing", len(tasks), len(todo))
        raise SystemExit(0 if not todo else 1)
    if args.max_chunks:
        todo = todo[: args.max_chunks]
    logger.info(
        "shard %d/%d: %d chunk tasks, %d to do",
        args.shard,
        args.num_shards,
        len(tasks),
        len(todo),
    )

    done = 0
    t_start = time.time()
    with multiprocessing.get_context(args.mp_context).Pool(
        args.workers, initializer=_init_worker
    ) as pool:
        for name, secs, status in pool.imap_unordered(
            _process_chunk, todo, chunksize=1
        ):
            done += 1
            if done % 20 == 0 or status != "ok":
                rate = done / max(time.time() - t_start, 1e-6)
                logger.info(
                    "%d/%d %s (%s, %.1fs) — %.2f chunks/s, ETA %.0f min",
                    done,
                    len(todo),
                    name,
                    status,
                    secs,
                    rate,
                    (len(todo) - done) / max(rate, 1e-6) / 60,
                )
    logger.info("done: %d chunks in %.1f min", done, (time.time() - t_start) / 60)


if __name__ == "__main__":
    main()
