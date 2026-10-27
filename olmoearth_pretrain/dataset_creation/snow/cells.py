"""Stage 3: pick the label cells and extract their ERA5-Land series.

Eligible cells have seasonal snow every year in the MODIS record and a
well-observed melt (``eligible``): a snow season in at least
``MIN_SEASON_SHARE`` of the water years, an SDD in at least ``MIN_SDD_YEARS``,
a climatological season of at least ``MIN_CLIM_SEASON_DAYS``, and a cell that
is land in ERA5-Land. Neighbouring cells share weather and snow, so one
eligible cell per ``BLOCK_CELLS x BLOCK_CELLS`` block (0.5 deg) is drawn at
random (seeded): a jittered grid, up to four cells per mirror chunk.

Series are read from the weka ERA5-Land mirror one 10 x 10 cell chunk column at
a time (every sampled cell of a chunk from a single read), with the band order,
no-data handling and 29 February filling of ``fluxnet.era5.Era5Mirror``, from
``SERIES_START`` (2000-01-01, the first day of the mirror's third time chunk,
so earlier chunks are never read: MODIS starts on 2000-02-18).

Outputs under ``<root>/meta``: ``cells.parquet`` (``cell_id, lat, lon, ilat,
ilon`` and the MODIS climatology), ``era5/<cell_id>.npy`` (``(T, 14)``
float32, raw ECMWF units, ``NODATA`` gaps) and ``era5_dates.npy`` (days since
1970-01-01, from ``SERIES_START``).

Usage::

    python -m olmoearth_pretrain.dataset_creation.snow.cells --workers 48
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..fluxnet.era5 import Era5Mirror
from .common import DEFAULT_ROOT, NODATA

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BLOCK_CELLS = 5
MIRROR_CHUNK = 10
SERIES_START = (date(2000, 1, 1) - date(1970, 1, 1)).days
MIN_SEASON_SHARE = 0.9
MIN_SDD_YEARS = 15
MIN_CLIM_SEASON_DAYS = 30


def eligible(cells: pd.DataFrame) -> pd.Series:
    """Cells with a seasonal snow cover every year and a well-defined melt.

    Args:
        cells: ``modis_cells.parquet``.

    Returns:
        Boolean mask.
    """
    season_days = cells["sdd_dowy_mean"] - cells["sad_dowy_mean"]
    return (
        (cells["season_share"] >= MIN_SEASON_SHARE)
        & (cells["n_sdd"] >= MIN_SDD_YEARS)
        & (season_days.fillna(cells["sdd_dowy_mean"]) >= MIN_CLIM_SEASON_DAYS)
    )


def era5_indices(
    mirror: Era5Mirror, lat10: np.ndarray, lon10: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror row / column of cell centres (exact grid matches, -1 when outside).

    Args:
        mirror: The opened mirror.
        lat10: Cell-centre latitudes, tenths of a degree.
        lon10: Cell-centre longitudes, tenths of a degree (-1800..1800).

    Returns:
        ``(ilat, ilon)``.
    """
    lat_index = {int(round(float(v) * 10)): i for i, v in enumerate(mirror.lat)}
    lon_index = {int(round(float(v) * 10)) % 3600: j for j, v in enumerate(mirror.lon)}
    ilat = np.array([lat_index.get(int(v), -1) for v in lat10])
    ilon = np.array([lon_index.get(int(v) % 3600, -1) for v in lon10])
    return ilat, ilon


def sample_cells(root: Path, mirror: Era5Mirror, seed: int) -> pd.DataFrame:
    """One random eligible, ERA5-land cell per 0.5 deg block.

    Args:
        root: Dataset root (reads ``meta/modis_cells.parquet``).
        mirror: The opened mirror.
        seed: Sampling seed.

    Returns:
        The cell table.
    """
    modis = pd.read_parquet(root / "meta" / "modis_cells.parquet")
    cand = modis[eligible(modis)].copy()
    cand["ilat"], cand["ilon"] = era5_indices(mirror, cand["lat10"].to_numpy(),
                                              cand["lon10"].to_numpy())  # fmt: skip
    cand = cand[(cand["ilat"] >= 0) & (cand["ilon"] >= 0)]
    land = np.zeros(len(cand), bool)
    # ERA5-Land land mask, read one chunk-aligned block at a time.
    chunk = (cand["ilat"] // MIRROR_CHUNK) * 10_000 + cand["ilon"] // MIRROR_CHUNK
    for key, rows in cand.groupby(chunk).indices.items():
        i0, j0 = (key // 10_000) * MIRROR_CHUNK, (key % 10_000) * MIRROR_CHUNK
        block = np.asarray(
            mirror.valid[i0 : i0 + MIRROR_CHUNK, j0 : j0 + MIRROR_CHUNK], bool
        )
        sub = cand.iloc[rows]
        land[rows] = block[sub["ilat"].to_numpy() - i0, sub["ilon"].to_numpy() - j0]
    cand = cand[land]
    cand["block"] = (cand["lat10"] // BLOCK_CELLS) * 10_000 + cand[
        "lon10"
    ] // BLOCK_CELLS
    rng = np.random.default_rng(seed)
    cand["_key"] = rng.random(len(cand))
    cells = cand.sort_values("_key").groupby("block").head(1)
    cells = cells.drop(columns=["_key", "block"]).sort_values(["lat10", "lon10"])
    cells = cells.rename(columns={"cell": "cell_id"}).reset_index(drop=True)
    logger.info(
        "%d MODIS cells, %d eligible, %d ERA5-Land land; %d sampled (one per %d x %d block) "
        "in %d chunk columns",
        len(modis), int(eligible(modis).sum()), len(cand), len(cells), BLOCK_CELLS, BLOCK_CELLS,
        int(((cells["ilat"] // MIRROR_CHUNK) * 10_000 + cells["ilon"] // MIRROR_CHUNK).nunique()),
    )  # fmt: skip
    return cells


_MIRROR: Era5Mirror | None = None


def _init_worker() -> None:
    """Open the mirror once per worker process."""
    global _MIRROR
    _MIRROR = Era5Mirror()


def block_series(mirror: Era5Mirror, i0: int, i1: int, j0: int, j1: int) -> np.ndarray:
    """Daily records of a block of cells from ``SERIES_START``, 29 February filled in.

    Same values per cell as ``Era5Mirror.series`` on the same days, from one read
    of the chunks after ``SERIES_START``.

    Args:
        mirror: The opened mirror.
        i0: First latitude index.
        i1: End latitude index (exclusive).
        j0: First longitude index.
        j1: End longitude index (exclusive).

    Returns:
        ``(len(dates), 14, i1 - i0, j1 - j0)`` float32 in ``ERA5L_BANDS`` order,
        ``dates`` being ``mirror.dates[mirror.dates >= SERIES_START]``.
    """
    k0 = int(np.searchsorted(mirror.native_dates, SERIES_START))
    native = np.asarray(mirror.data[:, k0:, i0:i1, j0:j1], dtype=np.float32)
    native = native[mirror.band_index].transpose(1, 0, 2, 3)  # (T, C, h, w)
    native = np.where(np.isfinite(native), native, NODATA).astype(np.float32)
    dates = mirror.dates[mirror.dates >= SERIES_START]
    rows = np.searchsorted(dates, mirror.native_dates[k0:])
    full = np.full((len(dates),) + native.shape[1:], NODATA, np.float32)
    full[rows] = native
    missing = np.setdiff1d(np.arange(len(dates)), rows)
    before, after = full[missing - 1], full[missing + 1]
    ok = (before != NODATA) & (after != NODATA)
    full[missing] = np.where(ok, (before + after) / 2, NODATA)
    return full


def extract_chunk(job: tuple[list[dict[str, Any]], str]) -> list[dict[str, Any]]:
    """Worker: write the series of every sampled cell in one mirror chunk column.

    Args:
        job: ``(cell rows sharing a chunk, output dir)``.

    Returns:
        Per cell: ``cell_id``, its count of days with any no-data band and its
        mean surface pressure (hPa, an elevation proxy).
    """
    cells, out_dir = job
    assert _MIRROR is not None
    i0 = min(c["ilat"] for c in cells)
    i1 = max(c["ilat"] for c in cells) + 1
    j0 = min(c["ilon"] for c in cells)
    j1 = max(c["ilon"] for c in cells) + 1
    block = block_series(_MIRROR, i0, i1, j0, j1)
    sp_band = 4  # ERA5L_BANDS index of "sp"
    out = []
    for c in cells:
        series = np.ascontiguousarray(block[:, :, c["ilat"] - i0, c["ilon"] - j0])
        np.save(Path(out_dir) / f"{c['cell_id']}.npy", series)
        sp = series[:, sp_band]
        out.append(dict(
            cell_id=c["cell_id"], n_nodata_days=int((series == NODATA).any(axis=1).sum()),
            sp_hpa=float(sp[sp != NODATA].mean() / 100) if (sp != NODATA).any() else np.nan,
        ))  # fmt: skip
    return out


def run(root: Path, workers: int, seed: int) -> None:
    """Sample the cells and write their series.

    Args:
        root: Dataset root.
        workers: Reader processes.
        seed: Cell sampling seed.
    """
    meta = root / "meta"
    out_dir = meta / "era5"
    out_dir.mkdir(parents=True, exist_ok=True)
    mirror = Era5Mirror()
    cells = sample_cells(root, mirror, seed)
    np.save(meta / "era5_dates.npy", mirror.dates[mirror.dates >= SERIES_START])
    chunk = (cells["ilat"] // MIRROR_CHUNK) * 10_000 + cells["ilon"] // MIRROR_CHUNK
    todo = [
        (group.to_dict("records"), str(out_dir)) for _, group in cells.groupby(chunk)
    ]
    logger.info("%d chunk columns to read (%d cells)", len(todo), len(cells))
    rows: list[dict[str, Any]] = []
    with multiprocessing.get_context("spawn").Pool(
        workers, initializer=_init_worker
    ) as pool:
        for k, part in enumerate(pool.imap_unordered(extract_chunk, todo), start=1):
            rows += part
            if k % 200 == 0:
                logger.info("[%d/%d chunks]", k, len(todo))
    cells = cells.merge(pd.DataFrame(rows), on="cell_id", how="left")
    cells.to_parquet(meta / "cells.parquet")
    logger.info("wrote %d cells", len(cells))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(Path(args.root), args.workers, args.seed)


if __name__ == "__main__":
    main()
