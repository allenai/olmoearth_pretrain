"""Stage 3: sample label cells over CONUS and extract their ERA5-Land series.

USDM categories are drawn as polygons hundreds of km across, so neighbouring
0.1 deg cells carry nearly the same label and weather; one cell per
``BLOCK_CELLS x BLOCK_CELLS`` block (0.5 deg) keeps the label field's spatial
detail without storing redundant series. Within each block one domain cell
(mostly inside CONUS, see ``rasterize``) that is land in ERA5-Land is drawn at
random (seeded), so the sample is a jittered grid.

Series are read from the weka ERA5-Land mirror one 10 x 10 cell chunk column at
a time (every sampled cell of a chunk from a single read), with the same band
order, no-data handling and 29 February filling as ``fluxnet.era5.Era5Mirror``.

Outputs under ``<root>/meta``: ``cells.parquet`` (``cell_id, row, col, lat,
lon, ilat, ilon``), ``era5/<cell_id>.npy`` (``(T_all, 14)`` float32, raw ECMWF
units, ``NODATA`` gaps) and ``era5_dates.npy`` (days since 1970-01-01).

Usage::

    python -m olmoearth_pretrain.dataset_creation.usdm.cells --workers 48
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..fluxnet.era5 import Era5Mirror
from .common import (
    DEFAULT_ROOT,
    GRID_COLS,
    GRID_ROWS,
    MIN_INSIDE_SUBCELLS,
    NODATA,
    grid_lat,
    grid_lon,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BLOCK_CELLS = 5
MIRROR_CHUNK = 10


def era5_indices(
    mirror: Era5Mirror, lat: np.ndarray, lon: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror row / column of cell centres (exact grid matches).

    Args:
        mirror: The opened mirror.
        lat: Cell-centre latitudes.
        lon: Cell-centre longitudes (-180..180).

    Returns:
        ``(ilat, ilon)``.
    """
    lat_index = {round(float(v), 1): i for i, v in enumerate(mirror.lat)}
    lon_index = {round(float(v) % 360.0, 1): j for j, v in enumerate(mirror.lon)}
    ilat = np.array([lat_index[round(float(v), 1)] for v in lat])
    ilon = np.array([lon_index[round(float(v) % 360.0, 1) % 360.0] for v in lon])
    return ilat, ilon


def sample_cells(root: Path, mirror: Era5Mirror, seed: int) -> pd.DataFrame:
    """One random domain + ERA5-land cell per 0.5 deg block.

    Args:
        root: Dataset root (reads ``meta/usdm_inside.npy``).
        mirror: The opened mirror.
        seed: Sampling seed.

    Returns:
        The cell table.
    """
    inside = np.load(root / "meta" / "usdm_inside.npy")
    rows, cols = np.nonzero(inside >= MIN_INSIDE_SUBCELLS)
    lat, lon = grid_lat(rows), grid_lon(cols)
    ilat, ilon = era5_indices(mirror, lat, lon)
    # ERA5-Land land mask over the grid's bounding box, read in one go.
    i0, i1, j0, j1 = ilat.min(), ilat.max() + 1, ilon.min(), ilon.max() + 1
    valid = np.asarray(mirror.valid[i0:i1, j0:j1], dtype=bool)
    land = valid[ilat - i0, ilon - j0]
    cand = pd.DataFrame(
        dict(row=rows, col=cols, lat=lat, lon=lon, ilat=ilat, ilon=ilon)
    )[land]
    cand["block"] = (cand["row"] // BLOCK_CELLS) * GRID_COLS + cand[
        "col"
    ] // BLOCK_CELLS
    rng = np.random.default_rng(seed)
    cand["_key"] = rng.random(len(cand))
    cells = cand.sort_values("_key").groupby("block").head(1)
    cells = cells.drop(columns=["_key", "block"]).sort_values(["row", "col"])
    cells.insert(
        0, "cell_id", [f"r{r:03d}c{c:03d}" for r, c in zip(cells["row"], cells["col"])]
    )
    logger.info(
        "%d domain cells, %d land in ERA5-Land, %d sampled (one per %d x %d block; grid %d x %d)",
        len(rows), int(land.sum()), len(cells), BLOCK_CELLS, BLOCK_CELLS, GRID_ROWS, GRID_COLS,
    )  # fmt: skip
    return cells.reset_index(drop=True)


_MIRROR: Era5Mirror | None = None


def _init_worker() -> None:
    """Open the mirror once per worker process."""
    global _MIRROR
    _MIRROR = Era5Mirror()


def block_series(mirror: Era5Mirror, i0: int, i1: int, j0: int, j1: int) -> np.ndarray:
    """Complete daily records of a block of cells, 29 February filled in.

    Same result per cell as ``Era5Mirror.series``, from one read.

    Args:
        mirror: The opened mirror.
        i0: First latitude index.
        i1: End latitude index (exclusive).
        j0: First longitude index.
        j1: End longitude index (exclusive).

    Returns:
        ``(len(dates), 14, i1 - i0, j1 - j0)`` float32 in ``ERA5L_BANDS`` order.
    """
    native = np.asarray(mirror.data[:, :, i0:i1, j0:j1], dtype=np.float32)
    native = native[mirror.band_index].transpose(1, 0, 2, 3)  # (T, C, h, w)
    native = np.where(np.isfinite(native), native, NODATA).astype(np.float32)
    full = np.full((len(mirror.dates),) + native.shape[1:], NODATA, np.float32)
    full[mirror._native_rows] = native
    missing = np.setdiff1d(np.arange(len(mirror.dates)), mirror._native_rows)
    before, after = full[missing - 1], full[missing + 1]
    ok = (before != NODATA) & (after != NODATA)
    full[missing] = np.where(ok, (before + after) / 2, NODATA)
    return full


def extract_chunk(job: tuple[list[dict[str, Any]], str]) -> list[dict[str, Any]]:
    """Worker: write the series of every sampled cell in one mirror chunk column.

    Args:
        job: ``(cell rows sharing a chunk, output dir)``.

    Returns:
        Per cell: ``cell_id`` and its count of days with any no-data band.
    """
    cells, out_dir = job
    assert _MIRROR is not None
    i0 = min(c["ilat"] for c in cells)
    i1 = max(c["ilat"] for c in cells) + 1
    j0 = min(c["ilon"] for c in cells)
    j1 = max(c["ilon"] for c in cells) + 1
    block = block_series(_MIRROR, i0, i1, j0, j1)
    out = []
    for c in cells:
        series = np.ascontiguousarray(block[:, :, c["ilat"] - i0, c["ilon"] - j0])
        np.save(Path(out_dir) / f"{c['cell_id']}.npy", series)
        out.append(
            dict(
                cell_id=c["cell_id"],
                n_nodata_days=int((series == NODATA).any(axis=1).sum()),
            )
        )
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
    np.save(meta / "era5_dates.npy", mirror.dates)
    chunk = (cells["ilat"] // MIRROR_CHUNK) * 10_000 + cells["ilon"] // MIRROR_CHUNK
    todo = [
        (group.to_dict("records"), str(out_dir))
        for _, group in cells.groupby(chunk)
        if not all((out_dir / f"{c}.npy").exists() for c in group["cell_id"])
    ]
    logger.info("%d chunk columns to read (%d cells)", len(todo), len(cells))
    rows: list[dict[str, Any]] = []
    with multiprocessing.get_context("spawn").Pool(
        workers, initializer=_init_worker
    ) as pool:
        for k, part in enumerate(pool.imap_unordered(extract_chunk, todo), start=1):
            rows += part
            if k % 100 == 0:
                logger.info("[%d/%d chunks]", k, len(todo))
    if rows:
        nodata = pd.DataFrame(rows)
        cells = cells.merge(nodata, on="cell_id", how="left")
    cells.to_parquet(meta / "cells.parquet")
    logger.info("wrote %d cells; dates %d days", len(cells), len(mirror.dates))


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
