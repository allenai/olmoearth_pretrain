"""Stage 2: rasterize the weekly USDM maps onto the CONUS ERA5-Land cell grid.

Each 0.1 deg cell (centred on an ERA5-Land cell) is split into ``SUB x SUB``
sub-cells; every sub-cell inside CONUS takes the USDM severity of its centre
(0 = none, 1 = D0, ..., 5 = D4) and the cell stores the sum over its inside
sub-cells. A cell's severity is that sum over its number of inside sub-cells:
an area average in [0, 5], which is the cell's DSCI / 100 (USDM's Drought
Severity and Coverage Index weights D0..D4 cumulatively, i.e. by category + 1).

Outputs under ``<root>/meta``:

* ``usdm_sum.npy``: ``(n_maps, GRID_ROWS, GRID_COLS)`` uint8 severity sums,
  ``OUTSIDE`` (255) for cells outside the domain;
* ``usdm_inside.npy``: ``(GRID_ROWS, GRID_COLS)`` uint8 inside sub-cell counts;
* ``usdm_dates.npy``: map dates (days since 1970-01-01);
* ``usdm_national.csv``: per map, the CONUS area fractions in D0+ .. D4+ and the
  DSCI, to check against the published USDM statistics.

Usage::

    python -m olmoearth_pretrain.dataset_creation.usdm.rasterize --workers 32
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from rasterio.transform import from_origin

from .common import (
    DEFAULT_ROOT,
    EPOCH_DATE,
    GRID_COLS,
    GRID_LATS,
    GRID_NORTH,
    GRID_RES,
    GRID_ROWS,
    GRID_WEST,
    MIN_INSIDE_SUBCELLS,
    NON_CONUS_STATES,
    OUTSIDE,
    SUB,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FINE_SHAPE = (GRID_ROWS * SUB, GRID_COLS * SUB)
FINE_TRANSFORM = from_origin(
    GRID_WEST - GRID_RES / 2, GRID_NORTH + GRID_RES / 2, GRID_RES / SUB, GRID_RES / SUB
)


def block_sum(fine: np.ndarray) -> np.ndarray:
    """Sum ``SUB x SUB`` sub-cell blocks into cells.

    Args:
        fine: ``FINE_SHAPE`` array.

    Returns:
        ``(GRID_ROWS, GRID_COLS)`` sums.
    """
    return fine.reshape(GRID_ROWS, SUB, GRID_COLS, SUB).sum(axis=(1, 3), dtype=np.int32)


def conus_inside(states_zip: Path) -> np.ndarray:
    """Sub-cells inside CONUS (lower 48 + DC).

    Args:
        states_zip: Census state boundary archive.

    Returns:
        ``FINE_SHAPE`` uint8 mask.
    """
    states = gpd.read_file(f"zip://{states_zip}").to_crs(4326)
    conus = states[~states["STUSPS"].isin(NON_CONUS_STATES)]
    return rasterize(
        conus.geometry, out_shape=FINE_SHAPE, transform=FINE_TRANSFORM,
        fill=0, default_value=1, dtype="uint8",
    )  # fmt: skip


_INSIDE: np.ndarray | None = None


def _init_worker(inside: np.ndarray) -> None:
    """Keep the CONUS mask in each worker.

    Args:
        inside: Sub-cell mask.
    """
    global _INSIDE
    _INSIDE = inside


def rasterize_map(path: str) -> np.ndarray:
    """Worker: severity sums of one weekly map.

    Args:
        path: Map archive.

    Returns:
        ``(GRID_ROWS, GRID_COLS)`` int32 sums of sub-cell severity (category + 1).
    """
    assert _INSIDE is not None
    frame = gpd.read_file(f"zip://{path}")
    if frame.crs is not None and frame.crs.to_epsg() != 4326:
        frame = frame.to_crs(4326)
    frame = frame[frame.geometry.notna() & ~frame.geometry.is_empty]
    if len(frame) == 0:
        return np.zeros((GRID_ROWS, GRID_COLS), np.int32)
    shapes = [(geom, int(dm) + 1) for geom, dm in zip(frame.geometry, frame["DM"])]
    # Categories are exclusive; drawing in ascending order keeps the worst
    # category on the rare overlapping edge anyway.
    shapes.sort(key=lambda s: s[1])
    fine = rasterize(
        shapes, out_shape=FINE_SHAPE, transform=FINE_TRANSFORM, fill=0, dtype="uint8"
    )
    return block_sum(fine * _INSIDE)


def national_stats(
    sums: np.ndarray, inside: np.ndarray, domain: np.ndarray
) -> pd.DataFrame:
    """Area-weighted CONUS drought fractions per map (approximate: per cell).

    Args:
        sums: ``(n_maps, rows, cols)`` severity sums.
        inside: Inside sub-cell counts.
        domain: Domain mask.

    Returns:
        Columns ``dsci`` and ``d0p``..``d4p`` (fraction of area at or above
        each category, from cell-average severities).
    """
    weights = (np.cos(np.radians(GRID_LATS))[:, None]
               * np.ones(GRID_COLS))[domain]  # fmt: skip
    weights = weights / weights.sum()
    sev = sums[:, domain] / inside[domain]
    out = {"dsci": 100 * (sev * weights).sum(axis=1)}
    for k in range(5):
        out[f"d{k}p"] = ((sev >= k + 1 - 1e-6) * weights).sum(axis=1)
    return pd.DataFrame(out)


def run(root: Path, workers: int) -> None:
    """Rasterize every downloaded map.

    Args:
        root: Dataset root.
        workers: Processes.
    """
    raw, meta = root / "raw", root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    status = pd.read_csv(raw / "download_status.csv")
    status = status[status["status"].isin(["ok", "cached"])].sort_values("date")
    dates = [date.fromisoformat(d) for d in status["date"]]
    paths = [str(raw / "maps" / f"USDM_{d:%Y%m%d}_M.zip") for d in dates]

    inside_fine = conus_inside(raw / "cb_2018_us_state_20m.zip")
    inside = block_sum(inside_fine).astype(np.uint8)
    domain = inside >= MIN_INSIDE_SUBCELLS
    sums = np.full((len(paths), GRID_ROWS, GRID_COLS), OUTSIDE, np.uint8)
    with multiprocessing.get_context("spawn").Pool(
        workers, initializer=_init_worker, initargs=(inside_fine,)
    ) as pool:
        for k, cell_sums in enumerate(pool.imap(rasterize_map, paths, chunksize=4)):
            sums[k][domain] = cell_sums[domain].astype(np.uint8)
            if (k + 1) % 200 == 0:
                logger.info("[%d/%d maps]", k + 1, len(paths))
    np.save(meta / "usdm_sum.npy", sums)
    np.save(meta / "usdm_inside.npy", inside)
    days = np.array([(d - EPOCH_DATE).days for d in dates])
    np.save(meta / "usdm_dates.npy", days)
    (meta / "grid.json").write_text(json.dumps(dict(
        north=GRID_NORTH, west=GRID_WEST, res=GRID_RES, rows=GRID_ROWS, cols=GRID_COLS,
        sub=SUB, min_inside_subcells=MIN_INSIDE_SUBCELLS, outside=OUTSIDE,
        domain_cells=int(domain.sum()), maps=len(dates),
        first=dates[0].isoformat(), last=dates[-1].isoformat(),
    ), indent=2))  # fmt: skip
    national = national_stats(sums, inside, domain)
    national.insert(0, "date", [d.isoformat() for d in dates])
    national.to_csv(meta / "usdm_national.csv", index=False)
    gaps = np.diff(days)
    logger.info(
        "%d maps %s..%s (gaps != 7 days: %d), %d domain cells; D1+ area: mean %.3f, max %.3f on %s",
        len(dates), dates[0], dates[-1], int((gaps != 7).sum()), int(domain.sum()),
        national["d1p"].mean(), national["d1p"].max(), national.loc[national["d1p"].idxmax(), "date"],
    )  # fmt: skip


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    run(Path(args.root), args.workers)


if __name__ == "__main__":
    main()
