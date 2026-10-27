"""Stage 2: snow cover phenology of every 0.1 deg cell and water year.

Per cell of every screened tile (``modis counts`` parts), the 8-day snow cover
fraction (SCF = snow / (snow + no snow) land pixels) is defined where at least
``MIN_CLEAR_FRACTION`` of the cell's land pixels were seen, gap-filled by linear
interpolation in time (clouds, polar night) and filtered with a 3-composite
running median (removes the single-composite snow commission errors of the
maximum-extent product; a monotonic melt curve is unchanged). Composites are
placed at their centre day.

Water years run October-September in the north and April-March in the south,
named by the calendar year of the melt (NH WY 2011 = 2010-10-01 .. 2011-09-30,
SH WY 2011 = 2011-04-01 .. 2012-03-31). In each water year:

* the snow season is the longest run of composites with SCF >= 0.5 (at least
  ``MIN_SEASON_SLOTS`` composites, else the year has no season);
* ``sdd`` (snow disappearance date) is the time SCF falls through 0.5 at the end
  of that run, linearly interpolated between composite centres: about the
  melt-out date of the cell's median pixel; ``sad`` (appearance) the time it
  rises through 0.5 at the start. Both are days since 1970-01-01; each is
  undefined when the run touches the water year's edge (censored);
* ``scd``: snow-covered days, the sum over the year's composites of SCF x
  composite length (cell-mean days with snow);
* ``season``: ``sdd - sad``; ``peak``: the year's highest SCF;
* ``sdd_gap``: days between the observed (not gap-filled) composites that
  bracket the SDD crossing, a label quality flag.

Day-of-water-year versions (``sdd_dowy``, ``sad_dowy``) count from the water
year's first day (1 = 1 October / 1 April). Leave-one-year-out climatologies
(``*_clim``: mean over the cell's other years where defined, at least
``MIN_CLIM_YEARS``) and anomalies (``*_anom``) are attached, so no label leaks
its own year into its reference.

Cells keep their own tile's pixels only; a cell split between tiles is kept
by the tile holding at least ``MIN_PIXEL_SHARE`` of its expected pixel count.

Outputs under ``<root>/meta``: ``phenology/<tile>.parquet`` (cell x water
year), ``scf/<tile>.npz`` (``lat10``, ``lon10``, ``slot_mid`` and the filtered
SCF series as uint8 x 200, 255 = no value, for every kept cell) and, merged,
``modis_cells.parquet`` (one row per cell with its climatology) and
``phenology.parquet``.

Usage::

    python -m olmoearth_pretrain.dataset_creation.snow.phenology --workers 48
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from .common import DEFAULT_ROOT, SPHERE_RADIUS, TILE_PX, TILE_SIZE_M
from .modis import FIRST_YEAR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MIN_CLEAR_FRACTION = 0.3
MIN_CLEAR_PIXELS = 20
MIN_PIXEL_SHARE = 0.8
MIN_LAND_FRACTION = 0.5
SCF_THRESHOLD = 0.5
MIN_SEASON_SLOTS = 4
MIN_CLIM_YEARS = 8
# Cells written to the phenology table: a snow season in at least this share of
# their water years (the eval's stricter filter is applied later).
MIN_SEASON_SHARE = 0.5
# Water years fully covered by MODIS/Terra (2000-02-18 onward) and the read.
NH_WY = (2001, 2026)
SH_WY = (2000, 2025)
CLIM_COLUMNS = ("sdd_dowy", "sad_dowy", "scd", "season")


def days(d: date) -> int:
    """Days since 1970-01-01.

    Args:
        d: Date.

    Returns:
        Day number.
    """
    return (d - date(1970, 1, 1)).days


def expected_pixels(lat10: np.ndarray) -> np.ndarray:
    """MODIS 500 m pixels in a full 0.1 deg cell (sinusoidal pixels are equal-area).

    Args:
        lat10: Cell-centre latitudes in tenths of a degree.

    Returns:
        Expected pixel counts.
    """
    pixel_area = (TILE_SIZE_M / TILE_PX) ** 2
    step = np.radians(0.1)
    lat = np.radians(lat10 / 10.0)
    area = SPHERE_RADIUS**2 * step * (np.sin(lat + step / 2) - np.sin(lat - step / 2))
    return area / pixel_area


def load_tile(root: Path, tile: str) -> dict[str, np.ndarray]:
    """Every year part of one tile, concatenated in time.

    Args:
        root: Dataset root.
        tile: ``hHHvVV``.

    Returns:
        ``lat10, lon10, npix, slot_start, slot_days, counts [T, n_cells, 3]``.
    """
    parts = sorted((root / "raw" / "modis" / "parts").glob(f"{tile}_*.npz"))
    loaded = [np.load(p) for p in parts]
    first = loaded[0]
    for part in loaded[1:]:
        if not np.array_equal(part["lat10"], first["lat10"]):
            raise ValueError(f"{tile}: cell grids differ between years")
    return dict(
        lat10=first["lat10"].astype(np.int32), lon10=first["lon10"].astype(np.int32),
        npix=first["npix"], slot_start=np.concatenate([p["slot_start"] for p in loaded]),
        slot_days=np.concatenate([p["slot_days"] for p in loaded]).astype(np.int32),
        counts=np.concatenate([p["counts"] for p in loaded]),
    )  # fmt: skip


def fill_series(scf: np.ndarray) -> np.ndarray:
    """Linear interpolation over missing composites, per cell.

    Args:
        scf: ``[T, n]`` with NaN gaps.

    Returns:
        Gap-filled copy (leading / trailing gaps take the nearest value; an
        all-missing column stays NaN).
    """
    out = scf.copy()
    t = np.arange(scf.shape[0])
    for j in range(scf.shape[1]):
        ok = np.isfinite(scf[:, j])
        if ok.any() and not ok.all():
            out[:, j] = np.interp(t, t[ok], scf[ok, j])
    return out


def crossing(t0: float, f0: float, t1: float, f1: float) -> float:
    """Time at which a linear segment crosses ``SCF_THRESHOLD``.

    Args:
        t0: First time.
        f0: Value at ``t0``.
        t1: Second time.
        f1: Value at ``t1``.

    Returns:
        The crossing time.
    """
    if f0 == f1:
        return (t0 + t1) / 2
    return t0 + (SCF_THRESHOLD - f0) / (f1 - f0) * (t1 - t0)


def water_years(north: bool) -> list[tuple[int, int, int]]:
    """Water years of a hemisphere.

    Args:
        north: Northern hemisphere cell.

    Returns:
        ``(wy, first day, end day exclusive)`` (days since 1970-01-01).
    """
    if north:
        return [(y, days(date(y - 1, 10, 1)), days(date(y, 10, 1)))
                for y in range(NH_WY[0], NH_WY[1] + 1)]  # fmt: skip
    return [(y, days(date(y, 4, 1)), days(date(y + 1, 4, 1)))
            for y in range(SH_WY[0], SH_WY[1] + 1)]  # fmt: skip


def cell_years(
    f: np.ndarray, observed: np.ndarray, mid: np.ndarray, length: np.ndarray,
    years: list[tuple[int, int, int]],
) -> list[dict[str, float]]:  # fmt: skip
    """Phenology of one cell in every water year.

    Args:
        f: Filtered, gap-filled SCF ``[T]``.
        observed: Whether each composite was observed (not gap-filled).
        mid: Composite centre days.
        length: Composite lengths (days).
        years: Output of :func:`water_years`.

    Returns:
        One dict per water year.
    """
    rows = []
    for wy, first, end in years:
        sel = np.flatnonzero((mid >= first) & (mid < end))
        row: dict[str, float] = dict(wy=wy, wy_start=first, n_slots=len(sel),
                                     n_obs=int(observed[sel].sum()))  # fmt: skip
        fs = f[sel]
        row["scd"] = float((fs * length[sel]).sum())
        row["peak"] = float(fs.max())
        snowy = fs >= SCF_THRESHOLD
        # Longest run of snow-covered composites.
        best_len, best_start, k = 0, -1, 0
        while k < len(sel):
            if snowy[k]:
                j = k
                while j + 1 < len(sel) and snowy[j + 1]:
                    j += 1
                if j - k + 1 > best_len:
                    best_len, best_start = j - k + 1, k
                k = j + 1
            else:
                k += 1
        row.update(sad=np.nan, sdd=np.nan, sdd_gap=np.nan, run_slots=best_len)
        if best_len >= MIN_SEASON_SLOTS:
            a, b = sel[best_start], sel[best_start + best_len - 1]
            if best_start > 0:
                row["sad"] = crossing(mid[a - 1], f[a - 1], mid[a], f[a])
            if best_start + best_len < len(sel):
                row["sdd"] = crossing(mid[b], f[b], mid[b + 1], f[b + 1])
                before = np.flatnonzero(observed[: b + 1])
                after = np.flatnonzero(observed[b + 1 :])
                if len(before) and len(after):
                    row["sdd_gap"] = float(mid[b + 1 + after[0]] - mid[before[-1]])
        rows.append(row)
    return rows


def loo_climatology(frame: pd.DataFrame, column: str) -> pd.Series:
    """Leave-one-year-out mean of a column per cell.

    Args:
        frame: Cell x water year rows (``cell`` key column).
        column: Column to average.

    Returns:
        Mean over the cell's other years where defined (NaN with fewer than
        ``MIN_CLIM_YEARS`` such years).
    """
    defined = frame[column].notna()
    total = frame[column].fillna(0.0).groupby(frame["cell"]).transform("sum")
    count = defined.groupby(frame["cell"]).transform("sum")
    others = count - defined.astype(int)
    mean = (total - frame[column].fillna(0.0)) / others.where(others > 0)
    return mean.where(others >= MIN_CLIM_YEARS)


def process_tile(job: tuple[str, str]) -> tuple[str, int, int]:
    """Worker: phenology table and SCF series of one tile.

    Args:
        job: ``(tile, dataset root)``.

    Returns:
        ``(tile, cells kept, cells written to the phenology table)``.
    """
    tile, root_str = job
    root = Path(root_str)
    data = load_tile(root, tile)
    counts = data["counts"].astype(np.float32)
    snow, clear = counts[..., 0], counts[..., 0] + counts[..., 1]
    land = clear + counts[..., 2]
    full = data["npix"] >= MIN_PIXEL_SHARE * expected_pixels(data["lat10"])
    # Land share from the composites where the cell was read at all.
    read = land.sum(axis=0) > 0
    land_frac = np.where(read, land.max(axis=0) / data["npix"], 0.0)
    keep = full & read & (land_frac >= MIN_LAND_FRACTION)
    idx = np.flatnonzero(keep)
    seen = (clear[:, idx] >= np.maximum(MIN_CLEAR_PIXELS, MIN_CLEAR_FRACTION * land[:, idx]))
    scf = np.where(seen, snow[:, idx] / np.maximum(clear[:, idx], 1), np.nan)
    filled = fill_series(scf)
    filtered = median_filter(np.nan_to_num(filled, nan=0.0), size=(3, 1), mode="nearest")
    filtered = np.where(np.isfinite(filled), filtered, np.nan)
    mid = data["slot_start"] + data["slot_days"] / 2.0
    lat10, lon10 = data["lat10"][idx], data["lon10"][idx]
    out_scf = root / "meta" / "scf" / f"{tile}.npz"
    out_scf.parent.mkdir(parents=True, exist_ok=True)
    stored = np.where(np.isfinite(filtered), np.rint(filtered * 200), 255).astype(np.uint8)
    np.savez_compressed(out_scf.with_suffix(".part.npz"), lat10=lat10, lon10=lon10,
                        slot_mid=mid, scf=stored, observed=seen)  # fmt: skip
    out_scf.with_suffix(".part.npz").rename(out_scf)
    rows = []
    for k in range(len(idx)):
        if not np.isfinite(filtered[:, k]).any():
            continue
        cell_rows = cell_years(filtered[:, k], seen[:, k], mid, data["slot_days"],
                               water_years(lat10[k] > 0))  # fmt: skip
        if np.mean([r["run_slots"] >= MIN_SEASON_SLOTS for r in cell_rows]) < MIN_SEASON_SHARE:
            continue
        for r in cell_rows:
            r.update(lat10=int(lat10[k]), lon10=int(lon10[k]))
        rows += cell_rows
    frame = pd.DataFrame(rows)
    if len(frame):
        frame.insert(0, "cell", [f"c{a:+05d}{b:+06d}" for a, b in zip(frame["lat10"], frame["lon10"])])
        frame["tile"] = tile
        frame["npix"] = frame["cell"].map(dict(zip(
            [f"c{a:+05d}{b:+06d}" for a, b in zip(lat10, lon10)], data["npix"][idx])))  # fmt: skip
        frame["sdd_dowy"] = frame["sdd"] - frame["wy_start"] + 1
        frame["sad_dowy"] = frame["sad"] - frame["wy_start"] + 1
        frame["season"] = frame["sdd"] - frame["sad"]
        for column in CLIM_COLUMNS:
            frame[f"{column}_clim"] = loo_climatology(frame, column)
            frame[f"{column}_anom"] = frame[column] - frame[f"{column}_clim"]
    out = root / "meta" / "phenology" / f"{tile}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out.with_suffix(".part.parquet"))
    out.with_suffix(".part.parquet").rename(out)
    return tile, len(idx), int(frame["cell"].nunique()) if len(frame) else 0


def merge(root: Path) -> None:
    """Merge the per-tile tables into ``phenology.parquet`` and ``modis_cells.parquet``.

    Args:
        root: Dataset root.
    """
    meta = root / "meta"
    frame = pd.concat(
        [pd.read_parquet(p) for p in sorted((meta / "phenology").glob("h*.parquet"))],
        ignore_index=True,
    )
    frame.to_parquet(meta / "phenology.parquet")
    grouped = frame.groupby("cell")
    cells = grouped.agg(
        lat10=("lat10", "first"), lon10=("lon10", "first"), tile=("tile", "first"),
        npix=("npix", "first"), n_wy=("wy", "size"),
        season_share=("run_slots", lambda s: float((s >= MIN_SEASON_SLOTS).mean())),
        n_sdd=("sdd", lambda s: int(s.notna().sum())),
        sdd_dowy_mean=("sdd_dowy", "mean"), sdd_dowy_sd=("sdd_dowy", "std"),
        sad_dowy_mean=("sad_dowy", "mean"), scd_mean=("scd", "mean"), scd_sd=("scd", "std"),
        peak_mean=("peak", "mean"), sdd_gap_median=("sdd_gap", "median"),
    ).reset_index()  # fmt: skip
    cells["lat"] = cells["lat10"] / 10.0
    cells["lon"] = cells["lon10"] / 10.0
    cells.to_parquet(meta / "modis_cells.parquet")
    logger.info(
        "phenology: %d cells x water years, %d cells (snow season in >= %.0f%% of years); "
        "%d with >= 15 SDD years; median SDD SD %.1f days",
        len(frame), len(cells), 100 * MIN_SEASON_SHARE, int((cells["n_sdd"] >= 15).sum()),
        float(cells.loc[cells["n_sdd"] >= 15, "sdd_dowy_sd"].median()),
    )  # fmt: skip


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tiles", default=None, help="hHHvVV list (default: every tile with parts)")
    args = ap.parse_args()
    root = Path(args.root)
    parts = sorted((root / "raw" / "modis" / "parts").glob("h*_*.npz"))
    have = sorted({p.name.split("_")[0] for p in parts})
    tiles = args.tiles.split(",") if args.tiles else have
    expected_years = 2026 - FIRST_YEAR + 1
    incomplete = [t for t in tiles
                  if sum(p.name.startswith(t + "_") for p in parts) != expected_years]  # fmt: skip
    if incomplete:
        raise ValueError(f"tiles without all {expected_years} year parts: {incomplete}")
    jobs = [(t, str(root)) for t in tiles]
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        for tile, kept, written in pool.imap_unordered(process_tile, jobs):
            logger.info("%s: %d cells, %d with seasonal snow", tile, kept, written)
    merge(root)


if __name__ == "__main__":
    main()
