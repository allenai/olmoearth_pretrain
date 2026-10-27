"""Stage 4: anchors, targets, space x time splits and the rslearn windows.

``candidates``: for every sampled cell and every weekly map date ``d`` (the
anchor) whose ERA5 window ``[d - 448, d)`` is inside the mirror and has no
all-missing day: the cell's USDM severity ``s`` (0 = none, 1 = D0 .. 5 = D4,
area-averaged over the 0.1 deg cell, see ``rasterize``) at ``d`` and at
``LAGS`` weeks before / after it. The map valid on ``d`` (Tuesday) is drawn from
data up to that morning, so the ERA5 window ending the day before holds no
information from after the map. Targets:

* ``sev``: ``s(d)``, the current drought state (a nowcast);
* ``dpast<L>``: ``s(d) - s(d - L weeks)``, the recent evolution, all inside the
  input window (a nowcast of the trajectory);
* ``dfut<L>``: ``s(d + L weeks) - s(d)``, the coming evolution (a forecast:
  the weather after ``d`` is unseen);
* ``sev_m<L>`` / ``sev_p<L>``: the lagged severities themselves;
* ``tpast8`` / ``tfut8``: ``dpast8`` / ``dfut8`` as improve / stable / worsen
  (``|delta| >= TREND_THRESHOLD``), ``drought``: ``sev >= 2`` (D1 or worse).

A lag is only defined when the map ``L`` weeks away exists (no gaps).

Splits cross space and time, so no test window shares a cell, a landscape or a
season with a training window: 3 deg blocks go to train / val / test (70 / 15 /
15 % of cells, within west / central / east CONUS), water years (October to
September) go to train / val / test by ``WY % 5`` (2 -> test, 4 -> val), and a
window's split is its block's split when its water year has the same split,
else ``none`` (unused). Test water years 2002, 2007, 2012, 2017, 2022 and val
2004, 2009, 2014, 2019, 2024 spread every split over the 2000-2026 record.

``windows``: two groups of pre-materialized windows (``era5_daily`` raster
``(14, 448, 1, 1)`` + ``label`` point with the targets as properties), as in
the FLUXNET eval:

* ``all``: candidates on a ``--spacing``-week anchor grid (per-cell random
  phase), for supervised training;
* ``eval``: a fixed subset (``meta/eval_keys.parquet``, reused on rebuilds) for
  the in-loop linear probes, drawn by :func:`select_eval`.

Tags: ``split``, ``cell_id``, ``block``, ``region``, ``wy``, ``anchor_date``.

Usage::

    python -m olmoearth_pretrain.dataset_creation.usdm.write_windows candidates --workers 32
    python -m olmoearth_pretrain.dataset_creation.usdm.write_windows windows --workers 48
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rasterio.crs import CRS
from rslearn.dataset import Dataset, Window
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import WGS84_PROJECTION, Projection, STGeometry
from rslearn.utils.raster_array import RasterArray, RasterMetadata
from rslearn.utils.raster_format import NumpyRasterFormat
from rslearn.utils.vector_format import GeojsonCoordinateMode, GeojsonVectorFormat
from shapely.geometry import Point
from upath import UPath

from .common import (
    DEFAULT_ROOT,
    EPOCH,
    ERA5_LAYER,
    ERA5L_BANDS,
    LABEL_LAYER,
    NODATA,
    OUTSIDE,
    WINDOW_DAYS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PAST_LAGS = (4, 8, 13, 26)
FUTURE_LAGS = (4, 8, 13)
TREND_THRESHOLD = 0.5
SPLIT_BLOCK_CELLS = 30  # 3 deg
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
WY_SPLIT = {2: "test", 4: "val"}  # water year mod 5 -> split; others train

SEV_COLUMNS = (
    ["sev"]
    + [f"sev_m{lag}" for lag in PAST_LAGS]
    + [f"sev_p{lag}" for lag in FUTURE_LAGS]
)
DELTA_COLUMNS = [f"dpast{lag}" for lag in PAST_LAGS] + [
    f"dfut{lag}" for lag in FUTURE_LAGS
]
NUMERIC_TARGETS = tuple(SEV_COLUMNS + DELTA_COLUMNS)
CLASS_TARGETS = ("tpast8", "tfut8", "drought")


# --- splits -------------------------------------------------------------------
def region_of(lon: float) -> str:
    """Coarse CONUS region used to stratify the spatial split.

    Args:
        lon: Longitude (deg).

    Returns:
        ``west`` (< 104 W), ``central`` (104-90 W) or ``east``.
    """
    if lon < -104.0:
        return "west"
    return "central" if lon < -90.0 else "east"


def spatial_split(cells: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Split block, region and split of every cell, by whole 3 deg blocks.

    Args:
        cells: Cell table (``row``, ``col``, ``lon``).
        seed: Shuffle seed.

    Returns:
        ``cells`` with ``block``, ``region`` and ``space_split`` columns.
    """
    cells = cells.copy()
    brow, bcol = cells["row"] // SPLIT_BLOCK_CELLS, cells["col"] // SPLIT_BLOCK_CELLS
    cells["block"] = [f"b{r:02d}_{c:02d}" for r, c in zip(brow, bcol)]
    block_lon = cells.groupby("block")["lon"].transform("mean")
    cells["region"] = [region_of(v) for v in block_lon]
    rng = np.random.default_rng(seed)
    cells["space_split"] = ""
    for _, region in cells.groupby("region"):
        sizes = region.groupby("block").size()
        order = rng.permutation(sizes.index.to_numpy())
        cumulative = sizes.loc[order].cumsum() / sizes.sum()
        for block, frac in cumulative.items():
            if frac <= SPLIT_FRACTIONS["train"] + 1e-9:
                name = "train"
            elif frac <= SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["val"] + 1e-9:
                name = "val"
            else:
                name = "test"
            cells.loc[region.index[region["block"] == block], "space_split"] = name
    return cells


def water_year(days: np.ndarray) -> np.ndarray:
    """Water year (October-September, named by its end) of dates.

    Args:
        days: Days since 1970-01-01.

    Returns:
        Water years.
    """
    dt = pd.to_datetime(days, unit="D")
    return (dt.year + (dt.month >= 10)).to_numpy()


def time_split(wy: np.ndarray) -> np.ndarray:
    """Split of each water year.

    Args:
        wy: Water years.

    Returns:
        ``train`` / ``val`` / ``test`` per entry.
    """
    return np.array([WY_SPLIT.get(int(y) % 5, "train") for y in wy])


# --- targets ------------------------------------------------------------------
def cell_severity(root: Path, cells: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Severity series of every cell.

    Args:
        root: Dataset root.
        cells: Cell table.

    Returns:
        ``(map dates, severity [n_maps, n_cells])``; severity in [0, 5].
    """
    meta = root / "meta"
    sums = np.load(meta / "usdm_sum.npy", mmap_mode="r")
    inside = np.load(meta / "usdm_inside.npy")
    rows, cols = cells["row"].to_numpy(), cells["col"].to_numpy()
    picked = np.asarray(sums[:, rows, cols])
    if (picked == OUTSIDE).any():
        raise ValueError("a sampled cell is outside the USDM domain")
    return np.load(meta / "usdm_dates.npy"), (picked / inside[rows, cols]).astype(
        np.float32
    )


def shifted(values: np.ndarray, dates: np.ndarray, weeks: int) -> np.ndarray:
    """Values ``weeks`` maps later (negative: earlier), NaN across gaps / ends.

    Args:
        values: ``[n_maps, ...]`` per map.
        dates: Map dates (days).
        weeks: Offset in weekly maps.

    Returns:
        Same shape; entry ``k`` is ``values[k + weeks]`` when that map is exactly
        ``7 * weeks`` days away.
    """
    out = np.full(values.shape, np.nan, np.float32)
    n = len(dates)
    k = np.arange(n)
    j = k + weeks
    ok = (j >= 0) & (j < n)
    ok[ok] &= (dates[j[ok]] - dates[k[ok]]) == 7 * weeks
    out[k[ok]] = values[j[ok]]
    return out


def trend(delta: pd.Series) -> pd.Series:
    """Improve / stable / worsen labels of a severity change.

    Args:
        delta: Severity change.

    Returns:
        String labels (NaN where ``delta`` is).
    """
    labels = np.where(delta >= TREND_THRESHOLD, "worsen",
                      np.where(delta <= -TREND_THRESHOLD, "improve", "stable"))  # fmt: skip
    return pd.Series(labels, index=delta.index).where(delta.notna())


def cell_candidates(
    job: tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray], str],
) -> pd.DataFrame:
    """Worker: candidate anchors of one cell with their targets.

    Args:
        job: ``(cell row, map dates, {column: per-map values}, meta dir)``.

    Returns:
        One row per anchor with every target defined.
    """
    cell, dates, columns, meta_dir = job
    meta = Path(meta_dir)
    era5_dates = np.load(meta / "era5_dates.npy")
    series = np.load(meta / "era5" / f"{cell['cell_id']}.npy", mmap_mode="r")
    missing_day = (np.asarray(series) == NODATA).all(axis=1)
    bad_prefix = np.concatenate([[0], np.cumsum(missing_day)])
    frame = pd.DataFrame(columns)
    frame.insert(0, "anchor", dates)
    t0 = np.searchsorted(era5_dates, dates - WINDOW_DAYS)
    inside = (dates - WINDOW_DAYS >= era5_dates[0]) & (
        t0 + WINDOW_DAYS <= len(era5_dates)
    )
    t0_safe = np.where(inside, t0, 0)
    keep = inside & ((bad_prefix[t0_safe + WINDOW_DAYS] - bad_prefix[t0_safe]) == 0)
    keep &= frame[list(NUMERIC_TARGETS)].notna().all(axis=1).to_numpy()
    frame.insert(1, "t0", t0)
    frame = frame[keep].reset_index(drop=True)
    frame.insert(0, "cell_id", cell["cell_id"])
    return frame


def build_candidates(
    root: Path, workers: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cell splits and every candidate anchor; writes both tables under ``meta``.

    Args:
        root: Dataset root.
        workers: Processes.
        seed: Split seed.

    Returns:
        ``(cells, candidates)``.
    """
    meta = root / "meta"
    cells = spatial_split(pd.read_parquet(meta / "cells.parquet"), seed)
    dates, sev = cell_severity(root, cells)
    lagged = {"sev": sev}
    for lag in PAST_LAGS:
        lagged[f"sev_m{lag}"] = shifted(sev, dates, -lag)
    for lag in FUTURE_LAGS:
        lagged[f"sev_p{lag}"] = shifted(sev, dates, lag)
    for lag in PAST_LAGS:
        lagged[f"dpast{lag}"] = lagged["sev"] - lagged[f"sev_m{lag}"]
    for lag in FUTURE_LAGS:
        lagged[f"dfut{lag}"] = lagged[f"sev_p{lag}"] - lagged["sev"]
    jobs = [
        (cell, dates, {k: v[:, i] for k, v in lagged.items()}, str(meta))
        for i, cell in enumerate(cells.to_dict("records"))
    ]
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        parts = list(pool.imap(cell_candidates, jobs, chunksize=8))
    candidates = pd.concat(parts, ignore_index=True)
    candidates["wy"] = water_year(candidates["anchor"].to_numpy())
    candidates = candidates.merge(
        cells[["cell_id", "block", "region", "space_split"]], on="cell_id"
    )
    time_part = time_split(candidates["wy"].to_numpy())
    candidates["split"] = np.where(candidates["space_split"].to_numpy() == time_part,
                                   time_part, "none")  # fmt: skip
    candidates = candidates.drop(columns=["space_split"])
    candidates["tpast8"] = trend(candidates["dpast8"])
    candidates["tfut8"] = trend(candidates["dfut8"])
    candidates["drought"] = np.where(candidates["sev"] >= 2.0, "yes", "no")
    candidates = candidates.sort_values(["cell_id", "anchor"], ignore_index=True)
    cells.to_parquet(meta / "cells_split.parquet")
    candidates.to_parquet(meta / "candidates.parquet")
    summary = candidates.groupby("split").agg(
        cells=("cell_id", "nunique"), windows=("anchor", "size"),
        sev_mean=("sev", "mean"), drought=("drought", lambda s: float((s == "yes").mean())),
        dpast8_sd=("dpast8", "std"), dfut8_sd=("dfut8", "std"),
    )  # fmt: skip
    logger.info("candidates:\n%s", summary)
    return cells, candidates


# --- subsets ------------------------------------------------------------------
def stratum_of(frame: pd.DataFrame) -> pd.Series:
    """Eval sampling stratum: drought state x recent / coming evolution.

    Args:
        frame: Candidate rows.

    Returns:
        Stratum label per row: ``<state>_<dynamics>`` with state ``wet`` (s = 0),
        ``dry`` (0 < s < 2) or ``drought`` (s >= 2) and dynamics ``moving`` when
        either the past or the coming 8-week change reaches ``TREND_THRESHOLD``,
        else ``steady``.
    """
    state = np.where(
        frame["sev"] <= 0, "wet", np.where(frame["sev"] < 2.0, "dry", "drought")
    )
    moving = (frame["dpast8"].abs() >= TREND_THRESHOLD) | (
        frame["dfut8"].abs() >= TREND_THRESHOLD
    )
    return pd.Series(state, index=frame.index) + np.where(moving, "_moving", "_steady")


# Share of each split's eval windows per stratum: drought dynamics are rare in a
# uniform draw (most cell-weeks are wet and steady), so moving windows are
# enriched; states are kept roughly in proportion so the current-state target
# stays meaningful.
STRATUM_SHARES = {
    "wet_steady": 0.20, "wet_moving": 0.10,
    "dry_steady": 0.10, "dry_moving": 0.20,
    "drought_steady": 0.15, "drought_moving": 0.25,
}  # fmt: skip


def balanced_subset(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Up to ``n`` rows spread as evenly as possible over cells and water years.

    Takes rows round-robin over cells (each cell's rows in random order, at most
    one per (cell, water year) per round), so every cell contributes before any
    contributes twice.

    Args:
        frame: Candidate rows.
        n: Rows wanted.
        seed: Shuffle seed.

    Returns:
        The selected rows.
    """
    shuffled = frame.sample(frac=1.0, random_state=seed)
    shuffled = shuffled.assign(_wy_rank=shuffled.groupby(["cell_id", "wy"]).cumcount())
    shuffled = shuffled.sort_values("_wy_rank", kind="stable")
    shuffled = shuffled.assign(_rank=shuffled.groupby("cell_id").cumcount())
    cell_order = {c: k for k, c in enumerate(shuffled["cell_id"].drop_duplicates())}
    shuffled = shuffled.assign(_cell=shuffled["cell_id"].map(cell_order))
    picked = shuffled.sort_values(["_rank", "_cell"]).head(n)
    return picked.drop(columns=["_wy_rank", "_rank", "_cell"])


def select_eval(
    candidates: pd.DataFrame, quotas: dict[str, int], seed: int
) -> pd.DataFrame:
    """The eval subset: per split, ``STRATUM_SHARES`` of each stratum, cell-balanced.

    Args:
        candidates: Output of :func:`build_candidates`.
        quotas: Windows per split.
        seed: Sampling seed.

    Returns:
        Selected rows with a ``stratum`` column.
    """
    parts = []
    for split, n in quotas.items():
        pool = candidates[candidates["split"] == split]
        pool = pool.assign(stratum=stratum_of(pool))
        for stratum, share in STRATUM_SHARES.items():
            part = pool[pool["stratum"] == stratum]
            want = int(round(n * share))
            if len(part) < want:
                logger.warning(
                    "%s %s: %d candidates < %d wanted", split, stratum, len(part), want
                )
            parts.append(balanced_subset(part, want, seed))
    return pd.concat(parts, ignore_index=True)


def select_all(candidates: pd.DataFrame, spacing_weeks: int, seed: int) -> pd.DataFrame:
    """The ``all`` group: every split's candidates on a per-cell phased anchor grid.

    Args:
        candidates: Output of :func:`build_candidates`.
        spacing_weeks: Anchor spacing in weeks.
        seed: Phase seed.

    Returns:
        Selected rows (split ``none`` excluded).
    """
    rng = np.random.default_rng(seed)
    cells = candidates["cell_id"].unique()
    phase = dict(zip(cells, rng.integers(0, spacing_weeks, len(cells))))
    week = (candidates["anchor"] // 7).to_numpy()
    on_grid = (week - candidates["cell_id"].map(phase).to_numpy()) % spacing_weeks == 0
    return candidates[on_grid & (candidates["split"] != "none").to_numpy()]


# --- windows ------------------------------------------------------------------
class WindowWriter:
    """Writes pre-materialized rslearn windows (ERA5 raster + target point)."""

    def __init__(self, ds_path: str) -> None:
        """Open the dataset and set up formats and projection.

        Args:
            ds_path: rslearn dataset root.
        """
        self.dataset = Dataset(UPath(ds_path))
        self.projection = Projection(CRS.from_epsg(4326), 0.1, -0.1)
        self.raster_format = NumpyRasterFormat()
        self.vector_format = GeojsonVectorFormat(
            coordinate_mode=GeojsonCoordinateMode.WGS84
        )

    def write(
        self, group: str, row: dict[str, Any], cell: dict[str, Any], block: np.ndarray
    ) -> None:
        """Save one window with its ERA5 block and target feature.

        Args:
            group: Window group (``all`` or ``eval``).
            row: Candidate row.
            cell: Cell row (coordinates).
            block: ``(448, 14)`` ERA5 rows of the window.
        """
        start = EPOCH + timedelta(days=int(row["anchor"]) - WINDOW_DAYS)
        end = EPOCH + timedelta(days=int(row["anchor"]))
        # The 0.1 deg pixel whose north-west corner is the cell centre.
        col = int(np.floor(cell["lon"] / 0.1 + 1e-6))
        pix_row = int(np.floor(cell["lat"] / -0.1 + 1e-6))
        targets: dict[str, Any] = {t: float(row[t]) for t in NUMERIC_TARGETS}
        targets.update({t: str(row[t]) for t in CLASS_TARGETS})
        options: dict[str, Any] = dict(
            split=row["split"], cell_id=row["cell_id"], block=row["block"],
            region=row["region"], wy=str(int(row["wy"])), anchor_date=end.strftime("%Y-%m-%d"),
            **targets,
        )  # fmt: skip
        if "stratum" in row:
            options["stratum"] = row["stratum"]
        window = Window(
            storage=self.dataset.storage, group=group, name=f"{row['cell_id']}_{end:%Y%m%d}",
            projection=self.projection, bounds=(col, pix_row, col + 1, pix_row + 1),
            time_range=(start, end), options=options,
        )  # fmt: skip
        window.save()
        raster = RasterArray(
            array=np.ascontiguousarray(block.T)[:, :, None, None].astype(np.float32),
            timestamps=[
                (start + timedelta(days=k), start + timedelta(days=k + 1))
                for k in range(WINDOW_DAYS)
            ],  # fmt: skip
            metadata=RasterMetadata(nodata_value=NODATA),
        )
        self.raster_format.encode_raster(
            window.get_raster_dir(ERA5_LAYER, list(ERA5L_BANDS)),
            self.projection,
            window.bounds,
            raster,
        )
        window.mark_layer_completed(ERA5_LAYER)
        # The cell centre lies on a pixel corner of the 0.1 deg window grid, so
        # the label point goes at the window pixel's centre (0.05 deg off the
        # ERA5 cell centre), safely inside the bounds rslearn reads labels from.
        centre = Point((col + 0.5) * 0.1, (pix_row + 0.5) * -0.1)
        feature = Feature(STGeometry(WGS84_PROJECTION, centre, (start, end)), targets)
        self.vector_format.encode_vector(window.get_layer_dir(LABEL_LAYER), [feature])
        window.mark_layer_completed(LABEL_LAYER)


def dataset_config() -> dict[str, Any]:
    """Rslearn ``config.json`` (pre-materialized layers, no data sources).

    Returns:
        The config dict.
    """
    band_set = {
        "bands": list(ERA5L_BANDS), "dtype": "float32", "nodata_value": NODATA,
        "format": {"class_path": "rslearn.utils.raster_format.NumpyRasterFormat"},
    }  # fmt: skip
    return {"layers": {ERA5_LAYER: {"type": "raster", "band_sets": [band_set]},
                       LABEL_LAYER: {"type": "vector"}}}  # fmt: skip


def write_cell(
    job: tuple[dict[str, Any], dict[str, list[dict[str, Any]]], str, str],
) -> dict[str, int]:
    """Worker: write every selected window of one cell.

    Args:
        job: ``(cell row, {group: candidate rows}, meta dir, dataset path)``.

    Returns:
        Windows written per group.
    """
    cell, rows_by_group, meta_dir, ds_path = job
    series = np.load(Path(meta_dir) / "era5" / f"{cell['cell_id']}.npy", mmap_mode="r")
    writer = WindowWriter(ds_path)
    counts: dict[str, int] = {}
    for group, rows in rows_by_group.items():
        for row in rows:
            t0 = int(row["t0"])
            writer.write(group, row, cell, np.asarray(series[t0 : t0 + WINDOW_DAYS]))
        counts[group] = len(rows)
    return counts


def write_all(
    root: Path, selected: dict[str, pd.DataFrame], cells: pd.DataFrame, workers: int
) -> None:
    """Write the selected windows of every group and their summaries.

    Args:
        root: Dataset root.
        selected: Rows per group.
        cells: Cell table.
        workers: Processes.
    """
    ds_path = root / "rslearn_dataset"
    ds_path.mkdir(parents=True, exist_ok=True)
    (ds_path / "config.json").write_text(json.dumps(dataset_config(), indent=2))
    cell_rows = cells.set_index("cell_id").to_dict("index")
    grouped = {g: dict(tuple(f.groupby("cell_id"))) for g, f in selected.items()}
    jobs = []
    for cell_id in sorted(set().union(*(set(f["cell_id"]) for f in selected.values()))):
        rows_by_group = {g: parts[cell_id].to_dict("records") for g, parts in grouped.items()
                         if cell_id in parts}  # fmt: skip
        jobs.append((dict(cell_rows[cell_id], cell_id=cell_id), rows_by_group,
                     str(root / "meta"), str(ds_path)))  # fmt: skip
    totals: dict[str, int] = {}
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for k, counts in enumerate(pool.imap_unordered(write_cell, jobs), start=1):
            for group, n in counts.items():
                totals[group] = totals.get(group, 0) + n
            if k % 200 == 0:
                logger.info("[%d/%d cells] %s", k, len(jobs), totals)
    stats: dict[str, Any] = {}
    for group, frame in selected.items():
        frame.to_parquet(root / "meta" / f"windows_{group}.parquet")
        stats[group] = {
            split: {"windows": len(part), "cells": int(part["cell_id"].nunique()),
                    **{t: dict(mean=float(part[t].mean()), std=float(part[t].std()))
                       for t in NUMERIC_TARGETS},
                    **{t: part[t].value_counts().to_dict() for t in CLASS_TARGETS}}
            for split, part in frame.groupby("split")
        }  # fmt: skip
    (root / "meta" / "label_stats.json").write_text(json.dumps(stats, indent=2))
    logger.info("windows written: %s", totals)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("step", choices=["candidates", "windows", "all"])
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--spacing",
        type=int,
        default=12,
        help="anchor spacing (weeks) of the 'all' group",
    )
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=1500)
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--groups", default="all,eval", help="window groups to write")
    args = ap.parse_args()
    root = Path(args.root)
    if args.step in ("candidates", "all"):
        cells, candidates = build_candidates(root, args.workers, args.seed)
    else:
        cells = pd.read_parquet(root / "meta" / "cells_split.parquet")
        candidates = pd.read_parquet(root / "meta" / "candidates.parquet")
    if args.step in ("windows", "all"):
        keys_path = root / "meta" / "eval_keys.parquet"
        if keys_path.exists():
            keys = pd.read_parquet(keys_path)
            eval_rows = candidates.merge(
                keys[["cell_id", "anchor", "stratum"]], on=["cell_id", "anchor"]
            )
            if len(eval_rows) != len(keys):
                raise ValueError(
                    f"{len(keys) - len(eval_rows)} eval keys are not candidates"
                )
        else:
            quotas = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
            eval_rows = select_eval(candidates, quotas, args.seed)
            eval_rows[["cell_id", "anchor", "split", "stratum"]].to_parquet(keys_path)
        selected = {
            "all": select_all(candidates, args.spacing, args.seed),
            "eval": eval_rows,
        }
        selected = {g: f for g, f in selected.items() if g in args.groups.split(",")}
        write_all(root, selected, cells, args.workers)


if __name__ == "__main__":
    main()
