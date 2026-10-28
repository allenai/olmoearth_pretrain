"""Stage 4: anchors, targets, space x time splits and the rslearn windows.

One label per sampled cell and water year (``phenology``). Each window family
places the ERA5 window ``[d - 448, d)`` differently relative to the melt:

* ``l<L>`` (forecast, ``LEADS``): ``d`` is ``L`` days before the cell's
  climatological melt-out date (leave-one-year-out mean SDD, the same reference
  as the anomaly), so every cell's history stops the same time ahead of its
  usual melt. Target ``sdd_anom``: this year's SDD minus that climatology,
  which equals the days from the window's end to melt-out minus ``L``; the
  encoder never sees the climatology, so the target is "how many days until the
  snow is gone", from the snowpack's history and the spring so far. Small
  leads include early years whose melt is already inside the window (``melted``).
* ``p<L>`` (lag, ``LAGS``): ``d`` is ``L`` days after the climatological melt-out
  date, so the melt is usually inside the window; ``sdd_anom`` is then ``L``
  minus the days since the snow disappeared: dating a past event of the
  history (ERA5-Land's own albedo shows its modelled melt, which is not
  MODIS's).
* ``post`` (hindcast): ``d`` is the first day of the next water year, so the
  window holds the whole season (and its 83 days before). Targets ``sdd_anom``,
  ``scd_anom``, ``sad_anom``, ``season_anom``, and the absolute ``sdd_dowy``,
  ``scd`` (the naive targets, dominated by each cell's climate). The ERA5-Land
  input has no snow variable, but its net / downward shortwave ratio is its own
  modelled albedo, so the hindcast measures how well the encoder reads a season
  it has seen.

Labels need an observed melt: ``sdd`` with a leave-one-year-out climatology and
bracketing observed composites at most ``MAX_SDD_GAP`` days apart, in a cell
with a stable seasonal snowpack (inter-annual SDD SD at most
``MAX_CELL_SDD_SD`` days, climatological season at least
``MIN_CELL_SEASON_DAYS``). ``sad`` and
``season`` (the ``post`` family's ``sad_anom`` / ``season_anom``) are defined
only when the season does not start before the water year (tag ``has_sad``).

Splits cross space and time, as in the USDM eval: 3 deg blocks go to train /
val / test (70 / 15 / 15 % of cells within each region), water years by
``WY % 5`` (2 -> test, 4 -> val, else train), and a window's split is its
block's when its water year has the same split, else ``none`` (unused).

``windows``: pre-materialized windows per family (``era5_daily`` raster
``(14, 448, 1, 1)`` + ``label`` point with the targets as properties, as in
the FLUXNET / USDM evals), in groups ``eval_<family>`` (the same fixed
(cell, water year) keys in every family, ``meta/eval_keys.parquet``) and
``all_<family>`` (up to ``--all-per-cell`` water years of every cell of the
train / val / test splits, for supervised training; ``--all-families``).

Tags: ``split``, ``cell_id``, ``block``, ``region``, ``wy``, ``anchor_date``,
``family``, ``has_sad``.

Usage::

    python -m olmoearth_pretrain.dataset_creation.snow.write_windows candidates --workers 32
    python -m olmoearth_pretrain.dataset_creation.snow.write_windows windows --workers 48
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
    WINDOW_DAYS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LEADS = (0, 30, 60, 90)
# Cells with a stable seasonal snowpack: in short or intermittent snow climates
# (much of the Tibetan Plateau, maritime lowlands) the longest snow run jumps
# between separate episodes from year to year, which shows up as SDD anomalies
# of +-60-120 days (kurtosis 15 over all cells, 2.7 with this filter; it keeps
# 87 % of the cell-years and leaves the station agreement unchanged).
MAX_CELL_SDD_SD = 20.0
MIN_CELL_SEASON_DAYS = 60.0
LAGS = (30, 60)
FAMILIES = (
    tuple(f"l{lead}" for lead in LEADS) + tuple(f"p{lag}" for lag in LAGS) + ("post",)
)
MAX_SDD_GAP = 24.0
SPLIT_BLOCK_DEG = 3
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
WY_SPLIT = {2: "test", 4: "val"}  # water year mod 5 -> split; others train

TARGETS = {
    "forecast": ("sdd_anom", "sdd_dowy", "days_to_melt"),
    "post": ("sdd_anom", "scd_anom", "sad_anom", "season_anom", "sdd_dowy", "scd"),
}
# Targets undefined for some windows (``has_sad``); written as NaN-free
# properties only where defined.
OPTIONAL_TARGETS = ("sad_anom", "season_anom")


def family_kind(family: str) -> str:
    """``forecast`` or ``post``.

    Args:
        family: Window family.

    Returns:
        Its kind.
    """
    return "post" if family == "post" else "forecast"


# --- splits -------------------------------------------------------------------
def region_of(lat: float, lon: float) -> str:
    """Coarse region used to stratify the spatial split.

    Args:
        lat: Latitude (deg).
        lon: Longitude (deg).

    Returns:
        ``south`` (southern hemisphere), ``america``, ``europe`` (30 W - 60 E),
        ``north_asia`` (east of 60 E, from 50 N) or ``south_asia``.
    """
    if lat < 0:
        return "south"
    if lon < -30.0:
        return "america"
    if lon < 60.0:
        return "europe"
    return "north_asia" if lat >= 50.0 else "south_asia"


def spatial_split(cells: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Split block, region and split of every cell, by whole 3 deg blocks.

    Args:
        cells: Cell table (``lat``, ``lon``).
        seed: Shuffle seed.

    Returns:
        ``cells`` with ``block``, ``region`` and ``space_split`` columns.
    """
    cells = cells.copy()
    brow = np.floor(cells["lat"] / SPLIT_BLOCK_DEG).astype(int)
    bcol = np.floor(cells["lon"] / SPLIT_BLOCK_DEG).astype(int)
    cells["block"] = [f"b{r:+03d}_{c:+04d}" for r, c in zip(brow, bcol)]
    block_lat = cells.groupby("block")["lat"].transform("mean")
    block_lon = cells.groupby("block")["lon"].transform("mean")
    cells["region"] = [region_of(a, b) for a, b in zip(block_lat, block_lon)]
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


# --- candidates ---------------------------------------------------------------
def family_anchor(rows: pd.DataFrame, family: str) -> np.ndarray:
    """Anchor (window end, exclusive; days since epoch) of each label row.

    Args:
        rows: Phenology rows.
        family: Window family.

    Returns:
        Anchors.
    """
    if family == "post":
        # First day of the next water year.
        start = pd.to_datetime(rows["wy_start"].to_numpy(), unit="D")
        end = start + pd.DateOffset(years=1)
        return (end - pd.Timestamp("1970-01-01")).days.to_numpy().astype(np.int64)
    lead = int(family[1:]) if family.startswith("l") else -int(family[1:])
    clim_date = rows["wy_start"] - 1 + rows["sdd_dowy_clim"]
    return np.rint(clim_date - lead).to_numpy().astype(np.int64)


def build_candidates(root: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cell splits and every candidate (cell, water year, family).

    Args:
        root: Dataset root.
        seed: Split seed.

    Returns:
        ``(cells, candidates)``; both written under ``meta``.
    """
    meta = root / "meta"
    cells = pd.read_parquet(meta / "cells.parquet")
    season = (cells["sdd_dowy_mean"] - cells["sad_dowy_mean"]).fillna(
        cells["sdd_dowy_mean"]
    )
    stable = (cells["sdd_dowy_sd"] <= MAX_CELL_SDD_SD) & (
        season >= MIN_CELL_SEASON_DAYS
    )
    logger.info(
        "%d of %d cells with a stable seasonal snowpack", int(stable.sum()), len(cells)
    )
    cells = spatial_split(cells[stable].reset_index(drop=True), seed)
    era5_dates = np.load(meta / "era5_dates.npy")
    phen = pd.read_parquet(meta / "phenology.parquet")
    # A date anomaly is the same in days of the water year as in days.
    phen = phen.rename(columns={"cell": "cell_id", "sdd_dowy_anom": "sdd_anom",
                                "sad_dowy_anom": "sad_anom"})  # fmt: skip
    phen = phen[phen["cell_id"].isin(cells["cell_id"])]
    ok = (phen["sdd"].notna() & phen["sdd_dowy_clim"].notna()
          & (phen["sdd_gap"] <= MAX_SDD_GAP))  # fmt: skip
    phen = phen[ok].reset_index(drop=True)
    bad_days = {}
    for cell_id in phen["cell_id"].unique():
        series = np.load(meta / "era5" / f"{cell_id}.npy", mmap_mode="r")
        missing = (np.asarray(series) == NODATA).all(axis=1)
        bad_days[cell_id] = np.concatenate([[0], np.cumsum(missing)])
    parts = []
    for family in FAMILIES:
        frame = phen.copy()
        frame["family"] = family
        frame["anchor"] = family_anchor(frame, family)
        t0 = np.searchsorted(era5_dates, frame["anchor"].to_numpy() - WINDOW_DAYS)
        inside = (frame["anchor"].to_numpy() - WINDOW_DAYS >= era5_dates[0]) & (
            frame["anchor"].to_numpy() <= era5_dates[-1] + 1
        )
        frame["t0"] = t0
        frame = frame[inside].reset_index(drop=True)
        complete = np.array([
            bad_days[c][t + WINDOW_DAYS] - bad_days[c][t] == 0
            for c, t in zip(frame["cell_id"], frame["t0"])
        ], bool)  # fmt: skip
        frame = frame[complete]
        parts.append(frame)
    candidates = pd.concat(parts, ignore_index=True)
    candidates["days_to_melt"] = candidates["sdd"] - candidates["anchor"]
    candidates["melted"] = candidates["days_to_melt"] <= 0
    candidates["has_sad"] = (
        candidates["sad_anom"].notna() & candidates["season_anom"].notna()
    )
    candidates = candidates.merge(
        cells[["cell_id", "block", "region", "space_split"]], on="cell_id"
    )
    time_part = np.array([WY_SPLIT.get(int(y) % 5, "train") for y in candidates["wy"]])
    candidates["split"] = np.where(candidates["space_split"].to_numpy() == time_part,
                                   time_part, "none")  # fmt: skip
    candidates = candidates.drop(columns=["space_split"])
    candidates = candidates.sort_values(["family", "cell_id", "wy"], ignore_index=True)
    cells.to_parquet(meta / "cells_split.parquet")
    candidates.to_parquet(meta / "candidates.parquet")
    summary = candidates.groupby(["family", "split"]).agg(
        cells=("cell_id", "nunique"), windows=("wy", "size"),
        sdd_anom_sd=("sdd_anom", "std"), melted=("melted", "mean"),
        days_to_melt=("days_to_melt", "mean"),
    )  # fmt: skip
    logger.info("candidates:\n%s", summary.to_string())
    return cells, candidates


# --- subsets ------------------------------------------------------------------
def balanced_subset(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Up to ``n`` (cell, water year) keys spread over cells, then years.

    Round-robin over cells (each cell's years in random order), so every cell
    contributes before any contributes twice.

    Args:
        frame: Candidate keys (``cell_id``, ``wy``).
        n: Keys wanted.
        seed: Shuffle seed.

    Returns:
        The selected rows.
    """
    shuffled = frame.sample(frac=1.0, random_state=seed)
    shuffled = shuffled.assign(_rank=shuffled.groupby("cell_id").cumcount())
    cell_order = {c: k for k, c in enumerate(shuffled["cell_id"].drop_duplicates())}
    shuffled = shuffled.assign(_cell=shuffled["cell_id"].map(cell_order))
    picked = shuffled.sort_values(["_rank", "_cell"]).head(n)
    return picked.drop(columns=["_rank", "_cell"])


def select_eval_keys(
    candidates: pd.DataFrame, quotas: dict[str, int], seed: int
) -> pd.DataFrame:
    """The eval keys: (cell, water year) pairs with a window in every family.

    Args:
        candidates: Output of :func:`build_candidates`.
        quotas: Keys per split.
        seed: Sampling seed.

    Returns:
        ``cell_id, wy, split`` rows.
    """
    per_key = candidates[candidates["split"] != "none"].groupby(
        ["cell_id", "wy", "split"]
    )
    families = per_key["family"].nunique().rename("n_families").reset_index()
    full = families[families["n_families"] == len(FAMILIES)]
    parts = []
    for split, n in quotas.items():
        pool = full[full["split"] == split]
        if len(pool) < n:
            logger.warning("%s: %d keys < %d wanted", split, len(pool), n)
        parts.append(balanced_subset(pool, n, seed))
    return pd.concat(parts, ignore_index=True)[["cell_id", "wy", "split"]]


def select_all(rows: pd.DataFrame, per_cell: int, seed: int) -> pd.DataFrame:
    """The ``all`` group of one family: up to ``per_cell`` water years per cell.

    Neighbouring years of a cell share its climate and much of its weather
    history (the windows overlap by 83 days), so a few years from every cell
    carry most of the information at a fraction of the windows.

    Args:
        rows: The family's candidates of the train / val / test splits.
        per_cell: Water years kept per cell (random, seeded).
        seed: Sampling seed.

    Returns:
        The selected rows.
    """
    shuffled = rows.sample(frac=1.0, random_state=seed)
    return shuffled[shuffled.groupby("cell_id").cumcount() < per_cell].sort_index()


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
            group: Window group.
            row: Candidate row.
            cell: Cell row (coordinates).
            block: ``(448, 14)`` ERA5 rows of the window.
        """
        start = EPOCH + timedelta(days=int(row["anchor"]) - WINDOW_DAYS)
        end = EPOCH + timedelta(days=int(row["anchor"]))
        # The 0.1 deg pixel whose north-west corner is the cell centre.
        col = int(np.floor(cell["lon"] / 0.1 + 1e-6))
        pix_row = int(np.floor(cell["lat"] / -0.1 + 1e-6))
        targets = {t: float(row[t]) for t in TARGETS[family_kind(row["family"])]
                   if not (t in OPTIONAL_TARGETS and not row["has_sad"])}  # fmt: skip
        options: dict[str, Any] = dict(
            split=row["split"], cell_id=row["cell_id"], block=row["block"],
            region=row["region"], wy=str(int(row["wy"])), anchor_date=end.strftime("%Y-%m-%d"),
            family=row["family"], **targets,
        )  # fmt: skip
        if row["has_sad"]:
            options["has_sad"] = ""
        window = Window(
            storage=self.dataset.storage, group=group,
            name=f"{row['cell_id']}_{int(row['wy'])}_{row['family']}",
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
    # Tables first: the embedding dumps and baselines read only these and the
    # cell series, so they can start while the windows are being written.
    for group, frame in selected.items():
        frame.to_parquet(root / "meta" / f"windows_{group}.parquet")
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
            if k % 500 == 0:
                logger.info("[%d/%d cells] %s", k, len(jobs), totals)
    stats: dict[str, Any] = {}
    for group, frame in selected.items():
        targets = TARGETS[family_kind(frame["family"].iloc[0])]
        stats[group] = {
            split: {"windows": len(part), "cells": int(part["cell_id"].nunique()),
                    "melted": float(part["melted"].mean()),
                    **{t: dict(mean=float(part[t].mean()), std=float(part[t].std()))
                       for t in targets}}
            for split, part in frame.groupby("split")
        }  # fmt: skip
    # Merged with earlier runs, so groups can be written by separate runs.
    stats_path = root / "meta" / "label_stats.json"
    if stats_path.exists():
        stats = {**json.loads(stats_path.read_text()), **stats}
    stats_path.write_text(json.dumps(stats, indent=2))
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
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=1500)
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--eval-families", default=",".join(FAMILIES))
    ap.add_argument(
        "--eval-group",
        default="eval",
        help="eval key set: meta/<name>_keys.parquet, window groups <name>_<family>",
    )
    ap.add_argument("--all-families", default="", help="families of the 'all' groups")
    ap.add_argument(
        "--all-per-cell",
        type=int,
        default=6,
        help="water years per cell in 'all' groups",
    )
    args = ap.parse_args()
    root = Path(args.root)
    if args.step in ("candidates", "all"):
        cells, candidates = build_candidates(root, args.seed)
    else:
        cells = pd.read_parquet(root / "meta" / "cells_split.parquet")
        candidates = pd.read_parquet(root / "meta" / "candidates.parquet")
    if args.step in ("windows", "all"):
        keys_path = root / "meta" / f"{args.eval_group}_keys.parquet"
        if keys_path.exists():
            keys = pd.read_parquet(keys_path)
        else:
            quotas = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
            keys = select_eval_keys(candidates, quotas, args.seed)
            keys.to_parquet(keys_path)
        selected: dict[str, pd.DataFrame] = {}
        for family in filter(None, args.eval_families.split(",")):
            rows = candidates[candidates["family"] == family].merge(
                keys[["cell_id", "wy"]], on=["cell_id", "wy"]
            )
            if len(rows) != len(keys):
                raise ValueError(
                    f"{family}: {len(keys) - len(rows)} eval keys lack a window"
                )
            selected[f"{args.eval_group}_{family}"] = rows
        for family in filter(None, args.all_families.split(",")):
            rows = candidates[
                (candidates["family"] == family) & (candidates["split"] != "none")
            ]
            selected[f"all_{family}"] = select_all(rows, args.all_per_cell, args.seed)
        write_all(root, selected, cells, args.workers)


if __name__ == "__main__":
    main()
