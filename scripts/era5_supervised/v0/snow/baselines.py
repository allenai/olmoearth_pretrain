"""Reference baselines for the snow eval windows, without any encoder.

Calibrates how much of each target simple models already explain, so probe
numbers can be read against them. Feature sets (per window family table):

* ``window``: hand-made summaries of the 448-day input window, as for FLUXNET
  and USDM: per-band means over the last 8 / 30 / 90 / 365 days and over the
  same 8 / 30 / 83 days one year earlier, plus the anchor's day of year (sin,
  cos). What a linear probe would need from the embedding.
* ``snow``: snow-physics summaries of the water year so far: cold-season
  precipitation (``tp`` on days with ``t2m`` < 0 C, i.e. snowfall), positive
  degree-days, mean ``t2m`` over the last 30 / 60 days, ERA5-Land's modelled
  albedo (``ssr`` / ``ssrd``) over the last 8 / 30 days, the day the albedo
  last fell below 0.4 (its own melt-out, days before the anchor) and the
  runoff of the last 30 days; plus the day of year.
* ``snow_clim``: the ``snow`` features as anomalies against the cell's own
  other water years (leave-one-year-out, same anchor offset). The encoder
  never sees the cell's climatology (no location input, one season of
  history), so this is an oracle for what a model that knew the cell's normal
  season could extract from the same weather.
* ``window+snow``.

Models: ridge (alpha by 5-fold CV over train blocks) and histogram gradient
boosting. ``zero``: predicting the climatology (anomaly 0).

Usage::

    python scripts/era5_supervised/v0/snow/baselines.py --root /weka/.../snow --table windows_eval_l30
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from olmoearth_pretrain.dataset_creation.snow.common import ERA5L_BANDS
from olmoearth_pretrain.dataset_creation.snow.write_windows import TARGETS, family_kind

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NODATA = -9999.0
WINDOW_DAYS = 448
SPANS = (8, 30, 90, 365)
LAST_YEAR_SPANS = (8, 30, 83)
BAND = {b: i for i, b in enumerate(ERA5L_BANDS)}
ALPHAS = np.logspace(-2, 4, 13)
MELT_ALBEDO = 0.4
FREEZING_K = 273.15


def doy_features(anchor: int) -> np.ndarray:
    """Day of year of the window's last day as (sin, cos).

    Args:
        anchor: Anchor date (days since epoch).

    Returns:
        Two features.
    """
    doy = (pd.Timestamp("1970-01-01") + pd.Timedelta(days=int(anchor) - 1)).day_of_year
    angle = 2 * np.pi * doy / 365.25
    return np.array([np.sin(angle), np.cos(angle)])


def window_features(block: np.ndarray, anchor: int) -> np.ndarray:
    """Hand-made summary of one 448-day window (as ``fluxnet/baselines.py``).

    Args:
        block: ``(448, 14)`` window, NaN for no data.
        anchor: Anchor date (days since epoch).

    Returns:
        ``14 * 7 + 2`` features.
    """
    parts = [np.nanmean(block[-span:], axis=0) for span in SPANS]
    parts += [np.nanmean(block[-365 - span : -365], axis=0) for span in LAST_YEAR_SPANS]
    return np.concatenate(parts + [doy_features(anchor)])


def snow_features(block: np.ndarray, season_days: int) -> np.ndarray:
    """Snow-physics summaries of the last ``season_days`` days of a window.

    Args:
        block: ``(448, 14)`` window, NaN for no data.
        season_days: Days since the water year's start (capped at 448).

    Returns:
        10 features.
    """
    season = block[-max(1, min(season_days, WINDOW_DAYS)) :]
    t2m, tp = season[:, BAND["t2m"]], season[:, BAND["tp"]]
    with np.errstate(invalid="ignore", divide="ignore"):
        albedo = 1.0 - block[:, BAND["ssr"]] / block[:, BAND["ssrd"]]
    snowy = np.flatnonzero(albedo >= MELT_ALBEDO)
    since_snow = WINDOW_DAYS - 1 - snowy[-1] if len(snowy) else WINDOW_DAYS
    return np.array([
        np.nansum(np.where(t2m < FREEZING_K, tp, 0.0)),
        np.nansum(np.clip(t2m - FREEZING_K, 0, None)),
        np.nanmean(block[-30:, BAND["t2m"]]), np.nanmean(block[-60:, BAND["t2m"]]),
        np.nanmean(albedo[-8:]), np.nanmean(albedo[-30:]), float(since_snow),
        np.nanmean(block[-30:, BAND["ro"]]), np.nansum(tp),
        np.nanmean(block[-30:, BAND["swvl1"]]),
    ])  # fmt: skip


def cell_features(
    job: tuple[str, list[tuple[int, int, int, int]], str],
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Worker: window, snow and leave-one-year-out snow anomaly features of one cell.

    Args:
        job: ``(cell_id, [(row index, t0, anchor, days since water year start)],
            meta dir)``.

    Returns:
        ``(row indices, window features, snow features, snow anomalies)``.
    """
    cell_id, rows, meta_dir = job
    series = np.load(Path(meta_dir) / "era5" / f"{cell_id}.npy")
    series = np.where(series == NODATA, np.nan, series).astype(np.float64)
    win, snow = [], []
    for _, t0, anchor, season_days in rows:
        block = series[t0 : t0 + WINDOW_DAYS]
        win.append(window_features(block, anchor))
        snow.append(snow_features(block, season_days))
    snow_arr = np.array(snow)
    # Leave-one-year-out: the cell's windows of one family sit at the same
    # offset from its climatological melt (or water year end), one per year.
    total, count = np.nansum(snow_arr, axis=0), np.isfinite(snow_arr).sum(axis=0)
    others = np.where(
        count > 1, (total - np.nan_to_num(snow_arr)) / np.maximum(count - 1, 1), np.nan
    )
    return [r[0] for r in rows], np.array(win), snow_arr, snow_arr - others


def build_features(
    root: Path, frame: pd.DataFrame, workers: int
) -> dict[str, np.ndarray]:
    """Feature sets for every row of a window table.

    Args:
        root: Dataset root.
        frame: Rows with ``cell_id, t0, anchor, wy_start``.
        workers: Processes.

    Returns:
        ``{set name: [N, F]}``.
    """
    season = (frame["anchor"] - frame["wy_start"]).to_numpy()
    jobs = [
        (cell_id, list(zip(rows.index, rows["t0"], rows["anchor"], season[rows.index])),
         str(root / "meta"))
        for cell_id, rows in frame.groupby("cell_id")
    ]  # fmt: skip
    win = np.zeros((len(frame), 14 * (len(SPANS) + len(LAST_YEAR_SPANS)) + 2))
    snow = np.zeros((len(frame), 10))
    anom = np.zeros((len(frame), 10))
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for idx, w, s, a in pool.imap_unordered(cell_features, jobs):
            win[idx], snow[idx], anom[idx] = w, s, a
    doy = np.stack([doy_features(a) for a in frame["anchor"]])
    win, snow, anom = np.nan_to_num(win), np.nan_to_num(snow), np.nan_to_num(anom)
    snow = np.hstack([snow, doy])
    return {
        "window": win, "snow": snow, "snow_clim": np.hstack([anom, doy]),
        "window+snow": np.hstack([win, snow]),
    }  # fmt: skip


def run(root: Path, table: str, workers: int) -> dict:
    """Fit and score every baseline for every target of the table's family.

    The leave-one-year-out anomalies need the cell's other years, so they are
    computed on the family's full candidate set and joined to the table.

    Args:
        root: Dataset root.
        table: Window table under ``meta``.
        workers: Feature processes.

    Returns:
        ``{target: {"<features>/<model>": {"val", "test"}}}``.
    """
    frame = pd.read_parquet(root / "meta" / f"{table}.parquet").reset_index(drop=True)
    family = frame["family"].iloc[0]
    pool_rows = pd.read_parquet(root / "meta" / "candidates.parquet")
    pool_rows = pool_rows[(pool_rows["family"] == family)
                          & pool_rows["cell_id"].isin(frame["cell_id"])].reset_index(drop=True)  # fmt: skip
    features = build_features(root, pool_rows, workers)
    key = pool_rows[["cell_id", "wy"]].reset_index().merge(frame[["cell_id", "wy"]].reset_index(),
                                                          on=["cell_id", "wy"], suffixes=("_pool", ""))  # fmt: skip
    order = key.sort_values("index")["index_pool"].to_numpy()
    features = {k: v[order] for k, v in features.items()}
    split = frame["split"].to_numpy()
    train = split == "train"
    results: dict = {}
    for target in TARGETS[family_kind(family)]:
        y = frame[target].to_numpy(np.float64)
        defined = np.isfinite(y)
        results[target] = {}
        for name, x in features.items():
            xs = StandardScaler().fit(x[train & defined]).transform(x)
            fit = train & defined
            fit_idx = np.flatnonzero(fit)
            target_folds = list(
                GroupKFold(5).split(fit_idx, groups=frame["block"].to_numpy()[fit_idx])
            )
            linear = RidgeCV(alphas=ALPHAS, cv=target_folds).fit(xs[fit], y[fit])
            gbm = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, random_state=0
            )
            gbm.fit(xs[fit], y[fit])
            for model_name, model in (("linear", linear), ("gbm", gbm)):
                results[target][f"{name}/{model_name}"] = {
                    s: float(r2_score(y[(split == s) & defined], model.predict(xs[(split == s) & defined])))
                    for s in ("val", "test")
                }  # fmt: skip
        if target.endswith("_anom"):
            results[target]["zero"] = {
                s: float(r2_score(y[(split == s) & defined], np.zeros(((split == s) & defined).sum())))
                for s in ("val", "test")
            }  # fmt: skip
        logger.info("%s %s: %s", family, target,
                    {k: round(v["val"], 3) for k, v in results[target].items()})  # fmt: skip
    return results


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default="/weka/dfive-default/rslearn-eai/datasets/snow")
    ap.add_argument("--tables", default="windows_eval_l0,windows_eval_l30,windows_eval_l60,"
                    "windows_eval_l90,windows_eval_post")  # fmt: skip
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    root = Path(args.root)
    for table in args.tables.split(","):
        results = run(root, table, args.workers)
        out = root / "meta" / f"baselines_{table}.json"
        out.write_text(json.dumps(results, indent=2))
        logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
