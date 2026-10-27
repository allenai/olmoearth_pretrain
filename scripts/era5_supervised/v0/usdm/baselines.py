"""Reference baselines for the USDM eval subset, without any encoder.

Calibrates how much of each target simple models already explain, so probe
numbers can be read against them. Feature sets:

* ``window``: hand-made summaries of the 448-day input window, as for FLUXNET:
  per-band means over the last 8 / 30 / 90 / 365 days and over the same 8 / 30
  / 83 days one year earlier, plus the anchor's day of year (sin, cos). What a
  linear probe would need from the embedding.
* ``clim``: standardized anomalies of the window's last 7-365 days against the
  cell's own 1991-2020 climatology at the same day of year (SPI-like
  precipitation, soil moisture, temperature, potential-minus-actual
  evaporation and runoff), plus the day of year. The encoder never sees the
  cell's climatology (no location input, one year of history), so this is an
  oracle for the weather part of the USDM's own inputs.
* ``window+clim``.
* ``state`` (forecast targets only): the true current and lagged USDM
  severities (``sev``, ``sev_m4``, ``sev_m8``), information no input carries;
  ``state+clim`` adds the anomalies.

Models: ridge (alpha by 5-fold CV over train blocks) and histogram gradient
boosting (regression), multinomial logistic regression / boosting (class
targets, macro F1). ``zero``: predicting no change (R2 of a persistence
forecast) for the change targets.

Usage::

    python scripts/era5_supervised/v0/usdm/baselines.py --root /weka/.../usdm
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from olmoearth_pretrain.dataset_creation.usdm.common import ERA5L_BANDS
from olmoearth_pretrain.dataset_creation.usdm.write_windows import NUMERIC_TARGETS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NODATA = -9999.0
WINDOW_DAYS = 448
SPANS = (8, 30, 90, 365)
LAST_YEAR_SPANS = (8, 30, 83)
CLASS_TARGETS = ("tpast8", "tfut8", "drought")
BAND = {b: i for i, b in enumerate(ERA5L_BANDS)}
# (name, band(s), aggregation lengths in days, sum or mean). ``e`` and ``pev``
# are negative upward fluxes in the ECMWF convention, so e - pev = PET - ET is
# the evaporative deficit (positive = demand exceeds supply).
ANOMALIES = (
    ("tp", ("tp",), (30, 60, 90, 180, 270, 365), "sum"),
    ("swvl1", ("swvl1",), (7, 30, 90), "mean"),
    ("swvl2", ("swvl2",), (7, 30, 90, 180), "mean"),
    ("t2m", ("t2m",), (30, 90), "mean"),
    ("deficit", ("e", "pev"), (30, 90, 180), "sum"),
    ("ro", ("ro",), (90, 365), "sum"),
)
CLIM_YEARS = (1991, 2020)
CLIM_HALF_WIDTH = 7
ALPHAS = np.logspace(-2, 4, 13)


def window_features(series: np.ndarray, t0: int, anchor: int) -> np.ndarray:
    """Hand-made summary of one 448-day window (as ``fluxnet/baselines.py``).

    Args:
        series: Cell's ``(T_all, 14)`` ERA5 series.
        t0: First row of the window.
        anchor: Anchor date (days since epoch).

    Returns:
        ``14 * 7 + 2`` features.
    """
    block = np.asarray(series[t0 : t0 + WINDOW_DAYS], dtype=np.float64)
    block = np.where(block == NODATA, np.nan, block)
    parts = [np.nanmean(block[-span:], axis=0) for span in SPANS]
    parts += [np.nanmean(block[-365 - span : -365], axis=0) for span in LAST_YEAR_SPANS]
    return np.concatenate(parts + [doy_features(anchor)])


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


def anomaly_table(
    series: np.ndarray, dates: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Standardized anomalies of every aggregate ending on every day.

    Args:
        series: Cell's ``(T_all, 14)`` ERA5 series.
        dates: Its dates (days since epoch).

    Returns:
        ``([T_all, n_features], names)``; row ``t`` covers days ending at ``t``.
    """
    values = np.where(series == NODATA, np.nan, series).astype(np.float64)
    dt = pd.to_datetime(dates, unit="D")
    doy = np.minimum(dt.dayofyear.to_numpy(), 365) - 1
    in_clim = (dt.year >= CLIM_YEARS[0]) & (dt.year <= CLIM_YEARS[1])
    # Climatology sample of each day of year: +-7 days, 1991-2020.
    near = [
        np.flatnonzero(
            (np.abs(((doy - d) + 182) % 365 - 182) <= CLIM_HALF_WIDTH) & in_clim
        )
        for d in range(365)
    ]
    cols, names = [], []
    for name, bands, lengths, how in ANOMALIES:
        daily = values[:, BAND[bands[0]]]
        if len(bands) == 2:
            daily = values[:, BAND[bands[0]]] - values[:, BAND[bands[1]]]
        csum = np.concatenate([[0.0], np.nancumsum(daily)])
        for length in lengths:
            agg = np.full(len(daily), np.nan)
            agg[length - 1 :] = csum[length:] - csum[:-length]
            if how == "mean":
                agg /= length
            mean, sd = np.full(365, np.nan), np.full(365, np.nan)
            for d in range(365):
                sample = agg[near[d]]
                sample = sample[np.isfinite(sample)]
                if len(sample) > 10:
                    mean[d], sd[d] = sample.mean(), sample.std()
            with np.errstate(invalid="ignore", divide="ignore"):
                cols.append((agg - mean[doy]) / np.where(sd > 0, sd, np.nan)[doy])
            names.append(f"{name}_{length}d")
    return np.stack(cols, axis=1), names


def cell_features(
    job: tuple[str, list[tuple[int, int, int]], str],
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Worker: window and climatology features of one cell's windows.

    Args:
        job: ``(cell_id, [(row index, t0, anchor)], meta dir)``.

    Returns:
        ``(row indices, window features, clim features)``.
    """
    cell_id, rows, meta_dir = job
    meta = Path(meta_dir)
    series = np.load(meta / "era5" / f"{cell_id}.npy")
    dates = np.load(meta / "era5_dates.npy")
    anomalies, _ = anomaly_table(series, dates)
    win, clim = [], []
    for _, t0, anchor in rows:
        win.append(window_features(series, t0, anchor))
        # The window's last day is row t0 + 447.
        clim.append(
            np.concatenate([anomalies[t0 + WINDOW_DAYS - 1], doy_features(anchor)])
        )
    return [r[0] for r in rows], np.array(win), np.array(clim)


def build_features(
    root: Path, frame: pd.DataFrame, workers: int
) -> dict[str, np.ndarray]:
    """Feature sets for every row of a window table.

    Args:
        root: Dataset root.
        frame: Rows with ``cell_id, t0, anchor`` and the targets.
        workers: Processes.

    Returns:
        ``{set name: [N, F]}``.
    """
    jobs = [
        (cell_id, list(zip(rows.index, rows["t0"], rows["anchor"])), str(root / "meta"))
        for cell_id, rows in frame.groupby("cell_id")
    ]
    win = np.zeros((len(frame), 14 * (len(SPANS) + len(LAST_YEAR_SPANS)) + 2))
    n_clim = sum(len(a[2]) for a in ANOMALIES) + 2
    clim = np.zeros((len(frame), n_clim))
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for idx, w, c in pool.imap_unordered(cell_features, jobs):
            win[idx], clim[idx] = w, c
    win, clim = np.nan_to_num(win), np.nan_to_num(clim)
    state = frame[["sev", "sev_m4", "sev_m8"]].to_numpy()
    return {
        "window": win, "clim": clim, "window+clim": np.hstack([win, clim]),
        "state": state, "state+clim": np.hstack([state, clim]),
    }  # fmt: skip


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination.

    Args:
        y: Targets.
        pred: Predictions.

    Returns:
        R2.
    """
    return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def run(root: Path, table: str, workers: int) -> dict:
    """Fit and score every baseline for every target.

    Args:
        root: Dataset root.
        table: Window table under ``meta``.
        workers: Feature processes.

    Returns:
        ``{target: {"<features>/<model>": {"val", "test"}}}``.
    """
    frame = pd.read_parquet(root / "meta" / f"{table}.parquet").reset_index(drop=True)
    features = build_features(root, frame, workers)
    split = frame["split"].to_numpy()
    train = split == "train"
    train_idx = np.flatnonzero(train)
    folds = list(
        GroupKFold(5).split(train_idx, groups=frame["block"].to_numpy()[train_idx])
    )
    results: dict = {}
    for target in list(NUMERIC_TARGETS) + list(CLASS_TARGETS):
        results[target] = {}
        is_class = target in CLASS_TARGETS
        y = frame[target].to_numpy()
        y = y.astype(str) if is_class else y.astype(np.float64)
        for name, x in features.items():
            if name.startswith("state") and not (
                target.startswith("dfut")
                or target.startswith("sev_p")
                or target == "tfut8"
            ):
                continue
            xs = StandardScaler().fit(x[train]).transform(x)
            if is_class:
                linear = LogisticRegression(C=0.1, max_iter=2000).fit(
                    xs[train], y[train]
                )
                gbm = HistGradientBoostingClassifier(
                    max_iter=300, learning_rate=0.05, random_state=0
                )
            else:
                linear = RidgeCV(alphas=ALPHAS, cv=folds).fit(xs[train], y[train])
                gbm = HistGradientBoostingRegressor(
                    max_iter=300, learning_rate=0.05, random_state=0
                )
            gbm.fit(xs[train], y[train])
            for model_name, model in (("linear", linear), ("gbm", gbm)):
                scores = {}
                for s in ("val", "test"):
                    sel = split == s
                    pred = model.predict(xs[sel])
                    scores[s] = (f1_score(y[sel], pred, average="macro") if is_class
                                 else r2(y[sel], pred))  # fmt: skip
                results[target][f"{name}/{model_name}"] = scores
        if target.startswith(("dpast", "dfut")):
            results[target]["zero"] = {s: r2(y[split == s], np.zeros((split == s).sum()))
                                       for s in ("val", "test")}  # fmt: skip
        logger.info(
            "%s: %s",
            target,
            {k: round(v["val"], 3) for k, v in results[target].items()},
        )
    return results


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default="/weka/dfive-default/rslearn-eai/datasets/usdm")
    ap.add_argument("--table", default="windows_eval")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    root = Path(args.root)
    results = run(root, args.table, args.workers)
    out = root / "meta" / f"baselines_{args.table}.json"
    out.write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
