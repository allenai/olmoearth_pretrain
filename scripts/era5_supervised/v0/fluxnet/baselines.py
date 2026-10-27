"""Reference baselines for the FLUXNET eval subset, without any encoder.

Calibrates how much of each target a simple model of the same 448-day ERA5
input already explains, so probe numbers can be read against it:

* ``mean``: the train-split mean (R2 = 0 by construction on train).
* ``ridge``: ridge regression (alpha by 5-fold CV over sites) on hand-crafted
  summaries of the window: per-band means over the last 8 / 30 / 90 / 365 days
  plus the anchor's day of year (sin, cos). A linear model on the summaries a
  linear probe would need from the embedding.
* ``gbm``: histogram gradient boosting on the same features (nonlinear).

Inputs come from the dataset's ``meta`` tables (``windows_eval.parquet`` and
the per-site ERA5 series), so this needs no GPU and no rslearn reads.

Usage::

    python scripts/era5_supervised/v0/fluxnet/baselines.py --root /weka/.../fluxnet
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NODATA = -9999.0
WINDOW_DAYS = 448
SPANS = (8, 30, 90, 365)
TARGETS = ("gpp", "gpp_dt", "le", "nee", "gpp_anom")


def window_features(series: np.ndarray, t0: int, anchor: int) -> np.ndarray:
    """Hand-crafted summary of one 448-day window.

    Args:
        series: Site's ``(T_all, 14)`` ERA5 series.
        t0: First row of the window.
        anchor: Anchor date (days since epoch; the window ends the day before).

    Returns:
        ``14 * len(SPANS) + 2`` features.
    """
    block = series[t0 : t0 + WINDOW_DAYS].astype(np.float64)
    block = np.where(block == NODATA, np.nan, block)
    parts = [np.nanmean(block[-span:], axis=0) for span in SPANS]
    doy = (pd.Timestamp("1970-01-01") + pd.Timedelta(days=int(anchor) - 1)).dayofyear
    angle = 2 * np.pi * doy / 365.25
    return np.concatenate(parts + [[np.sin(angle), np.cos(angle)]])


def build_features(root: Path, frame: pd.DataFrame) -> np.ndarray:
    """Features for every row of a window table.

    Args:
        root: Dataset root.
        frame: Rows with ``site_id, t0, anchor``.

    Returns:
        ``(len(frame), n_features)`` array.
    """
    out = np.zeros((len(frame), 14 * len(SPANS) + 2))
    for site_id, rows in frame.groupby("site_id"):
        series = np.load(root / "meta" / "era5" / f"{site_id}.npy")
        for k, row in zip(rows.index, rows.itertuples()):
            out[frame.index.get_loc(k)] = window_features(series, row.t0, row.anchor)
    return np.nan_to_num(out)


def scores(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """RMSE and R2.

    Args:
        y: Targets.
        pred: Predictions.

    Returns:
        ``{"rmse", "r2"}``.
    """
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    r2 = float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
    return {"rmse": rmse, "r2": r2}


def run(root: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Fit and score the baselines for every target.

    Args:
        root: Dataset root.

    Returns:
        ``{target: {"<model>/<split>": {"rmse", "r2"}}}``.
    """
    frame = pd.read_parquet(root / "meta" / "windows_eval.parquet").reset_index(
        drop=True
    )
    features = build_features(root, frame)
    results: dict[str, dict[str, dict[str, float]]] = {}
    for target in TARGETS:
        ok = frame[target].notna().to_numpy()
        train = ok & (frame["split"] == "train").to_numpy()
        scaler = StandardScaler().fit(features[train])
        x = scaler.transform(features)
        y = frame[target].to_numpy()
        groups = frame["site_id"].to_numpy()
        n_sites = len(np.unique(groups[train]))
        # Folds by site; with a single site, RidgeCV's own leave-one-out.
        splits = (
            list(GroupKFold(min(5, n_sites)).split(x[train], y[train], groups[train]))
            if n_sites >= 2
            else None
        )
        ridge = RidgeCV(alphas=np.logspace(-2, 4, 13), cv=splits)
        ridge.fit(x[train], y[train])
        gbm = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, random_state=0
        )
        gbm.fit(x[train], y[train])
        results[target] = {}
        for split in ("train", "val", "test"):
            sel = ok & (frame["split"] == split).to_numpy()
            if not sel.any():
                continue
            mean_pred = np.full(sel.sum(), y[train].mean())
            results[target][f"mean/{split}"] = scores(y[sel], mean_pred)
            results[target][f"ridge/{split}"] = scores(y[sel], ridge.predict(x[sel]))
            results[target][f"gbm/{split}"] = scores(y[sel], gbm.predict(x[sel]))
            results[target][f"n/{split}"] = {"n": int(sel.sum())}
        logger.info("%s (ridge alpha %.3g): %s", target, ridge.alpha_,
                    {k: v for k, v in results[target].items() if "/val" in k})  # fmt: skip
    return results


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--root", default="/weka/dfive-default/rslearn-eai/datasets/fluxnet"
    )
    args = ap.parse_args()
    root = Path(args.root)
    results = run(root)
    out = root / "meta" / "baselines.json"
    out.write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
