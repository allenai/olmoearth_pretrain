"""Offline target design for the FLUXNET eval on dumped encoder embeddings.

Scores candidate flux targets on the 7000 eval windows with a closed-form ridge
probe (standardized embeddings, alpha by 5-fold CV over train sites), for every
embedding set written by ``dump_embeddings.py`` and every pooling, so a target
can be judged by what an eval needs before it is written into the dataset:

* the random-projection floor (``init_*`` sets) vs the trained encoders,
* the gap relative to the seed spread of trained encoders,
* the spread between recipes (width arms, the 1306 trajectory),
* headroom against hand-made weather summaries (``baselines.py`` features).

Candidate targets are computed per window from ``meta/daily`` with the dataset's
validity rule (daily QC >= 0.5, at least 75 % valid days in the period):

* ``gpp8`` / ``gpp30``: mean GPP over the window's last 8 / 30 days;
* ``rgpp8`` / ``rgpp30``: the same divided by the site's 95th percentile of
  ``gpp8``, capped at 1.5 (a phenological state, vegetation level removed);
* ``gpp_fwd30``: mean GPP over the 30 days after the window (a forecast);
* ``rgpp_fwd30``: the same, site-normalized;
* ``le8`` / ``le30``, ``ef8`` / ``ef30`` (evaporative fraction LE / (LE + H));
* ``nee30``; ``anom8`` / ``anom30`` (leave-one-year-out day-of-year anomalies).

Writes ``<root>/meta/target_lab.json`` and prints a summary per target.

Usage::

    python scripts/era5_supervised/v0/fluxnet/target_lab.py --root /weka/.../fluxnet
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MIN_QC = 0.5
MIN_FRAC = 0.75
CLIM_HALF_WIDTH = 7
ALPHAS = np.logspace(-1, 5, 13)


def _trailing(values: pd.Series, days: int) -> pd.Series:
    """Trailing mean over ``days`` days with at least ``MIN_FRAC`` valid.

    Args:
        values: Daily series (NaN = invalid) on a complete daily index.
        days: Window length.

    Returns:
        Series whose value at day ``e`` covers ``[e - days + 1, e]``.
    """
    return values.rolling(days, min_periods=int(np.ceil(MIN_FRAC * days))).mean()


def _loo_anomaly(target: pd.Series) -> pd.Series:
    """Target minus its leave-one-year-out day-of-year climatology (>= 2 other years).

    Args:
        target: Daily-indexed target series.

    Returns:
        The anomaly series.
    """
    frame = pd.DataFrame({"v": target, "year": target.index.year,
                          "doy": np.minimum(target.index.dayofyear, 365) - 1})  # fmt: skip
    years = np.sort(frame["year"].unique())
    rows = frame["year"].map({y: k for k, y in enumerate(years)}).to_numpy()
    values = np.full((len(years), 365), np.nan)
    values[rows, frame["doy"].to_numpy()] = frame["v"].to_numpy()
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    sums, counts = np.zeros_like(filled), np.zeros_like(filled)
    for shift in range(-CLIM_HALF_WIDTH, CLIM_HALF_WIDTH + 1):
        sums += np.roll(filled, shift, axis=1)
        counts += np.roll(finite, shift, axis=1)
    other_n = (counts > 0).sum(axis=0, keepdims=True) - (counts > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        clim = (sums.sum(0, keepdims=True) - sums) / (
            counts.sum(0, keepdims=True) - counts
        )
    clim = np.where(other_n >= 2, clim, np.nan)
    return target - clim[rows, frame["doy"].to_numpy()]


def site_target_table(daily: pd.DataFrame) -> pd.DataFrame:
    """All candidate targets of one site, indexed by the window's last day.

    Args:
        daily: The site's daily table (``targets.DAILY_COLUMNS``).

    Returns:
        Frame of candidate targets on a complete daily index.
    """
    daily = daily.asfreq("D")
    gpp_day = daily[["GPP_NT_VUT_REF", "GPP_DT_VUT_REF"]].mean(axis=1, skipna=True)
    ok = daily["NEE_VUT_REF_QC"] >= MIN_QC
    gpp = gpp_day.where(ok)
    le_ok = daily["LE_F_MDS_QC"] >= MIN_QC
    le = daily["LE_F_MDS"].where(le_ok)
    h = daily["H_F_MDS"].where(daily["H_F_MDS_QC"] >= MIN_QC)
    out = pd.DataFrame(index=daily.index)
    out["gpp8"], out["gpp30"] = _trailing(gpp, 8), _trailing(gpp, 30)
    scale = out["gpp8"].quantile(0.95)
    scale = scale if scale and scale > 0.5 else np.nan
    # Capped at 1.5 as in the dataset (dryland towers with a tiny percentile).
    out["rgpp8"] = (out["gpp8"] / scale).clip(upper=1.5)
    out["rgpp30"] = (out["gpp30"] / scale).clip(upper=1.5)
    # Mean over the 30 days after the window's last day.
    fwd = _trailing(gpp, 30).shift(-30)
    out["gpp_fwd30"], out["rgpp_fwd30"] = fwd, (fwd / scale).clip(upper=1.5)
    out["le8"], out["le30"] = _trailing(le, 8), _trailing(le, 30)
    for days in (8, 30):
        le_m, h_m = _trailing(le, days), _trailing(h, days)
        denom = le_m + h_m
        out[f"ef{days}"] = (le_m / denom).where(denom > 20)
    out["nee30"] = _trailing(daily["NEE_VUT_REF"].where(ok), 30)
    out["anom8"], out["anom30"] = _loo_anomaly(out["gpp8"]), _loo_anomaly(out["gpp30"])
    return out


def window_targets(root: Path, windows: pd.DataFrame) -> pd.DataFrame:
    """Candidate targets for every eval window, aligned with ``windows``.

    Args:
        root: Dataset root.
        windows: ``site_id, anchor`` rows (embedding row order).

    Returns:
        Frame of targets, one row per window.
    """
    sites = pd.read_parquet(root / "meta" / "sites.parquet").set_index("site_id")
    parts = []
    for site_id, rows in windows.groupby("site_id"):
        table = site_target_table(
            pd.read_parquet(root / "meta" / "daily" / sites.loc[site_id, "daily_file"])
        )
        last_day = pd.to_datetime(rows["anchor"] - 1, unit="D")
        part = table.reindex(last_day.to_numpy())
        part.index = rows.index
        parts.append(part)
    return pd.concat(parts).loc[windows.index]


def ridge_scores(
    x: np.ndarray, y: np.ndarray, split: np.ndarray, groups: np.ndarray
) -> dict[str, float]:
    """Val / test R2 of a ridge probe fit on the train split.

    Args:
        x: Features ``[N, D]``.
        y: Target ``[N]`` (NaN = undefined).
        split: Split per row.
        groups: Site id per row (CV groups).

    Returns:
        ``{"val", "test", "alpha"}``.
    """
    ok = np.isfinite(y)
    train = ok & (split == "train")
    scaler = StandardScaler().fit(x[train])
    xs = scaler.transform(x)
    folds = list(GroupKFold(5).split(xs[train], y[train], groups[train]))
    model = RidgeCV(alphas=ALPHAS, cv=folds).fit(xs[train], y[train])
    out = {"alpha": float(model.alpha_)}
    for name in ("val", "test"):
        sel = ok & (split == name)
        resid = y[sel] - model.predict(xs[sel])
        out[name] = float(1 - (resid**2).sum() / ((y[sel] - y[sel].mean()) ** 2).sum())
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--root", default="/weka/dfive-default/rslearn-eai/datasets/fluxnet"
    )
    ap.add_argument("--poolings", default="mean,last4")
    args = ap.parse_args()
    root = Path(args.root)
    emb_dir = root / "embeddings"
    windows = pd.read_parquet(emb_dir / "windows.parquet")
    targets = window_targets(root, windows)
    split, groups = windows["split"].to_numpy(), windows["site_id"].to_numpy()
    logger.info("targets defined per window:\n%s", targets.notna().sum().to_string())
    results: dict[str, dict] = {}
    for path in sorted(emb_dir.glob("*.npz")):
        emb = np.load(path)
        for pooling in args.poolings.split(","):
            x = emb[pooling].astype(np.float32)
            for target in targets.columns:
                scores = ridge_scores(x, targets[target].to_numpy(), split, groups)
                results.setdefault(target, {}).setdefault(pooling, {})[path.stem] = (
                    scores
                )
        logger.info("scored %s", path.stem)
    stats = {
        t: dict(n=int(targets[t].notna().sum()), sd=float(targets[t].std()))
        for t in targets
    }
    (root / "meta" / "target_lab.json").write_text(
        json.dumps({"stats": stats, "results": results}, indent=1)
    )
    for target, by_pool in results.items():
        for pooling, sets in by_pool.items():
            floor = [v["val"] for k, v in sets.items() if k.startswith("init_")]
            trained = [
                v["val"] for k, v in sets.items() if k.startswith("era5enc_1601_doyc")
            ]
            logger.info("%-10s %-6s floor %.3f+-%.3f | 1601 %.3f+-%.3f | %s", target, pooling,
                        np.mean(floor) if floor else np.nan, np.std(floor) if floor else np.nan,
                        np.mean(trained) if trained else np.nan, np.std(trained) if trained else np.nan,
                        {k.replace("era5enc_", "")[:18]: round(v["val"], 3) for k, v in sets.items()
                         if not k.startswith(("init_", "era5enc_1601"))})  # fmt: skip


if __name__ == "__main__":
    main()
