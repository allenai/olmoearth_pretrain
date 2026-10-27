"""Offline target design for the USDM eval on dumped encoder embeddings.

Scores every candidate target of the eval windows (the ``write_windows``
targets) with closed-form probes on each embedding set written by
``dump_embeddings.py`` and each pooling, so a target can be judged by what an
eval needs before it is promoted to the in-loop basket:

* regression targets: ridge on standardized embeddings, alpha by 5-fold CV over
  train blocks; val / test R2;
* class targets (``tpast8``, ``tfut8``, ``drought``): multinomial logistic
  regression, C by the same CV (macro F1); val / test macro F1.

The summary per (target, pooling) gives the random-projection floor (``init_*``
sets), the step-50k arms (seed mean +- SD), the between-arm F (between-arm over
seed variance, arms with >= 2 seeds) and the 1306 trajectory (step 0..50k).

Writes ``<root>/<emb dir>/target_lab.json``.

Usage::

    python scripts/era5_supervised/v0/usdm/target_lab.py --root /weka/.../usdm
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

ALPHAS = np.logspace(-1, 5, 13)
CS = np.logspace(-4, 1, 6)
CLASS_TARGETS = ("tpast8", "tfut8", "drought")
SEED_RE = re.compile(r"_seed\d+")


def arm_of(name: str) -> str:
    """Arm of an embedding set name (``<run>[_seedN]@<step>`` or ``init_<x>[_seedN]``).

    Args:
        name: Embedding set name.

    Returns:
        The name without seed suffix (step kept).
    """
    return SEED_RE.sub("", name)


def regression_scores(
    x: np.ndarray, y: np.ndarray, split: np.ndarray, folds: list
) -> dict[str, float]:
    """Val / test R2 of a ridge probe fit on the train split.

    Args:
        x: Standardized features ``[N, D]``.
        y: Target ``[N]``.
        split: Split per row.
        folds: CV folds over the train rows.

    Returns:
        ``{"val", "test", "alpha"}``.
    """
    train = split == "train"
    model = RidgeCV(alphas=ALPHAS, cv=folds).fit(x[train], y[train])
    out = {"alpha": float(model.alpha_)}
    for name in ("val", "test"):
        sel = split == name
        resid = y[sel] - model.predict(x[sel])
        out[name] = float(1 - (resid**2).sum() / ((y[sel] - y[sel].mean()) ** 2).sum())
    return out


def class_scores(
    x: np.ndarray, y: np.ndarray, split: np.ndarray, folds: list
) -> dict[str, float]:
    """Val / test macro F1 of a logistic probe fit on the train split.

    Args:
        x: Standardized features ``[N, D]``.
        y: Class labels ``[N]``.
        split: Split per row.
        folds: CV folds over the train rows.

    Returns:
        ``{"val", "test", "C"}``.
    """
    train = np.flatnonzero(split == "train")
    best_c, best = CS[0], -1.0
    for c in CS:
        f1s = []
        for fit, held in folds:
            model = LogisticRegression(C=c, max_iter=500).fit(
                x[train[fit]], y[train[fit]]
            )
            f1s.append(
                f1_score(y[train[held]], model.predict(x[train[held]]), average="macro")
            )
        if np.mean(f1s) > best:
            best_c, best = c, float(np.mean(f1s))
    model = LogisticRegression(C=best_c, max_iter=1000).fit(x[train], y[train])
    out = {"C": float(best_c)}
    for name in ("val", "test"):
        sel = split == name
        out[name] = float(f1_score(y[sel], model.predict(x[sel]), average="macro"))
    return out


def score_set(
    x: np.ndarray, windows: pd.DataFrame, targets: list[str], folds: list
) -> dict[str, dict[str, float]]:
    """Every target's scores on one embedding matrix.

    Args:
        x: Raw features ``[N, D]``.
        windows: Window table (targets, ``split``).
        targets: Target columns.
        folds: CV folds over train rows.

    Returns:
        ``{target: scores}``.
    """
    split = windows["split"].to_numpy()
    xs = StandardScaler().fit(x[split == "train"]).transform(x)
    out = {}
    for target in targets:
        y = windows[target].to_numpy()
        if target in CLASS_TARGETS:
            out[target] = class_scores(xs, y.astype(str), split, folds)
        else:
            out[target] = regression_scores(xs, y.astype(np.float64), split, folds)
    return out


def score_job(job: tuple) -> tuple[str, str, dict]:
    """Worker: score the listed targets on one (embedding set, pooling).

    Args:
        job: ``(npz path, pooling, targets, windows, folds, threads)``.

    Returns:
        ``(set name, pooling, {target: scores})``.
    """
    path, pooling, todo, windows, folds, threads = job
    from threadpoolctl import threadpool_limits

    with threadpool_limits(threads):
        x = np.load(path)[pooling].astype(np.float32)
        return Path(path).stem, pooling, score_set(x, windows, todo, folds)


def between_arm_f(values: dict[str, float]) -> float:
    """Between-arm over within-arm (seed) variance, arms with >= 2 seeds.

    Args:
        values: ``{set name: value}``.

    Returns:
        F (NaN with fewer than two multi-seed arms).
    """
    groups: dict[str, list[float]] = {}
    for name, v in values.items():
        groups.setdefault(arm_of(name), []).append(v)
    arms = [np.array(v) for v in groups.values() if len(v) > 1]
    if len(arms) < 2:
        return float("nan")
    n = np.array([len(a) for a in arms])
    means = np.array([a.mean() for a in arms])
    grand = np.average(means, weights=n)
    between = np.sum(n * (means - grand) ** 2) / (len(arms) - 1)
    within = np.sum([((a - a.mean()) ** 2).sum() for a in arms]) / (n.sum() - len(arms))
    return float(between / within)


def summarize(results: dict, metric: str = "val") -> pd.DataFrame:
    """One row per (target, pooling): floor, 1601, best arm, F, 1306 trajectory.

    Args:
        results: ``{target: {pooling: {set: scores}}}``.
        metric: ``val`` or ``test``.

    Returns:
        Summary table.
    """
    rows = []
    for target, by_pool in results.items():
        for pooling, sets in by_pool.items():
            vals = {k: v[metric] for k, v in sets.items()}
            init = [v for k, v in vals.items() if k.startswith("init_")]
            trained = {k: v for k, v in vals.items() if k.endswith("@50000")}
            ref = [
                v for k, v in trained.items() if arm_of(k) == "era5enc_1601_doyc@50000"
            ]
            traj = {int(k.split("@")[1]): v for k, v in vals.items()
                    if k.startswith("era5enc_1306_halo75_nogate@")}  # fmt: skip
            arm_means = pd.Series(trained).groupby(lambda k: arm_of(k)).mean()
            floor = float(np.mean(init)) if init else np.nan
            floor_sd = float(np.std(init, ddof=1)) if len(init) > 1 else np.nan
            ref_mean = float(np.mean(ref)) if ref else np.nan
            ref_sd = float(np.std(ref, ddof=1)) if len(ref) > 1 else np.nan
            rows.append(dict(
                target=target, pooling=pooling, floor=floor, floor_sd=floor_sd,
                ref1601=ref_mean, ref_sd=ref_sd, gain=ref_mean - floor,
                gain_z=(ref_mean - floor) / np.sqrt((floor_sd**2 + ref_sd**2) / 2)
                if np.isfinite(floor_sd) and np.isfinite(ref_sd) else np.nan,
                arm_min=float(arm_means.min()) if len(arm_means) else np.nan,
                arm_max=float(arm_means.max()) if len(arm_means) else np.nan,
                F=between_arm_f(trained),
                traj=" ".join(f"{traj[s]:.3f}" for s in sorted(traj)),
            ))  # fmt: skip
    return pd.DataFrame(rows)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default="/weka/dfive-default/rslearn-eai/datasets/usdm")
    ap.add_argument("--emb-dir", default="embeddings")
    ap.add_argument("--table", default="windows_eval")
    ap.add_argument("--poolings", default="mean,last4")
    ap.add_argument("--targets", default=None, help="comma list (default: all)")
    ap.add_argument("--sets", default=None, help="regex on embedding set names")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    root = Path(args.root)
    emb_dir = root / args.emb_dir
    order = pd.read_parquet(emb_dir / "windows.parquet")
    table = pd.read_parquet(root / "meta" / f"{args.table}.parquet")
    windows = order.merge(
        table.drop(columns=["split"]), on=["cell_id", "anchor"], how="left"
    )
    assert len(windows) == len(order)
    from olmoearth_pretrain.dataset_creation.usdm.write_windows import (
        NUMERIC_TARGETS,
    )

    targets = (
        args.targets.split(",")
        if args.targets
        else list(NUMERIC_TARGETS) + list(CLASS_TARGETS)
    )
    split = windows["split"].to_numpy()
    train_idx = np.flatnonzero(split == "train")
    folds = list(
        GroupKFold(5).split(train_idx, groups=windows["block"].to_numpy()[train_idx])
    )
    out_path = emb_dir / "target_lab.json"
    results: dict = (
        json.loads(out_path.read_text())["results"] if out_path.exists() else {}
    )
    jobs = []
    for path in sorted(emb_dir.glob("*.npz")):
        if args.sets and not re.search(args.sets, path.stem):
            continue
        for pooling in args.poolings.split(","):
            todo = [
                t
                for t in targets
                if path.stem not in results.get(t, {}).get(pooling, {})
            ]
            if todo:
                jobs.append((str(path), pooling, todo, windows, folds, args.threads))
    logger.info("%d (set, pooling) jobs", len(jobs))
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        for k, (stem, pooling, scores) in enumerate(
            pool.imap_unordered(score_job, jobs), start=1
        ):
            for target, sc in scores.items():
                results.setdefault(target, {}).setdefault(pooling, {})[stem] = sc
            if k % 10 == 0 or k == len(jobs):
                logger.info("[%d/%d] scored %s %s", k, len(jobs), stem, pooling)
                out_path.write_text(json.dumps({"results": results}, indent=1))
    out_path.write_text(json.dumps({"results": results}, indent=1))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    for metric in ("val", "test"):
        print(
            f"\n== {metric}\n",
            summarize(results, metric).round(3).to_string(index=False),
        )


if __name__ == "__main__":
    main()
