"""Stage 4: tag a stratified subset of windows for the in-loop eval.

The in-loop evaluator embeds the *whole* train split of every eval task at each
eval interval, so the full CY-Bench splits (141k maize / 66k wheat train
windows) are far too large. Following the LFMC precedent (``lfmc_woody_eval``
= 3k ``oep_eval``-tagged windows), this script adds an ``oep_eval`` tag to a
stratified subsample per crop and split, and the ``cybench_<crop>_eval``
registry entries filter on it. The full-size entries (``cybench_<crop>``) stay
untouched.

Sampling is stratified by (country, harvest year) with allocation proportional
to stratum size (at least one per stratum while the budget allows), seeded.
Candidates come from the label table, so only the sampled windows'
``metadata.json`` files are read/written; windows that were skipped at export
time are simply not sampled.

Usage::

    python -m olmoearth_pretrain.dataset_creation.cybench.tag_eval_subset \
        --labels-root ... --ds-path /weka/.../cybench/rslearn_dataset \
        --n-train 3000 --n-val 1000 --n-test 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .common import (
    CROP_TO_AFI,
    DEFAULT_LABELS_ROOT,
    DEFAULT_ROOT,
    list_crop_countries,
    load_calendar,
    load_labels,
    safe_name,
    split_for_year,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def candidate_windows(labels_root: Path, crop: str, min_year: int) -> pd.DataFrame:
    """All (crop, cc, adm_id, year) labels that could have a window, with names + splits."""
    frames = []
    for c, cc in list_crop_countries(labels_root):
        if c != crop:
            continue
        labels = load_labels(labels_root, crop, cc)
        cal = load_calendar(labels_root, crop, cc)
        labels = labels[labels["year"] >= min_year]
        labels = labels[labels["adm_id"].isin(set(cal["adm_id"]))]
        if labels.empty:
            continue
        labels = labels.assign(cc=cc)
        frames.append(labels)
    df = pd.concat(frames, ignore_index=True)
    df["name"] = [
        safe_name(f"{crop}_{r.cc}_{r.adm_id}_{r.year}") for r in df.itertuples()
    ]
    df["split"] = df["year"].map(split_for_year)
    return df.drop_duplicates("name")


def stratified_sample(
    df: pd.DataFrame, n: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Sample ``n`` rows stratified by (cc, year), proportional with a floor of 1."""
    if len(df) <= n:
        return df
    sizes = df.groupby(["cc", "year"]).size()
    alloc = np.floor(sizes / sizes.sum() * n).astype(int)
    if len(sizes) <= n:
        alloc = alloc.clip(lower=1)
    # Distribute the remainder to the largest fractional parts.
    remainder = n - int(alloc.sum())
    if remainder > 0:
        frac = (sizes / sizes.sum() * n) - np.floor(sizes / sizes.sum() * n)
        for key in frac.sort_values(ascending=False).index[:remainder]:
            alloc[key] += 1
    elif remainder < 0:
        for key in alloc.sort_values(ascending=False).index[:-remainder]:
            alloc[key] -= 1
    parts = []
    for (cc, year), k in alloc.items():
        g = df[(df["cc"] == cc) & (df["year"] == year)]
        k = min(int(k), len(g))
        if k > 0:
            parts.append(g.iloc[rng.choice(len(g), size=k, replace=False)])
    return pd.concat(parts, ignore_index=True)


def _tag_window(args: tuple) -> bool:
    """Add the tag to one window's metadata.json (returns False if missing)."""
    meta_path, tag = args
    p = Path(meta_path)
    if not p.exists():
        return False
    meta = json.loads(p.read_text())
    opts = meta.setdefault("options", {})
    if tag in opts:
        return True
    opts[tag] = ""
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta))
    os.replace(tmp, p)
    return True


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--labels-root", default=str(DEFAULT_LABELS_ROOT))
    ap.add_argument("--ds-path", default=str(DEFAULT_ROOT / "rslearn_dataset"))
    ap.add_argument("--crops", nargs="+", default=sorted(CROP_TO_AFI))
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--tag", default="oep_eval")
    ap.add_argument("--min-year", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    ds_path = Path(args.ds_path)
    labels_root = Path(args.labels_root)
    budgets = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    rng = np.random.default_rng(args.seed)
    chosen_all = []
    for crop in args.crops:
        cands = candidate_windows(labels_root, crop, args.min_year)
        # Keep only windows that exist (skipped at export time otherwise).
        with ThreadPoolExecutor(args.workers) as ex:
            exists = list(
                ex.map(
                    lambda n: (
                        ds_path / "windows" / crop / n / "metadata.json"
                    ).exists(),
                    cands["name"],
                )
            )
        cands = cands[np.array(exists, dtype=bool)]
        logger.info("%s: %d existing candidate windows", crop, len(cands))
        for split, n in budgets.items():
            pool = cands[cands["split"] == split]
            chosen = stratified_sample(pool, n, rng).assign(crop=crop)
            logger.info(
                "%s/%s: %d of %d windows (%d countries, %d years)",
                crop,
                split,
                len(chosen),
                len(pool),
                chosen["cc"].nunique(),
                chosen["year"].nunique(),
            )
            chosen_all.append(chosen)
    chosen_df = pd.concat(chosen_all, ignore_index=True)

    jobs = [
        (str(ds_path / "windows" / r.crop / r.name / "metadata.json"), args.tag)
        for r in chosen_df.itertuples()
    ]
    with ThreadPoolExecutor(args.workers) as ex:
        ok = list(ex.map(_tag_window, jobs))
    logger.info("tagged %d / %d windows with %r", sum(ok), len(ok), args.tag)

    meta_dir = ds_path.parent / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    chosen_df.to_csv(meta_dir / f"eval_subset_{args.tag}.csv", index=False)

    # Any cached rslearn window index predates the new tag; drop it so the
    # next dataset load rebuilds with the tags visible.
    idx = ds_path / ".rslearn_dataset_index"
    if idx.exists():
        shutil.rmtree(idx)
        logger.info("removed stale index %s", idx)


if __name__ == "__main__":
    main()
