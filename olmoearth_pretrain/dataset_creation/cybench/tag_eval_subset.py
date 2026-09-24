"""Stage 4: tag the windows the eval registry entries filter on.

The in-loop evaluator embeds every train window of a task at each eval
interval, so the full CY-Bench splits are far too large. Two tag schemes are
written into the windows' ``metadata.json`` options (the ``cybench_*``
registry entries filter on them):

* pooled (default): ``oep_eval`` on a stratified (country x harvest year)
  sample of 3k train / 1k val / 1k test windows per crop, like
  ``lfmc_woody_eval``;
* ``--per-country CC ...``: ``loyo_split=test`` on every window of the
  country's last label year and ``loyo_split=train`` on a stratified sample
  of its earlier years -- one fold of CY-Bench's leave-one-year-out protocol.

Candidates come from the label table, so only the chosen windows are touched.
Any cached rslearn window index is dropped so the tags become visible.

Usage::

    python -m olmoearth_pretrain.dataset_creation.cybench.tag_eval_subset --n-train 3000 --n-val 1000 --n-test 1000
    python -m olmoearth_pretrain.dataset_creation.cybench.tag_eval_subset --per-country US DE AR --n-train 3000
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
    split_for_year,
    window_name,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# --- candidates ---------------------------------------------------------------
def candidate_windows(
    labels_root: Path, ds_path: Path, crop: str, min_year: int, workers: int
) -> pd.DataFrame:
    """Every existing window of ``crop``: columns ``cc, adm_id, year, name, split``."""
    frames = []
    for c, cc in list_crop_countries(labels_root):
        if c != crop:
            continue
        labels = load_labels(labels_root, crop, cc)
        labels = labels[
            (labels["year"] >= min_year)
            & labels["adm_id"].isin(load_calendar(labels_root, crop, cc)["adm_id"])
        ]
        if not labels.empty:
            frames.append(labels.assign(cc=cc))
    df = pd.concat(frames, ignore_index=True)
    df["name"] = [window_name(crop, r.cc, r.adm_id, r.year) for r in df.itertuples()]
    df["split"] = df["year"].map(split_for_year)
    df = df.drop_duplicates("name")
    with ThreadPoolExecutor(workers) as ex:
        exists = list(
            ex.map(
                lambda n: (ds_path / "windows" / crop / n / "metadata.json").exists(),
                df["name"],
            )
        )
    return df[np.array(exists, dtype=bool)]


def stratified_sample(
    df: pd.DataFrame, n: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Sample ``n`` rows stratified by (cc, year): proportional allocation, floor of 1 while the budget allows."""
    if len(df) <= n:
        return df
    sizes = df.groupby(["cc", "year"]).size()
    exact = sizes / sizes.sum() * n
    alloc = np.floor(exact).astype(int)
    if len(sizes) <= n:
        alloc = alloc.clip(lower=1)
    remainder = n - int(alloc.sum())  # hand out (or claw back) the rounding remainder
    if remainder > 0:
        alloc[
            (exact - np.floor(exact)).sort_values(ascending=False).index[:remainder]
        ] += 1
    elif remainder < 0:
        alloc[alloc.sort_values(ascending=False).index[:-remainder]] -= 1
    parts = []
    for (cc, year), k in alloc.items():
        group = df[(df["cc"] == cc) & (df["year"] == year)]
        if k > 0:
            parts.append(
                group.iloc[
                    rng.choice(len(group), size=min(int(k), len(group)), replace=False)
                ]
            )
    return pd.concat(parts, ignore_index=True)


# --- the two tagging schemes ------------------------------------------------
def pooled_subset(
    cands: pd.DataFrame, crop: str, budgets: dict[str, int], rng: np.random.Generator
) -> pd.DataFrame:
    """Stratified sample per split; every chosen window gets tag value ``""``."""
    chosen = []
    for split, n in budgets.items():
        pool = cands[cands["split"] == split]
        sample = stratified_sample(pool, n, rng)
        logger.info(
            "%s/%s: %d of %d windows (%d countries, %d years)",
            crop,
            split,
            len(sample),
            len(pool),
            sample["cc"].nunique(),
            sample["year"].nunique(),
        )
        chosen.append(sample.assign(value=""))
    return pd.concat(chosen, ignore_index=True)


def last_year_out(
    cands: pd.DataFrame,
    crop: str,
    countries: list[str],
    n_train: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Per country: value ``test`` for its whole last label year, ``train`` for a sample of earlier years."""
    chosen = []
    for cc in countries:
        pool = cands[cands["cc"] == cc]
        if pool.empty:
            logger.warning("%s/%s: no candidate windows", crop, cc)
            continue
        last = int(pool["year"].max())
        test = pool[pool["year"] == last].assign(value="test")
        train = stratified_sample(pool[pool["year"] < last], n_train, rng).assign(
            value="train"
        )
        logger.info("%s/%s: held-out year %d -> %d test windows; train %d of %d (years %d-%d)",
                    crop, cc, last, len(test), len(train), int((pool["year"] < last).sum()), int(pool["year"].min()), last - 1)  # fmt: skip
        chosen += [test, train]
    return pd.concat(chosen, ignore_index=True)


# --- applying tags ------------------------------------------------------------
def apply_tags(ds_path: Path, chosen: pd.DataFrame, tag: str, workers: int) -> None:
    """Write ``options[tag] = value`` into each chosen window's metadata.json, then drop the stale index."""
    jobs = [
        (ds_path / "windows" / r.crop / r.name / "metadata.json", tag, r.value)
        for r in chosen.itertuples()
    ]
    with ThreadPoolExecutor(workers) as ex:
        ok = list(ex.map(_set_option, jobs))
    logger.info("tagged %d / %d windows with %r", sum(ok), len(ok), tag)
    index = ds_path / ".rslearn_dataset_index"
    if index.exists():  # predates the new tag; rslearn rebuilds it on next load
        shutil.rmtree(index)
        logger.info("removed stale index %s", index)


def _set_option(job: tuple[Path, str, str]) -> bool:
    path, tag, value = job
    if not path.exists():
        return False
    meta = json.loads(path.read_text())
    options = meta.setdefault("options", {})
    if options.get(tag) != value:
        options[tag] = value
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta))
        os.replace(tmp, path)
    return True


# --- CLI ----------------------------------------------------------------------
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
    ap.add_argument("--tag", default="oep_eval", help="tag key for the pooled subset")
    ap.add_argument(
        "--per-country",
        nargs="*",
        default=None,
        help="last-year-out tags for these country codes instead of the pooled subset",
    )
    ap.add_argument("--lyo-tag", default="loyo_split", help="tag key for --per-country")
    ap.add_argument("--min-year", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    labels_root, ds_path = Path(args.labels_root), Path(args.ds_path)
    rng = np.random.default_rng(args.seed)
    tag = args.lyo_tag if args.per_country else args.tag

    chosen = []
    for crop in args.crops:
        cands = candidate_windows(
            labels_root, ds_path, crop, args.min_year, args.workers
        )
        logger.info("%s: %d existing candidate windows", crop, len(cands))
        if args.per_country:
            picked = last_year_out(cands, crop, args.per_country, args.n_train, rng)
        else:
            picked = pooled_subset(
                cands,
                crop,
                {"train": args.n_train, "val": args.n_val, "test": args.n_test},
                rng,
            )
        chosen.append(picked.assign(crop=crop))
    chosen_df = pd.concat(chosen, ignore_index=True)

    apply_tags(ds_path, chosen_df, tag, args.workers)
    meta_dir = ds_path.parent / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    chosen_df.to_csv(meta_dir / f"eval_subset_{tag}.csv", index=False)


if __name__ == "__main__":
    main()
