"""Table of scorer results (eval_results/*.json): LP test mean ± seed sd, kNN-20, AEF-trial best-of.

  python summarize.py [name ...]      # names are eval_results stems, e.g. raw_aef_step0; default all
  python summarize.py --metric knn    # lp (default) | knn | trials | lp_sd

Prints one column per result and one row per task, plus the mean over tasks. With --best, a
'best raw' column holds the best of raw_aef / raw_oe / raw_tess per task.
"""

import argparse
import glob
import json
import os

import numpy as np

ROOT = "/weka/dfive-default/hadriens/platonic_space/eval_results"
TASKS = (
    "ethiopia_crops",
    "africa_crop_mask",
    "canada_crops_fine",
    "canada_crops_coarse",
    "descals",
    "lcmap_lu",
    "glance",
    "us_trees",
)


def cell(r: dict, metric: str) -> float:
    """One task's number for a metric."""
    if metric == "lp":
        return r["lp"]["test"]
    if metric == "lp_sd":
        return float(np.std(list(r["lp"]["test_per_seed"].values())))
    if metric == "knn":
        return r["knn"]["test"]
    return r["trials_best_of"]


def main() -> None:
    """Print the table."""
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="*")
    p.add_argument("--metric", default="lp")
    p.add_argument("--best", action="store_true")
    a = p.parse_args()
    names = a.names or sorted(
        os.path.basename(f)[:-5] for f in glob.glob(f"{ROOT}/*.json")
    )
    res = {n: json.load(open(f"{ROOT}/{n}.json"))["tasks"] for n in names}
    cols = {
        n: {t: cell(res[n][t], a.metric) for t in TASKS if t in res[n]} for n in names
    }
    if a.best:
        raws = [f"raw_{m}_step0" for m in ("aef", "oe", "tess")]
        raw = {
            r: {
                t: cell(json.load(open(f"{ROOT}/{r}.json"))["tasks"][t], a.metric)
                for t in TASKS
            }
            for r in raws
        }
        cols = {"best raw": {t: max(raw[r][t] for r in raws) for t in TASKS}} | cols
    w = max(12, *(len(n) for n in cols))
    print(f"{'task':<20}" + "".join(f"{n:>{w + 1}}" for n in cols))
    for t in TASKS:
        print(
            f"{t:<20}"
            + "".join(f"{c.get(t, float('nan')):>{w + 1}.4f}" for c in cols.values())
        )
    print(
        f"{'mean':<20}"
        + "".join(
            f"{np.mean([c[t] for t in TASKS if t in c]):>{w + 1}.4f}"
            for c in cols.values()
        )
    )


if __name__ == "__main__":
    main()
