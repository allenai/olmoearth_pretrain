"""What a *shared* latent allows that raw products don't: probes trained across sources.

For one run's aligned latents z_aef, z_oe, z_tess (export.py, same common windows) and each
AEF task, the harness linear probe (score.py's run_lp_job: LR sweep, best val -> test) is
trained three ways and always tested on one source m:

  single   train on z_m                     (= the scorer's number for <run>_z<m>)
  multi    train on z_aef + z_oe + z_tess   (3x the labelled rows, labels repeated)
  from_<n> train on z_n only                (drop-in: a probe fit on another product)

Val and test rows are z_m in every mode, so only the probe's training data differs.

Runs on favyen's Strata venv and code (read-only), like score.sh:
  PYTHONPATH=$STRATA/olmoearth_strata:$STRATA/olmoearth_pretrain $STRATA/venvs/oe_strata/bin/python \
    multiview_probe.py --run paired_l128 --step 20000 --out mv.json
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

ROOT = "/weka/dfive-default/hadriens/platonic_space"
SOURCES = ("aef", "oe", "tess")


def load(run: str, step: int, dataset: str) -> dict:
    """Per source: the aligned split arrays (same windows and order for every source)."""
    from olmoearth_strata.evals.aef_tasks import align_embeddings, load_label_table
    from olmoearth_strata.evals.score import load_embeddings, transform_embeddings

    table = load_label_table(dataset)
    out = {}
    for m in SOURCES:
        ids, emb = load_embeddings(
            f"{ROOT}/eval_embeddings/{run}_z{m}/step{step}", dataset
        )
        out[m] = align_embeddings(
            table, ids, transform_embeddings(emb, "aef_power"), allow_missing=True
        )
    return out


def arrays_for(split_arrays: dict, target: str, mode: str) -> dict:
    """Train from the mode's sources, val / test from the target."""
    from olmoearth_strata.evals.aef_tasks import SplitArrays

    srcs = (
        SOURCES if mode == "multi" else (target,) if mode == "single" else (mode[5:],)
    )
    tr = [split_arrays[s]["train"] for s in srcs]
    train = SplitArrays(
        embeddings=np.concatenate([a.embeddings for a in tr]),
        labels=np.concatenate([a.labels for a in tr]),
        window_ids=np.concatenate([a.window_ids for a in tr]),
    )
    return {
        "train": train,
        "val": split_arrays[target]["val"],
        "test": split_arrays[target]["test"],
    }


def main() -> None:
    """All (task, target, mode, lr, seed) LP jobs in one pool; JSON of summaries out."""
    from olmoearth_strata.evals.aef_tasks import AEF_DATASETS
    from olmoearth_strata.evals.score import _summarize_lp, run_lp_job

    from olmoearth_pretrain.internal.full_eval_sweep import LP_LRs

    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--datasets", default=",".join(AEF_DATASETS))
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    modes = ["single", "multi"] + [f"from_{n}" for n in SOURCES]
    data = {ds: load(a.run, a.step, ds) for ds in a.datasets.split(",")}
    futures = {}
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn")) as ex:
        for ds, sa in data.items():
            for m in SOURCES:
                for mode in modes:
                    if mode == f"from_{m}":
                        continue
                    arr = arrays_for(sa, m, mode)
                    for lr in LP_LRs:
                        for seed in seeds:
                            futures[(ds, m, mode, lr, seed)] = ex.submit(
                                run_lp_job, ds, lr, arr, seed, 2
                            )
        res = {k: f.result() for k, f in futures.items()}
    rows = []
    for ds in data:
        for m in SOURCES:
            for mode in modes:
                lp = {
                    (lr, s): r
                    for (d, t, mo, lr, s), r in res.items()
                    if (d, t, mo) == (ds, m, mode)
                }
                if lp:
                    sm = _summarize_lp(lp, "balanced_accuracy")
                    rows.append(
                        {
                            "dataset": ds,
                            "target": m,
                            "mode": mode,
                            "test": sm["test"],
                            "val": sm["val"],
                        }
                    )
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    for m in SOURCES:
        print(
            f"target {m}: "
            + "  ".join(
                f"{mode} {np.mean([r['test'] for r in rows if r['target'] == m and r['mode'] == mode]):.4f}"
                for mode in modes
                if mode != f"from_{m}"
            )
        )


if __name__ == "__main__":
    main()
