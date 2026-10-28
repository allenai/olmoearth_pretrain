"""Label-count curves: AEF balanced trials at several per-class draws.

Professor's suggestion (board, 2026-10-28 20:14Z): a latent can't add information to its source,
but it may make that information easier to use, which would show most with few labels. For
each embedding dir and AEF task this draws n labelled points per class (n in --caps), fits AEF's
predictors (ridge lambda=0, kNN-5, kNN-20), and scores the remainder, over AEF's number of
draws (balanced_trial.py; the remainder includes test rows, as in AEF's protocol).

Runs on favyen's Strata venv and code (read-only), like score.sh:
  PYTHONPATH=$STRATA/olmoearth_strata:$STRATA/olmoearth_pretrain $STRATA/venvs/oe_strata/bin/python \
    low_label.py --dirs raw_aef/step0,paired_l128_zaef/step20000 --out curves.json
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

ROOT = "/weka/dfive-default/hadriens/platonic_space"


def quantization(d: str) -> str:
    """Same mapping as score.sh."""
    name = d.split("/")[0]
    if name in ("raw_aef", "raw_oe", "concat"):
        return "none"
    return "tessera" if name == "raw_tess" else "aef_power"


def one(d: str, dataset: str, cap: int) -> dict:
    """One (dir, task, per-class draw): mean balanced accuracy per predictor."""
    import torch
    from olmoearth_pretrain.evals.balanced_trial import (
        BalancedTrialConfig,
        run_balanced_trials,
    )
    from olmoearth_strata.evals.aef_tasks import (
        SPLITS,
        align_embeddings,
        load_label_table,
        task_name,
    )
    from olmoearth_strata.evals.score import (
        _eval_config,
        load_embeddings,
        transform_embeddings,
    )

    from olmoearth_pretrain.internal.all_evals import EMBEDDING_EVAL_TASKS

    torch.set_num_threads(2)
    ids, emb = load_embeddings(f"{ROOT}/eval_embeddings/{d}", dataset)
    emb = transform_embeddings(emb, quantization(d))
    arrays = align_embeddings(load_label_table(dataset), ids, emb, allow_missing=True)
    task = EMBEDDING_EVAL_TASKS[task_name(dataset, knn=True)]
    res = run_balanced_trials(
        config=_eval_config(task.dataset),
        embeddings_by_split={s: torch.from_numpy(arrays[s].embeddings) for s in SPLITS},
        labels_by_split={s: torch.from_numpy(arrays[s].labels) for s in SPLITS},
        trial_config=BalancedTrialConfig(n_per_class=cap, cap=max(cap, 300)),
        device=torch.device("cpu"),
    )
    return {
        "dir": d,
        "dataset": dataset,
        "cap": cap,
        **{p: float(r.metrics["balanced_accuracy"]) for p, r in res.results.items()},
    }


def main() -> None:
    """Every (dir, task, cap) in parallel; one JSON list out."""
    from olmoearth_strata.evals.aef_tasks import AEF_DATASETS

    p = argparse.ArgumentParser()
    p.add_argument(
        "--dirs", required=True, help="comma list of dirs under ROOT/eval_embeddings"
    )
    p.add_argument("--caps", default="10,30,100,300")
    p.add_argument("--datasets", default=",".join(AEF_DATASETS))
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    jobs = [
        (d, ds, int(c))
        for d in a.dirs.split(",")
        for ds in a.datasets.split(",")
        for c in a.caps.split(",")
    ]
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn")) as ex:
        rows = list(ex.map(one, *zip(*jobs)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    for r in rows:
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
