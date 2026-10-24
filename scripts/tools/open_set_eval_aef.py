r"""Open-set rejection on the AEF eval datasets, from dumped probe embeddings.

Reads ``<dump_root>/<product>/<task>.npz`` written by an embedding eval run with
``OE_DUMP_EMBEDDINGS_DIR`` set (evaluator_callback), i.e. exactly the
train/val/test embeddings a kNN/LP probe sees. Methods and metrics are those of
open_set_eval.py (imported from it); only the known/unknown splits differ,
since these datasets have no background class:

- K=1: each class alone is known (positives-only).
- K=C-1: each class alone is unknown (leave-one-out).
- half: ``--draws`` random draws of half the classes as known (C >= 4).

The unlabeled pool is a sample of the train split over every class, in the
dataset's own class mix; oracle uses every non-known train class as
negatives. With ``--report_split val`` (the default) the val split is halved
at random: one half sets the threshold (``--tpr`` of known samples kept), the
other is reported, and test is never read. ``test`` thresholds on val and
reports on test.

    python scripts/tools/open_set_eval_aef.py --dump_root /weka/.../aef_evals \
        --out_dir /weka/.../aef_evals/results
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
from open_set_eval import (  # noqa: E402
    SINGLE_CLASS_SKIP,
    Context,
    auroc,
    cap_per_class,
    fpr_at_tpr,
    scores_for,
)

logger = logging.getLogger(__name__)

METHODS = [
    "msp",
    "energy",
    "maha",
    "rmaha_unlab",
    "knn10",
    "cluster200",
    "unlab_neg",
    "unlab_neg_mlp",
    "unlab_neg_argmax",
    "unlab_neg_mlp_argmax",
    "oracle",
]


def known_sets(classes: list[int], draws: int, rng: np.random.Generator) -> list[tuple]:
    """(regime, known classes) pairs: K=1, leave-one-out, random halves."""
    sets = [("K1", (c,)) for c in classes]
    if len(classes) >= 3:
        sets += [("loo", tuple(x for x in classes if x != c)) for c in classes]
    if len(classes) >= 4:
        half = len(classes) // 2
        sets += [
            (
                "half",
                tuple(int(c) for c in sorted(rng.choice(classes, half, replace=False))),
            )
            for _ in range(draws)
        ]
    return sets


def run_task(npz: Path, product: str, args: argparse.Namespace) -> list[dict]:
    """Every method over every known set of one dumped task."""
    rng = np.random.default_rng(args.seed)
    d = np.load(npz)
    splits = (
        ("train", "val") if args.report_split == "val" else ("train", "val", "test")
    )
    raw = {}
    for s in splits:
        x = d[f"{s}_embeddings"].astype(np.float32)
        y = d[f"{s}_labels"].astype(np.int64)
        if x.size == 0:
            raise ValueError(f"{npz}: no {s} split")
        raw[s] = (x, y)
    if args.report_split == "val":
        # threshold on one random half of val, report on the other
        x, y = raw["val"]
        perm = np.random.default_rng(args.seed + 2).permutation(len(x))
        calib, report = perm[: len(x) // 2], perm[len(x) // 2 :]
        raw["val"], raw["test"] = (x[calib], y[calib]), (x[report], y[report])
    parts = {}
    for s, cap in (
        ("train", args.train_cap),
        ("val", args.eval_cap),
        ("test", args.eval_cap),
    ):
        x, y = raw[s]
        idx = cap_per_class(y, cap, rng)
        parts[s] = (x[idx], y[idx])
    xtr_all, ytr_all = parts["train"]
    xv, yv = parts["val"]
    xt, yt = parts["test"]
    x_full = d["train_embeddings"].astype(np.float32)
    x_unlab = x_full[
        np.random.default_rng(args.seed + 1).choice(
            len(x_full), min(args.unlab_size, len(x_full)), replace=False
        )
    ]
    classes = sorted(int(c) for c in np.unique(ytr_all))
    counts = np.bincount(ytr_all)
    dataset = str(d["dataset"])
    logger.info(
        f"{product}/{dataset}: {len(classes)} classes, train {len(xtr_all)}, "
        f"val {len(xv)}, test {len(xt)}, dim {xtr_all.shape[1]}"
    )

    kmeans = {}
    shared_fit_s = {}
    for n in sorted(
        {int(m[len("cluster") :]) for m in args.methods if m.startswith("cluster")}
    ):
        n_eff = min(n, len(x_unlab) // 5)
        t0 = time.perf_counter()
        model = MiniBatchKMeans(n_clusters=n_eff, random_state=args.seed, n_init=3).fit(
            x_unlab
        )
        kmeans[n] = (model, model.predict(x_unlab))
        shared_fit_s[f"cluster{n}"] = time.perf_counter() - t0

    rows = []
    for regime, ks in known_sets(classes, args.draws, rng):
        if min(counts[list(ks)]) < args.min_train:
            continue
        tr = np.isin(ytr_all, ks)
        vk = np.isin(yv, ks)
        kn = np.isin(yt, ks)
        if kn.all() or not vk.any():
            continue
        for method in args.methods:
            if len(ks) == 1 and method in SINGLE_CLASS_SKIP:
                continue
            ctx = Context(
                xtr=xtr_all[tr],
                ytr=ytr_all[tr],
                xval=xv[vk],
                yval=yv[vk],
                x_unlab=x_unlab,
                xneg=xtr_all[~tr] if method == "oracle" else xtr_all[:0],
                kmeans=kmeans,
                seed=args.seed,
                c=args.c,
            )
            w0, c0 = time.perf_counter(), time.process_time()
            f = scores_for(method, ctx)
            w1, c1 = time.perf_counter(), time.process_time()
            sv, _ = f(xv[vk])
            # *_argmax: plain softmax argmax over the K+1 outputs, no calibration
            thr = 0.0 if method.endswith("_argmax") else np.quantile(sv, 1 - args.tpr)
            w2, c2 = time.perf_counter(), time.process_time()
            st, pred = f(xt)
            w3, c3 = time.perf_counter(), time.process_time()
            row = {
                "product": product,
                "dataset": dataset,
                "task": npz.stem,
                "n_classes": len(classes),
                "regime": regime,
                "unlab_size": len(x_unlab),
                "report_split": args.report_split,
                "K": len(ks),
                "known": "-".join(map(str, ks)),
                "method": method,
                "closed_acc": float((pred[kn] == yt[kn]).mean()),
                "known_kept_test": float((st[kn] >= thr).mean()),
                "auroc": auroc(st[kn], st[~kn]),
                "fpr": fpr_at_tpr(st[kn], st[~kn], args.tpr),
                "open_acc": float(
                    np.r_[(st[kn] >= thr) & (pred[kn] == yt[kn]), st[~kn] < thr].mean()
                ),
                "n_train": len(ctx.xtr) + len(ctx.xneg),
                "fit_s": w1 - w0,
                "fit_cpu_s": c1 - c0,
                "shared_fit_s": shared_fit_s.get(method, 0.0),
                "calib_s": w2 - w1,
                "score_us_per_sample": 1e6 * (w3 - w2) / len(xt),
                "score_cpu_us_per_sample": 1e6 * (c3 - c2) / len(xt),
            }
            for c in classes:
                if c not in ks and (yt == c).any():
                    row[f"rej_{c}"] = float((st[yt == c] < thr).mean())
            rows.append(row)
    return rows


def main() -> None:
    """Score every dumped task of every product and write one CSV per product."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dump_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--products", nargs="+", default=["olmoearth", "aef", "tessera_v2"]
    )
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--train_cap", type=int, default=2000)
    parser.add_argument("--eval_cap", type=int, default=3000)
    parser.add_argument("--min_train", type=int, default=20)
    parser.add_argument("--unlab_size", type=int, default=20000)
    parser.add_argument("--tpr", type=float, default=0.95)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--report_split", choices=["val", "test"], default="val")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for product in args.products:
        rows = []
        for npz in sorted((Path(args.dump_root) / product).glob("*.npz")):
            rows += run_task(npz, product, args)
            pd.DataFrame(rows).to_csv(out_dir / f"{product}.csv", index=False)
        df = pd.DataFrame(rows)
        pd.set_option("display.width", 220)
        print(f"=== {product} ===")
        print(
            df.groupby(["dataset", "regime", "method"], sort=False)[
                [
                    "closed_acc",
                    "auroc",
                    "fpr",
                    "open_acc",
                    "fit_cpu_s",
                    "score_cpu_us_per_sample",
                ]
            ]
            .mean()
            .round(3)
            .to_string()
        )


if __name__ == "__main__":
    main()
