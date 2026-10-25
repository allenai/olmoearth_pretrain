r"""Map-level accuracy of open-set classification on PASTIS, with "unknown" as one more class.

open_set_eval.py scores rejection on class-balanced samples. This scores the
map a user would get: every known class plus one "unknown" class (all other
labels, background included; void excluded), evaluated in PASTIS's natural
class mix so that errors are weighted by area.

- pixel level: the dump's uniform per-window pixel sample;
- segment level: every connected-component segment, weighted by its size.

Validation only (as ``open_set_eval.py --report_split val``): methods are fit
on train, thresholds set on even val windows to keep ``--tpr`` of known
samples, and maps scored on odd val windows; test is never read.

For each (known set, method) it reports area-weighted accuracy, per-class
precision / recall / F1 (``f1_<class>``, ``f1_unknown``), macro F1 over the
K+1 classes, and the error split into: unknown accepted as a known class,
known rejected as unknown, known accepted with the wrong known class. Two
references: ``none`` (the linear probe with no rejection, i.e. a closed-set
map) and ``oracle`` (a probe that saw every unknown class as a negative).
``*_argmax`` methods skip calibration and take the plain softmax argmax over
the K+1 outputs (known if the best known class beats "other").

    python scripts/tools/open_set_map_metrics.py --npz pastis_open_set_v2.npz \
        --product olmoearth --level pixel --ks 1 --out map_metrics.csv
"""

import argparse
import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from open_set_eval import (  # noqa: E402
    BACKGROUND,
    CROPS,
    SINGLE_CLASS_SKIP,
    Context,
    cap_per_class,
    fixed_threshold,
    scores_for,
)

logger = logging.getLogger(__name__)

VOID = 19
UNKNOWN = -1
METHODS = [
    "none",
    "unlab_neg",
    "unlab_neg_mlp",
    "unlab_neg_argmax",
    "unlab_neg_mlp_argmax",
    "rmaha_unlab",
    "knn1",
    "knn10",
    "msp",
    "energy",
    "bg0",
    "oracle",
]
MAIN5 = (1, 2, 3, 4, 5)


def table(d: np.lib.npyio.NpzFile, prefix: str, product: str) -> dict[str, np.ndarray]:
    """Rows of one npz table with finite embeddings, plus their val half."""
    x = d[f"{prefix}_{product}"].astype(np.float32)
    window = d[f"{prefix}_window"]
    split = d["window_split"][window].astype(object)
    split[(split == "val") & (window % 2 == 1)] = "report"
    split[split == "val"] = "calib"
    ok = np.isfinite(x).all(axis=1)
    out = {
        "x": x[ok],
        "y": d[f"{prefix}_label"][ok],
        "split": split[ok].astype(str),
    }
    out["w"] = (
        d["seg_size"][ok].astype(np.float64) if prefix == "seg" else np.ones(ok.sum())
    )
    return out


def known_sets(args: argparse.Namespace, rng: np.random.Generator) -> list[tuple]:
    """Every crop for K=1, --max_pairs pairs for K=2, draws for larger K, plus main5."""
    sets = []
    for k in args.ks:
        if k == 1:
            sets += [(c,) for c in args.crops]
        elif k == 2:
            pairs = list(itertools.combinations(args.crops, 2))
            take = rng.choice(
                len(pairs), min(args.max_pairs, len(pairs)), replace=False
            )
            sets += [pairs[i] for i in sorted(take)]
        else:
            sets += [
                tuple(int(c) for c in sorted(rng.choice(args.crops, k, replace=False)))
                for _ in range(args.draws)
            ]
            if k == 5 and set(MAIN5) <= set(args.crops):
                sets.append(MAIN5)
    return list(dict.fromkeys(sets))


def map_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray, ks: tuple
) -> dict[str, float]:
    """Area-weighted accuracy, per-class P/R/F1 (incl. unknown), error split."""
    total = w.sum()
    out: dict[str, float] = {"accuracy": float(w[y_true == y_pred].sum() / total)}
    f1s = []
    for c in (*ks, UNKNOWN):
        tp = w[(y_true == c) & (y_pred == c)].sum()
        fp = w[(y_true != c) & (y_pred == c)].sum()
        fn = w[(y_true == c) & (y_pred != c)].sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        name = "unknown" if c == UNKNOWN else str(c)
        out[f"precision_{name}"] = float(p)
        out[f"recall_{name}"] = float(r)
        out[f"f1_{name}"] = float(f1)
        f1s.append(f1)
    out["macro_f1"] = float(np.mean(f1s))
    out["mean_f1_known"] = float(np.mean(f1s[:-1]))
    known = y_true != UNKNOWN
    out["err_unknown_accepted"] = float(w[~known & (y_pred != UNKNOWN)].sum() / total)
    out["err_known_rejected"] = float(w[known & (y_pred == UNKNOWN)].sum() / total)
    out["err_known_confused"] = float(
        w[known & (y_pred != UNKNOWN) & (y_pred != y_true)].sum() / total
    )
    out["unknown_area"] = float(w[~known].sum() / total)
    return out


def main() -> None:
    """Fit, threshold and score maps for every known set and method."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--npz", required=True)
    parser.add_argument("--product", default="olmoearth")
    parser.add_argument("--level", choices=["pixel", "segment"], default="pixel")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--draws", type=int, default=30)
    parser.add_argument("--max_pairs", type=int, default=40)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--train_cap", type=int, default=2000)
    parser.add_argument("--eval_cap", type=int, default=3000)
    parser.add_argument("--min_train", type=int, default=50)
    parser.add_argument("--unlab_size", type=int, default=20000)
    parser.add_argument(
        "--pool_npz",
        default=None,
        help="pixel level: take the unlabeled pool from this npz's box_<product> "
        "(sample_bbox_pool.py) instead of the dump's per-tile uniform sample",
    )
    parser.add_argument("--tpr", type=float, default=0.95)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument(
        "--crops",
        type=lambda v: [int(c) for c in v.split(",")],
        default=CROPS,
        help="comma-separated class ids that can be known (default: PASTIS crops 1-18)",
    )
    parser.add_argument("--void", type=int, default=VOID)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    d = np.load(args.npz)
    fit = table(d, "pix" if args.level == "pixel" else "seg", args.product)
    uni = table(d, "uni", args.product)
    if args.level == "pixel":
        ev = {k: v[uni["split"] == "report"] for k, v in uni.items()}
        pool = uni["x"][uni["split"] == "train"]
    else:
        ev = {k: v[fit["split"] == "report"] for k, v in fit.items()}
        pool = fit["x"][fit["split"] == "train"]
    if args.pool_npz:
        if args.level != "pixel":
            raise ValueError("--pool_npz is a pixel-level pool")
        box = np.load(args.pool_npz)[f"box_{args.product}"].astype(np.float32)
        pool = box[np.isfinite(box).all(axis=1)]
    keep = ev["y"] != args.void
    ev = {k: v[keep] for k, v in ev.items()}
    x_unlab = pool[
        np.random.default_rng(args.seed + 1).choice(
            len(pool), min(args.unlab_size, len(pool)), replace=False
        )
    ]
    tr = np.nonzero(fit["split"] == "train")[0]
    tr = tr[cap_per_class(fit["y"][tr], args.train_cap, rng)]
    ca = np.nonzero(fit["split"] == "calib")[0]
    ca = ca[cap_per_class(fit["y"][ca], args.eval_cap, rng)]
    xtr_all, ytr_all = fit["x"][tr], fit["y"][tr]
    xca, yca = fit["x"][ca], fit["y"][ca]
    counts = np.bincount(ytr_all, minlength=args.num_classes)
    logger.info(
        f"{args.product}/{args.level}: train {len(tr)}, calib {len(ca)}, "
        f"map rows {len(ev['y'])} (weight {ev['w'].sum():.0f}), pool {len(x_unlab)}"
    )

    rows = []
    for ks in known_sets(args, rng):
        if min(counts[list(ks)]) < args.min_train:
            continue
        trk = np.isin(ytr_all, ks)
        cak = np.isin(yca, ks)
        y_true = np.where(np.isin(ev["y"], ks), ev["y"], UNKNOWN)
        for method in args.methods:
            if len(ks) == 1 and method in SINGLE_CLASS_SKIP:
                continue
            base = "unlab_neg" if method == "none" else method
            neg = {
                "bg0": [BACKGROUND],
                "oracle": [c for c in range(args.num_classes) if c not in ks],
            }
            ctx = Context(
                xtr=xtr_all[trk],
                ytr=ytr_all[trk],
                xval=xca[cak],
                yval=yca[cak],
                x_unlab=x_unlab,
                xneg=xtr_all[np.isin(ytr_all, neg.get(method, []))],
                kmeans={},
                seed=args.seed,
                c=args.c,
            )
            f = scores_for(base, ctx)
            s, pred = f(ev["x"])
            if method == "none":
                y_pred = pred  # closed-set: every pixel gets a known class
            elif fixed_threshold(method) is not None:
                # argmax / Elkan-Noto: the score already encodes the decision
                y_pred = np.where(s >= fixed_threshold(method), pred, UNKNOWN)
            else:
                sv, _ = f(xca[cak])
                thr = np.quantile(sv, 1 - args.tpr)
                y_pred = np.where(s >= thr, pred, UNKNOWN)
            row = {
                "product": args.product,
                "level": args.level,
                "K": len(ks),
                "known": "-".join(map(str, ks)),
                "method": method,
                "pool": "bbox" if args.pool_npz else "tiles",
                "unlab_size": len(x_unlab),
                **map_metrics(y_true, y_pred, ev["w"], ks),
            }
            rows.append(row)
        logger.info(f"known {ks} done")
        pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    pd.set_option("display.width", 220)
    print(f"=== {args.product} / {args.level} ===")
    print(
        df.groupby(["K", "method"], sort=False)[
            [
                "accuracy",
                "macro_f1",
                "f1_unknown",
                "mean_f1_known",
                "err_unknown_accepted",
                "err_known_rejected",
                "err_known_confused",
                "unknown_area",
            ]
        ]
        .mean()
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
