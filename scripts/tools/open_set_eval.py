r"""Open-set ("unknown class") rejection on frozen embeddings, scored on PASTIS.

Reads the .npz from dump_pastis_open_set_pixels.py. For each known set (a few
of the 18 PASTIS crop classes) a readout is fit on the train split's known
classes only; every other class is unknown. Each method gives a score where
higher means "more likely known":

- msp / maxlogit / energy: post-hoc on a softmax linear probe
  (Hendrycks & Gimpel 2017; Hendrycks et al. 2022; Liu et al. 2020).
- maha: min Mahalanobis distance to a class mean, shared Ledoit-Wolf
  covariance (Lee et al. 2018). Shrinkage matters: 128 dims vs a few hundred
  samples per class.
- rmaha_known / rmaha_unlab: relative Mahalanobis (Ren et al. 2021), minus the
  distance under one Gaussian fit to the known train data, or to unlabeled
  train pixels of every class (unlabeled imagery is free in remote sensing).
- knn: minus the distance to the k-th nearest known train embedding after L2
  normalisation (Sun et al. 2022).
- bg0: a (K+1)-class probe with PASTIS background (label 0) as an explicit
  "other" class (outlier exposure); score = 1 - p(other). Class 0 is then a
  seen negative, so bg0's background columns are not open-set numbers; only
  its crop columns are.
- oracle: the same probe with every non-known train class as "other", i.e. the
  test-time unknowns were seen as negatives. Not an open-set method: it is the
  upper bound on how separable known vs unknown is in the embedding with a
  linear readout, and the reference for how much each method leaves behind.

Thresholds are set on val so that ``--tpr`` of known val samples are kept; the
fraction of each unknown class rejected at that threshold is then measured on
test. AUROC / FPR@TPR are threshold-free, on test, split into crop unknowns
(classes 1-18) and background (class 0).

    python scripts/tools/open_set_eval.py --npz pastis_open_set.npz \
        --product olmoearth --level pixel --out results.csv
"""

import argparse
import itertools
import logging

import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

CROPS = list(range(1, 19))
BACKGROUND = 0
METHODS = [
    "msp",
    "maxlogit",
    "energy",
    "maha",
    "rmaha_known",
    "rmaha_unlab",
    "knn",
    "bg0",
    "oracle",
]


def load(npz_path: str, product: str, level: str) -> tuple[np.ndarray, ...]:
    """(embeddings, labels, split) for one product, dropping invalid rows."""
    d = np.load(npz_path)
    prefix = "pix" if level == "pixel" else "seg"
    x = d[f"{prefix}_{product}"].astype(np.float32)
    y = d[f"{prefix}_label"]
    split = d["window_split"][d[f"{prefix}_window"]]
    ok = np.isfinite(x).all(axis=1)
    logger.info(f"{product}/{level}: {ok.sum()}/{len(ok)} valid rows")
    return x[ok], y[ok], split[ok]


def cap_per_class(y: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Indices with at most ``cap`` rows per class."""
    keep = []
    for c in np.unique(y):
        idx = np.nonzero(y == c)[0]
        keep.append(rng.choice(idx, min(cap, len(idx)), replace=False))
    return np.concatenate(keep)


def fit_logits(xtr: np.ndarray, ytr: np.ndarray, c: float):
    """A softmax probe; returns a function mapping x to (n, K) logits."""
    clf = LogisticRegression(C=c, max_iter=2000).fit(xtr, ytr)

    def logits(x: np.ndarray) -> np.ndarray:
        z = clf.decision_function(x)
        if z.ndim == 1:  # binary: sklearn keeps one logit; recentre to two
            z = np.stack([-z / 2, z / 2], axis=1)
        return z

    return logits, clf.classes_


def mahalanobis_sq(x: np.ndarray, mean: np.ndarray, prec: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distance of each row of x to mean."""
    diff = x - mean
    return np.einsum("nd,de,ne->n", diff, prec, diff)


def scores_for(
    method: str,
    xtr: np.ndarray,
    ytr: np.ndarray,
    x_unlab: np.ndarray,
    xneg: np.ndarray,
    args: argparse.Namespace,
):
    """Fit a method; returns score(x) -> (higher = known, closed-set prediction)."""
    known = np.unique(ytr)
    if method in ("msp", "maxlogit", "energy"):
        logits, classes = fit_logits(xtr, ytr, args.c)

        def f(x: np.ndarray):
            z = logits(x)
            pred = classes[z.argmax(axis=1)]
            if method == "msp":
                return softmax(z, axis=1).max(axis=1), pred
            if method == "maxlogit":
                return z.max(axis=1), pred
            return logsumexp(z, axis=1), pred

        return f
    if method.startswith(("maha", "rmaha")):
        means = np.stack([xtr[ytr == c].mean(axis=0) for c in known])
        centred = xtr - means[np.searchsorted(known, ytr)]
        prec = LedoitWolf().fit(centred).precision_
        ref = None
        if method != "maha":
            pool = xtr if method == "rmaha_known" else x_unlab
            lw = LedoitWolf().fit(pool)
            ref = (lw.location_, lw.precision_)

        def f(x: np.ndarray):
            d = np.stack([mahalanobis_sq(x, m, prec) for m in means], axis=1)
            if ref is not None:
                d = d - mahalanobis_sq(x, *ref)[:, None]
            return -d.min(axis=1), known[d.argmin(axis=1)]

        return f
    if method == "knn":

        def norm(a: np.ndarray) -> np.ndarray:
            return a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-8)

        nn = NearestNeighbors(n_neighbors=args.k).fit(norm(xtr))

        def f(x: np.ndarray):
            dist, idx = nn.kneighbors(norm(x))
            votes = ytr[idx]
            pred = np.array([np.bincount(v).argmax() for v in votes])
            return -dist[:, -1], pred

        return f
    if method in ("bg0", "oracle"):
        other = -1
        xx = np.concatenate([xtr, xneg])
        yy = np.concatenate([ytr, np.full(len(xneg), other)])
        logits, classes = fit_logits(xx, yy, args.c)
        o = int(np.nonzero(classes == other)[0][0])

        def f(x: np.ndarray):
            p = softmax(logits(x), axis=1)
            p_known = p.copy()
            p_known[:, o] = -1
            return 1 - p[:, o], classes[p_known.argmax(axis=1)]

        return f
    raise ValueError(method)


def fpr_at_tpr(s_known: np.ndarray, s_unknown: np.ndarray, tpr: float) -> float:
    """Fraction of unknowns kept at the threshold that keeps ``tpr`` of knowns."""
    thr = np.quantile(s_known, 1 - tpr)
    return float((s_unknown >= thr).mean())


def known_sets(args: argparse.Namespace, rng: np.random.Generator) -> list[tuple]:
    """Every pair of crop classes for K=2; random draws for larger K."""
    sets = []
    for k in args.ks:
        if k == 2:
            sets += list(itertools.combinations(CROPS, 2))
        else:
            sets += [
                tuple(sorted(rng.choice(CROPS, k, replace=False)))
                for _ in range(args.draws)
            ]
    return sets


def main() -> None:
    """Run every method over every known set and write a long-format CSV."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--npz", required=True)
    parser.add_argument("--product", default="olmoearth")
    parser.add_argument("--level", choices=["pixel", "segment"], default="pixel")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ks", type=int, nargs="+", default=[2, 5, 10])
    parser.add_argument("--draws", type=int, default=30)
    parser.add_argument("--train_cap", type=int, default=2000)
    parser.add_argument("--eval_cap", type=int, default=3000)
    parser.add_argument("--min_train", type=int, default=50)
    parser.add_argument("--tpr", type=float, default=0.95)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    x, y, split = load(args.npz, args.product, args.level)
    parts = {}
    for s, cap in (
        ("train", args.train_cap),
        ("val", args.eval_cap),
        ("test", args.eval_cap),
    ):
        idx = np.nonzero(split == s)[0]
        idx = idx[cap_per_class(y[idx], cap, rng)]
        parts[s] = (x[idx], y[idx])
        logger.info(f"{s}: {np.bincount(y[idx], minlength=19).tolist()}")
    xtr_all, ytr_all = parts["train"]
    x_unlab = xtr_all[rng.choice(len(xtr_all), min(20000, len(xtr_all)), replace=False)]
    xbg = xtr_all[ytr_all == BACKGROUND]
    train_counts = np.bincount(ytr_all, minlength=19)

    rows = []
    for ks in known_sets(args, rng):
        if min(train_counts[list(ks)]) < args.min_train:
            continue
        tr = np.isin(ytr_all, ks)
        xv, yv = parts["val"]
        xt, yt = parts["test"]
        for method in args.methods:
            xneg = xtr_all[~tr] if method == "oracle" else xbg
            f = scores_for(method, xtr_all[tr], ytr_all[tr], x_unlab, xneg, args)
            sv, _ = f(xv[np.isin(yv, ks)])
            thr = np.quantile(sv, 1 - args.tpr)
            st, pred = f(xt)
            kn = np.isin(yt, ks)
            crop_unk = ~kn & (yt != BACKGROUND)
            bg_unk = yt == BACKGROUND
            row = {
                "product": args.product,
                "level": args.level,
                "K": len(ks),
                "known": "-".join(map(str, ks)),
                "method": method,
                "closed_acc": float((pred[kn] == yt[kn]).mean()),
                "known_kept_test": float((st[kn] >= thr).mean()),
                "auroc_crop": roc_auc_score(
                    np.r_[np.ones(kn.sum()), np.zeros(crop_unk.sum())],
                    np.r_[st[kn], st[crop_unk]],
                ),
                "auroc_bg": roc_auc_score(
                    np.r_[np.ones(kn.sum()), np.zeros(bg_unk.sum())],
                    np.r_[st[kn], st[bg_unk]],
                ),
                "fpr_crop": fpr_at_tpr(st[kn], st[crop_unk], args.tpr),
                "fpr_bg": fpr_at_tpr(st[kn], st[bg_unk], args.tpr),
                # open-set accuracy: known kept and correct, unknown rejected
                "open_acc": float(
                    np.r_[(st[kn] >= thr) & (pred[kn] == yt[kn]), st[~kn] < thr].mean()
                ),
            }
            for c in range(19):
                if c not in ks and (yt == c).any():
                    row[f"rej_{c}"] = float((st[yt == c] < thr).mean())
            rows.append(row)
        logger.info(f"known {ks} done")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    summary = df.groupby(["K", "method"])[
        [
            "closed_acc",
            "known_kept_test",
            "auroc_crop",
            "fpr_crop",
            "auroc_bg",
            "fpr_bg",
            "open_acc",
        ]
    ].mean()
    pd.set_option("display.width", 200)
    print(f"=== {args.product} / {args.level} ===")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
