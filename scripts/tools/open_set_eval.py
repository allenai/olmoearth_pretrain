r"""Open-set ("unknown class") rejection on frozen embeddings, scored on PASTIS.

Reads the .npz from dump_pastis_open_set_pixels.py. For each known set (a few
of the 18 PASTIS crop classes) a readout is fit on the train split's known
classes; every other class is unknown. Each method gives a score where higher
means "more likely known", plus a closed-set prediction. Methods, by family:

Post-hoc on a softmax linear probe
- msp / maxlogit / energy (Hendrycks & Gimpel 2017; Hendrycks et al. 2022;
  Liu et al. 2020).
- react_energy: energy after clipping every embedding dim at its 90th train
  percentile (ReAct, Sun et al. 2021, applied to the embedding itself).
- conformal: class-conditional split conformal on probe probabilities,
  calibrated on known val; score = max_c (p_c - q_c), so "every class's
  prediction set is empty" <=> score < 0.
- vim: ViM (Wang et al. 2022), energy vs a virtual logit from the residual
  outside the principal subspace of the train embeddings.

Feature-space density / distance (no classifier)
- maha: min Mahalanobis distance to a class mean, shared Ledoit-Wolf
  covariance (Lee et al. 2018).
- rmaha_known / rmaha_unlab: relative Mahalanobis (Ren et al. 2021), minus the
  distance under one Gaussian fit to the known train data, or to unlabeled
  train pixels of every class (unlabeled imagery is free in remote sensing).
- qda: min Mahalanobis with a per-class Ledoit-Wolf covariance.
- gmm: max over classes of a per-class Gaussian-mixture log-likelihood.
- residual: minus the PCA reconstruction error on the known train data.
- knn1 / knn10 / knn50: minus the distance to the k-th nearest known train
  embedding after L2 normalisation (Sun et al. 2022).
- cknn: min over classes of the 10th-nearest-neighbour distance within that
  class.
- ocsvm: max over classes of a per-class one-class RBF SVM.
- iforest: isolation forest on the known train data.

Unlabeled data
- cluster{50,200}: k-means on unlabeled train pixels; score = labelled known
  train pixels per unlabeled pixel in the test point's cluster, so clusters
  with no labelled member score 0 ("novel class discovery" as rejection).
- unlab_neg: a (K+1)-class probe with unlabeled train pixels as "other"
  (positive-unlabeled: the pool also contains the known classes).

Ensembles
- mlp_ens / mlp_ens_mi: 5 small MLPs (different seeds); max mean probability,
  and minus the mutual information (disagreement).

Explicit negatives (outlier exposure); score = 1 - p(other)
- bg0: PASTIS background (label 0) as "other".
- bghalf: background plus a random half of the unknown crops as "other".
- oracle: every non-known train class as "other": the test-time unknowns were
  seen, so it is not an open-set method but the upper bound on known-vs-unknown
  separability with a linear readout.

Classes a method saw as negatives are recorded (neg_classes); auroc_crop_unseen
scores only the crop unknowns it never saw, which is the open-set number for
bg0/bghalf. Thresholds are set on val so that ``--tpr`` of known val samples
are kept; the fraction of each unknown class rejected at that threshold is
measured on test. AUROC / FPR@TPR are threshold-free, on test, split into crop
unknowns (classes 1-18) and background (class 0).

    python scripts/tools/open_set_eval.py --npz pastis_open_set.npz \
        --product olmoearth --level pixel --out results.csv
"""

import argparse
import itertools
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax
from sklearn.cluster import MiniBatchKMeans
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.svm import OneClassSVM

logger = logging.getLogger(__name__)

CROPS = list(range(1, 19))
BACKGROUND = 0
OTHER = -1
METHODS = [
    "msp",
    "maxlogit",
    "energy",
    "react_energy",
    "conformal",
    "vim",
    "maha",
    "rmaha_known",
    "rmaha_unlab",
    "qda",
    "gmm",
    "residual",
    "knn1",
    "knn10",
    "knn50",
    "cknn",
    "ocsvm",
    "iforest",
    "cluster50",
    "cluster200",
    "unlab_neg",
    "mlp_ens",
    "mlp_ens_mi",
    "bg0",
    "bghalf",
    "oracle",
]

Scorer = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass
class Context:
    """Everything a method may fit on for one known set."""

    xtr: np.ndarray  # known-class train embeddings
    ytr: np.ndarray
    xval: np.ndarray  # known-class val embeddings (conformal calibration only)
    yval: np.ndarray
    x_unlab: np.ndarray  # unlabeled train pixels of every class
    xneg: np.ndarray  # explicit negatives for bg0 / bghalf / oracle
    kmeans: dict[
        int, tuple[MiniBatchKMeans, np.ndarray]
    ]  # n -> (model, unlab assignment)
    seed: int
    c: float


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


def l2n(a: np.ndarray) -> np.ndarray:
    """Rows scaled to unit L2 norm."""
    return a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-8)


def fit_probe(xtr: np.ndarray, ytr: np.ndarray, c: float):
    """A softmax probe; returns (logits function, classes, W, b) with W (K, D)."""
    clf = LogisticRegression(C=c, max_iter=2000).fit(xtr, ytr)
    w, b = clf.coef_, clf.intercept_
    if len(clf.classes_) == 2:  # binary: sklearn keeps one logit; recentre to two
        w, b = np.stack([-w[0] / 2, w[0] / 2]), np.array([-b[0] / 2, b[0] / 2])

    def logits(x: np.ndarray) -> np.ndarray:
        """(n, K) logits."""
        return x @ w.T + b

    return logits, clf.classes_, w, b


def mahalanobis_sq(x: np.ndarray, mean: np.ndarray, prec: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distance of each row of x to mean."""
    diff = x - mean
    return np.einsum("nd,de,ne->n", diff, prec, diff)


def negatives_probe(ctx: Context, xneg: np.ndarray) -> Scorer:
    """(K+1)-class probe with xneg as "other"; score = 1 - p(other)."""
    xx = np.concatenate([ctx.xtr, xneg])
    yy = np.concatenate([ctx.ytr, np.full(len(xneg), OTHER)])
    logits, classes, _, _ = fit_probe(xx, yy, ctx.c)
    o = int(np.nonzero(classes == OTHER)[0][0])

    def f(x: np.ndarray):
        """Score and closed-set prediction among the known classes."""
        p = softmax(logits(x), axis=1)
        p_known = p.copy()
        p_known[:, o] = -1
        return 1 - p[:, o], classes[p_known.argmax(axis=1)]

    return f


def probe_scorer(ctx: Context, method: str) -> Scorer:
    """Msp / maxlogit / energy / react_energy / conformal / vim."""
    xtr = ctx.xtr
    clip = np.quantile(xtr, 0.9, axis=0) if method == "react_energy" else None
    if clip is not None:
        xtr = np.minimum(xtr, clip)
    logits, classes, w, b = fit_probe(xtr, ctx.ytr, ctx.c)

    if method == "conformal":
        # per-class threshold: 5% of known val samples of class c fall below q_c
        pv = softmax(logits(ctx.xval), axis=1)
        q = np.array(
            [
                np.quantile(pv[ctx.yval == c, i], 0.05)
                if (ctx.yval == c).any()
                else 0.5
                for i, c in enumerate(classes)
            ]
        )
    if method == "vim":
        # origin u with W u + b = 0; principal subspace of train features about u
        u = -np.linalg.pinv(w) @ b
        dim = max(1, xtr.shape[1] // 2)
        pca = PCA(n_components=dim).fit(xtr - u)
        basis = pca.components_

        def residual(x: np.ndarray) -> np.ndarray:
            """Norm of the part of x - u outside the principal subspace."""
            z = x - u
            return np.linalg.norm(z - (z @ basis.T) @ basis, axis=1)

        alpha = logits(xtr).max(axis=1).mean() / residual(xtr).mean().clip(1e-8)

    def f(x: np.ndarray):
        """Score and closed-set prediction."""
        if clip is not None:
            x = np.minimum(x, clip)
        z = logits(x)
        pred = classes[z.argmax(axis=1)]
        if method == "msp":
            return softmax(z, axis=1).max(axis=1), pred
        if method == "maxlogit":
            return z.max(axis=1), pred
        if method in ("energy", "react_energy"):
            return logsumexp(z, axis=1), pred
        if method == "conformal":
            return (softmax(z, axis=1) - q).max(axis=1), pred
        v = alpha * residual(x)
        return -softmax(np.c_[z, v], axis=1)[:, -1], pred

    return f


def density_scorer(ctx: Context, method: str) -> Scorer:
    """Maha / rmaha_* / qda / gmm / residual / ocsvm / iforest."""
    xtr, ytr = ctx.xtr, ctx.ytr
    known = np.unique(ytr)
    by_class = [xtr[ytr == c] for c in known]

    if method in ("maha", "rmaha_known", "rmaha_unlab", "qda"):
        means = np.stack([a.mean(axis=0) for a in by_class])
        if method == "qda":
            precs = [LedoitWolf().fit(a).precision_ for a in by_class]
        else:
            centred = xtr - means[np.searchsorted(known, ytr)]
            precs = [LedoitWolf().fit(centred).precision_] * len(known)
        ref = None
        if method.startswith("rmaha"):
            lw = LedoitWolf().fit(xtr if method == "rmaha_known" else ctx.x_unlab)
            ref = (lw.location_, lw.precision_)

        def f(x: np.ndarray):
            """Score and closed-set prediction (nearest class)."""
            d = np.stack(
                [mahalanobis_sq(x, m, p) for m, p in zip(means, precs)], axis=1
            )
            if ref is not None:
                d = d - mahalanobis_sq(x, *ref)[:, None]
            return -d.min(axis=1), known[d.argmin(axis=1)]

        return f

    if method == "gmm":
        reg = 1e-2 * float(xtr.var(axis=0).mean())
        gmms = [
            GaussianMixture(
                n_components=min(4, max(1, len(a) // 50)),
                covariance_type="full",
                reg_covar=reg,
                random_state=ctx.seed,
            ).fit(a)
            for a in by_class
        ]

        def f(x: np.ndarray):
            """Score and closed-set prediction (most likely class)."""
            ll = np.stack([g.score_samples(x) for g in gmms], axis=1)
            return ll.max(axis=1), known[ll.argmax(axis=1)]

        return f

    if method == "ocsvm":
        svms = [OneClassSVM(nu=0.05, gamma="scale").fit(a) for a in by_class]

        def f(x: np.ndarray):
            """Score and closed-set prediction (most inlying class)."""
            s = np.stack([m.decision_function(x) for m in svms], axis=1)
            return s.max(axis=1), known[s.argmax(axis=1)]

        return f

    # residual / iforest: class-agnostic scores; predict with a probe
    logits, classes, _, _ = fit_probe(xtr, ytr, ctx.c)
    if method == "residual":
        pca = PCA(n_components=max(1, xtr.shape[1] // 4)).fit(xtr)

        def score(x: np.ndarray) -> np.ndarray:
            """Minus PCA reconstruction error."""
            return -np.linalg.norm(x - pca.inverse_transform(pca.transform(x)), axis=1)

    else:
        forest = IsolationForest(random_state=ctx.seed).fit(xtr)
        score = forest.score_samples

    def f(x: np.ndarray):
        """Score and probe prediction."""
        return score(x), classes[logits(x).argmax(axis=1)]

    return f


def knn_scorer(ctx: Context, method: str) -> Scorer:
    """knn{k} (global) / cknn (per class, k=10), cosine on L2-normalised rows."""
    xtr, ytr = l2n(ctx.xtr), ctx.ytr
    known = np.unique(ytr)
    if method == "cknn":
        nns = [
            NearestNeighbors(n_neighbors=min(10, int((ytr == c).sum()))).fit(
                xtr[ytr == c]
            )
            for c in known
        ]

        def f(x: np.ndarray):
            """Score and closed-set prediction (closest class)."""
            d = np.stack([nn.kneighbors(l2n(x))[0][:, -1] for nn in nns], axis=1)
            return -d.min(axis=1), known[d.argmin(axis=1)]

        return f

    nn = NearestNeighbors(n_neighbors=int(method[3:])).fit(xtr)

    def f(x: np.ndarray):
        """Score and majority-vote prediction."""
        dist, idx = nn.kneighbors(l2n(x))
        votes = ytr[idx]
        pred = np.array([np.bincount(v).argmax() for v in votes])
        return -dist[:, -1], pred

    return f


def cluster_scorer(ctx: Context, method: str) -> Scorer:
    """Known-label density of the test point's unlabeled k-means cluster."""
    model, unlab_assign = ctx.kmeans[int(method[len("cluster") :])]
    n = model.n_clusters
    sizes = np.bincount(unlab_assign, minlength=n).clip(1)
    known = np.unique(ctx.ytr)
    tr_assign = model.predict(ctx.xtr)
    counts = np.stack(
        [np.bincount(tr_assign[ctx.ytr == c], minlength=n) for c in known], axis=1
    )
    density = counts.sum(axis=1) / sizes
    majority = known[counts.argmax(axis=1)]

    def f(x: np.ndarray):
        """Score and the cluster's majority known label."""
        a = model.predict(x)
        return density[a], majority[a]

    return f


def mlp_scorer(ctx: Context, method: str) -> Scorer:
    """Ensemble of 5 small MLP probes: mean max-probability or minus mutual info."""
    members = [
        MLPClassifier(
            hidden_layer_sizes=(256,),
            early_stopping=True,
            max_iter=200,
            random_state=ctx.seed + i,
        ).fit(ctx.xtr, ctx.ytr)
        for i in range(5)
    ]
    classes = members[0].classes_

    def f(x: np.ndarray):
        """Score and ensemble prediction."""
        p = np.stack([m.predict_proba(x) for m in members])  # (M, n, K)
        pm = p.mean(axis=0)
        pred = classes[pm.argmax(axis=1)]
        if method == "mlp_ens":
            return pm.max(axis=1), pred

        def ent(q: np.ndarray) -> np.ndarray:
            """Entropy along the last axis."""
            return -(q * np.log(q.clip(1e-12))).sum(axis=-1)

        return -(ent(pm) - ent(p).mean(axis=0)), pred

    return f


def scores_for(method: str, ctx: Context) -> Scorer:
    """Fit one method on the context; returns score(x) -> (score, prediction)."""
    if method in ("msp", "maxlogit", "energy", "react_energy", "conformal", "vim"):
        return probe_scorer(ctx, method)
    if method in (
        "maha",
        "rmaha_known",
        "rmaha_unlab",
        "qda",
        "gmm",
        "residual",
        "ocsvm",
        "iforest",
    ):
        return density_scorer(ctx, method)
    if method.startswith("knn") or method == "cknn":
        return knn_scorer(ctx, method)
    if method.startswith("cluster"):
        return cluster_scorer(ctx, method)
    if method.startswith("mlp_ens"):
        return mlp_scorer(ctx, method)
    if method == "unlab_neg":
        return negatives_probe(ctx, ctx.x_unlab)
    if method in ("bg0", "bghalf", "oracle"):
        return negatives_probe(ctx, ctx.xneg)
    raise ValueError(method)


def fpr_at_tpr(s_known: np.ndarray, s_unknown: np.ndarray, tpr: float) -> float:
    """Fraction of unknowns kept at the threshold that keeps ``tpr`` of knowns."""
    if len(s_unknown) == 0:
        return float("nan")
    thr = np.quantile(s_known, 1 - tpr)
    return float((s_unknown >= thr).mean())


def auroc(s_known: np.ndarray, s_unknown: np.ndarray) -> float:
    """AUROC of known (positive) vs unknown; NaN without unknowns."""
    if len(s_unknown) == 0:
        return float("nan")
    return float(
        roc_auc_score(
            np.r_[np.ones(len(s_known)), np.zeros(len(s_unknown))],
            np.r_[s_known, s_unknown],
        )
    )


def known_sets(args: argparse.Namespace, rng: np.random.Generator) -> list[tuple]:
    """Every pair of crop classes for K=2; random draws for larger K."""
    sets = []
    for k in args.ks:
        if k == 2:
            sets += list(itertools.combinations(CROPS, 2))
        else:
            sets += [
                tuple(int(c) for c in sorted(rng.choice(CROPS, k, replace=False)))
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
    xv, yv = parts["val"]
    xt, yt = parts["test"]
    x_unlab = xtr_all[rng.choice(len(xtr_all), min(20000, len(xtr_all)), replace=False)]
    train_counts = np.bincount(ytr_all, minlength=19)
    kmeans = {}
    for n in sorted(
        {int(m[len("cluster") :]) for m in args.methods if m.startswith("cluster")}
    ):
        model = MiniBatchKMeans(n_clusters=n, random_state=args.seed, n_init=3).fit(
            x_unlab
        )
        kmeans[n] = (model, model.predict(x_unlab))

    rows = []
    elapsed: dict[str, float] = {m: 0.0 for m in args.methods}
    for ks in known_sets(args, rng):
        if min(train_counts[list(ks)]) < args.min_train:
            continue
        tr = np.isin(ytr_all, ks)
        vk = np.isin(yv, ks)
        unknown_crops = [c for c in CROPS if c not in ks]
        half = sorted(
            int(c)
            for c in rng.choice(unknown_crops, len(unknown_crops) // 2, replace=False)
        )
        negs = {
            "bg0": [BACKGROUND],
            "bghalf": [BACKGROUND, *half],
            "oracle": [BACKGROUND, *unknown_crops],
        }
        for method in args.methods:
            t0 = time.time()
            neg_classes = negs.get(method, [])
            ctx = Context(
                xtr=xtr_all[tr],
                ytr=ytr_all[tr],
                xval=xv[vk],
                yval=yv[vk],
                x_unlab=x_unlab,
                xneg=xtr_all[np.isin(ytr_all, neg_classes)],
                kmeans=kmeans,
                seed=args.seed,
                c=args.c,
            )
            f = scores_for(method, ctx)
            sv, _ = f(xv[vk])
            thr = np.quantile(sv, 1 - args.tpr)
            st, pred = f(xt)
            kn = np.isin(yt, ks)
            crop_unk = ~kn & (yt != BACKGROUND)
            crop_unseen = crop_unk & ~np.isin(yt, neg_classes)
            bg_unk = yt == BACKGROUND
            row = {
                "product": args.product,
                "level": args.level,
                "K": len(ks),
                "known": "-".join(map(str, ks)),
                "method": method,
                "neg_classes": "-".join(map(str, neg_classes)),
                "closed_acc": float((pred[kn] == yt[kn]).mean()),
                "known_kept_test": float((st[kn] >= thr).mean()),
                "auroc_crop": auroc(st[kn], st[crop_unk]),
                "auroc_crop_unseen": auroc(st[kn], st[crop_unseen]),
                "auroc_bg": auroc(st[kn], st[bg_unk]),
                "fpr_crop": fpr_at_tpr(st[kn], st[crop_unk], args.tpr),
                "fpr_crop_unseen": fpr_at_tpr(st[kn], st[crop_unseen], args.tpr),
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
            elapsed[method] += time.time() - t0
        logger.info(f"known {ks} done")
        pd.DataFrame(rows).to_csv(args.out, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    logger.info(
        "seconds per method: "
        + ", ".join(
            f"{m}={t:.0f}" for m, t in sorted(elapsed.items(), key=lambda kv: -kv[1])
        )
    )
    summary = df.groupby(["K", "method"], sort=False)[
        [
            "closed_acc",
            "known_kept_test",
            "auroc_crop",
            "auroc_crop_unseen",
            "fpr_crop_unseen",
            "auroc_bg",
            "fpr_bg",
            "open_acc",
        ]
    ].mean()
    pd.set_option("display.width", 220)
    print(f"=== {args.product} / {args.level} ===")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
