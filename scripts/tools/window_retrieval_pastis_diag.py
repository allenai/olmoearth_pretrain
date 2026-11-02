"""Why do retrieved PASTIS embeddings substitute for fresh ones worse than the annual vector?

Reads the per-window int8 embeddings ``window_retrieval_pastis_bake.py embed``
wrote, rebuilds every variant with the same retriever and composition as
``bake``, and compares each substitute v with the fresh embedding f it stands
in for, per pixel (labelled, non-void pixels; val windows unless noted):

- ``cos``: mean cosine(v, f); ``gap_closed``: (cos_v - cos_annual) / (1 - cos_annual);
- ``rel_mse``: E||v - f||^2 / total variance of f;
- ``var_ratio``: total variance of v / total variance of f (< 1 = shrunk);
- ``slope``: least-squares slope of f on v, pooled over dims (> 1 = v compressed);
- ``mean_shift``: ||mean v - mean f|| / sqrt(total variance of f);
- ``norm_ratio``: median ||v|| / ||f||;
- class directions: LDA fit on fresh train pixels. ``err_in_lda``: share of the
  error energy ||v - f||^2 inside the LDA subspace, divided by the share of f's
  variance there (> 1 = errors concentrate where the classes are separated);
  ``lda_acc``: pixel accuracy of that fresh-trained LDA applied to v.
- classifiers trained on fresh train pixels and applied unchanged to v, scored by
  mIoU over the classes present (the LP's metric) and accuracy: the LDA, and a
  multinomial logistic regression on standardized inputs (closest to the LP);
  per-class IoU goes to the JSON.
- spectrum: the error ||v - f||^2 split over the principal components of fresh
  (val), as error energy / fresh variance per component, in bins of 8
  components (bin 0 = highest variance). Rising toward the last bins = v's
  error is relatively largest where fresh barely varies.
- penalty sweep: the logistic regression refit at each C in ``PENALTY_C``
  (larger C = weaker L2 penalty), mIoU on fresh and each substitute.

Usage::

    python scripts/tools/window_retrieval_pastis_diag.py --work_dir $W --retriever_dir $R --out $OUT
"""

import argparse
import glob
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from window_retrieval_pastis_bake import (  # noqa: E402
    SOURCES,
    TEMPLATE_DS,
    dequantize_embeddings,
    key_to_file,
    load_retriever,
    quantize_embeddings,
    variants_for,
)

logger = logging.getLogger(__name__)

VOID = 19
PENALTY_C = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
SPECTRUM_BIN = 8
PAIRS = {
    "fresh_maraug": ("annual2019", "ret_maraug"),
    "fresh_sepaug": ("annual2019", "avg_sepaug", "ret_sepaug"),
}


def read_label(window_dir: str) -> np.ndarray | None:
    """The window's PASTIS class raster [H, W], or None if not found."""
    import rasterio

    tifs = sorted(glob.glob(os.path.join(window_dir, "layers", "label", "*", "*.tif")))
    if not tifs:
        return None
    with rasterio.open(tifs[0]) as src:
        return src.read(1)


def window_split(window_dir: str) -> str | None:
    """The window's eval_split option."""
    with open(os.path.join(window_dir, "metadata.json")) as f:
        return json.load(f).get("options", {}).get("eval_split")


def collect(args: argparse.Namespace) -> dict[str, dict[str, np.ndarray]]:
    """Sampled pixels of every variant plus labels, per split."""
    model = load_retriever(args.retriever_dir)
    work = Path(args.work_dir)
    window_dirs = sorted(glob.glob(os.path.join(TEMPLATE_DS, "windows", "*", "*")))
    rng = np.random.default_rng(args.seed)
    seeds = rng.integers(0, 2**31, len(window_dirs))
    indomain = {}
    if args.indomain_dir:
        from window_retrieval_pastis_indomain import load_models

        indomain = load_models(args.indomain_dir, args.retriever_dir, "cpu")
    logger.info(
        f"{len(window_dirs)} windows; label files of the first: {glob.glob(os.path.join(window_dirs[0], 'layers', 'label', '**'), recursive=True)}"
    )

    def handle(i: int) -> tuple[str, dict[str, np.ndarray]] | None:
        wd = window_dirs[i]
        key = "/".join(Path(wd).parts[-2:])
        files = {s: work / s / key_to_file(key) for s in SOURCES}
        if not all(f.exists() for f in files.values()):
            return None
        split = window_split(wd)
        label = read_label(wd)
        if split is None or label is None:
            return None
        codes = {s: np.load(f) for s, f in files.items()}
        H, W, _ = codes["cal2019"].shape
        if label.shape != (H, W):
            return None
        ok = np.flatnonzero(label.reshape(-1) != VOID)
        if len(ok) == 0:
            return None
        pick = np.random.default_rng(seeds[i]).choice(
            ok, min(args.pixels, len(ok)), replace=False
        )
        # Retrieval is per pixel, so retrieve only the sampled pixels.
        sub = {s: c.reshape(H * W, -1)[pick][:, None, :] for s, c in codes.items()}
        out = variants_for(sub, model)
        if indomain:
            from window_retrieval_pastis_indomain import predict

            z = {
                s: dequantize_embeddings(torch.from_numpy(c[:, 0, :])).float()
                for s, c in sub.items()
            }
            with torch.no_grad():
                for name, m in indomain.items():
                    out[name] = predict(name, m, z)[:, None, :]
        # int8 round trip, as bake writes every variant (store parity).
        row = {
            v: dequantize_embeddings(quantize_embeddings(e[:, 0, :]))
            .numpy()
            .astype(np.float32)
            for v, e in out.items()
        }
        row["label"] = label.reshape(-1)[pick].astype(np.int64)
        return split, row

    parts: dict[str, list] = {}
    torch.set_num_threads(max(1, args.workers // 4))
    with ThreadPoolExecutor(args.workers) as pool:
        for n, res in enumerate(pool.map(handle, range(len(window_dirs))), 1):
            if res is not None:
                parts.setdefault(res[0], []).append(res[1])
            if n % 200 == 0:
                logger.info(f"  {n}/{len(window_dirs)} windows")
    return {
        split: {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
        for split, rows in parts.items()
    }


def stats(v: np.ndarray, f: np.ndarray) -> dict[str, float]:
    """Geometry of substitute v against fresh f, both [N, D]."""
    vn, fn = np.linalg.norm(v, axis=1), np.linalg.norm(f, axis=1)
    cos = (v * f).sum(1) / np.maximum(vn * fn, 1e-8)
    vc, fc = v - v.mean(0), f - f.mean(0)
    tv_f = (fc**2).sum(1).mean()
    return {
        "cos": float(cos.mean()),
        "rel_mse": float(((v - f) ** 2).sum(1).mean() / tv_f),
        "var_ratio": float((vc**2).sum(1).mean() / tv_f),
        "slope": float((vc * fc).sum() / (vc**2).sum()),
        "mean_shift": float(np.linalg.norm(v.mean(0) - f.mean(0)) / np.sqrt(tv_f)),
        "norm_ratio": float(np.median(vn / np.maximum(fn, 1e-8))),
    }


def spectrum(err: np.ndarray, eigvecs: np.ndarray, eigvals: np.ndarray) -> list[float]:
    """Error energy / fresh variance per principal component, averaged in bins."""
    ratio = ((err @ eigvecs) ** 2).mean(0) / np.maximum(eigvals, 1e-12)
    return [
        float(ratio[i : i + SPECTRUM_BIN].mean())
        for i in range(0, len(ratio), SPECTRUM_BIN)
    ]


def class_iou(pred: np.ndarray, label: np.ndarray) -> dict[int, float]:
    """IoU of every class present in ``label``."""
    return {
        int(c): float(
            ((pred == c) & (label == c)).sum() / ((pred == c) | (label == c)).sum()
        )
        for c in np.unique(label)
    }


def main() -> None:
    """Collect, compare, write JSON and a log table."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--retriever_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--pixels", type=int, default=256, help="labelled pixels sampled per window"
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--indomain_dir",
        help="also score window_retrieval_pastis_indomain.py's retrievers (paired with their fresh target)",
    )
    args = parser.parse_args()

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data = collect(args)
    logger.info({s: len(d["label"]) for s, d in data.items()})
    train, val = data["train"], data["val"]

    results: dict = {}
    if args.indomain_dir:
        PAIRS["fresh_maraug"] += ("ind_maraug_ft", "ind_maraug_scratch")
        PAIRS["fresh_sepaug"] += ("ind_sepaug_compft", "ind_sepaug_mlp")
    for fresh, subs in PAIRS.items():
        lda = LinearDiscriminantAnalysis().fit(train[fresh], train["label"])
        logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(
            train[fresh], train["label"]
        )
        classifiers = {"lda": lda, "logreg": logreg}
        # Orthonormal basis of the discriminant subspace, in embedding coordinates.
        basis, _ = np.linalg.qr(
            lda.scalings_[:, : lda.explained_variance_ratio_.shape[0]]
        )
        f = val[fresh]
        fc = f - f.mean(0)
        eigvals, eigvecs = np.linalg.eigh(fc.T @ fc / len(fc))
        order = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        f_share = ((fc @ basis) ** 2).sum() / (fc**2).sum()
        results[fresh] = {
            "fresh_lda_acc": float(lda.score(f, val["label"])),
            "fresh_var_share_in_lda": float(f_share),
            "class_pixels": {
                int(c): int(n)
                for c, n in zip(*np.unique(val["label"], return_counts=True))
            },
        }
        for name, clf in classifiers.items():
            iou = class_iou(clf.predict(f), val["label"])
            results[fresh][f"fresh_{name}_miou"] = float(np.mean(list(iou.values())))
            results[fresh][f"fresh_{name}_iou"] = iou
        cos_annual = None
        for v_name in subs:
            v = val[v_name]
            s = stats(v, f)
            if v_name == "annual2019":
                cos_annual = s["cos"]
            s["gap_closed"] = (
                (s["cos"] - cos_annual) / (1 - cos_annual)
                if cos_annual is not None
                else float("nan")
            )
            err = v - f
            s["err_in_lda"] = float(
                ((err @ basis) ** 2).sum() / (err**2).sum() / f_share
            )
            s["lda_acc"] = float(lda.score(v, val["label"]))
            for name, clf in classifiers.items():
                pred = clf.predict(v)
                iou = class_iou(pred, val["label"])
                s[f"{name}_miou"] = float(np.mean(list(iou.values())))
                s[f"{name}_iou"] = iou
            s["logreg_acc"] = float((pred == val["label"]).mean())
            s["spectrum"] = spectrum(v - f, eigvecs, eigvals)
            results[fresh][v_name] = s

        sweep = {}
        for c in PENALTY_C:
            clf = make_pipeline(
                StandardScaler(), LogisticRegression(C=c, max_iter=1000)
            ).fit(train[fresh], train["label"])
            sweep[c] = {
                v_name: float(
                    np.mean(
                        list(class_iou(clf.predict(val[v_name]), val["label"]).values())
                    )
                )
                for v_name in (fresh, *subs)
            }
            logger.info(
                f"{fresh} C={c:g}: "
                + " ".join(f"{k} {m:.4f}" for k, m in sweep[c].items())
            )
        results[fresh]["penalty_sweep"] = sweep

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    cols = (
        "cos",
        "gap_closed",
        "rel_mse",
        "var_ratio",
        "slope",
        "mean_shift",
        "norm_ratio",
        "err_in_lda",
        "lda_acc",
        "lda_miou",
        "logreg_acc",
        "logreg_miou",
    )
    for fresh, r in results.items():
        logger.info(
            f"{fresh}: fresh LDA acc {r['fresh_lda_acc']:.4f}, fresh var share in LDA {r['fresh_var_share_in_lda']:.3f}, "
            f"fresh LDA mIoU {r['fresh_lda_miou']:.4f}, fresh logreg mIoU {r['fresh_logreg_miou']:.4f}"
        )
        logger.info("  " + " ".join(f"{c:>10s}" for c in ("variant",) + cols))
        for v_name in PAIRS[fresh]:
            logger.info(
                "  "
                + f"{v_name:>10s} "
                + " ".join(f"{r[v_name][c]:10.4f}" for c in cols)
            )
        # Per-class logreg IoU: fresh, then each substitute minus fresh.
        logger.info(
            "  logreg IoU by class (pixels): fresh | "
            + " | ".join(PAIRS[fresh])
            + " (minus fresh)"
        )
        for c, n in r["class_pixels"].items():
            base = r["fresh_logreg_iou"][c]
            deltas = " ".join(
                f"{r[v]['logreg_iou'][c] - base:+.3f}" for v in PAIRS[fresh]
            )
            logger.info(f"    class {c:2d} ({n:6d}): {base:.3f} | {deltas}")
        logger.info(
            "  spectrum (error / fresh variance, bins of 8 PCs, high variance first):"
        )
        for v_name in PAIRS[fresh]:
            logger.info(
                f"    {v_name:>12s} "
                + " ".join(f"{x:.3f}" for x in r[v_name]["spectrum"])
            )
    logger.info(f"wrote {args.out}")


if __name__ == "__main__":
    main()
