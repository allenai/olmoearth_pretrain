"""Calibrate a native aligner's cosine scores into match probabilities.

Contrastive training ranks texts against each other, so a raw cosine has no absolute
meaning. This fits ``p = sigmoid(a * cosine + b)`` on held-out (region, text) pairs: a
region with each text view of its own class is a positive, with every other class's
views a negative, the two sides weighted equally. A page can then show where a query
is likely present rather than the top percentiles of whatever is in view.

Run:
  python -m olmoearth_pretrain.clipearth.calibrate
      --emb_dir <embed out_dir> --aligner <run>/aligner.pt --out <file>.json
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import torch

from olmoearth_pretrain.clipearth.train import (
    CATEGORIES,
    GENERIC_LABELS,
    VISDESC_KEY,
    Aligner,
    encode_texts,
    is_holdout,
    load_regions,
    norm_key,
    text_views,
)

logger = logging.getLogger(__name__)


def fit_logistic(x: np.ndarray, y: np.ndarray, steps: int = 200) -> tuple[float, float]:
    """Class-balanced one-feature logistic regression by Newton's method."""
    w = np.where(y == 1, 0.5 / max(y.sum(), 1), 0.5 / max((1 - y).sum(), 1))
    a, b = 1.0, 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(a * x + b)))
        g = w * (p - y)
        h = w * p * (1 - p)
        grad = np.array([(g * x).sum(), g.sum()])
        hess = np.array([[(h * x * x).sum(), (h * x).sum()], [(h * x).sum(), h.sum()]])
        step = np.linalg.solve(hess + 1e-9 * np.eye(2), grad)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-8:
            break
    return float(a), float(b)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Probability a random positive outscores a random negative."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pairs(
    image: torch.Tensor,
    labels: np.ndarray,
    text: torch.Tensor,
    text_class: np.ndarray,
    background: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Score of every region with every text view among the regions' classes.

    With ``background`` (mean and covariance of unit image embeddings), each text's
    cosine becomes a z-score against typical land, so texts on different scales share
    one cutoff.
    """
    keep = np.isin(text_class, np.unique(labels))
    kept = text[torch.from_numpy(keep)].numpy()
    scores = image.numpy() @ kept.T
    if background is not None:
        mean, cov = background
        scores = (scores - kept @ mean) / np.sqrt(
            np.einsum("ti,ij,tj->t", kept, cov, kept)
        )
    match = labels[:, None] == text_class[keep][None, :]
    return scores.ravel(), match.ravel().astype(np.float64)


def summarize(x: np.ndarray, y: np.ndarray, a: float, b: float) -> dict[str, float]:
    """Separation and the precision/recall of p >= 0.5 at the pairs' own base rate."""
    p = 1 / (1 + np.exp(-(a * x + b)))
    hit = p >= 0.5
    return {
        "auroc": auroc(x, y),
        "recall_at_0.5": float(hit[y == 1].mean()),
        "false_positive_rate_at_0.5": float(hit[y == 0].mean()),
        "precision_at_0.5": float(y[hit].mean()) if hit.any() else float("nan"),
        "positive_rate": float(y.mean()),
    }


FALSE_POSITIVE_RATES = (0.01, 0.005, 0.001)


def at_false_positive_rates(x: np.ndarray, y: np.ndarray) -> dict[str, dict]:
    """The cosine cutoff that flags each fraction of non-matches, and its recall."""
    negatives = np.sort(x[y == 0])
    out = {}
    for rate in FALSE_POSITIVE_RATES:
        cutoff = float(negatives[int(len(negatives) * (1 - rate))])
        out[str(rate)] = {
            "cosine": cutoff,
            "recall": float((x[y == 1] >= cutoff).mean()),
        }
    return out


def main() -> None:
    """Fit and report the calibration."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--aligner", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dims", type=int, default=None)
    parser.add_argument(
        "--background",
        help="npz with mean_<dims>, cov_<dims> of unit image embeddings; calibrates "
        "z-scores",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    ckpt = torch.load(args.aligner, map_location="cpu", weights_only=False)
    a_ = ckpt["args"]
    if not a_.get("native"):
        raise ValueError("calibration is for native aligners")
    data = load_regions(args.emb_dir, a_["embedding"])
    keep = np.array([norm_key(c) not in GENERIC_LABELS for c in data["class"]])
    data = {k: v[keep] for k, v in data.items()}
    unseen = np.array([is_holdout(s, a_["holdout_frac"]) for s in data["slug"]])
    val = (data["split"] == "val") & ~unseen

    class_names = ckpt["class_names"]
    index = {c: i for i, c in enumerate(class_names)}
    labels = np.array([index.get(c, -1) for c in data["class"]])
    visdesc = json.load(open(CATEGORIES)).get(VISDESC_KEY, {})
    texts, text_class = [], []
    for i, name in enumerate(class_names):
        for view in text_views(name, visdesc):
            texts.append(view)
            text_class.append(i)
    view_emb = torch.from_numpy(
        encode_texts(
            texts,
            a_["text_model"],
            "cpu",
            a_.get("text_prefix", ""),
            a_.get("text_dims"),
            a_.get("text_onnx_file", "model_q4.onnx"),
        )
    )
    model = Aligner(
        data["embedding"].shape[1],
        view_emb.shape[1],
        a_["dim"],
        a_["hidden"],
        True,
        a_.get("text_linear", False),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        image = model.embed_image(torch.from_numpy(data["embedding"]), args.dims)
        text = model.embed_text(view_emb, args.dims)

    text_class_np = np.array(text_class)
    background = None
    if args.background:
        stats = np.load(args.background)
        dims = int(image.shape[1])
        background = (stats[f"mean_{dims}"], stats[f"cov_{dims}"])
    x_val, y_val = pairs(image[val], labels[val], text, text_class_np, background)
    a, b = fit_logistic(x_val, y_val)
    x_unseen, y_unseen = pairs(
        image[unseen], labels[unseen], text, text_class_np, background
    )
    report = {
        "a": a,
        "b": b,
        "dims": args.dims or int(image.shape[1]),
        "score": "z" if background is not None else "cosine",
        "cosine_at_0.5": -b / a,
        "val": summarize(x_val, y_val, a, b),
        "unseen": summarize(x_unseen, y_unseen, a, b),
        "val_at_false_positive_rate": at_false_positive_rates(x_val, y_val),
        "unseen_at_false_positive_rate": at_false_positive_rates(x_unseen, y_unseen),
    }
    logger.info(json.dumps(report, indent=1))
    json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
