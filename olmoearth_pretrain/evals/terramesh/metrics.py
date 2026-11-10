"""Per-tile metrics for zero-shot text segmentation, and bootstrap summaries."""

from __future__ import annotations

import numpy as np

METRICS = ("ap", "auroc", "pointing", "iou", "prevalence")

# Fixed background prompts for the relative score (ClipEarth's eval_pixels).
BACKGROUNDS = (
    "trees",
    "forest",
    "grass",
    "shrubs",
    "crop fields",
    "bare soil",
    "sand",
    "rocks",
    "water",
    "buildings",
    "road",
    "snow",
)


def ranking_metrics(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Average precision and AUROC of scores ``s`` for binary labels ``y``.

    Ties are grouped, as sklearn's average_precision_score and roc_auc_score treat them.
    """
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]  # end of each tied run
    tp = np.cumsum(y)[last]
    fp = (last + 1) - tp
    pos, neg = tp[-1], fp[-1]
    recall = tp / pos
    precision = tp / (last + 1)
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    fpr, tpr = np.r_[0.0, fp / neg], np.r_[0.0, recall]
    auroc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    return ap, auroc


def relative(cos_q: np.ndarray, cos_b: np.ndarray, scale: float) -> np.ndarray:
    """Sigmoid of the query's logit against the log-sum-exp of the backgrounds'.

    Args:
        cos_q: [N] cosines with the query.
        cos_b: [N, K] cosines with the background prompts.
        scale: The head's logit scale.
    """
    lse = np.logaddexp.reduce(scale * cos_b, axis=1)
    return 1 / (1 + np.exp(-(scale * cos_q - lse)))


def tile_metrics(
    score: np.ndarray, rel: np.ndarray, mask: np.ndarray, valid: np.ndarray
) -> dict[str, float]:
    """AP, AUROC, pointing, IoU and prevalence of one tile; {} if it can't be scored.

    Args:
        score: [H, W] raw score (cosine) with the query.
        rel: [H, W] relative score; pixels above 0.5 are the predicted mask.
        mask: [H, W] bool ground truth.
        valid: [H, W] bool pixels with an embedding.
    """
    y, s = mask[valid], score[valid]
    if y.all() or not y.any():
        return {}
    pred = rel[valid] > 0.5
    ap, auroc = ranking_metrics(y, s)
    top = np.unravel_index(np.argmax(np.where(valid, score, -np.inf)), score.shape)
    return {
        "ap": ap,
        "auroc": auroc,
        "pointing": float(mask[top]),
        "iou": float((pred & y).sum() / max(1, (pred | y).sum())),
        "prevalence": float(y.mean()),
    }


def summarize(
    rows: list[dict], rng: np.random.Generator, bootstrap: int = 1000
) -> dict:
    """Mean of each metric over tiles, with a 95% bootstrap interval: [mean, lo, hi]."""
    out: dict = {"tiles": len(rows)}
    if not rows:
        return out
    for m in METRICS:
        v = np.array([r[m] for r in rows])
        boot = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(bootstrap)]
        out[m] = [
            round(float(v.mean()), 4),
            round(float(np.percentile(boot, 2.5)), 4),
            round(float(np.percentile(boot, 97.5)), 4),
        ]
    return out
