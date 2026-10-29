"""Calibrate a native aligner's cosine scores into match probabilities.

Contrastive training ranks texts against each other, so a raw cosine has no absolute
meaning. This fits ``p = sigmoid(a * cosine + b)`` on held-out (image, text) pairs, the
two sides weighted equally. A page can then show where a query is likely present rather
than the top percentiles of whatever is in view.

``--source class``: a region with each text view of its own class is a positive, with
every other class's views a negative. ``val`` regions fit, ``unseen`` datasets report.

``--source caption``: chips held out by geography in training, each with its own texts
(every level) as positives and other chips' texts as negatives. Another chip's text is
often also true here ("dense forest"), so a negative whose text is nearly the same as one
of the chip's own (``--ignore_similar`` in the frozen text space) is dropped rather than
counted as a false match. The held-out chips are split in two by id: one half fits and
is reported as ``val``, the other is reported as ``unseen`` and sets the cutoffs
``export_text_head`` writes. ``--include_pixels`` adds a random sample of the run's pixel
samples (``--pixels``), each with its own texts; pixels of one class share their texts,
so they are dropped as each other's negatives by the same rule.

A run trained with ``--context`` scores every sample with its context mean: a chip's own
mean for chips and quadrants, the stored context for pixel samples, and a class region's
own mean for regions.

Run:
  python -m olmoearth_pretrain.clipearth.calibrate
      --emb_dir <embed out_dir> --aligner <run>/aligner.pt --out <file>.json
  python -m olmoearth_pretrain.clipearth.calibrate --source caption
      --aligner <run>/aligner.pt --out <file>.json  # captions from the run's args
"""

from __future__ import annotations

import argparse
import hashlib
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
    load_pixel_samples,
    load_regions,
    native_view,
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


def score_matrix(
    image: np.ndarray,
    text: np.ndarray,
    background: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Cosine of every image with every text, ``[N, T]``.

    With ``background`` (mean and covariance of unit image embeddings), each text's
    cosine becomes a z-score against typical land, so texts on different scales share
    one cutoff.
    """
    scores = image @ text.T
    if background is not None:
        mean, cov = background
        scores = (scores - text @ mean) / np.sqrt(
            np.einsum("ti,ij,tj->t", text, cov, text)
        )
    return scores


def pairs(
    image: torch.Tensor,
    labels: np.ndarray,
    text: torch.Tensor,
    text_class: np.ndarray,
    background: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Score of every region with every text view among the regions' classes."""
    keep = np.isin(text_class, np.unique(labels))
    scores = score_matrix(
        image.numpy(), text[torch.from_numpy(keep)].numpy(), background
    )
    match = labels[:, None] == text_class[keep][None, :]
    return scores.ravel(), match.ravel().astype(np.float64)


def caption_pairs(
    image: np.ndarray,
    owner: np.ndarray,
    text: np.ndarray,
    raw_text: np.ndarray,
    ignore_similar: float,
    background: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Score of every chip with every text, dropping ambiguous negatives.

    Args:
        image: ``[N, D]`` chip embeddings.
        owner: ``[T]`` index of the chip each text describes.
        text: ``[T, D]`` projected texts.
        raw_text: ``[T, E]`` the same texts in the frozen text space, unit length.
        ignore_similar: a negative is dropped when its text has at least this cosine
            with one of the chip's own texts.
        background: as in ``score_matrix``.

    Returns:
        Scores, labels, and the fraction of negative pairs dropped.
    """
    scores = score_matrix(image, text, background)
    match = owner[None, :] == np.arange(len(image))[:, None]
    text_sim = raw_text @ raw_text.T
    closest_own = np.full((len(image), len(owner)), -np.inf, dtype=np.float32)
    for j, i in enumerate(owner):
        np.maximum(closest_own[i], text_sim[j], out=closest_own[i])
    ambiguous = ~match & (closest_own >= ignore_similar)
    keep = ~ambiguous
    dropped = float(ambiguous.sum() / max((~match).sum(), 1))
    return scores[keep], match[keep].astype(np.float64), dropped


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


def encode_with(a_: dict, texts: list[str]) -> torch.Tensor:
    """Frozen text embeddings, encoded as the run encoded them."""
    return torch.from_numpy(
        encode_texts(
            texts,
            a_["text_model"],
            "cpu",
            a_.get("text_prefix", ""),
            a_.get("text_dims"),
            a_.get("text_onnx_file", "model_q4.onnx"),
            a_.get("text_cache") or None,
        )
    )


def load_model(ckpt: dict, image_dim: int, text_dim: int) -> Aligner:
    """The trained native aligner."""
    a_ = ckpt["args"]
    context_widths = sorted(
        int(key.split(".")[1])
        for key in ckpt["state_dict"]
        if key.startswith("context.") and key.endswith(".weight")
    )
    model = Aligner(
        image_dim,
        text_dim,
        a_["dim"],
        a_["hidden"],
        True,
        a_.get("text_linear", False),
        context_widths or None,
    )
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def embed_native_images(
    model: Aligner,
    ckpt: dict,
    embedding: np.ndarray,
    dims: int | None,
    context: np.ndarray | None = None,
) -> torch.Tensor:
    """Image embeddings as the run trained and the lab scores them.

    Centred on the run's global mean vector if it was trained with one. A context model
    takes ``context``, or each embedding as its own context.
    """
    x = torch.from_numpy(embedding)
    width = dims or x.shape[1]
    center = ckpt.get("center") or {}
    c = torch.from_numpy(center[width]) if width in center else None
    view = None
    if len(model.context):
        source = embedding if context is None else context
        view = native_view(torch.from_numpy(source), dims, c)
    return model.embed_image(native_view(x, dims, c), dims, view)


def load_background(
    path: str | None, dims: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Background mean and covariance for ``dims``, if a file is given."""
    if not path:
        return None
    stats = np.load(path)
    return stats[f"mean_{dims}"], stats[f"cov_{dims}"]


def in_fit_half(chip_id: str) -> bool:
    """Deterministic half of the held-out chips that fits the calibration."""
    digest = hashlib.md5(chip_id.encode(), usedforsecurity=False).hexdigest()
    return int(digest, 16) % 2 == 0


def calibrate_captions(args: argparse.Namespace, ckpt: dict) -> dict:
    """Calibration report from chips held out by geography."""
    from olmoearth_pretrain.clipearth.captions import (
        geo_holdout,
        load_caption_samples,
    )

    a_ = ckpt["args"]
    caption_files = (args.captions or a_.get("captions", "")).split(",")
    chip_files = (args.caption_chips or a_.get("caption_chips", "")).split(",")
    if not caption_files[0]:
        raise ValueError("no --captions given and none recorded in the aligner")
    levels = [lv for lv in a_.get("levels", "name,short,detailed").split(",") if lv]
    sources = {"chip", "quad"} if args.include_quads else {"chip"}
    chips, texts, owner, text_level = [], [], [], []
    for captions_path, chips_path in zip(caption_files, chip_files):
        for sample in load_caption_samples(
            chips_path, captions_path, a_.get("min_visibility", 0.5)
        ):
            if sample["source"] not in sources or not geo_holdout(
                sample["lat"], sample["lon"], a_.get("caption_holdout_frac", 0.1)
            ):
                continue
            chips.append(sample)
            for level in levels:
                for t in sample["levels"].get(level, []):
                    texts.append(t)
                    owner.append(len(chips) - 1)
                    text_level.append(level)
    rng = np.random.default_rng(0)
    pixel_files = [p for p in (args.pixels or a_.get("pixels", "")).split(",") if p]
    pixel_texts = (args.pixel_texts or a_.get("pixel_texts", "")).split(",")
    pixel_samples = []
    if args.include_pixels:
        for pixels_path, texts_path in zip(pixel_files, pixel_texts):
            pixels = load_pixel_samples(pixels_path, texts_path, args.max_pixels)
            pixel_samples += [
                {
                    "group": f"{pixels_path}/{pixels['id'][i]}",
                    "embedding": pixels["embedding"][i],
                    "context": pixels["context"][i],
                    "levels": pixels["texts"][int(pixels["text_id"][i])],
                }
                for i in range(len(pixels["id"]))
            ]
    if len(chips) > args.max_chips:
        chosen = np.sort(rng.choice(len(chips), args.max_chips, replace=False))
        remap = {int(c): i for i, c in enumerate(chosen)}
        keep_text = [j for j, o in enumerate(owner) if o in remap]
        chips = [chips[c] for c in chosen]
        texts = [texts[j] for j in keep_text]
        text_level = [text_level[j] for j in keep_text]
        owner = [remap[owner[j]] for j in keep_text]
    for sample in pixel_samples:
        chips.append(sample)
        for level in levels:
            for t in sample["levels"].get(level, []):
                texts.append(t)
                owner.append(len(chips) - 1)
                text_level.append(level)
    logger.info(
        f"{len(chips) - len(pixel_samples)} held-out chips, {len(pixel_samples)} "
        f"pixel samples, {len(texts)} texts"
    )

    raw = encode_with(a_, texts)
    embedding = np.stack([c["embedding"] for c in chips]).astype(np.float32)
    context = np.stack([c["context"] for c in chips]).astype(np.float32)
    model = load_model(ckpt, embedding.shape[1], raw.shape[1])
    with torch.no_grad():
        image = embed_native_images(model, ckpt, embedding, args.dims, context).numpy()
        text = model.embed_text(raw, args.dims).numpy()
    background = load_background(args.background, int(image.shape[1]))
    owner_a, level_a = np.array(owner), np.array(text_level)
    fit = np.array([in_fit_half(c["group"]) for c in chips])

    def half(select: np.ndarray, text_mask: np.ndarray | None = None):
        rows = np.flatnonzero(select)
        cols = np.flatnonzero(
            np.isin(owner_a, rows) & (text_mask if text_mask is not None else True)
        )
        position = {r: i for i, r in enumerate(rows)}
        return caption_pairs(
            image[rows],
            np.array([position[o] for o in owner_a[cols]]),
            text[cols],
            raw.numpy()[cols],
            args.ignore_similar,
            background,
        )

    x_fit, y_fit, dropped_fit = half(fit)
    a, b = fit_logistic(x_fit, y_fit)
    x_rep, y_rep, dropped_rep = half(~fit)
    report = {
        "source": "caption",
        "a": a,
        "b": b,
        "dims": args.dims or int(image.shape[1]),
        "score": "z" if background is not None else "cosine",
        "cosine_at_0.5": -b / a,
        "ignore_similar": args.ignore_similar,
        "dropped_negative_fraction": {"val": dropped_fit, "unseen": dropped_rep},
        "num_chips": {"val": int(fit.sum()), "unseen": int((~fit).sum())},
        "num_pixel_samples": len(pixel_samples),
        "val": summarize(x_fit, y_fit, a, b),
        "unseen": summarize(x_rep, y_rep, a, b),
        "val_at_false_positive_rate": at_false_positive_rates(x_fit, y_fit),
        "unseen_at_false_positive_rate": at_false_positive_rates(x_rep, y_rep),
        "unseen_by_level": {},
    }
    for level in levels:
        x_l, y_l, _ = half(~fit, level_a == level)
        if y_l.any() and (y_l == 0).any():
            report["unseen_by_level"][level] = {
                **summarize(x_l, y_l, a, b),
                "at_false_positive_rate": at_false_positive_rates(x_l, y_l),
            }
    return report


def main() -> None:
    """Fit and report the calibration."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", choices=["class", "caption"], default="class")
    parser.add_argument("--emb_dir", default="")
    parser.add_argument("--aligner", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dims", type=int, default=None)
    parser.add_argument(
        "--background",
        help="npz with mean_<dims>, cov_<dims> of unit image embeddings; calibrates "
        "z-scores",
    )
    parser.add_argument("--captions", default="", help="default: the run's --captions")
    parser.add_argument("--caption_chips", default="")
    parser.add_argument("--include_quads", action="store_true")
    parser.add_argument("--include_pixels", action="store_true")
    parser.add_argument("--pixels", default="", help="default: the run's --pixels")
    parser.add_argument("--pixel_texts", default="")
    parser.add_argument(
        "--max_pixels", type=int, default=2000, help="pixel samples per --pixels file"
    )
    parser.add_argument("--ignore_similar", type=float, default=0.85)
    parser.add_argument("--max_chips", type=int, default=2000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    ckpt = torch.load(args.aligner, map_location="cpu", weights_only=False)
    a_ = ckpt["args"]
    if not a_.get("native"):
        raise ValueError("calibration is for native aligners")
    if args.source == "caption":
        report = calibrate_captions(args, ckpt)
        logger.info(json.dumps(report, indent=1))
        json.dump(report, open(args.out, "w"), indent=1)
        return
    if not args.emb_dir:
        parser.error("--source class needs --emb_dir")
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
    view_emb = encode_with(a_, texts)
    model = load_model(ckpt, data["embedding"].shape[1], view_emb.shape[1])
    with torch.no_grad():
        image = embed_native_images(model, ckpt, data["embedding"], args.dims)
        text = model.embed_text(view_emb, args.dims)

    text_class_np = np.array(text_class)
    background = load_background(args.background, int(image.shape[1]))
    x_val, y_val = pairs(image[val], labels[val], text, text_class_np, background)
    a, b = fit_logistic(x_val, y_val)
    x_unseen, y_unseen = pairs(
        image[unseen], labels[unseen], text, text_class_np, background
    )
    report = {
        "source": "class",
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
