"""Train a CLIP-style alignment between text and OlmoEarth region embeddings.

Inputs are the region embeddings written by ``clipearth.embed`` (the mean OlmoEarth
embedding over each labeled class in a window) and, for every class, several text
views: the class name, a templated phrase, and the short and detailed visual
descriptions in ``molmoearth_categories.json``.

Design, following ``scripts/clip/train_text_align.py``:

* The text encoder is a frozen sentence-transformer and OlmoEarth is frozen; only a
  projection on each side trains. A few thousand classes cannot train a text tower, and
  open-vocabulary queries rely on the pretrained text space.
* Image embeddings are centered on the training mean before projection; without that,
  one dominant direction makes every cosine similarity high.
* Loss: image-to-text cross-entropy over the classes in the batch, plus text-to-image
  with every region of the class as a positive. Many regions in a batch share a class,
  so plain CLIP's one-positive diagonal would push apart matching pairs.

Evaluation:

* ``val``: Molmo's held-out windows (same source datasets as training).
* ``unseen``: every window of a held-out set of source datasets (``--holdout_frac`` of
  dataset slugs, chosen by hash), never seen in training, scored only against the
  classes those datasets use.
* Metrics: zero-shot top-1/top-5 classification of regions, and mean average precision
  of text-to-region retrieval. A class-prototype classifier (mean training embedding per
  class) is reported for ``val`` as the bar the text side should approach.

Run (sentence-transformers is not a project dependency; install it into the project
environment so it keeps the locked torch, rather than ``uv run --with``):
  uv pip install sentence-transformers
  python -m olmoearth_pretrain.clipearth.train --emb_dir <embed out_dir> --out_dir <dir>
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)

CATEGORIES = "/weka/oe-training-default/mm-olmo/rs_datasets/molmoearth_categories.json"
VISDESC_KEY = "seg_open_set_visdesc"
# Labels that name no visual content; their regions are dropped.
GENERIC_LABELS = {
    "background",
    "other",
    "others",
    "unknown",
    "unlabeled",
    "unclassified",
    "no data",
    "nodata",
    "none",
    "non field",
    "not field",
    "void",
    "ignore",
}


def norm_key(text: str) -> str:
    """Lowercase alphanumerics separated by single spaces."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def load_regions(emb_dir: str, key: str) -> dict[str, np.ndarray]:
    """Concatenate every part written by ``clipearth.embed``."""
    paths = sorted(glob.glob(os.path.join(emb_dir, "*.npz")))
    if not paths:
        raise FileNotFoundError(f"no embedding parts in {emb_dir}")
    window_split: dict[str, str] = {}
    window_slug: dict[str, str] = {}
    windows, classes, embeddings = [], [], []
    for path in paths:
        part = np.load(path)
        window_split.update(zip(part["window"], part["window_split"]))
        window_slug.update(zip(part["window"], part["window_slug"]))
        if len(part["region_class"]):
            windows.append(part["region_window"])
            classes.append(part["region_class"])
            embeddings.append(part[key].astype(np.float32))
    window = np.concatenate(windows)
    return {
        "window": window,
        "class": np.concatenate(classes),
        "embedding": np.concatenate(embeddings),
        "split": np.array([window_split[w] for w in window]),
        "slug": np.array([window_slug[w] for w in window]),
    }


def text_views(name: str, visdesc: dict[str, dict]) -> list[str]:
    """The texts that describe one class."""
    views = [name, f"a satellite image of {name}"]
    desc = visdesc.get(name) or visdesc.get(norm_key(name)) or {}
    views += [desc[k] for k in ("short", "detailed") if desc.get(k)]
    return views


def encode_texts(texts: list[str], model_name: str, device: str) -> np.ndarray:
    """Frozen sentence-transformer embeddings, L2-normalized."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    return model.encode(
        texts, batch_size=256, normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)


def is_holdout(slug: str, fraction: float) -> bool:
    """Deterministically hold out ``fraction`` of source datasets."""
    return (
        int(hashlib.md5(slug.encode(), usedforsecurity=False).hexdigest(), 16) % 1000
        < fraction * 1000
    )


class Aligner(nn.Module):
    """Project image and text embeddings into a shared space."""

    def __init__(self, image_dim: int, text_dim: int, dim: int, hidden: int) -> None:
        """Build the two projections and the learnable temperature."""
        super().__init__()
        self.image = nn.Sequential(
            nn.LayerNorm(image_dim),
            nn.Linear(image_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.text = nn.Linear(text_dim, dim)
        self.logit_scale = nn.Parameter(
            torch.tensor(np.log(1 / 0.07), dtype=torch.float32)
        )

    def embed_image(self, x: Tensor) -> Tensor:
        """Normalized image-side embedding."""
        return F.normalize(self.image(x), dim=-1)

    def embed_text(self, t: Tensor) -> Tensor:
        """Normalized text-side embedding."""
        return F.normalize(self.text(t), dim=-1)


def contrastive_loss(
    image: Tensor, text: Tensor, labels: Tensor, scale: Tensor
) -> Tensor:
    """Symmetric loss with all regions of a class as positives for its text.

    Args:
        image: ``[B, D]`` normalized region embeddings.
        text: ``[U, D]`` normalized embeddings of the batch's ``U`` unique classes.
        labels: ``[B]`` index into ``text`` of each region's class.
        scale: logit scale.
    """
    logits = scale * image @ text.T  # [B, U]
    image_to_text = F.cross_entropy(logits, labels)
    positives = F.one_hot(labels, text.shape[0]).T.float()  # [U, B]
    log_prob = logits.T.log_softmax(dim=-1)  # over regions, per class
    text_to_image = -(positives * log_prob).sum(-1) / positives.sum(-1)
    return (image_to_text + text_to_image.mean()) / 2


def class_text_matrix(
    model: Aligner, view_emb: Tensor, class_views: list[list[int]]
) -> Tensor:
    """One normalized embedding per class: the mean of its projected views."""
    projected = model.embed_text(view_emb)
    return F.normalize(torch.stack([projected[v].mean(0) for v in class_views]), dim=-1)


def average_precision(scores: np.ndarray, relevant: np.ndarray) -> float:
    """AP of ranking ``scores`` descending against boolean ``relevant``."""
    order = np.argsort(-scores)
    hits = relevant[order]
    if not hits.any():
        return float("nan")
    precision = np.cumsum(hits) / (np.arange(len(hits)) + 1)
    return float((precision * hits).sum() / hits.sum())


def evaluate(
    image: Tensor, labels: np.ndarray, class_emb: Tensor, candidates: np.ndarray
) -> dict[str, float]:
    """Zero-shot classification and retrieval among ``candidates`` class indices."""
    scores = (image @ class_emb[candidates].T).cpu().numpy()  # [N, C]
    position = {c: i for i, c in enumerate(candidates)}
    target = np.array([position[c] for c in labels])
    ranks = (scores > scores[np.arange(len(target)), target][:, None]).sum(1)
    aps = [
        average_precision(scores[:, i], target == i)
        for i in range(len(candidates))
        if (target == i).any()
    ]
    per_class = [ranks[target == i] == 0 for i in np.unique(target)]
    return {
        "top1": float((ranks == 0).mean()),
        "top5": float((ranks < 5).mean()),
        "macro_top1": float(np.mean([p.mean() for p in per_class])),
        "retrieval_map": float(np.nanmean(aps)),
        "num_regions": len(target),
        "num_classes": len(candidates),
    }


def main() -> None:
    """Train and evaluate the aligner."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--emb_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--embedding", default="region_loc", choices=["region_loc", "region_student"]
    )
    parser.add_argument("--text_model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--holdout_frac", type=float, default=0.1)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    data = load_regions(args.emb_dir, args.embedding)
    keep = np.array([norm_key(c) not in GENERIC_LABELS for c in data["class"]])
    data = {k: v[keep] for k, v in data.items()}
    unseen = np.array([is_holdout(s, args.holdout_frac) for s in data["slug"]])
    train = (data["split"] == "train") & ~unseen
    val = (data["split"] == "val") & ~unseen
    logger.info(
        f"regions: {train.sum()} train, {val.sum()} val, {unseen.sum()} unseen "
        f"({len(set(data['slug'][unseen]))} held-out datasets)"
    )

    class_names = sorted(set(data["class"]))
    class_index = {c: i for i, c in enumerate(class_names)}
    labels = np.array([class_index[c] for c in data["class"]])
    visdesc = json.load(open(CATEGORIES)).get(VISDESC_KEY, {})
    texts: list[str] = []
    class_views: list[list[int]] = []
    for name in class_names:
        views = text_views(name, visdesc)
        class_views.append(list(range(len(texts), len(texts) + len(views))))
        texts += views
    logger.info(f"{len(class_names)} classes, {len(texts)} text views")
    device = torch.device(args.device)
    view_emb = torch.from_numpy(encode_texts(texts, args.text_model, args.device)).to(
        device
    )

    mean = data["embedding"][train].mean(0)
    image = torch.from_numpy(data["embedding"] - mean).to(device)
    model = Aligner(image.shape[1], view_emb.shape[1], args.dim, args.hidden).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.steps, pct_start=0.05
    )

    train_idx = np.flatnonzero(train)
    labels_t = torch.from_numpy(labels).to(device)
    view_count = np.array([len(v) for v in class_views])
    view_start = np.array([v[0] for v in class_views])

    def run_eval() -> dict:
        model.eval()
        with torch.no_grad():
            class_emb = class_text_matrix(model, view_emb, class_views)
            projected = model.embed_image(image)
            seen_classes = np.unique(labels[train | val])
            metrics = {
                "val": evaluate(projected[val], labels[val], class_emb, seen_classes)
            }
            if unseen.any():
                metrics["unseen"] = evaluate(
                    projected[unseen],
                    labels[unseen],
                    class_emb,
                    np.unique(labels[unseen]),
                )
            # Bar for the text side: mean training embedding per class.
            prototypes = torch.zeros(len(class_names), image.shape[1], device=device)
            prototypes.index_add_(0, labels_t[train], image[train])
            prototypes = F.normalize(prototypes, dim=-1)
            # Only val regions of classes that have training regions can be scored.
            scorable = np.flatnonzero(val)[np.isin(labels[val], labels[train])]
            metrics["val_prototype"] = evaluate(
                F.normalize(image[scorable], dim=-1),
                labels[scorable],
                prototypes,
                np.unique(labels[train]),
            )
        model.train()
        return metrics

    history = []
    for step in range(1, args.steps + 1):
        batch = rng.choice(
            train_idx, size=min(args.batch_size, len(train_idx)), replace=False
        )
        batch_classes, batch_labels = np.unique(labels[batch], return_inverse=True)
        views = view_start[batch_classes] + rng.integers(0, view_count[batch_classes])
        loss = contrastive_loss(
            model.embed_image(image[batch]),
            model.embed_text(view_emb[views]),
            torch.from_numpy(batch_labels).to(device),
            model.logit_scale.exp().clamp(max=100),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % args.eval_every == 0 or step == args.steps:
            metrics = run_eval()
            history.append({"step": step, "loss": float(loss), **metrics})
            logger.info(json.dumps(history[-1]))

    torch.save(
        {
            "state_dict": model.state_dict(),
            "image_mean": mean,
            "class_names": class_names,
            "args": vars(args),
        },
        os.path.join(args.out_dir, "aligner.pt"),
    )
    json.dump(history, open(os.path.join(args.out_dir, "metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
