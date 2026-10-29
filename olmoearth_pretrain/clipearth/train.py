"""Train a CLIP-style alignment between text and OlmoEarth region embeddings.

Two sources of samples, usable together or alone:

* Class regions: the region embeddings written by ``clipearth.embed`` (the mean OlmoEarth
  embedding over each labeled class in a window), with the class name, a templated
  phrase, and the short and detailed visual descriptions in ``molmoearth_categories.json``.
* Captioned chips (``--captions``, see ``clipearth.captions``): archive chips and their
  quadrants, with GPT-5.5 captions.
* OSM regions (``--osm_regions``/``--osm_texts``): mean archive embeddings inside sampled
  OpenStreetMap areas, with texts from their tags. Drawn alongside class regions; a
  0.5-degree block holdout (``osm_val``) is scored by per-category retrieval.
* Pixel samples (``--pixels``/``--pixel_texts``): the 3x3 mean of unit pixel vectors
  around a labelled pixel (class windows, OSM polygons, GEA-17 points), with the label's
  texts. Drawn alongside class regions.

With ``--context`` the image side also sees the mean over the 2.56 km around each sample
(``Aligner``); captioned chips and quadrants take their chip's mean, and class and OSM
regions, which have no surroundings stored, their own mean.

Every sample's texts are grouped into levels of detail (``--levels``: a name, a short
phrase, and a description of at most two sentences). Each step computes the loss once per
level, with one text of that level per sample, so every image learns all its levels.
Identical texts are one text, so two chips both labelled "dense forest" are positives for
it rather than negatives of each other.

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
    paths = sorted(
        p
        for p in glob.glob(os.path.join(emb_dir, "*.npz"))
        if not p.endswith(".tmp.npz")
    )
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


def class_levels(name: str, visdesc: dict[str, dict]) -> dict[str, list[str]]:
    """The texts that describe one class, by level of detail."""
    desc = visdesc.get(name) or visdesc.get(norm_key(name)) or {}
    return {
        "name": [name, f"a satellite image of {name}"],
        "short": [desc["short"]] if desc.get("short") else [],
        "detailed": [desc["detailed"]] if desc.get("detailed") else [],
    }


def text_views(name: str, visdesc: dict[str, dict]) -> list[str]:
    """The texts that describe one class."""
    return [t for texts in class_levels(name, visdesc).values() for t in texts]


def load_osm_regions(regions_path: str, texts_path: str) -> dict:
    """OSM regions and their texts, joined on ``id``.

    Embeddings are means of unit pixel vectors, stored as a unit direction and its
    length, as for caption chips.
    """
    with np.load(regions_path) as stored:
        regions = {key: stored[key] for key in stored.files}
    texts = {}
    with open(texts_path) as f:
        for line in f:
            record = json.loads(line)
            texts[int(record["id"])] = record
    keep = np.array([int(i) in texts for i in regions["id"]])
    regions = {k: v[keep] for k, v in regions.items()}
    norm = regions.get("embedding_norm", np.ones(len(regions["id"]), np.float32))
    return {
        "embedding": regions["embedding"].astype(np.float32) * norm[:, None],
        "split": regions["split"].astype(str),
        "category": np.array([texts[int(i)]["category"] for i in regions["id"]]),
        "levels": [texts[int(i)]["levels"] for i in regions["id"]],
    }


def load_pixel_samples(
    npz_path: str, texts_path: str, max_samples: int | None = None, seed: int = 0
) -> dict:
    """Pixel samples and their texts (``v5/FORMAT.md``).

    ``pixel`` and ``context`` are means of unit pixel vectors stored as a unit direction
    and its length, as for OSM regions. The texts file has one line per ``text_id``;
    samples whose text is missing or whose category names no visual content are
    dropped. ``split`` indexes ``split_names``; a source without splits ("") is all
    ``train``. ``max_samples`` keeps a random subset of the usable samples, in file
    order. Returns per-sample arrays and ``texts``, text id -> levels.
    """
    with np.load(npz_path) as stored:
        arrays = {key: stored[key] for key in stored.files}
    texts = {}
    with open(texts_path) as f:
        for line in f:
            record = json.loads(line)
            if norm_key(record.get("category", "")) not in GENERIC_LABELS:
                texts[int(record["text_id"])] = record["levels"]
    keep = np.flatnonzero(
        np.isin(arrays["text_id"], np.fromiter(texts, np.int64, len(texts)))
    )
    if max_samples is not None and len(keep) > max_samples:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(keep, max_samples, replace=False))
    n = len(keep)

    def scaled(key: str) -> np.ndarray:
        norm = arrays.get(f"{key}_norm", np.ones(len(arrays["id"]), np.float32))[keep]
        return arrays[key][keep].astype(np.float32) * norm[:, None]

    split = (
        arrays["split_names"][arrays["split"][keep]].astype(str)
        if "split" in arrays
        else np.full(n, "")
    )
    split = np.where(split == "", "train", split)
    return {
        "id": arrays["id"][keep],
        "embedding": scaled("pixel"),
        "context": scaled("context"),
        "split": split,
        "text_id": arrays["text_id"][keep],
        "texts": texts,
    }


# Where a sample without a text at some level takes one from instead.
LEVEL_FALLBACK = {
    "name": ("name", "short", "detailed"),
    "short": ("short", "name", "detailed"),
    "detailed": ("detailed", "short", "name"),
}


def _cache_key(model_name: str, onnx_file: str, prefix: str, text: str) -> str:
    raw = f"{model_name}|{onnx_file}|{prefix}|{text}".encode()
    return hashlib.sha1(raw, usedforsecurity=False).hexdigest()


def _load_cache(path: str) -> dict[str, np.ndarray]:
    if not os.path.exists(path):
        return {}
    stored = np.load(path)
    return dict(zip(stored["keys"].tolist(), stored["emb"]))


def _save_cache(path: str, new: dict[str, np.ndarray]) -> None:
    """Merge ``new`` into the cache file on disk and replace it atomically.

    Another run may have added to the file since it was read.
    """
    merged = {**_load_cache(path), **new}
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez(
        tmp,
        keys=np.array(list(merged)),
        emb=np.stack(list(merged.values())).astype(np.float16),
    )
    os.replace(tmp, path)


def encode_texts(
    texts: list[str],
    model_name: str,
    device: str,
    prefix: str = "",
    dims: int | None = None,
    onnx_file: str = "model_q4.onnx",
    cache: str | None = None,
) -> np.ndarray:
    """Frozen text embeddings, L2-normalized after keeping the first ``dims``.

    With ``cache`` (an .npz path), full embeddings are looked up by model, ONNX file,
    prefix and text, and only missing texts are encoded; the cache stores float16.
    """
    if not cache:
        emb = _encode_raw(texts, model_name, device, prefix, onnx_file)
    else:
        keys = [_cache_key(model_name, onnx_file, prefix, t) for t in texts]
        stored = _load_cache(cache)
        missing = [i for i, k in enumerate(keys) if k not in stored]
        if missing:
            todo = list(dict.fromkeys(texts[i] for i in missing))
            logger.info(
                f"encoding {len(todo)} texts not in the cache ({len(stored)} cached)"
            )
            new = _encode_raw(todo, model_name, device, prefix, onnx_file)
            fresh = {
                _cache_key(model_name, onnx_file, prefix, t): e
                for t, e in zip(todo, new)
            }
            _save_cache(cache, fresh)
            stored.update({k: v.astype(np.float16) for k, v in fresh.items()})
        emb = np.stack([stored[k] for k in keys]).astype(np.float32)
    emb = emb[:, :dims].astype(np.float32)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def _encode_raw(
    texts: list[str],
    model_name: str,
    device: str,
    prefix: str = "",
    onnx_file: str = "model_q4.onnx",
) -> np.ndarray:
    """Full, unnormalized text embeddings.

    ``onnx:<dir>`` runs ``<dir>/onnx/<onnx_file>`` (its ``sentence_embedding``
    output), the same file a browser runs through transformers.js; anything else
    is a sentence-transformers model.
    """
    texts = [prefix + t for t in texts]
    if model_name.startswith("onnx:"):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        path = model_name.removeprefix("onnx:")
        tokenizer = AutoTokenizer.from_pretrained(path)  # nosec B615 - local dir
        options = ort.SessionOptions()
        options.intra_op_num_threads = 16
        session = ort.InferenceSession(f"{path}/onnx/{onnx_file}", options)
        output = [o.name for o in session.get_outputs()].index("sentence_embedding")
        batches = []
        for start in range(0, len(texts), 64):
            enc = tokenizer(
                texts[start : start + 64], padding=True, return_tensors="np"
            )
            feeds = {i.name: enc[i.name] for i in session.get_inputs()}
            batches.append(session.run(None, feeds)[output])
        emb = np.concatenate(batches)
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device=device)
        emb = model.encode(texts, batch_size=256, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


def is_holdout(slug: str, fraction: float) -> bool:
    """Deterministically hold out ``fraction`` of source datasets."""
    return (
        int(hashlib.md5(slug.encode(), usedforsecurity=False).hexdigest(), 16) % 1000
        < fraction * 1000
    )


class Aligner(nn.Module):
    """Project image and text embeddings into a shared space.

    With ``native`` the image embedding is used as it is and only text is projected,
    into the image space, so text vectors can be scored directly against stored
    OlmoEarth embeddings. ``dims`` keeps a leading (Matryoshka) prefix of each.

    ``context_widths`` adds a linear context head per image width, summed onto the image
    side: ``normalize(x + W c + b)`` for pixel ``x`` and context mean ``c``. It starts at
    zero, so training begins from the pixel-only aligner, and the text head is unchanged.
    """

    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        dim: int,
        hidden: int,
        native: bool = False,
        text_linear: bool = False,
        context_widths: list[int] | None = None,
    ) -> None:
        """Build the projections and the learnable temperature."""
        super().__init__()
        self.context = nn.ModuleDict()
        for width in context_widths or []:
            head = nn.Linear(width, width)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.context[str(width)] = head
        self.native = native
        if native:
            self.image: nn.Module = nn.Identity()
            self.text: nn.Module = (
                nn.Linear(text_dim, image_dim)
                if text_linear
                else nn.Sequential(
                    nn.Linear(text_dim, hidden), nn.GELU(), nn.Linear(hidden, image_dim)
                )
            )
        else:
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

    def embed_image(
        self, x: Tensor, dims: int | None = None, context: Tensor | None = None
    ) -> Tensor:
        """Normalized image-side embedding; ``context`` is already ``dims`` wide."""
        y = self.image(x)[..., :dims]
        if context is not None:
            y = y + self.context[str(y.shape[-1])](context)
        return F.normalize(y, dim=-1)

    def embed_text(self, t: Tensor, dims: int | None = None) -> Tensor:
        """Normalized text-side embedding."""
        return F.normalize(self.text(t)[..., :dims], dim=-1)


def load_center(path: str) -> dict[int, np.ndarray]:
    """Global mean unit vector per embedding width, from ``center_<dims>`` keys."""
    stats = np.load(path)
    return {
        int(k.removeprefix("center_")): stats[k].astype(np.float32)
        for k in stats.files
        if k.startswith("center_")
    }


def native_view(
    embedding: Tensor, dims: int | None, center: Tensor | None = None
) -> Tensor:
    """Embeddings as a native search scores them at width ``dims``.

    The leading ``dims`` are renormalized, as the Studio lab does for a narrower read,
    then the global mean vector is subtracted when centring. ``Aligner.embed_image``
    normalizes the result.
    """
    x = F.normalize(embedding[..., :dims], dim=-1)
    return x if center is None else x - center


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
    model: Aligner,
    view_emb: Tensor,
    class_views: list[list[int]],
    dims: int | None = None,
) -> Tensor:
    """One normalized embedding per class: the mean of its projected views."""
    projected = model.embed_text(view_emb, dims)
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


def caption_retrieval(image: Tensor, text: Tensor) -> dict[str, float]:
    """Text-to-image retrieval where text ``i`` describes image ``i``."""
    scores = text @ image.T  # [N, N]
    ranks = (scores > scores.diagonal()[:, None]).sum(1).cpu().numpy()
    return {
        "recall_at_1": float((ranks < 1).mean()),
        "recall_at_10": float((ranks < 10).mean()),
        "median_rank": float(np.median(ranks) + 1),
        "num_images": len(ranks),
    }


def main() -> None:
    """Train and evaluate the aligner."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--emb_dir", default="", help="class regions from clipearth.embed"
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--embedding", default="region_loc", choices=["region_loc", "region_student"]
    )
    parser.add_argument(
        "--captions",
        default="",
        help="comma-separated captions.jsonl files from the v2 pipeline's caption.py",
    )
    parser.add_argument(
        "--caption_chips",
        default="",
        help="comma-separated chips .npz files, one per --captions file",
    )
    parser.add_argument(
        "--class_boost",
        default="",
        help="REGEX=FACTOR: class regions whose dataset slug matches REGEX are drawn FACTOR "
        "times as often, e.g. to keep crop names' exposure when captions take batch space",
    )
    parser.add_argument(
        "--caption_weights",
        default="",
        help="comma-separated sampling weight per --captions file (default 1 each); a "
        "chip from a file with weight 0.3 is drawn 0.3x as often, e.g. to down-weight "
        "the cleaned GPT-4.1 descriptions. Retrieval is evaluated on the first file only.",
    )
    parser.add_argument("--osm_regions", default="", help="osm_regions.npz")
    parser.add_argument("--osm_texts", default="", help="osm_texts.jsonl")
    parser.add_argument(
        "--osm_weight",
        type=float,
        default=1.0,
        help="sampling weight of an OSM region relative to a class region; OSM regions "
        "share the class-region part of each batch",
    )
    parser.add_argument(
        "--pixels", default="", help="comma-separated pixel sample .npz files"
    )
    parser.add_argument(
        "--pixel_texts",
        default="",
        help="comma-separated texts .jsonl files, one per --pixels file",
    )
    parser.add_argument(
        "--pixel_weights",
        default="",
        help="comma-separated sampling weight per --pixels file, relative to a class "
        "region (default 1 each); pixel samples share the class-region part of each batch",
    )
    parser.add_argument(
        "--class_weight",
        type=float,
        default=1.0,
        help="sampling weight of a class region (times --class_boost)",
    )
    parser.add_argument(
        "--context",
        action="store_true",
        help="with --native, the image side also takes each sample's 2.56 km context "
        "mean through a linear head per width",
    )
    parser.add_argument("--min_visibility", type=float, default=0.5)
    parser.add_argument(
        "--caption_holdout_frac",
        type=float,
        default=0.1,
        help="fraction of 5-degree cells whose chips are held out for evaluation",
    )
    parser.add_argument(
        "--caption_frac",
        type=float,
        default=0.5,
        help="share of each batch drawn from captions when class regions are also used",
    )
    parser.add_argument(
        "--quad_frac",
        type=float,
        default=0.3,
        help="share of caption samples that are quadrants rather than whole chips",
    )
    parser.add_argument("--levels", default="name,short,detailed")
    parser.add_argument("--text_model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument(
        "--text_cache", default="", help=".npz cache of text embeddings, shared by runs"
    )
    parser.add_argument("--text_prefix", default="")
    parser.add_argument("--text_dims", type=int, default=None)
    parser.add_argument("--text_linear", action="store_true")
    parser.add_argument(
        "--text_onnx_file",
        default="model_q4.onnx",
        help="with an onnx: text model; browsers need model_no_gather_q4.onnx",
    )
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--holdout_frac", type=float, default=0.1)
    parser.add_argument(
        "--train_on_all",
        action="store_true",
        help="train on every sample, held-out splits included; the logged held-out "
        "metrics then measure fit, not generalization",
    )
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument(
        "--save_every",
        type=int,
        default=2000,
        help="write out_dir/checkpoint.pt every N steps (0: never); a rerun with "
        "the same out_dir resumes from it",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--native",
        action="store_true",
        help="score text against the raw image embedding (no image projection)",
    )
    parser.add_argument(
        "--matryoshka",
        default="",
        help="with --native, also train these leading dims, e.g. 64",
    )
    parser.add_argument(
        "--center",
        default="",
        help="with --native, npz of center_<dims> global mean unit vectors: score "
        "normalize(x - center) instead of x, here and in the Studio lab",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    levels = [lv for lv in args.levels.split(",") if lv]
    if not (args.emb_dir or args.captions or args.osm_regions or args.pixels):
        parser.error("give --emb_dir, --captions, --osm_regions, --pixels, or several")
    if bool(args.osm_regions) != bool(args.osm_texts):
        parser.error("--osm_regions and --osm_texts go together")
    if args.context and not args.native:
        parser.error("--context needs --native")

    # One table of samples for both sources. A group is what shares texts: a class, a
    # chip or a quadrant. Texts are deduplicated by string across all groups.
    text_index: dict[str, int] = {}
    group_levels: list[dict[str, list[int]]] = []
    embeddings: list[np.ndarray] = []
    # Each sample's context mean (``--context``); its own embedding where none is stored.
    contexts: list[np.ndarray] = []
    sample_group: list[int] = []
    split: list[str] = []
    source: list[str] = []
    # Sampling weight and --captions file index (-1 for class regions) per sample.
    weight: list[float] = []
    origin: list[int] = []

    def add_group(by_level: dict[str, list[str]]) -> int:
        group_levels.append(
            {
                lv: [text_index.setdefault(t, len(text_index)) for t in texts]
                for lv, texts in by_level.items()
                if texts
            }
        )
        return len(group_levels) - 1

    class_names: list[str] = []
    labels = np.zeros(0, dtype=int)
    if args.emb_dir:
        data = load_regions(args.emb_dir, args.embedding)
        keep = np.array([norm_key(c) not in GENERIC_LABELS for c in data["class"]])
        data = {k: v[keep] for k, v in data.items()}
        unseen = np.array([is_holdout(s, args.holdout_frac) for s in data["slug"]])
        class_names = sorted(set(data["class"]))
        class_index = {c: i for i, c in enumerate(class_names)}
        labels = np.array([class_index[c] for c in data["class"]])
        visdesc = json.load(open(CATEGORIES)).get(VISDESC_KEY, {})
        class_group = [add_group(class_levels(n, visdesc)) for n in class_names]
        embeddings += list(data["embedding"])
        contexts += list(data["embedding"])
        sample_group += [class_group[i] for i in labels]
        split += ["unseen" if u else str(s) for u, s in zip(unseen, data["split"])]
        source += ["class"] * len(labels)
        if args.class_boost:
            pattern, factor = args.class_boost.rsplit("=", 1)
            boosted = np.array([bool(re.search(pattern, s)) for s in data["slug"]])
            weight += list(np.where(boosted, float(factor), 1.0) * args.class_weight)
            logger.info(
                f"class boost x{factor}: {boosted.sum()} regions in "
                f"{len(set(data['slug'][boosted]))} datasets"
            )
        else:
            weight += [args.class_weight] * len(labels)
        origin += [-1] * len(labels)
        logger.info(
            f"class regions: {(np.array(split) == 'train').sum()} train, "
            f"{(np.array(split) == 'val').sum()} val, {unseen.sum()} unseen "
            f"({len(set(data['slug'][unseen]))} held-out datasets), "
            f"{len(class_names)} classes"
        )

    if args.captions:
        from olmoearth_pretrain.clipearth.captions import (
            geo_holdout,
            load_caption_samples,
        )

        caption_files = args.captions.split(",")
        chip_files = args.caption_chips.split(",")
        if len(caption_files) != len(chip_files):
            parser.error("--captions and --caption_chips need one file each")
        file_weights = (
            [float(w) for w in args.caption_weights.split(",")]
            if args.caption_weights
            else [1.0] * len(caption_files)
        )
        if len(file_weights) != len(caption_files):
            parser.error("--caption_weights needs one weight per --captions file")
        for file_index, (captions_path, chips_path) in enumerate(
            zip(caption_files, chip_files)
        ):
            for sample in load_caption_samples(
                chips_path, captions_path, args.min_visibility
            ):
                embeddings.append(sample["embedding"])
                contexts.append(sample["context"])
                sample_group.append(add_group(sample["levels"]))
                held = geo_holdout(
                    sample["lat"], sample["lon"], args.caption_holdout_frac
                )
                split.append("val_caption" if held else "train")
                source.append(sample["source"])
                weight.append(file_weights[file_index])
                origin.append(file_index)
        src = np.array(source)
        logger.info(
            f"caption samples: {(src == 'chip').sum()} chips, {(src == 'quad').sum()} "
            f"quadrants, {(np.array(split) == 'val_caption').sum()} held out by geography"
        )

    osm_categories: list[str] = []
    osm_labels = np.full(len(split), -1)
    if args.osm_regions:
        osm = load_osm_regions(args.osm_regions, args.osm_texts)
        osm_categories = sorted(set(osm["category"]))
        osm_index = {c: i for i, c in enumerate(osm_categories)}
        osm_labels = np.concatenate(
            [osm_labels, [osm_index[c] for c in osm["category"]]]
        )
        embeddings += list(osm["embedding"])
        contexts += list(osm["embedding"])
        sample_group += [add_group(levels_) for levels_ in osm["levels"]]
        split += list(osm["split"])
        source += ["osm"] * len(osm["split"])
        weight += [args.osm_weight] * len(osm["split"])
        origin += [-1] * len(osm["split"])
        osm_views = [
            [
                text_index.setdefault(t, len(text_index))
                for t in (c, f"a satellite image of {c}")
            ]
            for c in osm_categories
        ]
        logger.info(
            f"osm regions: {(osm['split'] == 'train').sum()} train, "
            f"{(osm['split'] == 'osm_val').sum()} osm_val, {len(osm_categories)} categories"
        )

    pixel_files = [p for p in args.pixels.split(",") if p]
    pixel_names: list[str] = []
    if pixel_files:
        pixel_text_files = args.pixel_texts.split(",")
        if len(pixel_text_files) != len(pixel_files):
            parser.error("--pixels and --pixel_texts need one file each")
        pixel_weights = (
            [float(w) for w in args.pixel_weights.split(",")]
            if args.pixel_weights
            else [1.0] * len(pixel_files)
        )
        if len(pixel_weights) != len(pixel_files):
            parser.error("--pixel_weights needs one weight per --pixels file")
        # Pixels with the same texts, in any file, are one group, so one text is picked
        # for all of them in a batch rather than different views that would be negatives.
        group_of: dict[str, int] = {}
        for file_index, (pixels_path, texts_path) in enumerate(
            zip(pixel_files, pixel_text_files)
        ):
            pixels = load_pixel_samples(pixels_path, texts_path)
            n = len(pixels["split"])
            embeddings += list(pixels["embedding"])
            contexts += list(pixels["context"])
            text_group = {}
            for text_id, by_level in pixels["texts"].items():
                key = json.dumps(by_level, sort_keys=True)
                if key not in group_of:
                    group_of[key] = add_group(by_level)
                text_group[text_id] = group_of[key]
            ids = np.fromiter(text_group, np.int64, len(text_group))
            groups = np.fromiter(text_group.values(), np.int64, len(text_group))
            order = np.argsort(ids)
            sample_group += list(
                groups[order][np.searchsorted(ids[order], pixels["text_id"])]
            )
            split += list(pixels["split"])
            source += ["pixel"] * n
            weight += [pixel_weights[file_index]] * n
            origin += [file_index] * n
            pixel_names.append(os.path.basename(pixels_path).removesuffix(".npz"))
            logger.info(f"pixel samples: {n} from {pixels_path}")

    split_a, source_a = np.array(split), np.array(source)
    weight_a, origin_a = np.array(weight), np.array(origin)
    group_a = np.array(sample_group)
    texts = [""] * len(text_index)
    for t, i in text_index.items():
        texts[i] = t
    logger.info(f"{len(group_levels)} groups, {len(texts)} unique texts")
    # Candidate texts per group and level, after falling back to the nearest level.
    candidates = [
        {
            lv: next((g[f] for f in LEVEL_FALLBACK.get(lv, (lv,)) if g.get(f)), [])
            for lv in levels
        }
        for g in group_levels
    ]

    device = torch.device(args.device)
    view_emb = torch.from_numpy(
        encode_texts(
            texts,
            args.text_model,
            args.device,
            args.text_prefix,
            args.text_dims,
            args.text_onnx_file,
            args.text_cache or None,
        )
    ).to(device)

    emb = np.stack(embeddings).astype(np.float32)
    del embeddings
    train = np.ones(len(split_a), bool) if args.train_on_all else split_a == "train"
    # Native mode scores raw embeddings, as a search over stored embeddings would.
    mean = np.zeros_like(emb[0]) if args.native else emb[train].mean(0)
    image = torch.from_numpy(emb - mean).to(device)
    model = Aligner(
        image.shape[1],
        view_emb.shape[1],
        args.dim,
        args.hidden,
        args.native,
        args.text_linear,
        [int(d) for d in args.matryoshka.split(",") if d] + [emb.shape[1]]
        if args.context
        else None,
    ).to(device)
    prefixes: list[int | None] = [None] + [
        int(d) for d in args.matryoshka.split(",") if d and args.native
    ]
    if args.center and not args.native:
        parser.error("--center needs --native")
    center = load_center(args.center) if args.center else {}
    # The image side each prefix trains and is scored on.
    images = {
        dims: native_view(
            image,
            dims,
            torch.from_numpy(center[dims or image.shape[1]]).to(device)
            if center
            else None,
        )
        if args.native
        else image
        for dims in prefixes
    }
    # The context side of each prefix, viewed as the image side is.
    context_views = {}
    if args.context:
        context = torch.from_numpy(np.stack(contexts).astype(np.float32)).to(device)
        context_views = {
            dims: native_view(
                context,
                dims,
                torch.from_numpy(center[dims or context.shape[1]]).to(device)
                if center
                else None,
            )
            for dims in prefixes
        }
    del contexts

    def project(dims: int | None, rows: Tensor | np.ndarray | None = None) -> Tensor:
        """Image-side embeddings at ``dims`` of all samples, or of ``rows``."""
        x = images[dims] if rows is None else images[dims][rows]
        c = context_views.get(dims)
        if c is not None and rows is not None:
            c = c[rows]
        return model.embed_image(x, dims, c)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.steps, pct_start=0.05
    )

    pools = {
        name: np.flatnonzero(train & (source_a == name))
        for name in ("class", "chip", "quad")
    }
    # OSM regions and pixel samples are drawn with the class regions, by weight.
    pools["class"] = np.flatnonzero(
        train & np.isin(source_a, ["class", "osm", "pixel"])
    )
    caption_frac = args.caption_frac if len(pools["class"]) else 1.0
    if not len(pools["chip"]) and not len(pools["quad"]):
        caption_frac = 0.0
    quad_frac = args.quad_frac if len(pools["chip"]) else 1.0
    if not len(pools["quad"]):
        quad_frac = 0.0

    # Drawing probability within each pool, in proportion to the sample weights.
    draw_p = {k: weight_a[v] / weight_a[v].sum() for k, v in pools.items() if len(v)}
    for k in ("chip", "quad"):
        if k in draw_p and len(set(origin_a[pools[k]])) > 1:
            share = {
                int(o): float(draw_p[k][origin_a[pools[k]] == o].sum())
                for o in np.unique(origin_a[pools[k]])
            }
            logger.info(f"{k} draws by --captions file: {share}")
    if "class" in draw_p:
        region_source = source_a[pools["class"]]
        share = {
            str(name): float(draw_p["class"][region_source == name].sum())
            for name in np.unique(region_source)
        }
        for o, name in enumerate(pixel_names):
            share[name] = float(
                draw_p["class"][
                    (region_source == "pixel") & (origin_a[pools["class"]] == o)
                ].sum()
            )
        logger.info(f"region draws by source: {share}")

    def draw_batch() -> np.ndarray:
        n_caption = round(args.batch_size * caption_frac)
        n_quad = round(n_caption * quad_frac)
        counts = {
            "class": args.batch_size - n_caption,
            "quad": n_quad,
            "chip": n_caption - n_quad,
        }
        parts = [draw(k, n) for k, n in counts.items() if n and len(pools[k])]
        return np.concatenate(parts)

    # rng.choice(replace=False, p=...) costs O(pool) per call, ~0.25 s for 10M pixel
    # samples; inverse-CDF draws with repeats redrawn sample the same distribution.
    draw_cdf = {k: np.cumsum(p) for k, p in draw_p.items()}

    def draw(pool: str, n: int) -> np.ndarray:
        """``n`` distinct samples of ``pool`` by weight (all of it if smaller)."""
        if n >= len(pools[pool]):
            return pools[pool]
        cdf = draw_cdf[pool]
        picked = np.zeros(0, dtype=int)
        while len(picked) < n:
            u = rng.random(n - len(picked)) * cdf[-1]
            new = np.searchsorted(cdf, u, side="right").clip(max=len(cdf) - 1)
            picked = np.unique(np.concatenate([picked, new]))
        return pools[pool][picked]

    def level_texts(batch: np.ndarray, level: str) -> tuple[np.ndarray, np.ndarray]:
        """Pick one text of ``level`` per group in the batch.

        Returns the unique text ids and each sample's index into them.
        """
        groups, inverse = np.unique(group_a[batch], return_inverse=True)
        picked = np.array(
            [
                cands[rng.integers(len(cands))]
                if (cands := candidates[g][level])
                else -1
                for g in groups
            ]
        )[inverse]
        return np.unique(picked, return_inverse=True)

    class_mask = source_a == "class"
    class_split = np.where(class_mask, split_a, "")
    labels_all = np.full(len(split_a), -1)
    labels_all[: len(labels)] = labels
    # Class groups were added first, so they are the first len(class_names) groups.
    class_views = [
        [i for lv in ("name", "short", "detailed") for i in g.get(lv, [])]
        for g in group_levels[: len(class_names)]
    ]

    # Fit only: a fixed subset of each --pixels file against the first-level texts in it.
    pixel_eval = []
    eval_rng = np.random.default_rng(args.seed + 1)
    for o, name in enumerate(pixel_names):
        rows = np.flatnonzero((source_a == "pixel") & (origin_a == o))
        rows = rows[[bool(candidates[g][levels[0]]) for g in group_a[rows]]]
        rows = np.sort(eval_rng.choice(rows, min(len(rows), 5000), replace=False))
        first = [candidates[g][levels[0]][0] for g in group_a[rows]]
        text_ids, target = np.unique(first, return_inverse=True)
        pixel_eval.append((name, rows, text_ids, target))

    # The samples scored below; only these are projected at each evaluation.
    val_rows = np.flatnonzero(class_split == "val")
    unseen_rows = np.flatnonzero(class_split == "unseen")
    held = np.flatnonzero(
        (split_a == "val_caption") & (source_a == "chip") & (origin_a == 0)
    )
    scored_rows = np.unique(
        np.concatenate(
            [val_rows, unseen_rows, held] + [rows for _, rows, _, _ in pixel_eval]
        )
    ).astype(int)

    def run_eval() -> dict:
        model.eval()
        metrics: dict = {}
        with torch.no_grad():
            for dims in prefixes:
                suffix = f"_{dims}" if dims else ""
                scored = project(dims, scored_rows)

                def projected_rows(rows: np.ndarray) -> Tensor:
                    return scored[np.searchsorted(scored_rows, rows)]

                if class_names:
                    seen_classes = np.unique(
                        labels_all[(class_split == "train") | (class_split == "val")]
                    )
                    class_emb = class_text_matrix(model, view_emb, class_views, dims)
                    metrics["val" + suffix] = evaluate(
                        projected_rows(val_rows),
                        labels_all[val_rows],
                        class_emb,
                        seen_classes,
                    )
                    if len(unseen_rows):
                        metrics["unseen" + suffix] = evaluate(
                            projected_rows(unseen_rows),
                            labels_all[unseen_rows],
                            class_emb,
                            np.unique(labels_all[unseen_rows]),
                        )
                if len(held):
                    for lv in levels:
                        with_text = [i for i in held if candidates[group_a[i]][lv]]
                        first = [candidates[group_a[i]][lv][0] for i in with_text]
                        if len(with_text) > 1:
                            metrics[f"caption_{lv}{suffix}"] = caption_retrieval(
                                projected_rows(np.array(with_text)),
                                model.embed_text(view_emb[first], dims),
                            )
                for name, rows, text_ids, target in pixel_eval:
                    metrics[name + suffix] = evaluate(
                        projected_rows(rows),
                        target,
                        model.embed_text(view_emb[text_ids], dims),
                        np.arange(len(text_ids)),
                    )
            osm_val = np.flatnonzero((source_a == "osm") & (split_a == "osm_val"))
            if len(osm_val):
                for dims in prefixes:
                    suffix = f"_{dims}" if dims else ""
                    category_emb = class_text_matrix(model, view_emb, osm_views, dims)
                    candidates_val = np.unique(osm_labels[osm_val])
                    projected = project(dims, osm_val)
                    metrics["osm_val" + suffix] = evaluate(
                        projected, osm_labels[osm_val], category_emb, candidates_val
                    )
                    scores = (projected @ category_emb.T).cpu().numpy()
                    metrics["osm_val_category_ap" + suffix] = {
                        osm_categories[c]: average_precision(
                            scores[:, c], osm_labels[osm_val] == c
                        )
                        for c in candidates_val
                    }
            if class_names:
                class_train = class_split == "train"
                val = class_split == "val"
                prototypes = torch.zeros(
                    len(class_names), image.shape[1], device=device
                )
                prototypes.index_add_(
                    0,
                    torch.from_numpy(labels_all[class_train]).to(device),
                    image[torch.from_numpy(class_train).to(device)],
                )
                prototypes = F.normalize(prototypes, dim=-1)
                # Only val regions of classes that have training regions can be scored.
                scorable = np.flatnonzero(val)[
                    np.isin(labels_all[val], labels_all[class_train])
                ]
                metrics["val_prototype"] = evaluate(
                    F.normalize(image[scorable], dim=-1),
                    labels_all[scorable],
                    prototypes,
                    np.unique(labels_all[class_train]),
                )
        model.train()
        return metrics

    history = []
    start = 1
    ckpt_path = os.path.join(args.out_dir, "checkpoint.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        schedule.load_state_dict(ckpt["schedule"])
        rng.bit_generator.state = ckpt["rng"]
        torch.set_rng_state(ckpt["torch_rng"])
        history = ckpt["history"]
        start = ckpt["step"] + 1
        logger.info(f"resumed from {ckpt_path} at step {ckpt['step']}")
    for step in range(start, args.steps + 1):
        batch = draw_batch()
        batch_t = torch.from_numpy(batch).to(device)
        scale = model.logit_scale.exp().clamp(max=100)
        losses = []
        for level in levels:
            unique_texts, targets = level_texts(batch, level)
            keep = unique_texts[targets] >= 0
            if not keep.all():
                # Groups with no text at any level sit out this level.
                unique_texts, targets = np.unique(
                    unique_texts[targets][keep], return_inverse=True
                )
                level_batch = batch_t[torch.from_numpy(keep).to(device)]
            else:
                level_batch = batch_t
            targets_t = torch.from_numpy(targets).to(device)
            for dims in prefixes:
                losses.append(
                    contrastive_loss(
                        project(dims, level_batch),
                        model.embed_text(view_emb[unique_texts], dims),
                        targets_t,
                        scale,
                    )
                )
        loss = sum(losses) / len(losses)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % args.eval_every == 0 or step == args.steps:
            metrics = run_eval()
            history.append({"step": step, "loss": float(loss), **metrics})
            logger.info(json.dumps(history[-1]))
        if args.save_every and step % args.save_every == 0 and step < args.steps:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "schedule": schedule.state_dict(),
                    "rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "history": history,
                    "step": step,
                },
                ckpt_path + ".tmp",
            )
            os.replace(ckpt_path + ".tmp", ckpt_path)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "image_mean": mean,
            "center": center,
            "class_names": class_names,
            "osm_categories": osm_categories,
            "args": vars(args),
        },
        os.path.join(args.out_dir, "aligner.pt"),
    )
    json.dump(history, open(os.path.join(args.out_dir, "metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
