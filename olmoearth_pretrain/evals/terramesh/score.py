"""Score TerraMesh-Masks tiles zero-shot with text queries.

  python -m olmoearth_pretrain.evals.terramesh.score --emb <dir> --masks <dir> \
      --annotations <parquet> --out <json>

``--emb`` is embed's output, with ``head.json`` saying how to embed queries into the
vectors' space. Per tile, the query is the tile's own class_label (fixed in advance; the
synonyms are scored separately, never best-of):

* ap, auroc: ranking the mask's pixels above the rest of the tile by raw score (cosine,
  3 x 3 smoothed). Chance AP is the mask's share of the tile.
* pointing: whether the tile's highest raw score falls inside the mask.
* iou: of the pixels whose relative score (the query against fixed background prompts,
  see metrics.relative) exceeds 0.5. Backgrounds whose text is close to the query
  (cosine >= BACKGROUND_EXCLUDE) are left out.

Means come with 95% bootstrap intervals over tiles, overall and per TerraMesh group.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

from olmoearth_pretrain.evals.terramesh.metrics import (
    BACKGROUNDS,
    METRICS,
    relative,
    summarize,
    tile_metrics,
)

ANNOTATIONS = "/weka/oe-training-default/mm-olmo/rs_datasets/terramesh_seg_eval/annotations.parquet"
BACKGROUND_EXCLUDE = 0.8


def read_annotations(path: str) -> pd.DataFrame:
    """The eval's tiles; synonyms are stored as the text of a Python list."""
    tiles = pd.read_parquet(path)
    tiles["class_synonyms"] = tiles.class_synonyms.map(ast.literal_eval)
    return tiles


def sentence_encoder(info: dict) -> Callable[[list[str]], np.ndarray]:
    """Embed texts as the head's training texts were: unit vectors, truncated."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(info["model"], device=device)

    def encode(texts: list[str]) -> np.ndarray:
        emb = model.encode(
            texts, prompt_name=info["prompt_name"], convert_to_numpy=True
        ).astype(np.float32)[:, : info["dim"]]
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    return encode


def score_tiles(
    tiles: pd.DataFrame,
    emb_dir: str,
    masks_dir: str,
    text: dict[str, np.ndarray],
    scale: float,
) -> tuple[list[dict], list[dict]]:
    """Per-tile metrics for the class labels and, separately, for the synonyms.

    Args:
        tiles: Annotations of the tiles to score.
        emb_dir: <file_id>.npz with vectors [H, W, D] and valid [H, W].
        masks_dir: <file_id>.tif masks.
        text: Unit text vector of every label, synonym and background prompt.
        scale: The head's logit scale, for the relative score.
    """
    import rasterio

    bg = np.stack([text[b] for b in BACKGROUNDS])

    def one(x3: np.ndarray, q: np.ndarray, mask: np.ndarray, valid: np.ndarray) -> dict:
        keep = (bg @ q) < BACKGROUND_EXCLUDE
        cos = x3.reshape(-1, x3.shape[-1]) @ np.concatenate([q[None], bg[keep]]).T
        rel = relative(cos[:, 0], cos[:, 1:], scale).reshape(mask.shape)
        return tile_metrics(cos[:, 0].reshape(mask.shape), rel, mask, valid)

    rows, synonym_rows = [], []
    for tile in tiles.itertuples():
        z = np.load(os.path.join(emb_dir, f"{tile.file_id}.npz"))
        valid = z["valid"]
        x3 = uniform_filter(
            z["vectors"].astype(np.float32), size=(3, 3, 1), mode="nearest"
        )
        with rasterio.open(os.path.join(masks_dir, f"{tile.file_id}.tif")) as src:
            mask = src.read(1) > 0
        m = one(x3, text[tile.class_label], mask, valid)
        if m:
            rows.append(
                {
                    "file_id": tile.file_id,
                    "group": tile.group_name,
                    "label": tile.class_label,
                    **m,
                }
            )
        for syn in tile.class_synonyms:
            ms = one(x3, text[syn], mask, valid)
            if ms:
                synonym_rows.append(
                    {"file_id": tile.file_id, "group": tile.group_name, **ms}
                )
    return rows, synonym_rows


def report(rows: list[dict], synonym_rows: list[dict], seed: int = 0) -> dict:
    """Summaries overall, for synonyms and per group, plus the per-tile rows."""
    rng = np.random.default_rng(seed)
    groups = sorted({r["group"] for r in rows})
    return {
        "class_label": summarize(rows, rng),
        "synonyms": summarize(synonym_rows, rng),
        "by_group": {
            g: summarize([r for r in rows if r["group"] == g], rng) for g in groups
        },
        "per_tile": rows,
    }


def main() -> None:
    """Score one embeddings dir and write the report."""
    p = argparse.ArgumentParser()
    p.add_argument("--emb", required=True)
    p.add_argument("--masks", required=True)
    p.add_argument("--annotations", default=ANNOTATIONS)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    info = json.load(open(os.path.join(a.emb, "head.json")))
    tiles = read_annotations(a.annotations)
    tiles = tiles[
        [os.path.exists(os.path.join(a.emb, f"{f}.npz")) for f in tiles.file_id]
    ]
    texts = sorted(
        set(tiles.class_label)
        | {s for syns in tiles.class_synonyms for s in syns}
        | set(BACKGROUNDS)
    )
    text = dict(zip(texts, sentence_encoder(info)(texts)))
    rows, synonym_rows = score_tiles(tiles, a.emb, a.masks, text, info["scale"])
    out = {"embeddings": a.emb, "head": info, "tiles_with_embeddings": len(tiles)}
    out.update(report(rows, synonym_rows))
    json.dump(out, open(a.out, "w"), indent=1)
    s = out["class_label"]
    print({m: s[m][0] for m in METRICS if m in s}, "tiles", s["tiles"])


if __name__ == "__main__":
    main()
