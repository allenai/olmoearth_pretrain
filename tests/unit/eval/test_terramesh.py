"""Tests for the TerraMesh-Masks zero-shot eval."""

import os

import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from olmoearth_pretrain.evals.terramesh.embed import SIZE, TILE, from_tiles, to_tiles
from olmoearth_pretrain.evals.terramesh.metrics import (
    BACKGROUNDS,
    ranking_metrics,
    relative,
    summarize,
    tile_metrics,
)
from olmoearth_pretrain.evals.terramesh.score import report, score_tiles


def _brute_force(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """AP over distinct thresholds and AUROC as P(pos > neg) + P(tie) / 2."""
    ap, prev_recall = 0.0, 0.0
    for t in np.unique(s)[::-1]:
        pred = s >= t
        recall = (pred & y).sum() / y.sum()
        ap += (recall - prev_recall) * (pred & y).sum() / pred.sum()
        prev_recall = recall
    pos, neg = s[y], s[~y]
    diff = pos[:, None] - neg[None, :]
    auroc = ((diff > 0).sum() + (diff == 0).sum() / 2) / diff.size
    return ap, auroc


@pytest.mark.parametrize("levels", [None, 5])
def test_ranking_metrics_match_definitions(levels: int | None) -> None:
    """Exact AP and AUROC, with and without ties."""
    rng = np.random.default_rng(0)
    s = rng.normal(size=300)
    if levels is not None:
        s = np.round(s * levels) / levels
    y = rng.random(300) < 0.3
    ap, auroc = ranking_metrics(y, s + y * 0.5)
    want_ap, want_auroc = _brute_force(y, s + y * 0.5)
    assert ap == pytest.approx(want_ap)
    assert auroc == pytest.approx(want_auroc)


def test_relative_is_softmax_share_against_backgrounds() -> None:
    """With one background, the relative score is the two-way softmax."""
    rel = relative(np.array([0.3]), np.array([[0.1]]), 10.0)
    assert rel[0] == pytest.approx(1 / (1 + np.exp(-2.0)))


def test_tile_metrics_perfect_and_unscorable() -> None:
    """A score equal to the mask is perfect; a tile without negatives is skipped."""
    mask = np.zeros((8, 8), dtype=bool)
    mask[:3, :3] = True
    valid = np.ones_like(mask)
    m = tile_metrics(mask.astype(float), mask.astype(float), mask, valid)
    assert m["ap"] == 1.0 and m["auroc"] == 1.0
    assert m["pointing"] == 1.0 and m["iou"] == 1.0
    assert m["prevalence"] == pytest.approx(9 / 64)
    assert tile_metrics(mask.astype(float), mask.astype(float), mask, mask) == {}


def test_summarize_interval_brackets_mean() -> None:
    """The bootstrap interval contains the mean."""
    rows = [
        {"ap": a, "auroc": a, "pointing": 1.0, "iou": a, "prevalence": 0.1}
        for a in np.linspace(0, 1, 20)
    ]
    s = summarize(rows, np.random.default_rng(0), bootstrap=200)
    mean, lo, hi = s["ap"]
    assert s["tiles"] == 20 and lo <= mean <= hi


def test_tiles_round_trip() -> None:
    """Cutting into 16 px tiles and stitching one token per pixel is the identity."""
    x = torch.randn(1, SIZE, SIZE, 3)
    tiles = to_tiles(x)
    assert tiles.shape == ((SIZE // TILE) ** 2, TILE, TILE, 3)
    assert torch.equal(from_tiles(tiles), x[0])


def test_from_tiles_repeats_coarse_tokens() -> None:
    """A g x g token grid per tile is repeated up to pixels."""
    tokens = torch.randn((SIZE // TILE) ** 2, 8, 8, 2)
    grid = from_tiles(tokens)
    assert grid.shape == (SIZE, SIZE, 2)
    assert torch.equal(grid[0, 0], grid[1, 1])
    assert torch.equal(grid[0, 0], tokens[0, 0, 0])
    assert torch.equal(grid[0, TILE], tokens[1, 0, 0])


def test_score_tiles_finds_the_mask(tmp_path: str) -> None:
    """Pixels whose vector is the label's text vector score as a perfect match."""
    dim = 16
    rng = np.random.default_rng(0)
    texts = ["vineyard", "grapes", *BACKGROUNDS]
    vecs = rng.normal(size=(len(texts), dim))
    text = dict(zip(texts, vecs / np.linalg.norm(vecs, axis=1, keepdims=True)))
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1
    vectors = np.where(mask[..., None] > 0, text["vineyard"], text["water"])
    emb, masks = os.path.join(tmp_path, "emb"), os.path.join(tmp_path, "masks")
    os.makedirs(emb)
    os.makedirs(masks)
    np.savez(
        os.path.join(emb, "t0.npz"),
        vectors=vectors.astype(np.float16),
        valid=np.ones((32, 32), dtype=bool),
    )
    with rasterio.open(
        os.path.join(masks, "t0.tif"), "w", driver="GTiff", height=32, width=32,
        count=1, dtype="uint8", transform=from_origin(0, 320, 10, 10),
    ) as dst:  # fmt: skip
        dst.write(mask, 1)
    tiles = pd.DataFrame(
        {
            "file_id": ["t0"],
            "group_name": ["g"],
            "class_label": ["vineyard"],
            "class_synonyms": [["grapes"]],
        }
    )
    rows, synonym_rows = score_tiles(tiles, emb, masks, text, scale=50.0)
    assert rows[0]["ap"] == pytest.approx(1.0)
    assert rows[0]["pointing"] == 1.0
    assert rows[0]["iou"] > 0.5
    assert len(synonym_rows) == 1
    out = report(rows, synonym_rows)
    assert out["class_label"]["tiles"] == 1 and set(out["by_group"]) == {"g"}
