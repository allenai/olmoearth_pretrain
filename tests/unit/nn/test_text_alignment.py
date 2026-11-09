"""Tests for the text alignment head."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from olmoearth_pretrain.data.caption_embeddings import CaptionEmbeddings
from olmoearth_pretrain.data.captions import caption_cell_raster
from olmoearth_pretrain.data.constants import MISSING_VALUE
from olmoearth_pretrain.nn.text_alignment import (
    LabelEmbeddings,
    TextAlignmentHead,
    label_alignment_loss,
    pool_caption_cells,
    pool_labels,
    text_alignment_loss,
)

DIM = 8


def write_embeddings(path: Path, num_windows: int, seed: int = 0) -> np.ndarray:
    """Random unit caption embeddings [num_windows, 17, DIM] in a caption_embeddings dir."""
    array = np.random.default_rng(seed).normal(size=(num_windows, 17, DIM))
    array /= np.linalg.norm(array, axis=-1, keepdims=True)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "embeddings.npy", array.astype(np.float16))
    (path / "meta.json").write_text(json.dumps({"dim": DIM}))
    return array


@pytest.fixture
def embeddings(tmp_path: Path) -> CaptionEmbeddings:
    """Embeddings for 4 windows."""
    write_embeddings(tmp_path / "emb", 4)
    return CaptionEmbeddings(str(tmp_path / "emb"))


def raster(
    window: int, size: int = 8, present: np.ndarray | None = None
) -> torch.Tensor:
    """caption_cell for one window, every cell captioned unless ``present`` says not."""
    present = np.ones(16, dtype=bool) if present is None else present
    return torch.from_numpy(caption_cell_raster(window, present, size, size))


def test_pool_averages_tokens_per_sample_and_cell(
    embeddings: CaptionEmbeddings,
) -> None:
    """Each captioned (sample, cell) gets its tokens' mean and its caption."""
    present = np.ones(16, dtype=bool)
    present[0] = False
    caption_cell = torch.stack([raster(1, present=present), raster(3)])
    # 4 x 4 location tokens over 8 x 8 pixels: one token per cell here.
    tokens = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4, 1)
    pooled, text = pool_caption_cells(tokens, caption_cell, embeddings)
    assert pooled.shape == (31, 1)
    assert pooled[0, 0] == 1  # sample 0 starts at cell 1
    assert pooled[15, 0] == 16  # sample 1, cell 0
    expected, _ = embeddings.lookup(torch.tensor([[3.0, 0.0]]))
    torch.testing.assert_close(text[15], expected[0])

    # Two location tokens per cell side: four tokens pool into each cell.
    fine = torch.ones(2, 8, 8, 1)
    fine[0, :2, :2] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])[..., None]
    pooled, _ = pool_caption_cells(fine, caption_cell, embeddings)
    assert pooled.shape == (31, 1)
    assert pooled[0, 0] == 1  # cell 0 of sample 0 has no caption: dropped


def test_head_does_not_use_near_identical_captions_as_negatives(
    embeddings: CaptionEmbeddings,
) -> None:
    """A caption repeated in another cell can't count against its own match."""
    head = TextAlignmentHead(embeddings=embeddings, embedding_dim=DIM)
    with torch.no_grad():
        head.proj.weight.copy_(torch.eye(DIM))
        head.proj.bias.zero_()
    text = torch.eye(DIM)[:3]
    text[1] = text[0]
    loss, metrics = head(text.clone(), text)
    assert torch.isfinite(loss)
    assert metrics["text/top1"] == 1

    unmasked = TextAlignmentHead(
        embeddings=embeddings, embedding_dim=DIM, same_text_threshold=2.0
    )
    unmasked.load_state_dict(head.state_dict())
    assert unmasked(text.clone(), text)[0] > loss


def test_loss_without_two_captioned_cells_is_zero_with_gradients(
    embeddings: CaptionEmbeddings,
) -> None:
    """Too few cells still runs the head, so every rank reduces the same gradients."""
    head = TextAlignmentHead(embeddings=embeddings, embedding_dim=3)
    present = np.zeros(16, dtype=bool)
    present[5] = True
    caption_cell = raster(2, present=present)[None]
    tokens = torch.randn(1, 16, 3, requires_grad=True)
    loss, metrics = text_alignment_loss(head, tokens, caption_cell, pixels=2)
    loss.backward()
    assert loss == 0
    assert metrics["text/cells"] == 1
    assert head.proj.weight.grad is not None


@pytest.fixture
def labels(tmp_path: Path) -> LabelEmbeddings:
    """Embeddings of 6 labels: 0 and 1 share a text, the rest are distinct."""
    table = np.eye(DIM, dtype=np.float32)[:6]
    table[1] = table[0]
    path = tmp_path / "labels"
    path.mkdir()
    np.save(path / "embeddings.npy", table)
    (path / "meta.json").write_text(json.dumps({"dim": DIM}))
    return LabelEmbeddings(str(path))


def test_pool_labels_averages_tokens_per_sample_and_label() -> None:
    """A token joins the group of every label at its centre pixel, per sample."""
    label = torch.full((2, 2, 2, 1, 4), float(MISSING_VALUE))
    label[0, :, :, 0, 0] = 3  # all of sample 0 is label 3 ...
    label[0, 0, 0, 0, 1] = 5  # ... and its first token is also label 5
    label[1, :, 1, 0, 0] = 3  # sample 1's right column is label 3
    tokens = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2, 1)
    pooled, ids = pool_labels(tokens, label)
    assert ids.tolist() == [3, 5, 3]
    assert pooled[:, 0].tolist() == [1.5, 0.0, 6.0]


def test_label_loss_ignores_near_identical_labels(
    embeddings: CaptionEmbeddings, labels: LabelEmbeddings
) -> None:
    """A label sharing the target's text is not a negative, even as a distractor."""
    head = TextAlignmentHead(embeddings=embeddings, embedding_dim=DIM, labels=labels)
    with torch.no_grad():
        head.proj.weight.copy_(torch.eye(DIM))
        head.proj.bias.zero_()
    candidates = labels.get(torch.arange(6))
    targets = torch.tensor([0, 2])
    loss, metrics = head(candidates[targets].clone(), candidates, targets)
    assert metrics["text/label_top1"] == 1
    unmasked = TextAlignmentHead(
        embeddings=embeddings,
        embedding_dim=DIM,
        labels=labels,
        same_text_threshold=2.0,
    )
    unmasked.load_state_dict(head.state_dict())
    assert unmasked(candidates[targets].clone(), candidates, targets)[0] > loss


def test_label_loss_without_labels_is_zero_with_gradients(
    embeddings: CaptionEmbeddings, labels: LabelEmbeddings
) -> None:
    """No labelled token still runs the head, so every rank reduces the same gradients."""
    head = TextAlignmentHead(embeddings=embeddings, embedding_dim=3, labels=labels)
    label = torch.full((1, 4, 4, 1, 4), float(MISSING_VALUE))
    tokens = torch.randn(1, 4, 3, requires_grad=True)
    loss, metrics = label_alignment_loss(head, tokens, label, pixels=2)
    loss.backward()
    assert loss == 0
    assert metrics["text/labels"] == 0
    assert head.proj.weight.grad is not None
