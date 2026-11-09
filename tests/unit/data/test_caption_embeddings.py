"""Tests for caption embeddings."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import sentence_transformers
import torch

from olmoearth_pretrain.data import caption_embeddings
from olmoearth_pretrain.data.caption_embeddings import CaptionEmbeddings, embed_store
from olmoearth_pretrain.data.captions import build_caption_store
from olmoearth_pretrain.data.constants import MISSING_VALUE

DIM = 4


class FakeModel:
    """Embeds a text as a fixed vector picked by its length."""

    calls = 0

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Ignore the model name and options."""

    def get_sentence_embedding_dimension(self) -> int:
        """The embedding size."""
        return DIM

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        """One vector per text."""
        FakeModel.calls += 1
        return np.stack([fake_vector(t) for t in texts])


def fake_vector(text: str) -> np.ndarray:
    """What FakeModel returns for ``text``."""
    v = np.array([len(text), 1, 2, 3], dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def embedded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Embeddings of a 3-window store: window 1 captioned, cell 2 empty."""
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeModel)
    monkeypatch.setattr(caption_embeddings, "CHUNK", 2)
    captions = tmp_path / "captions"
    captions.mkdir()
    cells = [f"cell {'x' * i}" for i in range(16)]
    cells[2] = ""
    row = {"id": 1, "window": "a whole window", "cells": cells}
    (captions / "c.jsonl").write_text(json.dumps(row) + "\n")
    build_caption_store(str(captions), str(tmp_path / "store"), "h5", num_windows=3)
    out = str(tmp_path / "emb")
    os.makedirs(out)
    embed_store(str(tmp_path / "store"), out, "fake", None, batch_size=8)
    return out


def test_embed_store_places_each_caption_by_window_and_row(embedded: str) -> None:
    """Rows follow window then cells; empty captions and windows stay zero."""
    array = np.load(os.path.join(embedded, "embeddings.npy"))
    assert array.shape == (3, 17, DIM)
    assert array.dtype == np.float16
    np.testing.assert_allclose(array[1, 0], fake_vector("a whole window"), atol=1e-3)
    np.testing.assert_allclose(array[1, 6], fake_vector("cell xxxxx"), atol=1e-3)
    assert not array[1, 3].any()
    assert not array[0].any() and not array[2].any()


def test_embed_store_resumes_without_redoing_chunks(embedded: str) -> None:
    """A rerun skips chunks already marked done."""
    FakeModel.calls = 0
    store = os.path.join(os.path.dirname(embedded), "store")
    embed_store(store, embedded, "fake", None, batch_size=8)
    assert FakeModel.calls == 0


def test_lookup_returns_unit_vectors_and_masks_missing_keys(embedded: str) -> None:
    """Keys of captioned cells get unit vectors; missing or empty ones are invalid."""
    keys = torch.tensor(
        [[1.0, 5.0], [1.0, 2.0], [MISSING_VALUE, MISSING_VALUE], [0.0, 3.0]]
    )
    vec, valid = CaptionEmbeddings(embedded, dim=2).lookup(keys)
    assert vec.shape == (4, 2)
    assert valid.tolist() == [True, False, False, False]
    expected = fake_vector("cell xxxxx")[:2]
    np.testing.assert_allclose(vec[0], expected / np.linalg.norm(expected), atol=1e-3)
    assert not vec[1:].any()


def test_write_rows_lands_on_disk_without_touching_neighbours(tmp_path: Path) -> None:
    """Rows go to their byte offset and read back through a fresh, unmapped load."""
    path = str(tmp_path / "a.npy")
    np.lib.format.open_memmap(path, "w+", np.float16, (4, 17, DIM)).flush()
    offset = np.load(path, mmap_mode="r").offset
    block = np.ones((2, 17, DIM), dtype=np.float16)
    caption_embeddings.write_rows(path, offset + 17 * DIM * 2, block)
    array = np.load(path)
    assert not array[0].any() and not array[3].any()
    assert (array[1:3] == 1).all()
