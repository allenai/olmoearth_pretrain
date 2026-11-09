"""Tests for grid captions and the caption_cell raster."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from upath import UPath

from olmoearth_pretrain.data.captions import (
    CaptionStore,
    build_caption_store,
    caption_cell_raster,
    patch_caption_keys,
)
from olmoearth_pretrain.data.collate import collate_single_masked_batched
from olmoearth_pretrain.data.constants import MISSING_VALUE
from olmoearth_pretrain.data.dataset import GetItemArgs, OlmoEarthDataset
from olmoearth_pretrain.dataset.convert_to_h5py import ConvertToH5py
from olmoearth_pretrain.datatypes import MaskValue, OlmoEarthSample
from olmoearth_pretrain.train.masking import RandomMaskingStrategy

CELLS = [f"cell {i} é" for i in range(16)]


def _store(tmp_path: Path, h5py_dir: str = "h5") -> str:
    captions = tmp_path / "captions"
    captions.mkdir()
    rows = [
        {"id": 1, "window": "a town", "cells": CELLS},
        {"id": 3, "window": "fields", "cells": [""] * 15 + ["a pond"]},
    ]
    (captions / "captions_0.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    out = str(tmp_path / "store")
    build_caption_store(str(captions), out, h5py_dir, num_windows=5)
    return out


def test_store_returns_captions_by_window_id(tmp_path: Path) -> None:
    """Captions come back by window id; uncaptioned ids are all masked."""
    store = CaptionStore(_store(tmp_path))
    assert len(store) == 5
    assert store.get(1) == ("a town", CELLS)
    assert store.get(3)[1][15] == "a pond"
    assert store.get(2) == ("", [""] * 16)
    assert store.cells_present(3).tolist() == [False] * 15 + [True]
    assert not store.cells_present(4).any()


def test_raster_marks_cells_and_masks_uncaptioned_ones() -> None:
    """Pixels carry their window and cell; cells without a caption are missing."""
    present = np.ones(16, dtype=bool)
    present[5] = False
    raster = caption_cell_raster(7, present, 8, 8)
    assert raster.shape == (8, 8, 1, 2)
    assert raster.dtype == np.float32
    assert (raster[0, 0, 0] == [7, 0]).all()
    assert (raster[7, 7, 0] == [7, 15]).all()
    assert (raster[0, 6, 0] == [7, 3]).all()
    assert (raster[2:4, 2:4] == MISSING_VALUE).all()


def _write_sample(h5py_dir: Path, index: int) -> None:
    t = 2
    with h5py.File(
        h5py_dir / ConvertToH5py.sample_file_pattern.format(index=index), "w"
    ) as f:
        f["sentinel2_l2a"] = np.ones((128, 128, t, 12), dtype=np.float32)
        f["timestamps"] = np.array([[1, 0, 2020], [1, 1, 2020]])


def test_getitem_crop_keeps_each_token_on_its_cell(tmp_path: Path) -> None:
    """A crop takes the matching piece of the raster; uncaptioned windows are missing."""
    h5py_dir = tmp_path / "h5"
    h5py_dir.mkdir()
    for i in (1, 2):
        _write_sample(h5py_dir, i)
    dataset = OlmoEarthDataset(
        h5py_dir=UPath(h5py_dir),
        training_modalities=["sentinel2_l2a", "caption_cell"],
        dtype=np.float32,
        normalize=False,
        caption_dir=_store(tmp_path, str(h5py_dir)),
    )
    args = GetItemArgs(idx=1, patch_size=8, sampled_hw_p=6, token_budget=10_000)

    np.random.seed(0)
    _, sample = dataset[args]
    cell, s2 = sample.caption_cell, sample.sentinel2_l2a
    assert isinstance(cell, np.ndarray) and isinstance(s2, np.ndarray)
    assert cell.shape == (48, 48, 1, 2)
    assert s2.shape[:2] == (48, 48)
    full = caption_cell_raster(1, np.ones(16, dtype=bool), 128, 128)
    assert any(
        np.array_equal(full[h : h + 48, w : w + 48], cell)
        for h in range(81)
        for w in range(81)
    )

    keys = patch_caption_keys(torch.from_numpy(cell[None]), 8)
    assert keys.shape == (1, 6, 6, 2)
    assert (keys[..., 0] == 1).all()

    _, uncaptioned = dataset[args._replace(idx=2)]
    assert isinstance(uncaptioned.caption_cell, np.ndarray)
    assert np.all(uncaptioned.caption_cell == MISSING_VALUE)


def test_caption_cell_needs_float32(tmp_path: Path) -> None:
    """Window ids don't survive float16, so the dataset refuses it."""
    with pytest.raises(ValueError, match="float32"):
        OlmoEarthDataset(
            h5py_dir=UPath(tmp_path),
            training_modalities=["caption_cell"],
            dtype=np.float16,
            caption_dir=_store(tmp_path),
        )


def test_collate_keeps_caption_cell_out_of_masking() -> None:
    """caption_cell survives collation unmasked by the strategy, MISSING to the model."""
    present = np.ones(16, dtype=bool)
    samples = [
        (
            2,
            OlmoEarthSample(
                sentinel2_l2a=np.ones((8, 8, 2, 12), dtype=np.float32),
                caption_cell=caption_cell_raster(i, present, 8, 8),
                timestamps=np.array([[1, 0, 2020], [1, 1, 2020]]),
            ),
        )
        for i in (4, 9)
    ]
    _, masked = collate_single_masked_batched(samples, None, RandomMaskingStrategy())
    assert masked.caption_cell is not None and masked.caption_cell_mask is not None
    assert masked.caption_cell.shape == (2, 8, 8, 1, 2)
    assert (masked.caption_cell[1, ..., 0] == 9).all()
    assert masked.caption_cell_mask.shape == (2, 8, 8, 1, 1)
    assert (masked.caption_cell_mask == MaskValue.MISSING.value).all()
    assert masked.sentinel2_l2a_mask is not None
