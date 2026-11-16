"""Tests for held-out classes and locations in text targets."""

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest
from upath import UPath

from olmoearth_pretrain.data.captions import build_caption_store
from olmoearth_pretrain.data.constants import MISSING_VALUE
from olmoearth_pretrain.data.dataset import GetItemArgs, OlmoEarthDataset
from olmoearth_pretrain.data.text_holdout import TextHoldout
from olmoearth_pretrain.data.text_labels import (
    LAND_COVER,
    OSM_CHANNEL,
    LabelVocabulary,
    layer_label_names,
)
from olmoearth_pretrain.dataset.convert_to_h5py import ConvertToH5py
from olmoearth_pretrain.nn.text_alignment import LabelEmbeddings

HELD = (63.22200293249243, 44.87249059739375)
BUFFER = (25.989827788396326, 171.82118383886257)
TRAIN = (-23.57879364187751, -158.38514727754563)
TEST_POINTS = (
    "/weka/dfive-default/agent/piperbots/2026-11-16-piperw-clean-splits/"
    "split_test_points.json"
)


def _spec(path: Path, held_names: list[str]) -> str:
    spec = {
        "held_names": held_names,
        "region_deg": 2.0,
        "buffer_deg": 0.5,
        "held_fraction": 0.1,
    }
    path.write_text(json.dumps(spec))
    return str(path)


def test_location_split_matches_clipearth(tmp_path: Path) -> None:
    """Points from ClipEarth's split land where its locations.py put them."""
    holdout = TextHoldout.load(_spec(tmp_path / "spec.json", []))
    lat, lon = zip(HELD, BUFFER, TRAIN)
    assert holdout.location_split(np.array(lat), np.array(lon)).tolist() == [
        "held",
        "buffer",
        "train",
    ]
    assert holdout.masked_locations(np.array(lat), np.array(lon)).tolist() == [
        True,
        True,
        False,
    ]


@pytest.mark.skipif(not os.path.exists(TEST_POINTS), reason="needs the split's points")
def test_location_split_matches_all_test_points(tmp_path: Path) -> None:
    """All of the split's recorded test points."""
    points = json.load(open(TEST_POINTS))
    holdout = TextHoldout.load(_spec(tmp_path / "spec.json", []))
    split = holdout.location_split(np.array(points["lat"]), np.array(points["lon"]))
    assert split.tolist() == points["split"]


def test_held_label_ids_normalize_names(tmp_path: Path) -> None:
    """Held names match vocabulary names however they are written."""
    vocab = LabelVocabulary(["Open water", "Wind_turbine", "Tree cover"])
    holdout = TextHoldout.load(
        _spec(tmp_path / "spec.json", ["wind turbine", "tree cover", "not in vocab"])
    )
    assert holdout.held_label_ids(vocab).tolist() == [1, 2]


def _write(path: Path, label: np.ndarray | None) -> None:
    with h5py.File(path, "w") as f:
        f["sentinel2_l2a"] = np.ones((8, 8, 2, 12), dtype=np.uint16)
        f["worldcover"] = np.full((8, 8, 1, 1), 80, dtype=np.uint8)
        f["timestamps"] = np.array([[1, 0, 2020], [1, 1, 2020]])
        if label is not None:
            f["text_label"] = label


def test_dataset_withholds_held_labels_and_locations(tmp_path: Path) -> None:
    """Held labels become missing; windows in held or buffer locations lose all labels."""
    vocab_dir = tmp_path / "vocab"
    vocab = LabelVocabulary([*layer_label_names(), "solar farm"])
    vocab.save(str(vocab_dir))
    h5py_dir = tmp_path / "h5"
    h5py_dir.mkdir()
    stored = np.full((8, 8, 1, 4), MISSING_VALUE, dtype=np.float32)
    stored[..., 0, LAND_COVER] = vocab.id("solar farm")
    stored[:4, :, 0, OSM_CHANNEL] = vocab.id("wind turbine")
    for i, point in enumerate((TRAIN, HELD, BUFFER)):
        _write(h5py_dir / ConvertToH5py.sample_file_pattern.format(index=i), stored)
    _write(h5py_dir / ConvertToH5py.sample_file_pattern.format(index=3), None)
    np.save(
        h5py_dir / ConvertToH5py.latlon_distribution_fname,
        np.array([TRAIN, HELD, BUFFER, TRAIN]),
    )
    dataset = OlmoEarthDataset(
        h5py_dir=UPath(h5py_dir),
        training_modalities=["sentinel2_l2a", "worldcover", "text_label"],
        dtype=np.float32,
        normalize=False,
        label_vocab_dir=str(vocab_dir),
        text_holdout_spec=_spec(tmp_path / "spec.json", ["solar farm", "open water"]),
    )
    args = GetItemArgs(idx=0, patch_size=1, sampled_hw_p=8, token_budget=10_000)
    _, kept = dataset[args]
    assert kept.text_label is not None
    assert (kept.text_label[..., LAND_COVER] == MISSING_VALUE).all()
    assert (kept.text_label[:4, :, 0, OSM_CHANNEL] == vocab.id("wind turbine")).all()
    for idx in (1, 2, 3):
        # Held and buffer windows lose everything; window 3's only label is held out.
        _, sample = dataset[args._replace(idx=idx)]
        assert sample.text_label is None or (sample.text_label == MISSING_VALUE).all()


def test_held_labels_are_never_distractors(tmp_path: Path) -> None:
    """LabelEmbeddings leaves the spec's held labels out of its distractor pool."""
    path = tmp_path / "labels"
    LabelVocabulary(["a", "b", "c", "d"]).save(str(path))
    np.save(path / "embeddings.npy", np.eye(4, dtype=np.float32))
    (path / "meta.json").write_text(json.dumps({"dim": 4}))
    labels = LabelEmbeddings(str(path), holdout_spec=_spec(tmp_path / "s.json", ["b"]))
    assert labels.distractor_ids.tolist() == [0, 2, 3]
    assert LabelEmbeddings(str(path)).distractor_ids.tolist() == [0, 1, 2, 3]


def test_dataset_withholds_captions_in_held_locations(tmp_path: Path) -> None:
    """A captioned window in a held-out region gets no caption_cell; others keep theirs."""
    h5py_dir = tmp_path / "h5"
    h5py_dir.mkdir()
    for i in (0, 1):
        _write(h5py_dir / ConvertToH5py.sample_file_pattern.format(index=i), None)
    np.save(h5py_dir / ConvertToH5py.latlon_distribution_fname, np.array([TRAIN, HELD]))
    captions = tmp_path / "captions"
    captions.mkdir()
    rows = [{"id": i, "window": "a town", "cells": ["a field"] * 16} for i in (0, 1)]
    (captions / "c.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    build_caption_store(str(captions), str(tmp_path / "store"), str(h5py_dir), 2)
    dataset = OlmoEarthDataset(
        h5py_dir=UPath(h5py_dir),
        training_modalities=["sentinel2_l2a", "caption_cell"],
        dtype=np.float32,
        normalize=False,
        caption_dir=str(tmp_path / "store"),
        text_holdout_spec=_spec(tmp_path / "spec.json", []),
    )
    args = GetItemArgs(idx=0, patch_size=1, sampled_hw_p=8, token_budget=10_000)
    _, kept = dataset[args]
    _, held = dataset[args._replace(idx=1)]
    assert kept.caption_cell is not None and (kept.caption_cell != MISSING_VALUE).all()
    assert held.caption_cell is None or (held.caption_cell == MISSING_VALUE).all()
