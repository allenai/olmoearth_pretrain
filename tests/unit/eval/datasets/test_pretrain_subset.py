"""Tests for pretrain-subset eval labels."""

import numpy as np
import pytest
import torch

from olmoearth_pretrain.evals.datasets.configs import DATASET_TO_CONFIG
from olmoearth_pretrain.evals.datasets.pretrain_subset import (
    GLO30_BAND_ASPECT,
    GLO30_BAND_ELEVATION,
    GLO30_BAND_SLOPE,
    GLO30_LABEL_ASPECT_COS,
    GLO30_LABEL_ASPECT_SIN,
    GLO30_TARGET_MODALITY,
    PretrainSubsetDataset,
)
from olmoearth_pretrain.internal.all_evals import EVAL_TASKS


def test_split_indices_are_disjoint_and_deterministic() -> None:
    """A single shuffled population is sliced by configured per-split counts."""
    train = PretrainSubsetDataset._select_split_indices(
        total=100,
        split="train",
        seed=7,
        train_samples=10,
        valid_samples=8,
        test_samples=6,
    )
    valid = PretrainSubsetDataset._select_split_indices(
        total=100,
        split="valid",
        seed=7,
        train_samples=10,
        valid_samples=8,
        test_samples=6,
    )
    test = PretrainSubsetDataset._select_split_indices(
        total=100,
        split="test",
        seed=7,
        train_samples=10,
        valid_samples=8,
        test_samples=6,
    )
    train_again = PretrainSubsetDataset._select_split_indices(
        total=100,
        split="train",
        seed=7,
        train_samples=10,
        valid_samples=8,
        test_samples=6,
    )

    assert train == train_again
    assert len(train) == 10
    assert len(valid) == 8
    assert len(test) == 6
    assert not (set(train) & set(valid))
    assert not (set(train) & set(test))
    assert not (set(valid) & set(test))

    shuffled = np.random.RandomState(7).permutation(100).tolist()
    assert train == shuffled[:10]
    assert valid == shuffled[10:18]
    assert test == shuffled[18:24]


def test_worldcover_label_maps_class_codes() -> None:
    """WorldCover raw class codes map to contiguous labels."""
    raw = torch.tensor([[[[10], [20]], [[95], [100]]]])

    label = PretrainSubsetDataset._worldcover_label(raw)

    assert torch.equal(label, torch.tensor([[0, 1], [9, 10]]))


def test_osm_label_uses_argmax_and_ignores_empty_pixels() -> None:
    """OSM multi-channel rasters become single-class segmentation labels."""
    raw = torch.zeros(2, 2, 3)
    raw[0, 0, 1] = 1
    raw[0, 1, 2] = 1

    label = PretrainSubsetDataset._osm_label(raw)

    expected = torch.tensor([[1, 2], [-1, -1]])
    assert torch.equal(label, expected)


def test_srtm_label_is_continuous() -> None:
    """SRTM labels should stay floating-point elevations."""
    raw = torch.tensor([[[[1.5], [2.0]], [[3.25], [4.0]]]])

    label = PretrainSubsetDataset._srtm_label(raw)

    assert label.dtype == torch.float32
    assert torch.equal(label, torch.tensor([[1.5, 2.0], [3.25, 4.0]]))


def test_meta_canopy_label_is_continuous() -> None:
    """Meta canopy labels should stay floating-point heights in meters."""
    raw = torch.tensor([[[[0.0], [7.0]], [[19.0], [28.0]]]])

    label = PretrainSubsetDataset._meta_canopy_label(raw)

    assert label.dtype == torch.float32
    assert torch.equal(label, torch.tensor([[0.0, 7.0], [19.0, 28.0]]))


def _glo30_raw(elevation: float, slope: float, aspect: float) -> torch.Tensor:
    """Build a [1, 1, 1, 1, 3] glo30 target holding one pixel."""
    return torch.tensor([[[[[elevation, slope, aspect]]]]])


def test_glo30_label_requires_a_band_index() -> None:
    """glo30 carries 3 bands, so a single-channel probe must name one."""
    with pytest.raises(ValueError, match="target_band_index"):
        PretrainSubsetDataset._glo30_label(_glo30_raw(100.0, 5.0, 90.0), None)


def test_glo30_label_rejects_unknown_band_index() -> None:
    """Out-of-range band indices are a config error, not a silent wrap-around."""
    with pytest.raises(ValueError, match="Unsupported glo30 target_band_index"):
        PretrainSubsetDataset._glo30_label(_glo30_raw(100.0, 5.0, 90.0), 7)


def test_glo30_label_selects_stored_bands() -> None:
    """Elevation and slope come straight out of the stored band axis."""
    raw = torch.tensor([[[[100.0, 5.0, 90.0], [200.0, 10.0, 180.0]]]])

    elevation = PretrainSubsetDataset._glo30_label(raw, GLO30_BAND_ELEVATION)
    slope = PretrainSubsetDataset._glo30_label(raw, GLO30_BAND_SLOPE)

    assert torch.equal(elevation, torch.tensor([100.0, 200.0]))
    assert torch.equal(slope, torch.tensor([5.0, 10.0]))


@pytest.mark.parametrize(
    "aspect_deg,expected_sin,expected_cos",
    [
        (0.0, 0.0, 1.0),
        (90.0, 1.0, 0.0),
        (180.0, 0.0, -1.0),
        (270.0, -1.0, 0.0),
    ],
)
def test_glo30_aspect_sincos_encodes_compass_bearing(
    aspect_deg: float, expected_sin: float, expected_cos: float
) -> None:
    """The virtual aspect bands are sin/cos of the bearing in degrees."""
    raw = _glo30_raw(100.0, 5.0, aspect_deg)

    sin = PretrainSubsetDataset._glo30_label(raw, GLO30_LABEL_ASPECT_SIN)
    cos = PretrainSubsetDataset._glo30_label(raw, GLO30_LABEL_ASPECT_COS)

    assert sin.item() == pytest.approx(expected_sin, abs=1e-6)
    assert cos.item() == pytest.approx(expected_cos, abs=1e-6)


def test_glo30_aspect_sincos_is_continuous_across_north() -> None:
    """359 and 1 degrees are 2 degrees apart, and must encode that way.

    Raw degrees put them at opposite ends of the target range, which is the
    whole reason the probes regress sin/cos instead.
    """
    just_west = _glo30_raw(100.0, 5.0, 359.0)
    just_east = _glo30_raw(100.0, 5.0, 1.0)

    for band in (GLO30_LABEL_ASPECT_SIN, GLO30_LABEL_ASPECT_COS):
        west = PretrainSubsetDataset._glo30_label(just_west, band)
        east = PretrainSubsetDataset._glo30_label(just_east, band)
        assert abs(west.item() - east.item()) < 0.05

    raw_west = PretrainSubsetDataset._glo30_label(just_west, GLO30_BAND_ASPECT)
    raw_east = PretrainSubsetDataset._glo30_label(just_east, GLO30_BAND_ASPECT)
    assert abs(raw_west.item() - raw_east.item()) == pytest.approx(358.0)


@pytest.mark.parametrize(
    "band_index",
    [GLO30_BAND_ASPECT, GLO30_LABEL_ASPECT_SIN, GLO30_LABEL_ASPECT_COS],
)
def test_glo30_flat_pixels_are_nan_for_every_aspect_band(band_index: int) -> None:
    """Flat pixels have no bearing, so they must drop out of the loss.

    Left unmasked, the -1 sentinel normalizes to almost exactly due north.
    """
    raw = torch.tensor([[[[100.0, 0.0, -1.0], [200.0, 10.0, 90.0]]]])

    label = PretrainSubsetDataset._glo30_label(raw, band_index)

    assert torch.isnan(label[0])
    assert torch.isfinite(label[1])


def test_glo30_flat_pixels_do_not_affect_elevation_or_slope() -> None:
    """The -1 sentinel is aspect-only; elevation and slope stay valid there."""
    raw = torch.tensor([[[[100.0, 0.0, -1.0], [200.0, 10.0, 90.0]]]])

    for band_index in (GLO30_BAND_ELEVATION, GLO30_BAND_SLOPE):
        label = PretrainSubsetDataset._glo30_label(raw, band_index)
        assert torch.isfinite(label).all()


def test_glo30_probes_are_wired_to_valid_band_indices() -> None:
    """Every registered glo30 probe names a band the label builder supports."""
    valid = {
        GLO30_BAND_ELEVATION,
        GLO30_BAND_SLOPE,
        GLO30_BAND_ASPECT,
        GLO30_LABEL_ASPECT_SIN,
        GLO30_LABEL_ASPECT_COS,
    }
    glo30_tasks = {
        name: task
        for name, task in EVAL_TASKS.items()
        if task.pretrain_target_modality == GLO30_TARGET_MODALITY
    }
    assert glo30_tasks, "expected glo30 probes to be registered"
    for name, task in glo30_tasks.items():
        assert task.pretrain_target_band_index in valid, name
        assert task.dataset in DATASET_TO_CONFIG, name

    # Aspect is probed via sin/cos; no probe should regress raw degrees.
    aspect_tasks = [
        name
        for name, task in glo30_tasks.items()
        if task.pretrain_target_band_index == GLO30_BAND_ASPECT
    ]
    assert not aspect_tasks, (
        f"raw-degree aspect probes should be retired: {aspect_tasks}"
    )


def test_geographic_split_is_deterministic() -> None:
    """Geographic splits should not depend on Python's randomized hash seed."""
    latlons = torch.stack(
        [
            torch.linspace(-40, 40, 100),
            torch.linspace(-120, 120, 100),
        ],
        dim=1,
    ).numpy()
    candidate_positions = torch.arange(100).numpy()

    split = PretrainSubsetDataset._geographic_split_positions(
        latlons=latlons,
        candidate_positions=candidate_positions,
        split="train",
        seed=7,
        train_samples=10,
        valid_samples=8,
        test_samples=6,
    )
    split_again = PretrainSubsetDataset._geographic_split_positions(
        latlons=latlons,
        candidate_positions=candidate_positions,
        split="train",
        seed=7,
        train_samples=10,
        valid_samples=8,
        test_samples=6,
    )

    assert split.tolist() == split_again.tolist()


def test_geographic_splits_are_disjoint() -> None:
    """Geographic train/valid/test splits should assign bins once, without overlap."""
    latlons = torch.stack(
        [
            torch.linspace(-40, 40, 100),
            torch.linspace(-120, 120, 100),
        ],
        dim=1,
    ).numpy()
    candidate_positions = torch.arange(100).numpy()

    kwargs = dict(
        latlons=latlons,
        candidate_positions=candidate_positions,
        seed=7,
        train_samples=40,
        valid_samples=10,
        test_samples=10,
    )
    train = PretrainSubsetDataset._geographic_split_positions(split="train", **kwargs)
    valid = PretrainSubsetDataset._geographic_split_positions(split="valid", **kwargs)
    test = PretrainSubsetDataset._geographic_split_positions(split="test", **kwargs)

    assert not (set(train.tolist()) & set(valid.tolist()))
    assert not (set(train.tolist()) & set(test.tolist()))
    assert not (set(valid.tolist()) & set(test.tolist()))
