"""Tests for low-label fraction helpers."""

import json
from pathlib import Path

import pytest
import torch

from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.evals.datasets import scale_train_samples
from olmoearth_pretrain.evals.datasets.utils import (
    build_band_stat_arrays,
    build_masked_eval_sample,
    eval_collate_fn_variable_time,
    load_label_fraction_partition_indices,
    resolve_label_fraction_partition,
)
from olmoearth_pretrain.modalities import Modality


def test_scale_train_samples() -> None:
    """Pretrain probes scale train samples from their configured full-data count."""
    assert scale_train_samples(6144, 0.1) == 614
    assert scale_train_samples(8, 0.01) == 1


def test_resolve_label_fraction_partition() -> None:
    """Supported low-label fractions map to partition basenames."""
    assert resolve_label_fraction_partition(0.01) == "0.01x_train"
    assert resolve_label_fraction_partition(1.0) is None


def test_resolve_label_fraction_partition_rejects_unknown_fraction() -> None:
    """Unsupported fractions should fail with a useful message."""
    with pytest.raises(ValueError, match="Supported values are: 0.01, 0.02"):
        resolve_label_fraction_partition(0.03, {0.01: "0.01x_train", 0.02: None})


def test_load_label_fraction_partition_indices(tmp_path: Path) -> None:
    """Partition index files are loaded from the conventional filename."""
    partition_path = tmp_path / "0.01x_train_partition.json"
    partition_path.write_text(json.dumps([4, 1, 8]))

    assert load_label_fraction_partition_indices(tmp_path, "0.01x_train") == [4, 1, 8]


def test_build_band_stat_arrays_preserves_band_order() -> None:
    """Band stat arrays should follow the requested band order."""
    stats = {
        "b": {"mean": 2.0, "std": 20.0, "min": -2.0, "max": 200.0},
        "a": {"mean": 1.0, "std": 10.0, "min": -1.0, "max": 100.0},
    }

    means, stds, mins, maxs = build_band_stat_arrays(stats, ["a", "b"])

    assert means.tolist() == [1.0, 2.0]
    assert stds.tolist() == [10.0, 20.0]
    assert mins.tolist() == [-1.0, -2.0]
    assert maxs.tolist() == [100.0, 200.0]


def test_build_band_stat_arrays_rejects_missing_band() -> None:
    """Missing band stats should fail before silently misnormalizing."""
    with pytest.raises(KeyError, match="b not found"):
        build_band_stat_arrays(
            {"a": {"mean": 1.0, "std": 1.0, "min": 0.0, "max": 2.0}}, ["a", "b"]
        )


def test_build_masked_eval_sample_adds_modality_masks() -> None:
    """Eval sample helper should build the standard modality masks."""
    sample = build_masked_eval_sample(
        {
            Modality.SENTINEL2_L2A.name: torch.ones(
                2, 2, 1, Modality.SENTINEL2_L2A.num_bands
            )
        },
        timestamps=torch.ones(1, 3, dtype=torch.long),
    )

    assert sample.sentinel2_l2a is not None
    assert sample.sentinel2_l2a_mask is not None
    assert sample.sentinel2_l2a_mask.shape == (
        2,
        2,
        1,
        Modality.SENTINEL2_L2A.num_band_sets,
    )


def test_build_masked_eval_sample_rejects_empty_modalities() -> None:
    """Eval samples need at least one modality tensor."""
    with pytest.raises(ValueError, match="At least one modality"):
        build_masked_eval_sample({}, timestamps=torch.ones(1, 3, dtype=torch.long))


def test_eval_collate_variable_time_pads_masks_by_band_set_count() -> None:
    """Variable-time eval masks should pad over band sets, not raw bands."""
    sample_short = MaskedOlmoEarthSample(
        timestamps=torch.ones(1, 3),
        sentinel2_l2a=torch.ones(2, 2, 1, Modality.SENTINEL2_L2A.num_bands),
        sentinel2_l2a_mask=torch.zeros(
            2, 2, 1, Modality.SENTINEL2_L2A.num_band_sets, dtype=torch.long
        ),
    )
    sample_long = MaskedOlmoEarthSample(
        timestamps=torch.ones(3, 3),
        sentinel2_l2a=torch.ones(2, 2, 3, Modality.SENTINEL2_L2A.num_bands),
        sentinel2_l2a_mask=torch.zeros(
            2, 2, 3, Modality.SENTINEL2_L2A.num_band_sets, dtype=torch.long
        ),
    )

    collated, targets = eval_collate_fn_variable_time(
        [(sample_short, torch.tensor(0)), (sample_long, torch.tensor(1))]
    )

    assert targets.tolist() == [0, 1]
    assert collated.sentinel2_l2a is not None
    assert collated.sentinel2_l2a_mask is not None
    assert collated.sentinel2_l2a.shape == (
        2,
        2,
        2,
        3,
        Modality.SENTINEL2_L2A.num_bands,
    )
    assert collated.sentinel2_l2a_mask.shape == (
        2,
        2,
        2,
        3,
        Modality.SENTINEL2_L2A.num_band_sets,
    )
    assert (
        collated.sentinel2_l2a_mask[0, :, :, 1:, :] == MaskValue.MISSING.value
    ).all()
