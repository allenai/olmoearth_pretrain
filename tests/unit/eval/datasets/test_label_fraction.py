"""Tests for low-label fraction helpers."""

import pytest

from olmoearth_pretrain.evals.datasets import (
    EvalDatasetPartition,
    fraction_to_partition,
    scale_train_samples,
)


def test_fraction_to_partition_maps_supported_values() -> None:
    """Supported label fractions map to existing partition names."""
    assert fraction_to_partition(1.0) == EvalDatasetPartition.TRAIN1X
    assert fraction_to_partition(0.1) == EvalDatasetPartition.TRAIN_010X
    assert fraction_to_partition(0.01) == EvalDatasetPartition.TRAIN_001X


def test_fraction_to_partition_rejects_unknown_values() -> None:
    """Only fractions with existing partition files are accepted."""
    with pytest.raises(ValueError, match="Unsupported label_fraction"):
        fraction_to_partition(0.3)


def test_scale_train_samples() -> None:
    """Pretrain probes scale train samples from their configured full-data count."""
    assert scale_train_samples(6144, 0.1) == 614
    assert scale_train_samples(8, 0.01) == 1

