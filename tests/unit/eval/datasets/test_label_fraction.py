"""Tests for low-label fraction helpers."""

from olmoearth_pretrain.evals.datasets import scale_train_samples


def test_scale_train_samples() -> None:
    """Pretrain probes scale train samples from their configured full-data count."""
    assert scale_train_samples(6144, 0.1) == 614
    assert scale_train_samples(8, 0.01) == 1
