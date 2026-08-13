"""Unit tests for probe modules in linear_probe.py."""

import pytest
import torch

from olmoearth_pretrain.evals.datasets.configs import TaskType
from olmoearth_pretrain.evals.linear_probe import (
    InterpolateLinearProbe,
    LinearProbe,
    ProbeInputNorm,
)


class TestLinearProbeClassification:
    """Tests for LinearProbe."""

    def test_output_shape_classification(self) -> None:
        """Classification probe: (B, D) -> (B, C)."""
        probe = LinearProbe(in_dim=32, num_classes=5, task_type=TaskType.CLASSIFICATION)
        x = torch.randn(4, 32)
        logits = probe(x)["logits"]
        assert logits.shape == (4, 5)

    def test_output_shape_segmentation(self) -> None:
        """Segmentation probe: (B, H_p, W_p, D) -> (B, C, H, W)."""
        probe = LinearProbe(
            in_dim=32,
            num_classes=5,
            task_type=TaskType.SEGMENTATION,
            num_output_pixels_per_side_of_patch=4,
        )
        # 8 patches per dim * 4 pixels per patch = 32 output pixels per dim
        x = torch.randn(2, 8, 8, 32)
        logits = probe(x)["logits"]
        assert logits.shape == (2, 5, 32, 32)

    def test_output_shape_scalar_regression(self) -> None:
        """Scalar regression probe: (B, D) -> (B,)."""
        probe = LinearProbe(
            in_dim=32, num_classes=1, task_type=TaskType.WINDOW_REGRESSION
        )
        x = torch.randn(4, 32)
        logits = probe(x)["logits"]
        assert logits.shape == (4,)


class TestInterpolateLinearProbe:
    """Tests for InterpolateLinearProbe."""

    def test_output_shape_segmentation(self) -> None:
        """InterpolateLinearProbe: (B, H_p, W_p, D) -> (B, C, H, W) via bilinear upsample."""
        probe = InterpolateLinearProbe(
            in_dim=32,
            num_classes=5,
            task_type=TaskType.SEGMENTATION,
            num_output_pixels_per_side_of_patch=4,
        )
        x = torch.randn(2, 8, 8, 32)
        logits = probe(x)["logits"]
        # 8 patches * 4 pixels per patch = 32
        assert logits.shape == (2, 5, 32, 32)

    def test_rejects_classification(self) -> None:
        """InterpolateLinearProbe should reject non-segmentation tasks."""
        with pytest.raises(ValueError, match="only supports segmentation"):
            InterpolateLinearProbe(
                in_dim=32,
                num_classes=5,
                task_type=TaskType.CLASSIFICATION,
                num_output_pixels_per_side_of_patch=4,
            )


class TestProbeInputNorm:
    """Tests for the linear probe's configurable input norm."""

    def test_classification_defaults_to_batchnorm(self) -> None:
        """Historical behavior is unchanged when nothing is configured."""
        probe = LinearProbe(in_dim=16, num_classes=4, task_type=TaskType.CLASSIFICATION)
        assert isinstance(probe.batchnorm, torch.nn.BatchNorm1d)

    def test_classification_none_drops_batchnorm(self) -> None:
        """NONE scores classification exactly like the dense probes."""
        probe = LinearProbe(
            in_dim=16,
            num_classes=4,
            task_type=TaskType.CLASSIFICATION,
            input_norm=ProbeInputNorm.NONE,
        )
        assert isinstance(probe.batchnorm, torch.nn.Identity)

    def test_segmentation_never_normalizes(self) -> None:
        """Dense probes have no input norm under either setting."""
        for norm in (ProbeInputNorm.BATCHNORM, ProbeInputNorm.NONE):
            probe = LinearProbe(
                in_dim=16,
                num_classes=4,
                task_type=TaskType.SEGMENTATION,
                num_output_pixels_per_side_of_patch=2,
                input_norm=norm,
            )
            assert isinstance(probe.batchnorm, torch.nn.Identity)

    def test_none_is_scale_sensitive(self) -> None:
        """Without BatchNorm the probe sees raw feature scale -- the point of the arm.

        With BatchNorm the same features scaled by a constant give identical
        logits (in eval mode the running stats absorb it after enough batches;
        here we check the training-mode batch statistics do).
        """
        x = torch.randn(32, 16)
        bn = LinearProbe(in_dim=16, num_classes=4, task_type=TaskType.CLASSIFICATION)
        none = LinearProbe(
            in_dim=16,
            num_classes=4,
            task_type=TaskType.CLASSIFICATION,
            input_norm=ProbeInputNorm.NONE,
        )
        bn.train()
        none.train()
        assert torch.allclose(bn(x)["logits"], bn(x * 10)["logits"], atol=1e-4)
        assert not torch.allclose(none(x)["logits"], none(x * 10)["logits"], atol=1e-4)
