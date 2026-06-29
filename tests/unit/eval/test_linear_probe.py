"""Unit tests for probe modules in linear_probe.py."""

import pytest
import torch

from olmoearth_pretrain.evals.datasets.configs import TaskType
from olmoearth_pretrain.evals.linear_probe import (
    InterpolateLinearProbe,
    LinearProbe,
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
