"""Tests for multi-layer decoder heads."""

import torch

from olmoearth_pretrain.evals.finetune.heads import (
    HeadType,
    MultiLayerClassificationHead,
    MultiLayerSegmentationHead,
)


class TestMultiLayerClassificationHead:
    """Tests for MultiLayerClassificationHead."""

    def test_output_shape(self) -> None:
        """Output should be (B, num_classes) from spatial input."""
        head = MultiLayerClassificationHead(emb_dim=768, num_classes=10)
        x = torch.randn(4, 8, 8, 768)
        out = head(x)
        assert out.shape == (4, 10)

    def test_custom_channels(self) -> None:
        """Should work with non-default conv and fc channels."""
        head = MultiLayerClassificationHead(
            emb_dim=256, num_classes=5, conv_channels=64, fc_channels=128
        )
        x = torch.randn(2, 4, 4, 256)
        out = head(x)
        assert out.shape == (2, 5)

    def test_spatial_invariant(self) -> None:
        """Should handle different spatial sizes due to global max pool."""
        head = MultiLayerClassificationHead(emb_dim=64, num_classes=3)
        out_small = head(torch.randn(1, 4, 4, 64))
        out_large = head(torch.randn(1, 16, 16, 64))
        assert out_small.shape == (1, 3)
        assert out_large.shape == (1, 3)

    def test_gradient_flows(self) -> None:
        """Gradients should flow through all layers."""
        head = MultiLayerClassificationHead(emb_dim=64, num_classes=3)
        x = torch.randn(2, 4, 4, 64, requires_grad=True)
        out = head(x)
        out.sum().backward()
        assert x.grad is not None
        for p in head.parameters():
            assert p.grad is not None


class TestMultiLayerSegmentationHead:
    """Tests for MultiLayerSegmentationHead."""

    def test_output_shape_default_channels(self) -> None:
        """Output should be 4x the spatial input size with default channels."""
        head = MultiLayerSegmentationHead(emb_dim=768, num_classes=10)
        # Input at 4x downsample: 16x16 patches for a 64x64 image
        x = torch.randn(2, 16, 16, 768)
        out = head(x)
        assert out.shape == (2, 10, 64, 64)

    def test_output_shape_small(self) -> None:
        """Should work with small spatial dims."""
        head = MultiLayerSegmentationHead(emb_dim=128, num_classes=5)
        x = torch.randn(1, 4, 4, 128)
        out = head(x)
        assert out.shape == (1, 5, 16, 16)

    def test_custom_channels(self) -> None:
        """Should work with custom channel configuration."""
        head = MultiLayerSegmentationHead(
            emb_dim=256, num_classes=3, channels=(128, 64)
        )
        # Two stages with one upsample -> 2x spatial increase
        x = torch.randn(2, 8, 8, 256)
        out = head(x)
        assert out.shape == (2, 3, 16, 16)

    def test_gradient_flows(self) -> None:
        """Gradients should flow through all layers."""
        head = MultiLayerSegmentationHead(emb_dim=64, num_classes=3, channels=(32, 16))
        x = torch.randn(1, 4, 4, 64, requires_grad=True)
        out = head(x)
        out.sum().backward()
        assert x.grad is not None
        for p in head.parameters():
            assert p.grad is not None


class TestHeadType:
    """Tests for HeadType enum."""

    def test_default_is_linear(self) -> None:
        """LINEAR value should be 'linear'."""
        assert HeadType.LINEAR == "linear"

    def test_multi_layer_value(self) -> None:
        """MULTI_LAYER value should be 'multi_layer'."""
        assert HeadType.MULTI_LAYER == "multi_layer"

    def test_from_string(self) -> None:
        """HeadType should be constructible from string values."""
        assert HeadType("linear") == HeadType.LINEAR
        assert HeadType("multi_layer") == HeadType.MULTI_LAYER
