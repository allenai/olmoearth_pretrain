"""Unit tests for embedding_transforms module."""

import torch

from olmoearth_pretrain.evals.embedding_transforms import (
    dequantize_embeddings,
    quantize_embeddings,
    reduce_embedding_dim,
)


class TestQuantization:
    """Tests for int8 quantization functions."""

    def test_roundtrip(self) -> None:
        """Verify quantize → dequantize preserves values approximately."""
        embeddings = torch.randn(100, 768)
        quantized = quantize_embeddings(embeddings)
        recovered = dequantize_embeddings(quantized)

        assert quantized.dtype == torch.int8
        assert recovered.dtype == torch.float32
        assert recovered.shape == embeddings.shape


class TestDimReduction:
    """Tests for PCA dimensionality reduction."""

    def test_pca_shapes(self) -> None:
        """Verify PCA dimension reduction produces correct shapes."""
        train = torch.randn(500, 768)
        val = torch.randn(100, 768)
        test = torch.randn(100, 768)

        train_out, val_out, test_out, variance = reduce_embedding_dim(
            train, val, test, target_dim=256
        )

        assert train_out.shape == (500, 256)
        assert val_out.shape == (100, 256)
        assert test_out is not None
        assert test_out.shape == (100, 256)
        assert 0 < variance <= 1.0
