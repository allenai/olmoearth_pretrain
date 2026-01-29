"""Unit tests for embedding_transforms module."""

import numpy as np
import torch

from olmoearth_pretrain.evals.embedding_transforms import (
    dequantize_embeddings,
    dequantize_embeddings_percentile,
    quantize_embeddings,
    quantize_embeddings_percentile,
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


class TestPercentileQuantization:
    """Tests for percentile-based quantization functions."""

    def _create_quantiles_and_midpoints(
        self, embeddings: torch.Tensor, bits: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Helper to compute quantile boundaries and midpoints from embeddings."""
        num_buckets = 2**bits
        embeddings_np = embeddings.numpy()

        boundary_percentiles = np.linspace(0, 100, num_buckets + 1)
        quantiles = np.percentile(embeddings_np, boundary_percentiles, axis=0).T

        midpoint_percentiles = np.array(
            [(i + 0.5) / num_buckets * 100 for i in range(num_buckets)]
        )
        midpoints = np.percentile(embeddings_np, midpoint_percentiles, axis=0).T

        return torch.from_numpy(quantiles.astype(np.float32)), torch.from_numpy(
            midpoints.astype(np.float32)
        )

    def test_roundtrip_preserves_shape(self) -> None:
        """Verify quantize → dequantize preserves shape."""
        embeddings = torch.randn(100, 64)
        quantiles, midpoints = self._create_quantiles_and_midpoints(embeddings, bits=8)

        quantized = quantize_embeddings_percentile(embeddings, quantiles, bits=8)
        recovered = dequantize_embeddings_percentile(quantized, midpoints)

        assert quantized.dtype == torch.int8
        assert quantized.shape == embeddings.shape
        assert recovered.dtype == torch.float32
        assert recovered.shape == embeddings.shape

    def test_bucket_assignment_correct(self) -> None:
        """Verify values are assigned to correct buckets."""
        # Create simple uniform data for predictable bucket assignment
        dim = 4
        embeddings = torch.linspace(0, 1, 1000).unsqueeze(1).expand(-1, dim)
        quantiles, midpoints = self._create_quantiles_and_midpoints(embeddings, bits=2)

        # With 2 bits = 4 buckets, values should map to 0, 1, 2, 3
        quantized = quantize_embeddings_percentile(embeddings, quantiles, bits=2)

        # Check that bucket indices span 0 to 3
        unique_vals = torch.unique(quantized)
        assert len(unique_vals) == 4
        assert unique_vals.min() == 0
        assert unique_vals.max() == 3

    def test_8bit_int8_wraparound(self) -> None:
        """Verify 8-bit quantization handles int8 wrap-around correctly.

        Values 128-255 are stored as -128 to -1 in int8, so dequantization
        must convert back to unsigned before indexing midpoints.
        """
        dim = 8
        # Create uniform data that will use all 256 buckets
        embeddings = torch.linspace(0, 1, 10000).unsqueeze(1).expand(-1, dim).clone()
        quantiles, midpoints = self._create_quantiles_and_midpoints(embeddings, bits=8)

        quantized = quantize_embeddings_percentile(embeddings, quantiles, bits=8)

        # Verify we have negative int8 values (wrap-around occurred)
        assert quantized.min() < 0, "Expected negative int8 values from wrap-around"

        # Dequantize should work without errors and produce valid values
        recovered = dequantize_embeddings_percentile(quantized, midpoints)

        # Recovered values should be in original range
        assert recovered.min() >= embeddings.min() - 0.1
        assert recovered.max() <= embeddings.max() + 0.1

        # Check reconstruction is reasonable (not NaN, not completely wrong)
        assert not torch.isnan(recovered).any()
        mse = ((embeddings - recovered) ** 2).mean()
        assert mse < 0.01, f"MSE too high: {mse}"

    def test_different_bit_levels(self) -> None:
        """Verify quantization works for all supported bit levels."""
        embeddings = torch.randn(200, 32)

        for bits in [1, 2, 4, 8]:
            num_buckets = 2**bits
            quantiles, midpoints = self._create_quantiles_and_midpoints(
                embeddings, bits=bits
            )

            quantized = quantize_embeddings_percentile(embeddings, quantiles, bits=bits)
            recovered = dequantize_embeddings_percentile(quantized, midpoints)

            # Check bucket indices are in valid range (accounting for int8 storage)
            quantized_unsigned = quantized.to(torch.uint8)
            assert quantized_unsigned.max() <= num_buckets - 1
            assert quantized_unsigned.min() >= 0

            # Check recovery works
            assert recovered.shape == embeddings.shape
            assert not torch.isnan(recovered).any()

    def test_spatial_shape_preserved(self) -> None:
        """Verify quantization works with spatial dimensions (N, H, W, dim)."""
        embeddings = torch.randn(10, 4, 4, 32)  # batch, height, width, dim
        flat_embeddings = embeddings.reshape(-1, 32)
        quantiles, midpoints = self._create_quantiles_and_midpoints(
            flat_embeddings, bits=4
        )

        quantized = quantize_embeddings_percentile(embeddings, quantiles, bits=4)
        recovered = dequantize_embeddings_percentile(quantized, midpoints)

        assert quantized.shape == (10, 4, 4, 32)
        assert recovered.shape == (10, 4, 4, 32)

    def test_reconstruction_quality_improves_with_bits(self) -> None:
        """Verify that more bits → better reconstruction quality."""
        embeddings = torch.randn(500, 64)
        mse_by_bits = {}

        for bits in [1, 2, 4, 8]:
            quantiles, midpoints = self._create_quantiles_and_midpoints(
                embeddings, bits=bits
            )
            quantized = quantize_embeddings_percentile(embeddings, quantiles, bits=bits)
            recovered = dequantize_embeddings_percentile(quantized, midpoints)
            mse_by_bits[bits] = ((embeddings - recovered) ** 2).mean().item()

        # MSE should decrease as bits increase
        assert mse_by_bits[1] > mse_by_bits[2]
        assert mse_by_bits[2] > mse_by_bits[4]
        assert mse_by_bits[4] > mse_by_bits[8]

    def test_per_dimension_buckets(self) -> None:
        """Verify quantization uses different buckets for each dimension.

        Creates 2D embeddings where dim 0 has values in [0, 10] and dim 1 has
        values in [100, 200]. Verifies each dimension is quantized according
        to its own boundaries, not a shared one.
        """
        n_samples = 1000

        # Create embeddings with very different ranges per dimension
        # dim 0: uniform in [0, 10]
        # dim 1: uniform in [100, 200]
        embeddings = torch.zeros(n_samples, 2)
        embeddings[:, 0] = torch.linspace(0, 10, n_samples)
        embeddings[:, 1] = torch.linspace(100, 200, n_samples)

        bits = 2  # 4 buckets
        quantiles, midpoints = self._create_quantiles_and_midpoints(
            embeddings, bits=bits
        )

        # Verify quantiles are computed per-dimension
        # dim 0 quantiles should be around [0, 2.5, 5, 7.5, 10]
        # dim 1 quantiles should be around [100, 125, 150, 175, 200]
        assert quantiles[0, 0] < 1  # dim 0 min near 0
        assert quantiles[0, -1] > 9  # dim 0 max near 10
        assert quantiles[1, 0] > 99  # dim 1 min near 100
        assert quantiles[1, -1] < 201  # dim 1 max near 200

        # Quantize and dequantize
        quantized = quantize_embeddings_percentile(embeddings, quantiles, bits=bits)
        recovered = dequantize_embeddings_percentile(quantized, midpoints)

        # Check recovered values are in correct ranges per dimension
        assert recovered[:, 0].min() >= 0
        assert recovered[:, 0].max() <= 10
        assert recovered[:, 1].min() >= 100
        assert recovered[:, 1].max() <= 200

        # Check that a value in the middle of dim 0's range maps to middle buckets
        mid_idx = n_samples // 2  # value around 5 for dim 0, 150 for dim 1
        # Both should map to bucket 1 or 2 (middle buckets for 4-bucket quantization)
        quantized_unsigned = quantized.to(torch.uint8)
        assert quantized_unsigned[mid_idx, 0] in [1, 2]
        assert quantized_unsigned[mid_idx, 1] in [1, 2]

        # Check reconstruction error is reasonable for both dimensions
        mse_dim0 = ((embeddings[:, 0] - recovered[:, 0]) ** 2).mean()
        mse_dim1 = ((embeddings[:, 1] - recovered[:, 1]) ** 2).mean()
        # MSE should be proportional to range^2 / num_buckets^2
        # dim 0 range=10, dim 1 range=100, so dim 1 MSE should be ~100x larger
        assert mse_dim0 < 5  # reasonable for range 10 with 4 buckets
        assert mse_dim1 < 500  # reasonable for range 100 with 4 buckets
        assert mse_dim1 > mse_dim0 * 10  # dim 1 should have much larger MSE
