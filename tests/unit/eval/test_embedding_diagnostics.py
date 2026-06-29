"""Unit tests for embedding diagnostics."""

import pytest
import torch

from olmoearth_pretrain.evals.embedding_diagnostics import (
    compute_embedding_diagnostics,
    compute_spatial_embedding_diagnostics,
    effective_rank,
    embedding_norm_stats,
    pairwise_cosine_stats,
    uniformity,
)


class TestEffectiveRank:
    """Tests for effective_rank function."""

    def test_identity_matrix(self) -> None:
        """Identity matrix has effective rank = D."""
        D = 16
        embeddings = torch.eye(D)
        rank = effective_rank(embeddings)
        assert abs(rank - D) < 0.01

    def test_rank_one(self) -> None:
        """Repeated row gives rank 1."""
        row = torch.randn(1, 32)
        embeddings = row.expand(100, -1)
        rank = effective_rank(embeddings)
        assert abs(rank - 1.0) < 0.01

    def test_rank_two(self) -> None:
        """Two distinct directions give rank ~2."""
        N, D = 100, 32
        a = torch.randn(1, D)
        b = torch.randn(1, D)
        coeffs = torch.randn(N, 2)
        embeddings = coeffs[:, 0:1] * a + coeffs[:, 1:2] * b
        rank = effective_rank(embeddings)
        assert 1.5 < rank < 2.5

    def test_random_full_rank(self) -> None:
        """Random Gaussian has high effective rank."""
        N, D = 200, 64
        embeddings = torch.randn(N, D)
        rank = effective_rank(embeddings)
        assert rank > D * 0.7


class TestUniformity:
    """Tests for uniformity function."""

    def test_uniform_better_than_collapsed(self) -> None:
        """Random embeddings have lower (better) uniformity than collapsed."""
        uniform = torch.randn(100, 32)
        collapsed = torch.randn(100, 32) * 0.01 + torch.randn(1, 32)
        u_uniform = uniformity(uniform)
        u_collapsed = uniformity(collapsed)
        assert u_uniform < u_collapsed

    def test_identical_embeddings(self) -> None:
        """Identical embeddings have uniformity near 0 (worst)."""
        row = torch.randn(1, 32)
        embeddings = row.expand(50, -1)
        u = uniformity(embeddings)
        assert u > -0.1


class TestPairwiseCosineStats:
    """Tests for pairwise_cosine_stats function."""

    def test_identical_embeddings(self) -> None:
        """All identical gives cosine sim = 1."""
        row = torch.randn(1, 32)
        embeddings = row.expand(50, -1)
        stats = pairwise_cosine_stats(embeddings)
        assert abs(stats["cosine_sim_mean"] - 1.0) < 0.01
        assert stats["cosine_sim_std"] < 0.01

    def test_random_embeddings(self) -> None:
        """Random embeddings have mean cosine sim near 0."""
        embeddings = torch.randn(200, 64)
        stats = pairwise_cosine_stats(embeddings)
        assert abs(stats["cosine_sim_mean"]) < 0.2


class TestEmbeddingNormStats:
    """Tests for embedding_norm_stats function."""

    def test_unit_norm(self) -> None:
        """L2-normalized embeddings have norms ~1."""
        embeddings = torch.nn.functional.normalize(torch.randn(50, 32), dim=-1)
        stats = embedding_norm_stats(embeddings)
        assert abs(stats["norm_mean"] - 1.0) < 0.01
        assert stats["norm_std"] < 0.01


class TestComputeEmbeddingDiagnostics:
    """Tests for compute_embedding_diagnostics function."""

    def test_returns_all_keys(self) -> None:
        """All expected metric keys are present."""
        embeddings = torch.randn(50, 32)
        metrics = compute_embedding_diagnostics(embeddings)
        expected = {
            "effective_rank",
            "embedding_dim",
            "num_samples",
            "norm_mean",
            "norm_std",
            "norm_min",
            "norm_max",
            "uniformity",
            "cosine_sim_mean",
            "cosine_sim_std",
            "cosine_sim_min",
            "cosine_sim_max",
        }
        assert expected == set(metrics.keys())

    def test_rejects_non_2d(self) -> None:
        """Non-2D input raises ValueError."""
        with pytest.raises(ValueError, match="2D"):
            compute_embedding_diagnostics(torch.randn(3, 4, 5))

    def test_single_sample_returns_empty(self) -> None:
        """Single sample returns empty dict."""
        metrics = compute_embedding_diagnostics(torch.randn(1, 32))
        assert metrics == {}

    def test_few_samples_skips_pairwise(self) -> None:
        """With < 4 samples, pairwise metrics are skipped."""
        metrics = compute_embedding_diagnostics(torch.randn(3, 32))
        assert "uniformity" not in metrics
        assert "effective_rank" in metrics


class TestSpatialEmbeddingDiagnostics:
    """Tests for compute_spatial_embedding_diagnostics."""

    def test_returns_all_prefixes(self) -> None:
        """Global, inter, and intra prefixes are present."""
        embeddings = torch.randn(10, 16, 64)  # 10 images, 16 patches, 64-dim
        metrics = compute_spatial_embedding_diagnostics(embeddings)
        prefixes = {k.split("_")[0] for k in metrics}
        assert {"global", "inter", "intra"}.issubset(prefixes)

    def test_4d_input(self) -> None:
        """Handles [N, H, W, D] input by flattening spatial dims."""
        embeddings = torch.randn(8, 4, 4, 32)  # 8 images, 4x4 grid, 32-dim
        metrics = compute_spatial_embedding_diagnostics(embeddings)
        assert "intra_num_patches" in metrics
        assert metrics["intra_num_patches"] == 16.0

    def test_collapsed_patches_detected(self) -> None:
        """Identical patches within images give high intra cosine sim."""
        N, P, D = 10, 16, 64
        per_image = torch.randn(N, 1, D)
        embeddings = per_image.expand(N, P, D)
        metrics = compute_spatial_embedding_diagnostics(embeddings)
        assert metrics["intra_cosine_sim_mean"] > 0.99

    def test_diverse_patches(self) -> None:
        """Random patches give healthy intra-sample diversity."""
        embeddings = torch.randn(10, 16, 64)
        metrics = compute_spatial_embedding_diagnostics(embeddings)
        assert metrics["intra_cosine_sim_mean"] < 0.5

    def test_rejects_2d(self) -> None:
        """2D input raises ValueError."""
        with pytest.raises(ValueError, match="3\\+ dim"):
            compute_spatial_embedding_diagnostics(torch.randn(10, 64))
