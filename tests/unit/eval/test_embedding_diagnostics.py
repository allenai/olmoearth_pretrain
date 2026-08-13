"""Unit tests for embedding diagnostics."""

import pytest
import torch

from olmoearth_pretrain.evals.embedding_diagnostics import (
    MAX_PIPELINE_ROWS,
    compute_embedding_diagnostics,
    compute_pipeline_diagnostics,
    compute_spatial_embedding_diagnostics,
    effective_rank,
    embedding_norm_stats,
    pairwise_cosine_stats,
    per_dim_stats,
    quantization_clip_stats,
    round_trip_stats,
    uniformity,
)
from olmoearth_pretrain.evals.embedding_transforms import (
    dequantize_embeddings,
    quantize_embeddings,
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


class TestPerDimStats:
    """Tests for per-dimension scale and shared-offset stats."""

    def test_isotropic_dims(self) -> None:
        """Standard normal dims have a per-dim std ratio near 1."""
        stats = per_dim_stats(torch.randn(4096, 32))
        assert 0.8 < stats["dim_std_ratio"] < 1.5
        assert stats["mean_component_ratio"] < 0.1

    def test_anisotropy_detected(self) -> None:
        """One inflated dimension shows up in the std ratio."""
        rows = torch.randn(4096, 32)
        rows[:, 0] *= 50
        stats = per_dim_stats(rows)
        assert stats["dim_std_ratio"] > 20

    def test_shared_offset_detected(self) -> None:
        """A large constant offset dominates the typical embedding."""
        rows = torch.randn(1024, 32) * 0.01 + 10.0
        stats = per_dim_stats(rows)
        assert stats["mean_component_ratio"] > 0.99


class TestQuantizationClipStats:
    """Tests for int8 saturation accounting."""

    def test_unit_norm_embeddings_do_not_clip(self) -> None:
        """AEF-style unit-L2 embeddings sit well inside the int8 range."""
        rows = torch.nn.functional.normalize(torch.randn(512, 64), dim=-1)
        assert quantization_clip_stats(rows)["clip_fraction"] == 0.0

    def test_layernorm_scale_embeddings_clip(self) -> None:
        """Per-coordinate std ~1 saturates roughly a third of coordinates."""
        stats = quantization_clip_stats(torch.randn(4096, 64))
        assert 0.25 < stats["clip_fraction"] < 0.40
        assert stats["clip_fraction_rows"] > 0.99


class TestRoundTripStats:
    """Tests for quantization round-trip damage."""

    def test_identical_is_lossless(self) -> None:
        """Comparing a tensor to itself reports no damage."""
        rows = torch.randn(64, 32)
        stats = round_trip_stats(rows, rows)
        assert stats["roundtrip_cosine_mean"] > 0.999
        assert stats["roundtrip_rel_mse"] < 1e-6

    def test_clipping_shows_as_damage(self) -> None:
        """The int8 round trip damages LayerNorm-scale embeddings more than unit ones."""
        wide = torch.randn(512, 64)
        unit = torch.nn.functional.normalize(torch.randn(512, 64), dim=-1)
        wide_damage = round_trip_stats(
            wide, dequantize_embeddings(quantize_embeddings(wide))
        )
        unit_damage = round_trip_stats(
            unit, dequantize_embeddings(quantize_embeddings(unit))
        )
        assert wide_damage["roundtrip_rel_mse"] > unit_damage["roundtrip_rel_mse"]
        assert (
            wide_damage["roundtrip_cosine_mean"] < unit_damage["roundtrip_cosine_mean"]
        )


class TestPipelineDiagnostics:
    """Tests for the extract -> normalize -> quantize pipeline diagnostics."""

    def test_raw_only(self) -> None:
        """Without a normalizer or quantization, only raw_* keys are emitted."""
        metrics = compute_pipeline_diagnostics(torch.randn(128, 32))
        assert any(k.startswith("raw_") for k in metrics)
        assert not any(k.startswith(("norm_", "roundtrip_")) for k in metrics)

    def test_all_stages(self) -> None:
        """Each stage contributes its own prefix."""
        raw = torch.randn(128, 32)
        normalized = torch.nn.functional.normalize(raw, dim=-1)
        round_tripped = dequantize_embeddings(quantize_embeddings(normalized))
        metrics = compute_pipeline_diagnostics(raw, normalized, round_tripped)
        assert metrics["raw_clip_fraction"] > metrics["norm_clip_fraction"]
        assert metrics["roundtrip_cosine_mean"] > 0.99

    def test_spatial_input(self) -> None:
        """Spatial embeddings are flattened to rows before analysis."""
        metrics = compute_pipeline_diagnostics(torch.randn(8, 4, 4, 32))
        assert metrics["raw_embedding_dim"] == 32.0

    def test_subsample_keeps_views_aligned(self) -> None:
        """Above the row cap, all three views are cut with the same indices."""
        raw = torch.randn(MAX_PIPELINE_ROWS * 2, 16)
        metrics = compute_pipeline_diagnostics(raw, raw, raw)
        # Misaligned rows would break the elementwise comparison, not match it.
        assert metrics["roundtrip_cosine_mean"] > 0.999
