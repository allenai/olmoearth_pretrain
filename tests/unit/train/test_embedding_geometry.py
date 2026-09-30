"""Unit tests for the embedding-geometry collapse monitors."""

import pytest
import torch
import torch.nn.functional as F

from olmoearth_pretrain.train.embedding_geometry import (
    effective_rank,
    spread_and_mean_cosine,
)


def _rotated_low_rank(n: int, d: int, k: int, seed: int = 0) -> torch.Tensor:
    """Isotropic rank-``k`` samples in a random ``k``-dim subspace of ``R^d``."""
    g = torch.Generator().manual_seed(seed)
    basis, _ = torch.linalg.qr(torch.randn(d, k, generator=g))
    return torch.randn(n, k, generator=g) @ basis.T


def test_identical_vectors_are_fully_collapsed() -> None:
    x = torch.randn(1, 16).repeat(64, 1)
    spread, mean_cos = spread_and_mean_cosine(x)
    assert spread < 1e-5
    assert abs(float(mean_cos) - 1.0) < 1e-5


def test_isotropic_cloud_is_spread_and_uncorrelated() -> None:
    torch.manual_seed(0)
    spread, mean_cos = spread_and_mean_cosine(torch.randn(4096, 32))
    assert 0.95 < float(spread) <= 1.0 + 1e-6
    assert abs(float(mean_cos)) < 0.01


def test_mean_cosine_matches_brute_force() -> None:
    torch.manual_seed(1)
    x = torch.randn(9, 5) + 0.5
    _, mean_cos = spread_and_mean_cosine(x)
    unit = F.normalize(x, dim=-1)
    sim = unit @ unit.T
    off_diag = sim[~torch.eye(9, dtype=torch.bool)].mean()
    torch.testing.assert_close(mean_cos, off_diag)


def test_shared_offset_raises_mean_cosine() -> None:
    torch.manual_seed(2)
    x = torch.randn(2048, 64) / 8.0
    x[:, 0] += 1.0
    spread, mean_cos = spread_and_mean_cosine(x)
    assert float(mean_cos) > 0.4
    assert float(spread) < 0.8


def test_effective_rank_sees_what_spread_misses() -> None:
    """A rotated rank-5 cloud looks spread out but has effective rank ~5."""
    x = _rotated_low_rank(n=4000, d=64, k=5)
    spread, _ = spread_and_mean_cosine(x)
    erank, top10 = effective_rank(x, top_k=10)
    assert float(spread) > 0.8
    assert abs(erank - 5.0) < 0.3
    assert top10 > 0.999


def test_effective_rank_of_isotropic_cloud_is_near_full() -> None:
    torch.manual_seed(3)
    erank, top10 = effective_rank(torch.randn(8000, 32), top_k=10)
    assert erank > 30.0
    assert top10 < 0.4


def test_inputs_need_two_samples() -> None:
    with pytest.raises(ValueError, match="N >= 2"):
        spread_and_mean_cosine(torch.randn(1, 8))
    with pytest.raises(ValueError, match="N >= 2"):
        effective_rank(torch.randn(1, 8))
