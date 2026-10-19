"""Tests for TileGroupSampler."""

import pytest
import torch

from olmoearth_pretrain.evals.datasets.tile_sampler import TileGroupSampler


def sample_epoch(seed: int = 0, **kwargs: int) -> list[int]:
    """One epoch of indices from a seeded sampler."""
    generator = torch.Generator().manual_seed(seed)
    return list(TileGroupSampler(generator=generator, **kwargs))


def test_yields_every_tile_once() -> None:
    """An epoch is a permutation of all tile indices."""
    idx = sample_epoch(num_samples=10, tiles_per_sample=16, group_size=3)
    assert sorted(idx) == list(range(160))


def test_consecutive_tiles_come_from_one_group() -> None:
    """Each run of group_size * tiles indices spans exactly that group's samples."""
    tiles, group = 16, 4
    idx = sample_epoch(num_samples=12, tiles_per_sample=tiles, group_size=group)
    for start in range(0, len(idx), group * tiles):
        bases = {i // tiles for i in idx[start : start + group * tiles]}
        assert len(bases) == group


def test_last_group_may_be_partial() -> None:
    """A sample count not divisible by group_size still yields every tile."""
    idx = sample_epoch(num_samples=5, tiles_per_sample=4, group_size=2)
    assert sorted(idx) == list(range(20))
    assert len({i // 4 for i in idx[-4:]}) == 1


def test_shuffles_across_epochs_and_is_seeded() -> None:
    """Same seed, same order; successive epochs of one sampler differ."""
    kwargs = {"num_samples": 8, "tiles_per_sample": 4, "group_size": 2}
    assert sample_epoch(seed=1, **kwargs) == sample_epoch(seed=1, **kwargs)
    sampler = TileGroupSampler(generator=torch.Generator().manual_seed(1), **kwargs)
    assert list(sampler) != list(sampler)


def test_rejects_empty_group() -> None:
    """group_size must be positive."""
    with pytest.raises(ValueError, match="group_size"):
        TileGroupSampler(num_samples=4, tiles_per_sample=4, group_size=0)
