"""Shuffling sampler for tiled datasets that keeps each batch's tiles to a few stored samples."""

from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class TileGroupSampler(Sampler[int]):
    """Shuffle stored samples into groups of ``group_size``, then tiles within each group.

    Tiled datasets index a stored sample's tiles consecutively (``base * tiles + tile``).
    A uniform shuffle makes nearly every tile read its whole stored sample from disk;
    drawing tiles from a few samples at a time lets a per-worker cache of
    ``group_size`` samples serve them.
    """

    def __init__(
        self,
        num_samples: int,
        tiles_per_sample: int,
        group_size: int,
        generator: torch.Generator | None = None,
    ) -> None:
        """Sample all ``num_samples * tiles_per_sample`` indices once per epoch."""
        if group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {group_size}")
        self.num_samples = num_samples
        self.tiles_per_sample = tiles_per_sample
        self.group_size = group_size
        self.generator = generator

    def __len__(self) -> int:
        """Number of tiles per epoch."""
        return self.num_samples * self.tiles_per_sample

    def __iter__(self) -> Iterator[int]:
        """Yield every tile index once, grouped by stored sample."""
        order = torch.randperm(self.num_samples, generator=self.generator)
        tiles = torch.arange(self.tiles_per_sample)
        for group in order.split(self.group_size):
            idx = (group[:, None] * self.tiles_per_sample + tiles).flatten()
            yield from idx[torch.randperm(len(idx), generator=self.generator)].tolist()
