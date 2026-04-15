"""Tests for the random-negative class sampler."""

from __future__ import annotations

import random
from typing import cast

import torch

from olmoearth_pretrain.datatypes import OlmoEarthSample
from olmoearth_pretrain.open_set.catalog import (
    BandIndexExtractor,
    ClassEntry,
    ClassRegistry,
)
from olmoearth_pretrain.open_set.data.sampler import RandomNegativeSampler


def _registry(num_bands: int) -> ClassRegistry:
    """Build a registry with ``num_bands`` OSM-style entries."""
    return ClassRegistry(
        [
            ClassEntry(
                text=f"c{i}",
                source="openstreetmap_raster",
                extractor=BandIndexExtractor(i),
            )
            for i in range(num_bands)
        ]
    )


def _sample_with_present_classes(
    present_per_image: list[list[int]],
    num_bands: int,
    h: int = 4,
    w: int = 4,
) -> OlmoEarthSample:
    """Build a batch where ``present_per_image[i]`` lists which bands are nonzero in image i."""
    b = len(present_per_image)
    osm = torch.zeros(b, h, w, 1, num_bands)
    for i, present in enumerate(present_per_image):
        for band in present:
            osm[i, 0, 0, 0, band] = 1.0
    return OlmoEarthSample(openstreetmap_raster=osm)


class TestRandomNegativeSampler:
    """Sampler returns one selection per image, with capped pos / neg counts."""

    def test_one_selection_per_image(self) -> None:
        """For a 3-image batch we get 3 ``PerImageClassSelection`` records."""
        reg = _registry(num_bands=4)
        sampler = RandomNegativeSampler(reg, k_pos=1, k_neg=1, seed=0)
        sample = _sample_with_present_classes([[0], [1, 2], []], num_bands=4)
        sels = sampler.sample(sample, rng=random.Random(0))
        assert [s.image_index for s in sels] == [0, 1, 2]

    def test_positives_are_present_negatives_are_absent(self) -> None:
        """For every image, every sampled positive is present and negative is absent."""
        reg = _registry(num_bands=5)
        sampler = RandomNegativeSampler(reg, k_pos=2, k_neg=2, seed=42)
        present = [[0, 1, 2], [3], [0, 4]]
        sample = _sample_with_present_classes(present, num_bands=5)
        sels = sampler.sample(sample, rng=random.Random(0))

        for img_idx, sel in enumerate(sels):
            pos_indices = {
                cast(BandIndexExtractor, e.extractor).band_index for e in sel.positives
            }
            neg_indices = {
                cast(BandIndexExtractor, e.extractor).band_index for e in sel.negatives
            }
            assert pos_indices.issubset(set(present[img_idx]))
            assert neg_indices.isdisjoint(set(present[img_idx]))

    def test_capped_when_fewer_present_than_k_pos(self) -> None:
        """If only one class is present we sample at most one positive."""
        reg = _registry(num_bands=4)
        sampler = RandomNegativeSampler(reg, k_pos=3, k_neg=1, seed=0)
        sample = _sample_with_present_classes([[2]], num_bands=4)
        sels = sampler.sample(sample, rng=random.Random(0))
        assert len(sels[0].positives) == 1
        assert len(sels[0].negatives) == 1

    def test_image_with_no_positives_yields_only_negatives(self) -> None:
        """Empty-image case: positives is empty, negatives still drawn from the pool."""
        reg = _registry(num_bands=3)
        sampler = RandomNegativeSampler(reg, k_pos=2, k_neg=2, seed=0)
        sample = _sample_with_present_classes([[]], num_bands=3)
        sels = sampler.sample(sample, rng=random.Random(0))
        assert sels[0].positives == []
        assert len(sels[0].negatives) == 2

    def test_skips_classes_whose_source_is_absent(self) -> None:
        """An entry from a source not on the batch is excluded from both pools."""
        reg = ClassRegistry(
            [
                ClassEntry(
                    text="osm0",
                    source="openstreetmap_raster",
                    extractor=BandIndexExtractor(0),
                ),
                ClassEntry(
                    text="cdl0",
                    source="cdl",
                    extractor=BandIndexExtractor(0),
                ),
            ]
        )
        sampler = RandomNegativeSampler(reg, k_pos=1, k_neg=1, seed=0)
        sample = _sample_with_present_classes([[0]], num_bands=1)
        sels = sampler.sample(sample, rng=random.Random(0))
        sources_seen = {e.source for e in sels[0].all_entries}
        assert sources_seen == {"openstreetmap_raster"}
