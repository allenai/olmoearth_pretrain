"""Per-batch class sampling for binary segmentation training.

For each image we sample ``k_pos`` classes that *are* present (positive
supervision) and ``k_neg`` classes that *are not* present (negative
supervision). The model is then asked to produce a binary mask for each
sampled (image, class) pair.

This MVP uses random negatives. Hard-negative mining (sample negatives
similar to the positives in SigLIP space) is left for a follow-up; the
``ClassSampler`` protocol is general enough to support it without changing
the call sites.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import torch

from olmoearth_pretrain.config import Config
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, OlmoEarthSample
from olmoearth_pretrain.open_set.catalog.registry import ClassEntry, ClassRegistry

# The sampler reads modality tensors and the batch_size off a sample. Both
# OlmoEarthSample and MaskedOlmoEarthSample expose the same field-name and
# ``batch_size`` / ``modalities`` interface, so accepting either is correct.
SampleLike = OlmoEarthSample | MaskedOlmoEarthSample


@dataclass(frozen=True)
class PerImageClassSelection:
    """Sampled class assignments for one image in a batch."""

    image_index: int
    positives: list[ClassEntry]
    negatives: list[ClassEntry]

    @property
    def all_entries(self) -> list[ClassEntry]:
        """Positives followed by negatives."""
        return [*self.positives, *self.negatives]


class ClassSampler(Protocol):
    """Strategy that picks ``k_pos`` positive + ``k_neg`` negative classes per image."""

    def sample(
        self,
        sample: SampleLike,
        rng: random.Random | None = None,
    ) -> list[PerImageClassSelection]:
        """Return one ``PerImageClassSelection`` per image in ``sample``."""
        ...


def _present_classes_per_image(
    sample: SampleLike,
    candidate_entries: list[ClassEntry],
) -> list[list[ClassEntry]]:
    """For every image in the batch, list which candidate classes are present.

    "Present" means: the source modality is on the sample, *and* the binary
    mask produced by the entry's extractor has at least one positive pixel.

    Extractor results are cached per source so we run each extractor at most
    once per batch.
    """
    batch_size = sample.batch_size
    extractor_cache: dict[tuple[str, str], torch.Tensor] = {}
    by_source: dict[str, list[ClassEntry]] = {}
    for entry in candidate_entries:
        by_source.setdefault(entry.source, []).append(entry)

    presence: list[list[ClassEntry]] = [[] for _ in range(batch_size)]
    for source, entries in by_source.items():
        tensor = getattr(sample, source, None)
        if tensor is None:
            continue
        for entry in entries:
            key = (source, entry.text)
            if key not in extractor_cache:
                extractor_cache[key] = entry.extractor(tensor)  # [B, H, W]
            mask = extractor_cache[key]
            # Reduce over H, W: image present iff any positive pixel.
            per_image_present = mask.flatten(1).any(dim=1)
            for image_index in range(batch_size):
                if bool(per_image_present[image_index].item()):
                    presence[image_index].append(entry)
    return presence


class RandomNegativeSampler:
    """Sample positives uniformly at random and negatives uniformly at random.

    A "negative" for image ``i`` is a class entry from the same registry
    whose extractor produces an all-zero mask on image ``i`` (i.e. the class
    is genuinely absent from that image). Sources missing from the batch are
    excluded from both pools.
    """

    def __init__(
        self,
        registry: ClassRegistry,
        k_pos: int = 2,
        k_neg: int = 2,
        seed: int | None = None,
        restrict_negatives_to_present_sources: bool = True,
    ) -> None:
        """Initialize the sampler.

        Args:
            registry: Catalog of all classes the model knows about.
            k_pos: Number of positive classes to sample per image.
            k_neg: Number of negative classes to sample per image.
            seed: Optional seed; if None the sampler defers to the rng passed
                into ``sample`` (or builds a fresh one each call).
            restrict_negatives_to_present_sources: If True, negatives are drawn
                only from sources that appear on the batch. This is the safe
                default — drawing a negative from an absent source produces
                a useless all-zero target and wastes a forward pass.
        """
        self._registry = registry
        self._k_pos = k_pos
        self._k_neg = k_neg
        self._seed = seed
        self._restrict_neg_to_present = restrict_negatives_to_present_sources

    @property
    def registry(self) -> ClassRegistry:
        """The wrapped registry."""
        return self._registry

    @property
    def k_pos(self) -> int:
        """Number of positive classes per image."""
        return self._k_pos

    @property
    def k_neg(self) -> int:
        """Number of negative classes per image."""
        return self._k_neg

    def _candidate_entries(self, sample: SampleLike) -> list[ClassEntry]:
        """All entries whose source is present on the batch."""
        present_sources = set(sample.modalities)
        return [e for e in self._registry if e.source in present_sources]

    def _sample_without_replacement(
        self, pool: Iterable[ClassEntry], k: int, rng: random.Random
    ) -> list[ClassEntry]:
        pool_list = list(pool)
        if k <= 0 or not pool_list:
            return []
        if k >= len(pool_list):
            return pool_list
        return rng.sample(pool_list, k)

    def sample(
        self,
        sample: SampleLike,
        rng: random.Random | None = None,
    ) -> list[PerImageClassSelection]:
        """Sample per-image positive and negative classes."""
        if rng is None:
            rng = random.Random(self._seed)

        candidate_entries = self._candidate_entries(sample)
        presence = _present_classes_per_image(sample, candidate_entries)

        if self._restrict_neg_to_present:
            negative_pool_global = candidate_entries
        else:
            negative_pool_global = list(self._registry)

        selections: list[PerImageClassSelection] = []
        for image_index, present_entries in enumerate(presence):
            present_keys = {(e.source, e.text) for e in present_entries}
            absent_pool = [
                e
                for e in negative_pool_global
                if (e.source, e.text) not in present_keys
            ]
            positives = self._sample_without_replacement(
                present_entries, self._k_pos, rng
            )
            negatives = self._sample_without_replacement(absent_pool, self._k_neg, rng)
            selections.append(
                PerImageClassSelection(
                    image_index=image_index,
                    positives=positives,
                    negatives=negatives,
                )
            )
        return selections


@dataclass
class RandomNegativeSamplerConfig(Config):
    """Serializable configuration for :class:`RandomNegativeSampler`."""

    k_pos: int = 2
    k_neg: int = 2
    seed: int | None = None
    restrict_negatives_to_present_sources: bool = True

    def build(self, registry: ClassRegistry) -> RandomNegativeSampler:
        """Build the sampler against ``registry``."""
        return RandomNegativeSampler(
            registry=registry,
            k_pos=self.k_pos,
            k_neg=self.k_neg,
            seed=self.seed,
            restrict_negatives_to_present_sources=self.restrict_negatives_to_present_sources,
        )
