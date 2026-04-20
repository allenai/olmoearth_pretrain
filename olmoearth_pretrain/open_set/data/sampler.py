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
from olmoearth_pretrain.datatypes import (
    MISSING_VALUE,
    MaskedOlmoEarthSample,
    OlmoEarthSample,
)
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
            target = extractor_cache[key]
            if entry.is_regression:
                # Regression: present if any non-missing pixel exists.
                per_image_present = (target != MISSING_VALUE).flatten(1).any(dim=1)
            else:
                # Classification: present iff any positive pixel.
                per_image_present = target.flatten(1).any(dim=1)
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
        """Sample per-image positive and negative classes.

        Regression entries are always included as positives when present and
        never sampled as negatives.  Classification/binary_classification
        entries follow the original k_pos/k_neg random sampling.
        """
        if rng is None:
            rng = random.Random(self._seed)

        candidate_entries = self._candidate_entries(sample)
        presence = _present_classes_per_image(sample, candidate_entries)

        # Separate classification and regression candidates for negative pool.
        classification_candidates = [
            e for e in candidate_entries if not e.is_regression
        ]
        if self._restrict_neg_to_present:
            negative_pool_global = classification_candidates
        else:
            negative_pool_global = [e for e in self._registry if not e.is_regression]

        selections: list[PerImageClassSelection] = []
        for image_index, present_entries in enumerate(presence):
            # Split present entries into classification and regression.
            present_classification = [e for e in present_entries if not e.is_regression]
            present_regression = [e for e in present_entries if e.is_regression]

            present_keys = {(e.source, e.text) for e in present_entries}
            absent_pool = [
                e
                for e in negative_pool_global
                if (e.source, e.text) not in present_keys
            ]
            # Sample classification positives/negatives.
            sampled_positives = self._sample_without_replacement(
                present_classification, self._k_pos, rng
            )
            negatives = self._sample_without_replacement(absent_pool, self._k_neg, rng)
            # All present regression entries are always included as positives.
            positives = sampled_positives + present_regression
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


class PerBatchClassSampler:
    """Sample a fixed set of classes once per batch (shared across all images).

    Unlike :class:`RandomNegativeSampler` (which samples per image), this
    samples ``k_per_batch`` classification classes once per batch and gives
    every image the same class list.  Each (image, class) pair contributes
    supervision based on that image's per-pixel ground truth — a class that
    happens to be absent from a given image just produces an all-zero target
    (valid BCE supervision toward "not present").

    All regression entries whose source is present on the batch are always
    included (they are few and information-dense).

    When all ranks share the same ``seed`` and call :meth:`sample` the same
    number of times, they produce the same class list every step — so every
    rank runs the decoder for the same set of classes and no cross-rank
    synchronization is needed.

    Compared to :class:`RandomNegativeSampler`, this produces far fewer
    decoder forwards (``k_per_batch`` vs. up to ``B * (k_pos + k_neg)``
    classes in the batch union).
    """

    def __init__(
        self,
        registry: ClassRegistry,
        k_per_batch: int = 4,
        seed: int | None = None,
    ) -> None:
        """Initialize the sampler.

        Args:
            registry: Catalog of all classes the model knows about.
            k_per_batch: Number of classification classes to sample per batch.
                Regression entries are always included when present in
                addition to this.
            seed: Optional seed.  For FSDP/distributed training, every rank
                must pass the same seed so that sampled classes stay in sync.
        """
        self._registry = registry
        self._k_per_batch = k_per_batch
        self._seed = seed

    @property
    def registry(self) -> ClassRegistry:
        """The wrapped registry."""
        return self._registry

    @property
    def k_per_batch(self) -> int:
        """Number of classification classes sampled per batch."""
        return self._k_per_batch

    def _candidate_entries(self, sample: SampleLike) -> list[ClassEntry]:
        """All entries whose source is present on the batch."""
        present_sources = set(sample.modalities)
        return [e for e in self._registry if e.source in present_sources]

    def sample(
        self,
        sample: SampleLike,
        rng: random.Random | None = None,
    ) -> list[PerImageClassSelection]:
        """Sample ``k_per_batch`` classes for the whole batch.

        Every image in the batch gets the same class list.  Classes are
        returned in the ``positives`` slot (loss computation does not
        distinguish positive vs. negative — it supervises directly against
        the per-image ground-truth mask).
        """
        if rng is None:
            rng = random.Random(self._seed)

        candidates = self._candidate_entries(sample)
        classification_candidates = [e for e in candidates if not e.is_regression]
        regression_candidates = [e for e in candidates if e.is_regression]

        k = min(self._k_per_batch, len(classification_candidates))
        if k > 0:
            sampled_classification = rng.sample(classification_candidates, k)
        else:
            sampled_classification = []

        sampled = sampled_classification + regression_candidates

        # Give every image the same class list.  The list is a fresh ``list``
        # per image to avoid any aliasing surprises downstream.
        return [
            PerImageClassSelection(
                image_index=i,
                positives=list(sampled),
                negatives=[],
            )
            for i in range(sample.batch_size)
        ]


@dataclass
class PerBatchClassSamplerConfig(Config):
    """Serializable configuration for :class:`PerBatchClassSampler`."""

    k_per_batch: int = 4
    seed: int | None = None

    def build(self, registry: ClassRegistry) -> PerBatchClassSampler:
        """Build the sampler against ``registry``."""
        return PerBatchClassSampler(
            registry=registry,
            k_per_batch=self.k_per_batch,
            seed=self.seed,
        )
