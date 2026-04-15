"""Class catalog: unified description of every class we can train against.

Every label source (OSM, CDL, WorldCover, ...) lands here as a list of
``ClassEntry`` records. Each record bundles the class name (used to embed the
class in text-encoder space) with a small ``ClassExtractor`` callable that
turns the source's modality tensor into a binary mask for that class.

This is the single source of truth that the sampler, dataset and model all
read from.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Protocol

import torch


class ClassExtractor(Protocol):
    """Turn a modality tensor into a binary mask for a single class.

    The input is the raw modality tensor as it appears on an
    ``OlmoEarthSample`` for spatial single-timestep modalities — shape
    ``[B, H, W, 1, num_bands]`` — and the output is a binary float mask of
    shape ``[B, H, W]`` with values in ``{0.0, 1.0}``.

    Implementations must be deterministic and pure (no global state).
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Extract a binary mask for this class."""
        ...


@dataclass(frozen=True)
class BandIndexExtractor:
    """Pick a single band from a multi-channel raster and threshold it.

    Used for sources where each band already corresponds to one class
    (OSM, WorldCereal).
    """

    band_index: int
    threshold: float = 0.0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Extract a binary mask from a single band."""
        # tensor: [B, H, W, 1, num_bands]
        if tensor.ndim != 5:
            raise ValueError(
                f"BandIndexExtractor expects [B, H, W, 1, C] tensor, got shape {tuple(tensor.shape)}"
            )
        band = tensor[..., 0, self.band_index]
        return (band > self.threshold).float()


@dataclass(frozen=True)
class ValueEqExtractor:
    """Treat a single-band raster as class IDs, mask pixels equal to ``class_id``.

    Used for sources like WorldCover, CDL, EuroCrops where all classes share
    one channel and the pixel value is a category code.
    """

    class_id: int

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Extract a binary mask via class-ID equality."""
        if tensor.ndim != 5:
            raise ValueError(
                f"ValueEqExtractor expects [B, H, W, 1, C] tensor, got shape {tuple(tensor.shape)}"
            )
        # Single-band class-ID raster.
        band = tensor[..., 0, 0]
        return (band == self.class_id).float()


@dataclass(frozen=True)
class ClassEntry:
    """One classifiable class from a single label source."""

    text: str
    """Canonical name passed to the text encoder, e.g. "maize"."""

    source: str
    """Modality field name on ``OlmoEarthSample`` (e.g. "openstreetmap_raster")."""

    extractor: ClassExtractor
    """Callable producing a binary mask from the source modality tensor."""

    synonyms: tuple[str, ...] = field(default_factory=tuple)
    """Optional alternate phrasings used for prompt ensembling."""

    def all_prompts(self) -> tuple[str, ...]:
        """Return the canonical text plus any synonyms, in order."""
        return (self.text, *self.synonyms)


class ClassRegistry:
    """Append-only collection of ClassEntries with lookup helpers."""

    def __init__(self, entries: Iterable[ClassEntry] = ()) -> None:
        """Initialize with an iterable of class entries.

        Raises:
            ValueError: if any two entries share the same ``(source, text)`` pair.
        """
        self._entries: list[ClassEntry] = list(entries)
        seen: set[tuple[str, str]] = set()
        for entry in self._entries:
            key = (entry.source, entry.text)
            if key in seen:
                raise ValueError(f"Duplicate class entry for {key}")
            seen.add(key)

    @property
    def entries(self) -> list[ClassEntry]:
        """Return all entries."""
        return list(self._entries)

    def __len__(self) -> int:
        """Number of registered classes."""
        return len(self._entries)

    def __iter__(self) -> Iterator[ClassEntry]:
        """Iterate over entries."""
        return iter(self._entries)

    def by_source(self, source: str) -> list[ClassEntry]:
        """Return all entries whose source is ``source``."""
        return [e for e in self._entries if e.source == source]

    def sources(self) -> list[str]:
        """Return the set of distinct source names, preserving first-seen order."""
        seen: list[str] = []
        for e in self._entries:
            if e.source not in seen:
                seen.append(e.source)
        return seen

    def by_text(self, text: str, source: str | None = None) -> ClassEntry:
        """Look up an entry by text. If ``source`` is given, scope the lookup."""
        for e in self._entries:
            if e.text == text and (source is None or e.source == source):
                return e
        scope = f" in source {source}" if source else ""
        raise KeyError(f"No class entry for text {text!r}{scope}")
