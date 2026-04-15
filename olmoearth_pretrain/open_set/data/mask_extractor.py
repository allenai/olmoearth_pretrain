"""Turn an ``OlmoEarthSample`` (or batch) into per-class binary masks.

Given a batch and a list of ``(image_index, ClassEntry)`` selections, this
module produces a ``BatchClassAssignment``: a flat list of
(image_index, class_entry, mask) records that the training step can iterate
over directly.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

from olmoearth_pretrain.datatypes import OlmoEarthSample
from olmoearth_pretrain.open_set.catalog.registry import ClassEntry


@dataclass
class ClassAssignment:
    """One (image, class) supervision record."""

    image_index: int
    class_entry: ClassEntry
    mask: torch.Tensor  # [H, W] binary float


@dataclass
class BatchClassAssignment:
    """All (image, class) supervision records for a single batch."""

    assignments: list[ClassAssignment]

    def __len__(self) -> int:
        """Number of (image, class) records."""
        return len(self.assignments)

    def __iter__(self) -> Iterator[ClassAssignment]:
        """Iterate over ``ClassAssignment`` records."""
        return iter(self.assignments)


def _get_source_tensor(sample: OlmoEarthSample, source: str) -> torch.Tensor | None:
    """Return ``sample.<source>`` if present, else None."""
    return getattr(sample, source, None)


def extract_per_image_class_assignments(
    sample: OlmoEarthSample,
    selections: list[tuple[int, ClassEntry]],
) -> BatchClassAssignment:
    """Extract per-image binary masks for the given selections.

    Args:
        sample: The full batch.
        selections: List of (image_index, class_entry) pairs to extract masks for.

    Returns:
        A ``BatchClassAssignment`` containing one ``ClassAssignment`` per
        selection. Selections referencing a source that is missing from the
        batch raise ``KeyError`` — the sampler is responsible for only picking
        classes whose source is present.
    """
    # Cache binary masks per (source, class_text) so we do not redo extraction
    # if the same class is selected for multiple images.
    cache: dict[tuple[str, str], torch.Tensor] = {}
    assignments: list[ClassAssignment] = []
    for image_index, entry in selections:
        cache_key = (entry.source, entry.text)
        if cache_key not in cache:
            tensor = _get_source_tensor(sample, entry.source)
            if tensor is None:
                raise KeyError(
                    f"Source {entry.source!r} not present on batch — sampler "
                    f"should not have produced this selection."
                )
            cache[cache_key] = entry.extractor(tensor)  # [B, H, W]
        mask_batch = cache[cache_key]
        assignments.append(
            ClassAssignment(
                image_index=image_index,
                class_entry=entry,
                mask=mask_batch[image_index],
            )
        )
    return BatchClassAssignment(assignments=assignments)
