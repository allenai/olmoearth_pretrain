"""WorldCereal class catalog.

WorldCereal provides 8 binary classification bands, each indicating
whether a particular crop condition is present.
"""

from __future__ import annotations

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.open_set.catalog.registry import (
    BandIndexExtractor,
    ClassEntry,
)

_SOURCE = Modality.WORLDCEREAL.name

# Band names from Modality.WORLDCEREAL.band_order → human-readable text.
_WORLDCEREAL_BANDS: list[tuple[int, str]] = [
    (0, "annual temporary crops"),
    (1, "irrigated maize main season"),
    (2, "maize main season"),
    (3, "irrigated maize second season"),
    (4, "maize second season"),
    (5, "spring cereals"),
    (6, "irrigated winter cereals"),
    (7, "winter cereals"),
]


def build_worldcereal_entries() -> list[ClassEntry]:
    """Build a ClassEntry for every WorldCereal band."""
    entries: list[ClassEntry] = []
    for band_idx, text in _WORLDCEREAL_BANDS:
        entries.append(
            ClassEntry(
                text=text,
                source=_SOURCE,
                extractor=BandIndexExtractor(band_index=band_idx, threshold=0.0),
                task_type="binary_classification",
            )
        )
    return entries
