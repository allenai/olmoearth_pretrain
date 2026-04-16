"""WorldCover class catalog.

ESA WorldCover 2021 provides 11 land-cover classes at 10 m resolution.
After min-max normalization the class IDs (10, 20, ..., 100) map to
(0.1, 0.2, ..., 1.0).
"""

from __future__ import annotations

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.open_set.catalog.registry import (
    ClassEntry,
    NormalizedValueEqExtractor,
)

_SOURCE = Modality.WORLDCOVER.name

# ESA WorldCover class IDs → normalized values → human-readable labels.
_WORLDCOVER_CLASSES: list[tuple[float, str]] = [
    (0.1, "tree cover"),
    (0.2, "shrubland"),
    (0.3, "grassland"),
    (0.4, "cropland"),
    (0.5, "built-up"),
    (0.6, "bare or sparse vegetation"),
    (0.7, "snow and ice"),
    (0.8, "permanent water bodies"),
    (0.9, "herbaceous wetland"),
    (0.95, "mangroves"),
    (1.0, "moss and lichen"),
]


def build_worldcover_entries() -> list[ClassEntry]:
    """Build a ClassEntry for every ESA WorldCover land-cover class."""
    entries: list[ClassEntry] = []
    for class_value, text in _WORLDCOVER_CLASSES:
        entries.append(
            ClassEntry(
                text=text,
                source=_SOURCE,
                extractor=NormalizedValueEqExtractor(class_value=class_value),
                task_type="classification",
            )
        )
    return entries
