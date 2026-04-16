"""Regression target catalog.

Regression entries represent continuous-valued map modalities: SRTM (elevation),
WRI Canopy Height, and CDL (treated as regression of the normalized class ID).
"""

from __future__ import annotations

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.open_set.catalog.registry import (
    ClassEntry,
    RegressionExtractor,
)


def build_srtm_entries() -> list[ClassEntry]:
    """SRTM digital elevation model — single regression target."""
    return [
        ClassEntry(
            text="elevation",
            source=Modality.SRTM.name,
            extractor=RegressionExtractor(band_index=0),
            task_type="regression",
        )
    ]


def build_canopy_height_entries() -> list[ClassEntry]:
    """WRI canopy-height map — single regression target."""
    return [
        ClassEntry(
            text="canopy height",
            source=Modality.WRI_CANOPY_HEIGHT_MAP.name,
            extractor=RegressionExtractor(band_index=0),
            task_type="regression",
        )
    ]


def build_cdl_entries() -> list[ClassEntry]:
    """USDA Cropland Data Layer — treated as regression of the normalized class ID."""
    return [
        ClassEntry(
            text="crop type",
            source=Modality.CDL.name,
            extractor=RegressionExtractor(band_index=0),
            task_type="regression",
        )
    ]
