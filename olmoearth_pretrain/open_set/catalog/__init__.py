"""Catalog of classifiable classes and regression targets across OlmoEarth label sources.

Each ``ClassEntry`` says: given the string ``text`` (e.g. "maize"), the
ground-truth target is produced by applying ``extractor`` to the modality
tensor named ``source`` on a sample.
"""

from olmoearth_pretrain.open_set.catalog.osm import build_osm_entries
from olmoearth_pretrain.open_set.catalog.registry import (
    BandIndexExtractor,
    ClassEntry,
    ClassExtractor,
    ClassRegistry,
    NormalizedValueEqExtractor,
    RegressionExtractor,
    ValueEqExtractor,
)
from olmoearth_pretrain.open_set.catalog.regression import (
    build_canopy_height_entries,
    build_cdl_entries,
    build_srtm_entries,
)
from olmoearth_pretrain.open_set.catalog.worldcereal import build_worldcereal_entries
from olmoearth_pretrain.open_set.catalog.worldcover import build_worldcover_entries


def build_default_registry() -> ClassRegistry:
    """Build the default registry containing all currently-supported sources."""
    entries: list[ClassEntry] = []
    entries.extend(build_osm_entries())
    entries.extend(build_worldcover_entries())
    entries.extend(build_worldcereal_entries())
    entries.extend(build_srtm_entries())
    entries.extend(build_canopy_height_entries())
    entries.extend(build_cdl_entries())
    return ClassRegistry(entries)


__all__ = [
    "BandIndexExtractor",
    "ClassEntry",
    "ClassExtractor",
    "ClassRegistry",
    "NormalizedValueEqExtractor",
    "RegressionExtractor",
    "ValueEqExtractor",
    "build_canopy_height_entries",
    "build_cdl_entries",
    "build_default_registry",
    "build_osm_entries",
    "build_srtm_entries",
    "build_worldcereal_entries",
    "build_worldcover_entries",
]
