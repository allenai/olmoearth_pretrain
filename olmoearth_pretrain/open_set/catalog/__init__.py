"""Catalog of classifiable classes across OlmoEarth label sources.

Each ``ClassEntry`` says: given the string ``text`` (e.g. "maize"), the binary
ground-truth mask is produced by applying ``extractor`` to the modality tensor
named ``source`` on a sample.
"""

from olmoearth_pretrain.open_set.catalog.osm import build_osm_entries
from olmoearth_pretrain.open_set.catalog.registry import (
    BandIndexExtractor,
    ClassEntry,
    ClassExtractor,
    ClassRegistry,
    ValueEqExtractor,
)


def build_default_registry() -> ClassRegistry:
    """Build the default registry containing all currently-supported sources.

    For the MVP this is OSM only. Add new sources here as we wire them in.
    """
    entries = []
    entries.extend(build_osm_entries())
    return ClassRegistry(entries)


__all__ = [
    "BandIndexExtractor",
    "ClassEntry",
    "ClassExtractor",
    "ClassRegistry",
    "ValueEqExtractor",
    "build_default_registry",
    "build_osm_entries",
]
