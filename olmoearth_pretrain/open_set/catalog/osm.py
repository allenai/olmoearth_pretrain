"""OSM class catalog.

OSM bands in ``Modality.OPENSTREETMAP_RASTER`` are already named after the
classes they represent, so each band gives us one ClassEntry directly.
"""

from __future__ import annotations

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.open_set.catalog.registry import (
    BandIndexExtractor,
    ClassEntry,
)

# A few classes benefit from a more natural-language form for the text encoder
# (e.g. SigLIP prompts). The raw band names use snake_case which is fine, but
# spaces tend to produce slightly better embeddings for compound terms.
_PROMPT_OVERRIDES: dict[str, str] = {
    "aerialway_pylon": "aerialway pylon",
    "amenity_fuel": "fuel station",
    "communications_tower": "communications tower",
    "generator_wind": "wind turbine",
    "petroleum_well": "petroleum well",
    "power_plant": "power plant",
    "power_substation": "power substation",
    "power_tower": "power tower",
    "satellite_dish": "satellite dish",
    "storage_tank": "storage tank",
    "water_tower": "water tower",
}

_SOURCE = Modality.OPENSTREETMAP_RASTER.name


def _band_to_text(band: str) -> str:
    """Convert an OSM band name into a human-readable prompt."""
    return _PROMPT_OVERRIDES.get(band, band.replace("_", " "))


def build_osm_entries() -> list[ClassEntry]:
    """Build a ClassEntry for every band in OPENSTREETMAP_RASTER."""
    entries: list[ClassEntry] = []
    band_order = Modality.OPENSTREETMAP_RASTER.band_order
    for idx, band in enumerate(band_order):
        text = _band_to_text(band)
        entries.append(
            ClassEntry(
                text=text,
                source=_SOURCE,
                extractor=BandIndexExtractor(band_index=idx, threshold=0.0),
            )
        )
    return entries
