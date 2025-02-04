"""Constants shared across the helios package."""

from enum import Enum


# Modalities supported by helios
class Modality(Enum):
    """Modality information."""

    NAIP = {"name": "naip", "resolution": 0.625, "bands": ["R", "G", "B", "IR"]}
    S1 = {"name": "sentinel1", "resolution": 10, "bands": ["VV", "VH"]}
    S2 = {
        "name": "sentinel2",
        "resolution": 10,
        "bands": [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B10",
            "B11",
            "B12",
            "B13",
        ],
    }
    LANDSAT = {
        "name": "landsat",
        "resolution": 10,
        "bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"],
    }
    WORLDCOVER = {"name": "worldcover", "resolution": 10, "bands": ["B1"]}
    OSM = {
        "name": "openstreetmap",
        "resolution": 2.5,  # Original resolution is 10m, but upsample to 2.5m
        "bands": [
            "aerialway_pylon",
            "aerodrome",
            "airstrip",
            "amenity_fuel",
            "building",
            "chimney",
            "communications_tower",
            "crane",
            "flagpole",
            "fountain",
            "generator_wind",
            "helipad",
            "highway",
            "leisure",
            "lighthouse",
            "obelisk",
            "observatory",
            "parking",
            "petroleum_well",
            "power_plant",
            "power_substation",
            "power_tower",
            "river",
            "runway",
            "satellite_dish",
            "silo",
            "storage_tank",
            "taxiway",
            "water_tower",
            "works",
        ],  # OSM has been converted to raster format, each category is a band
    }
