"""Semantic class name mappings for map modalities.

These are used to generate text embeddings for contrastive targets.
"""

# ESA WorldCover v2 classes.
# Keys are the raw pixel values (10, 20, ..., 100).
WORLDCOVER_CLASSES: dict[int, str] = {
    10: "tree cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built-up area",
    60: "bare or sparse vegetation",
    70: "snow and ice",
    80: "permanent water bodies",
    90: "herbaceous wetland",
    95: "mangroves",
    100: "moss and lichen",
}

# OpenStreetMap raster categories.
# Order must match Modality.OPENSTREETMAP_RASTER.band_order.
OSM_RASTER_CLASSES: list[str] = [
    "aerial tramway pylon",
    "aerodrome",
    "airstrip",
    "fuel station",
    "building",
    "chimney",
    "communications tower",
    "crane",
    "flagpole",
    "fountain",
    "wind turbine",
    "helipad",
    "highway or road",
    "leisure area or park",
    "lighthouse",
    "obelisk",
    "observatory",
    "parking area",
    "petroleum well",
    "power plant",
    "power substation",
    "power transmission tower",
    "river",
    "runway",
    "satellite dish",
    "silo",
    "storage tank",
    "taxiway",
    "water tower",
    "industrial works or factory",
]

# WorldCereal crop type classification task descriptions.
# Order must match Modality.WORLDCEREAL.band_order.
WORLDCEREAL_CLASSES: list[str] = [
    "annual temporary crops",
    "irrigated maize main season",
    "maize main season",
    "irrigated maize second season",
    "maize second season",
    "spring cereals",
    "irrigated winter cereals",
    "winter cereals",
]
