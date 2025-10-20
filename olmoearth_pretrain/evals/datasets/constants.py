"""Useful constants for evals."""

from olmoearth_pretrain.data.constants import Modality

EVAL_S2_BAND_NAMES = [
    "01 - Coastal aerosol",
    "02 - Blue",
    "03 - Green",
    "04 - Red",
    "05 - Vegetation Red Edge",
    "06 - Vegetation Red Edge",
    "07 - Vegetation Red Edge",
    "08 - NIR",
    "08A - Vegetation Red Edge",
    "09 - Water vapour",
    "10 - SWIR - Cirrus",
    "11 - SWIR",
    "12 - SWIR",
]

EVAL_S2_L2A_BAND_NAMES = [b for b in EVAL_S2_BAND_NAMES if b != "10 - SWIR - Cirrus"]

EVAL_S1_BAND_NAMES = [
    "vv",
    "vh",
]

EVAL_L8_BAND_NAMES = [
    "B1",
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B9",
    "B10",
    "B11",
]

EVAL_SRTM_BAND_NAMES = ["srtm"]


# Get the corresponding index from either Sentinel2 L1C or L2A band names
def _eval_s2_band_index_from_helios_name(
    helios_name: str, band_names: list[str]
) -> int:
    for idx, band_name in enumerate(band_names):
        if helios_name.endswith(band_name.split(" ")[0][-2:]):
            return idx
    raise ValueError(f"Unmatched band name {helios_name}")


def _eval_s1_band_index_from_helios_name(helios_name: str) -> int:
    for idx, band_name in enumerate(EVAL_S1_BAND_NAMES):
        if helios_name == band_name:
            return idx
    raise ValueError(f"Unmatched band name {helios_name}")


def _eval_l8_band_index_from_helios_name(helios_name: str) -> int:
    for idx, band_name in enumerate(EVAL_L8_BAND_NAMES):
        if helios_name == band_name:
            return idx
    raise ValueError(f"Unmatched band name {helios_name}")


EVAL_TO_HELIOS_S2_BANDS = [
    _eval_s2_band_index_from_helios_name(b, EVAL_S2_BAND_NAMES)
    for b in Modality.SENTINEL2_L2A.band_order
]

EVAL_TO_HELIOS_S2_L2A_BANDS = [
    _eval_s2_band_index_from_helios_name(b, EVAL_S2_L2A_BAND_NAMES)
    for b in Modality.SENTINEL2_L2A.band_order
]

EVAL_TO_HELIOS_S1_BANDS = [
    _eval_s1_band_index_from_helios_name(b) for b in Modality.SENTINEL1.band_order
]

EVAL_TO_HELIOS_L8_BANDS = [
    _eval_l8_band_index_from_helios_name(b) for b in Modality.LANDSAT.band_order
]

# one to one mapping
EVAL_TO_HELIOS_SRTM_BANDS = [0]
