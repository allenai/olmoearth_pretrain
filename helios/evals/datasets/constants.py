"""Useful constants for evals."""

from helios.data.constants import Modality

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

EVAL_S1_BAND_NAMES = [
    "vv",
    "vh",
]


def _eval_s2_band_index_from_helios_name(helios_name: str) -> int:
    for idx, band_name in enumerate(EVAL_S2_BAND_NAMES):
        if helios_name.endswith(band_name.split(" ")[0][-2:]):
            return idx
    raise ValueError(f"Unmatched band name {helios_name}")


def _eval_s1_band_index_from_helios_name(helios_name: str) -> int:
    for idx, band_name in enumerate(EVAL_S1_BAND_NAMES):
        if helios_name == band_name:
            return idx
    raise ValueError(f"Unmatched band name {helios_name}")


# For now, with Sentinel2 L2A, we will drop the B10 band in the eval datasets
EVAL_TO_HELIOS_S2_BANDS = [
    _eval_s2_band_index_from_helios_name(b) for b in Modality.SENTINEL2_L2A.band_order
]

EVAL_TO_HELIOS_S1_BANDS = [
    _eval_s1_band_index_from_helios_name(b) for b in Modality.SENTINEL1.band_order
]
