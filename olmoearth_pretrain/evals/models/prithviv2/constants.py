"""Constants for Prithvi v2 model variants."""

from enum import StrEnum


class PrithviV2Models(StrEnum):
    """Names for different Prithvi v2 models on Hugging Face."""

    VIT_300 = "Prithvi-EO-2.0-300M"
    VIT_600 = "Prithvi-EO-2.0-600M"


MODEL_TO_HF_INFO = {
    PrithviV2Models.VIT_300: {
        "hf_hub_id": f"ibm-nasa-geospatial/{PrithviV2Models.VIT_300.value}",
        "weights": "Prithvi_EO_V2_300M.pt",
        "revision": "b2f2520ab889f42a25c5361ba18761fcb4ea44ad",
    },
    PrithviV2Models.VIT_600: {
        "hf_hub_id": f"ibm-nasa-geospatial/{PrithviV2Models.VIT_600.value}",
        "weights": "Prithvi_EO_V2_600M.pt",
        "revision": "87f15784813828dc37aa3197a143cd4689e4d080",
    },
}


__all__ = ["MODEL_TO_HF_INFO", "PrithviV2Models"]
