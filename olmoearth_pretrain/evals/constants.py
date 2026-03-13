"""Constants shared across eval modules."""

from olmoearth_pretrain.data.constants import Modality, ModalitySpec

# rslearn layer name -> OlmoEarth ModalitySpec (single source of truth)
RSLEARN_TO_OLMOEARTH: dict[str, ModalitySpec] = {
    "sentinel2": Modality.SENTINEL2_L2A,
    "sentinel2_l2a": Modality.SENTINEL2_L2A,
    "sentinel1": Modality.SENTINEL1,
    "sentinel1_ascending": Modality.SENTINEL1,
    "sentinel1_descending": Modality.SENTINEL1,
    "landsat": Modality.LANDSAT,
}
