"""Constants shared across eval modules."""

from olmoearth_pretrain.modalities import Modality, ModalitySpec

# rslearn layer name -> OlmoEarth ModalitySpec (single source of truth)
RSLEARN_TO_OLMOEARTH: dict[str, ModalitySpec] = {
    "sentinel2": Modality.SENTINEL2_L2A,
    "sentinel2_l2a": Modality.SENTINEL2_L2A,
    "sentinel1": Modality.SENTINEL1,
    "sentinel1_ascending": Modality.SENTINEL1,
    "sentinel1_descending": Modality.SENTINEL1,
    "landsat": Modality.LANDSAT,
}

RSLEARN_LAYER_PREFIXES = ("pre_", "post_")


def resolve_rslearn_layer_name(layer_name: str) -> str | None:
    """Resolve an rslearn layer name to a key in ``RSLEARN_TO_OLMOEARTH``."""
    if layer_name in RSLEARN_TO_OLMOEARTH:
        return layer_name

    for prefix in RSLEARN_LAYER_PREFIXES:
        if layer_name.startswith(prefix):
            stripped = layer_name[len(prefix) :]
            if stripped in RSLEARN_TO_OLMOEARTH:
                return stripped

    return None


def resolve_rslearn_modality(layer_name: str) -> ModalitySpec | None:
    """Resolve an rslearn layer name to an OlmoEarth modality, if known."""
    resolved = resolve_rslearn_layer_name(layer_name)
    if resolved is None:
        return None
    return RSLEARN_TO_OLMOEARTH[resolved]
