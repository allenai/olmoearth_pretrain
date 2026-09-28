"""Constants shared across eval modules."""

import logging

from olmoearth_pretrain.data.constants import Modality, ModalitySpec

logger = logging.getLogger(__name__)

# rslearn layer name -> OlmoEarth ModalitySpec (single source of truth)
RSLEARN_TO_OLMOEARTH: dict[str, ModalitySpec] = {
    "sentinel2": Modality.SENTINEL2_L2A,
    "sentinel2_l2a": Modality.SENTINEL2_L2A,
    "sentinel1": Modality.SENTINEL1,
    "sentinel1_ascending": Modality.SENTINEL1,
    "sentinel1_descending": Modality.SENTINEL1,
    "landsat": Modality.LANDSAT,
    # Daily ERA5-Land time series — used by the dedicated ERA5 encoder
    # (see olmoearth_pretrain/nn/era5_encoder.py). Canonical rslearn layer
    # name is ``era5l_day_10`` (matches Modality.ERA5L_DAY_10.name); the
    # ``era5_land`` / ``era5_land_daily`` aliases are kept defensively in
    # case existing rslearn dataset configs spell it differently.
    "era5l_day_10": Modality.ERA5L_DAY_10,
    "era5_land_daily": Modality.ERA5L_DAY_10,
    "era5d_historical": Modality.ERA5L_DAY_10,
}


def resolve_rslearn_layer_name(layer_name: str) -> str | None:
    """Resolve an rslearn layer name to a key in RSLEARN_TO_OLMOEARTH.

    Tries, in order:
      1. Direct lookup (e.g. ``"sentinel2_l2a"``).
      2. Prefix stripping: ``"pre_"`` / ``"post_"`` (e.g. ``"pre_sentinel2"``).
      3. Progressive suffix stripping: (e.g. ``"pre_sentinel2_feb"``)
      4. Prefix stripping *then* suffix stripping

    Returns:
        The matched key in ``RSLEARN_TO_OLMOEARTH``, or ``None``.
    """
    if layer_name in RSLEARN_TO_OLMOEARTH:
        return layer_name

    # Build candidates: original + prefix-stripped variants
    candidates = [layer_name]
    for prefix in ("pre_", "post_"):
        if layer_name.startswith(prefix):
            candidates.append(layer_name[len(prefix) :])

    for candidate in candidates:
        current = candidate
        while "_" in current:
            current = current.rsplit("_", 1)[0]
            if current in RSLEARN_TO_OLMOEARTH:
                logger.info(
                    "Resolved rslearn layer %r -> %r (modality: %s)",
                    layer_name,
                    current,
                    RSLEARN_TO_OLMOEARTH[current].name,
                )
                return current

    return None
