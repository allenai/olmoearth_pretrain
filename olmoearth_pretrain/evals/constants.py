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
    # Daily ERA5-Land time series — used by the dedicated ERA5 encoder
    # (see olmoearth_pretrain/nn/era5_encoder.py). Canonical rslearn layer
    # name is ``era5l_day_10`` (matches Modality.ERA5L_DAY_10.name); the
    # ``era5_land`` / ``era5_land_daily`` aliases are kept defensively in
    # case existing rslearn dataset configs spell it differently.
    "era5l_day_10": Modality.ERA5L_DAY_10,
    "era5_land_daily": Modality.ERA5L_DAY_10,
    "era5_land": Modality.ERA5L_DAY_10,
}
