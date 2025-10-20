"""Constants for downloading dinov3 models from torch hub."""

import os
from enum import StrEnum


class DinoV3Models(StrEnum):
    """Names for different DInoV3 images on torch hub."""

    BASE_WEB = "dinov3_vitb16"
    LARGE_WEB = "dinov3_vitl16"
    HUGE_PLUS_WEB = "dinov3_vith16plus"
    FULL_7B_WEB = "dinov3_vit7b16"
    LARGE_SATELLITE = "dinov3_vitl16_sat"
    FULL_7B_SATELLITE = "dinov3_vit7b16_sat"


# if non Ai2, user set the DINOV3_REPO_DIR and DINOV3_CHECKPOINT_DIR environment variables
REPO_DIR = os.environ.get(
    "DINOV3_REPO_DIR",
    "/weka/dfive-default/helios/models/dinov3/repo/dinov3",
)
CHECKPOINT_DIR = os.environ.get(
    "DINOV3_CHECKPOINT_DIR",
    "/weka/dfive-default/helios/models/dinov3/checkpoints",
)
MODEL_TO_TORCHHUB_ID_AND_WEIGHTS_URL = {
    DinoV3Models.LARGE_SATELLITE: (
        DinoV3Models.LARGE_SATELLITE.replace("_sat", ""),
        f"{CHECKPOINT_DIR}/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
    ),
    DinoV3Models.BASE_WEB: (
        DinoV3Models.BASE_WEB,
        f"{CHECKPOINT_DIR}/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    ),  # maybe redownload this
    DinoV3Models.LARGE_WEB: (
        DinoV3Models.LARGE_WEB,
        f"{CHECKPOINT_DIR}/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    ),
    DinoV3Models.HUGE_PLUS_WEB: (
        DinoV3Models.HUGE_PLUS_WEB,
        f"{CHECKPOINT_DIR}/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    ),
    DinoV3Models.FULL_7B_WEB: (
        DinoV3Models.FULL_7B_WEB,
        f"{CHECKPOINT_DIR}/dinov3_vit7b16_pretrain_lvd1689m-a955f4.pth",
    ),
    DinoV3Models.FULL_7B_SATELLITE: (
        DinoV3Models.FULL_7B_SATELLITE.replace("_sat", ""),
        f"{CHECKPOINT_DIR}/dinov3_vit7b16_pretrain_sat493m-a6675841.pth",
    ),
}
