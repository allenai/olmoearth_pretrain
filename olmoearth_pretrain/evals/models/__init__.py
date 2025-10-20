"""Models for evals."""

from enum import StrEnum
from typing import Any

from olmoearth_pretrain.evals.models.anysat.anysat import AnySat, AnySatConfig
from olmoearth_pretrain.evals.models.clay.clay import Clay, ClayConfig
from olmoearth_pretrain.evals.models.copernicusfm.copernicusfm import (
    CopernicusFM,
    CopernicusFMConfig,
)
from olmoearth_pretrain.evals.models.croma.croma import CROMA_SIZES, Croma, CromaConfig
from olmoearth_pretrain.evals.models.dinov2.dinov2 import DINOv2, DINOv2Config
from olmoearth_pretrain.evals.models.dinov3.constants import DinoV3Models
from olmoearth_pretrain.evals.models.dinov3.dinov3 import DINOv3, DINOv3Config
from olmoearth_pretrain.evals.models.galileo import GalileoConfig, GalileoWrapper
from olmoearth_pretrain.evals.models.galileo.single_file_galileo import (
    MODEL_SIZE_TO_WEKA_PATH as GALILEO_MODEL_SIZE_TO_WEKA_PATH,
)
from olmoearth_pretrain.evals.models.panopticon.panopticon import (
    Panopticon,
    PanopticonConfig,
)
from olmoearth_pretrain.evals.models.presto.presto import PrestoConfig, PrestoWrapper
from olmoearth_pretrain.evals.models.prithviv2.prithviv2 import (
    PrithviV2,
    PrithviV2Config,
    PrithviV2Models,
)
from olmoearth_pretrain.evals.models.satlas.satlas import Satlas, SatlasConfig
from olmoearth_pretrain.evals.models.terramind.terramind import (
    TERRAMIND_SIZES,
    Terramind,
    TerramindConfig,
)
from olmoearth_pretrain.evals.models.tessera.tessera import Tessera, TesseraConfig


class BaselineModelName(StrEnum):
    """Enum for baseline model names."""

    DINO_V3 = "dino_v3"
    PANOPTICON = "panopticon"
    GALILEO = "galileo"
    SATLAS = "satlas"
    CROMA = "croma"
    COPERNICUSFM = "copernicusfm"
    PRESTO = "presto"
    ANYSAT = "anysat"
    TESSERA = "tessera"
    PRITHVI_V2 = "prithvi_v2"
    TERRAMIND = "terramind"
    CLAY = "clay"


MODELS_WITH_MULTIPLE_SIZES: dict[BaselineModelName, Any] = {
    BaselineModelName.CROMA: CROMA_SIZES,
    BaselineModelName.DINO_V3: list(DinoV3Models),
    BaselineModelName.GALILEO: GALILEO_MODEL_SIZE_TO_WEKA_PATH.keys(),
    BaselineModelName.PRITHVI_V2: list(PrithviV2Models),
    BaselineModelName.TERRAMIND: TERRAMIND_SIZES,
}


def get_launch_script_path(model_name: str) -> str:
    """Get the launch script path for a model."""
    if model_name == "dino_v2":
        # TODO: Remove as not mantained since dinov3 came out
        return "olmoearth_pretrain/evals/models/dinov2/dinov2_launch.py"
    elif model_name == BaselineModelName.DINO_V3:
        return "olmoearth_pretrain/evals/models/dinov3/dino_v3_launch.py"
    elif model_name == BaselineModelName.GALILEO:
        return "olmoearth_pretrain/evals/models/galileo/galileo_launch.py"
    elif model_name == BaselineModelName.PANOPTICON:
        return "olmoearth_pretrain/evals/models/panopticon/panopticon_launch.py"
    elif model_name == BaselineModelName.TERRAMIND:
        return "olmoearth_pretrain/evals/models/terramind/terramind_launch.py"
    elif model_name == BaselineModelName.SATLAS:
        return "olmoearth_pretrain/evals/models/satlas/satlas_launch.py"
    elif model_name == BaselineModelName.CROMA:
        return "olmoearth_pretrain/evals/models/croma/croma_launch.py"
    elif model_name == BaselineModelName.CLAY:
        return "olmoearth_pretrain/evals/models/clay/clay_launch.py"
    elif model_name == BaselineModelName.COPERNICUSFM:
        return "olmoearth_pretrain/evals/models/copernicusfm/copernicusfm_launch.py"
    elif model_name == BaselineModelName.PRESTO:
        return "olmoearth_pretrain/evals/models/presto/presto_launch.py"
    elif model_name == BaselineModelName.ANYSAT:
        return "olmoearth_pretrain/evals/models/anysat/anysat_launch.py"
    elif model_name == BaselineModelName.TESSERA:
        return "olmoearth_pretrain/evals/models/tessera/tessera_launch.py"
    elif model_name == BaselineModelName.PRITHVI_V2:
        return "olmoearth_pretrain/evals/models/prithviv2/prithviv2_launch.py"
    else:
        raise ValueError(f"Invalid model name: {model_name}")


# TODO: assert that they all store a patch_size variable and supported modalities
__all__ = [
    "DINOv2",
    "DINOv2Config",
    "Panopticon",
    "PanopticonConfig",
    "GalileoWrapper",
    "GalileoConfig",
    "DINOv3",
    "DINOv3Config",
    "Terramind",
    "TerramindConfig",
    "Satlas",
    "SatlasConfig",
    "Croma",
    "CromaConfig",
    "Clay",
    "ClayConfig",
    "CopernicusFM",
    "CopernicusFMConfig",
    "PrestoWrapper",
    "PrestoConfig",
    "AnySat",
    "AnySatConfig",
    "Tessera",
    "TesseraConfig",
    "PrithviV2",
    "PrithviV2Config",
]
