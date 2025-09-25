"""Models for evals."""

from helios.evals.models.anysat.anysat import AnySat, AnySatConfig
from helios.evals.models.copernicusfm.copernicusfm import (
    CopernicusFM,
    CopernicusFMConfig,
)
from helios.evals.models.croma.croma import Croma, CromaConfig
from helios.evals.models.dinov2.dinov2 import DINOv2, DINOv2Config
from helios.evals.models.dinov3.dinov3 import DINOv3, DINOv3Config
from helios.evals.models.galileo import GalileoConfig, GalileoWrapper
from helios.evals.models.panopticon.panopticon import Panopticon, PanopticonConfig
from helios.evals.models.presto.presto import PrestoConfig, PrestoWrapper
from helios.evals.models.prithviv2.prithviv2 import PrithviV2, PrithviV2Config
from helios.evals.models.satlas.satlas import Satlas, SatlasConfig
from helios.evals.models.tessera.tessera import Tessera, TesseraConfig


def get_launch_script_path(model_name: str) -> str:
    """Get the launch script path for a model."""
    if model_name == "dino_v2":
        return "helios/evals/models/dinov2/dinov2_launch.py"
    elif model_name == "dino_v3":
        return "helios/evals/models/dinov3/dinov3_launch.py"
    elif model_name == "galileo":
        return "helios/evals/models/galileo/galileo_launch.py"
    elif model_name == "panopticon":
        return "helios/evals/models/panopticon/panopticon_launch.py"
    elif model_name == "satlas":
        return "helios/evals/models/satlas/satlas_launch.py"
    elif model_name == "croma":
        return "helios/evals/models/croma/croma_launch.py"
    elif model_name == "copernicusfm":
        return "helios/evals/models/copernicusfm/copernicusfm_launch.py"
    elif model_name == "presto":
        return "helios/evals/models/presto/presto_launch.py"
    elif model_name == "anysat":
        return "helios/evals/models/anysat/anysat_launch.py"
    elif model_name == "tessera":
        return "helios/evals/models/tessera/tessera_launch.py"
    elif model_name == "prithvi_v2":
        return "helios/evals/models/prithviv2/prithviv2_launch.py"
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
    "Satlas",
    "SatlasConfig",
    "Croma",
    "CromaConfig",
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
