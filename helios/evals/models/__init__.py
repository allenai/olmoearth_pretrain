"""Models for evals."""

from helios.evals.models.dinov2.dinov2 import DINOv2, DINOv2Config
from enum import StrEnum
from helios.evals.models.dinov3.dinov3 import DINOv3, DINOv3Config
from helios.evals.models.galileo import GalileoConfig, GalileoWrapper
from helios.evals.models.panopticon.panopticon import Panopticon, PanopticonConfig
from helios.evals.models.dofav2.dofa_v2 import DOFAv2, DOFAv2Config

class BaselineModels(StrEnum):
    DINOv2 = "dino_v2"
    DINOv3 = "dino_v3"
    Galileo = "galileo"
    Panopticon = "panopticon"
    DOFAv2 = "dofa_v2"

def get_launch_script_path(model_name: str) -> str:
    """Get the launch script path for a model."""
    if model_name == BaselineModels.DINOv2:
        return "helios/evals/models/dinov2/dinov2_launch.py"
    elif model_name == BaselineModels.DINOv3:
        return "helios/evals/models/dinov3/dinov3_launch.py"
    elif model_name == BaselineModels.Galileo:
        return "helios/evals/models/galileo/galileo_launch.py"
    elif model_name == BaselineModels.Panopticon:
        return "helios/evals/models/panopticon/panopticon_launch.py"
    elif model_name == BaselineModels.DOFAv2:
        return "helios/evals/models/dofav2/dofa_v2_launch.py"
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
    "DOFAv2",
    "DOFAv2Config",
]
