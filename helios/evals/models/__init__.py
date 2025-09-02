"""Models for evals."""

from helios.evals.models.dinov2.dinov2 import DINOv2, DINOv2Config
from helios.evals.models.dinov3.dinov3 import DINOv3, DINOv3Config
from helios.evals.models.galileo import GalileoConfig, GalileoWrapper
from helios.evals.models.panopticon.panopticon import Panopticon, PanopticonConfig


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
]
