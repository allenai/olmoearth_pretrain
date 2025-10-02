"""Callbacks for the trainer specific to Helios."""

from .evaluator_callback import DownstreamEvaluatorCallbackConfig
from .speed_monitor import OlmoEarthSpeedMonitorCallback, HeliosSpeedMonitorCallback
from .wandb import OlmoEarthWandBCallback, HeliosWandBCallback

__all__ = [
    "DownstreamEvaluatorCallbackConfig",
    "OlmoEarthSpeedMonitorCallback",
    "OlmoEarthWandBCallback",
    # Aliases
    "HeliosSpeedMonitorCallback",
    "HeliosWandBCallback",
]
