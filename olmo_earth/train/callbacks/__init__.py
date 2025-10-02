"""Callbacks for the trainer specific to Helios."""

from .evaluator_callback import DownstreamEvaluatorCallbackConfig
from .speed_monitor import HeliosSpeedMonitorCallback, OlmoEarthSpeedMonitorCallback
from .wandb import HeliosWandBCallback, OlmoEarthWandBCallback

__all__ = [
    "DownstreamEvaluatorCallbackConfig",
    "OlmoEarthSpeedMonitorCallback",
    "OlmoEarthWandBCallback",
    # Aliases
    "HeliosSpeedMonitorCallback",
    "HeliosWandBCallback",
]
