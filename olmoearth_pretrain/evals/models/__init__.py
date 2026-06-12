"""Models for evals.

Metadata is imported eagerly. Concrete model adapters are loaded lazily so
metadata-only callers do not need optional baseline model dependencies.
"""

from importlib import import_module
from typing import Any

from olmoearth_pretrain.evals.models.registry import (
    BASELINE_MODEL_SPECS,
    BASELINE_MODEL_SPECS_BY_NAME,
    MODELS_WITH_MULTIPLE_SIZES,
    BaselineModelName,
    BaselineModelSpec,
    build_registered_model_config,
    get_launch_script_path,
    get_lazy_model_exports,
    make_registered_build_model_config,
)

_LAZY_EXPORTS = get_lazy_model_exports()


def __getattr__(name: str) -> Any:
    """Lazily import concrete eval model adapters."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_LAZY_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "MODELS_WITH_MULTIPLE_SIZES",
    "BASELINE_MODEL_SPECS",
    "BASELINE_MODEL_SPECS_BY_NAME",
    "BaselineModelSpec",
    "BaselineModelName",
    "DinoV3Models",
    "PrithviV2Models",
    "build_registered_model_config",
    "make_registered_build_model_config",
    "get_launch_script_path",
]
