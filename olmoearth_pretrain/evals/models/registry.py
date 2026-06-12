"""Lightweight eval model metadata.

This module intentionally avoids importing concrete model adapters, since those
often require optional third-party dependencies.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from typing import Any

from olmoearth_pretrain.evals.models.dinov3.constants import DinoV3Models
from olmoearth_pretrain.evals.models.prithviv2.constants import PrithviV2Models


class BaselineModelName(StrEnum):
    """Enum for baseline model names."""

    DINO_V3 = "dino_v3"
    PANOPTICON = "panopticon"
    GALILEO = "galileo"
    SATLAS = "satlas"
    CROMA = "croma"
    PRESTO = "presto"
    ANYSAT = "anysat"
    TESSERA = "tessera"
    PRITHVI_V2 = "prithvi_v2"
    TERRAMIND = "terramind"
    CLAY = "clay"


@dataclass(frozen=True)
class BaselineModelSpec:
    """Lightweight metadata for an eval baseline adapter."""

    name: BaselineModelName
    adapter_class_name: str
    adapter_module: str
    config_class_name: str
    wrapper_class_name: str
    launch_script_path: str
    module_prefix: str
    size_names: tuple[str, ...] = ()
    extra_lazy_exports: tuple[tuple[str, str], ...] = ()
    launch_config_kwargs: tuple[tuple[str, Any], ...] = ()


BASELINE_MODEL_SPECS: tuple[BaselineModelSpec, ...] = (
    BaselineModelSpec(
        name=BaselineModelName.DINO_V3,
        adapter_class_name="DINOv3",
        adapter_module="olmoearth_pretrain.evals.models.dinov3.dinov3",
        config_class_name="DINOv3Config",
        wrapper_class_name="DINOv3EvalWrapper",
        launch_script_path="olmoearth_pretrain/evals/models/dinov3/dino_v3_launch.py",
        module_prefix="olmoearth_pretrain.evals.models.dinov3",
        size_names=tuple(model.value for model in DinoV3Models),
        extra_lazy_exports=(
            ("DinoV3Models", "olmoearth_pretrain.evals.models.dinov3.constants"),
        ),
        launch_config_kwargs=(
            ("apply_normalization", True),
            ("size", DinoV3Models.LARGE_SATELLITE),
        ),
    ),
    BaselineModelSpec(
        name=BaselineModelName.PANOPTICON,
        adapter_class_name="Panopticon",
        adapter_module="olmoearth_pretrain.evals.models.panopticon.panopticon",
        config_class_name="PanopticonConfig",
        wrapper_class_name="PanopticonEvalWrapper",
        launch_script_path="olmoearth_pretrain/evals/models/panopticon/panopticon_launch.py",
        module_prefix="olmoearth_pretrain.evals.models.panopticon",
    ),
    BaselineModelSpec(
        name=BaselineModelName.GALILEO,
        adapter_class_name="GalileoWrapper",
        adapter_module="olmoearth_pretrain.evals.models.galileo",
        config_class_name="GalileoConfig",
        wrapper_class_name="GalileoEvalWrapper",
        launch_script_path="olmoearth_pretrain/evals/models/galileo/galileo_launch.py",
        module_prefix="olmoearth_pretrain.evals.models.galileo",
        size_names=("nano", "tiny", "base"),
    ),
    BaselineModelSpec(
        name=BaselineModelName.SATLAS,
        adapter_class_name="Satlas",
        adapter_module="olmoearth_pretrain.evals.models.satlas.satlas",
        config_class_name="SatlasConfig",
        wrapper_class_name="SatlasEvalWrapper",
        launch_script_path="olmoearth_pretrain/evals/models/satlas/satlas_launch.py",
        module_prefix="olmoearth_pretrain.evals.models.satlas",
    ),
    BaselineModelSpec(
        name=BaselineModelName.CROMA,
        adapter_class_name="Croma",
        adapter_module="olmoearth_pretrain.evals.models.croma.croma",
        config_class_name="CromaConfig",
        wrapper_class_name="CromaEvalWrapper",
        launch_script_path="olmoearth_pretrain/evals/models/croma/croma_launch.py",
        module_prefix="olmoearth_pretrain.evals.models.croma",
        size_names=("base", "large"),
    ),
    BaselineModelSpec(
        name=BaselineModelName.PRESTO,
        adapter_class_name="PrestoWrapper",
        adapter_module="olmoearth_pretrain.evals.models.presto.presto",
        config_class_name="PrestoConfig",
        wrapper_class_name="PrestoEvalWrapper",
        launch_script_path="olmoearth_pretrain/evals/models/presto/presto_launch.py",
        module_prefix="olmoearth_pretrain.evals.models.presto",
    ),
    BaselineModelSpec(
        name=BaselineModelName.ANYSAT,
        adapter_class_name="AnySat",
        adapter_module="olmoearth_pretrain.evals.models.anysat.anysat",
        config_class_name="AnySatConfig",
        wrapper_class_name="AnySatEvalWrapper",
        launch_script_path="olmoearth_pretrain/evals/models/anysat/anysat_launch.py",
        module_prefix="olmoearth_pretrain.evals.models.anysat",
    ),
    BaselineModelSpec(
        name=BaselineModelName.TESSERA,
        adapter_class_name="Tessera",
        adapter_module="olmoearth_pretrain.evals.models.tessera.tessera",
        config_class_name="TesseraConfig",
        wrapper_class_name="TesseraEvalWrapper",
        launch_script_path="olmoearth_pretrain/evals/models/tessera/tessera_launch.py",
        module_prefix="olmoearth_pretrain.evals.models.tessera",
    ),
    BaselineModelSpec(
        name=BaselineModelName.PRITHVI_V2,
        adapter_class_name="PrithviV2",
        adapter_module="olmoearth_pretrain.evals.models.prithviv2.prithviv2",
        config_class_name="PrithviV2Config",
        wrapper_class_name="PrithviV2EvalWrapper",
        launch_script_path="olmoearth_pretrain/evals/models/prithviv2/prithviv2_launch.py",
        module_prefix="olmoearth_pretrain.evals.models.prithviv2",
        size_names=tuple(model.value for model in PrithviV2Models),
        extra_lazy_exports=(
            (
                "PrithviV2Models",
                "olmoearth_pretrain.evals.models.prithviv2.constants",
            ),
        ),
    ),
    BaselineModelSpec(
        name=BaselineModelName.TERRAMIND,
        adapter_class_name="Terramind",
        adapter_module="olmoearth_pretrain.evals.models.terramind.terramind",
        config_class_name="TerramindConfig",
        wrapper_class_name="TerramindEvalWrapper",
        launch_script_path="olmoearth_pretrain/evals/models/terramind/terramind_launch.py",
        module_prefix="olmoearth_pretrain.evals.models.terramind",
        size_names=("base", "large"),
    ),
    BaselineModelSpec(
        name=BaselineModelName.CLAY,
        adapter_class_name="Clay",
        adapter_module="olmoearth_pretrain.evals.models.clay.clay",
        config_class_name="ClayConfig",
        wrapper_class_name="ClayEvalWrapper",
        launch_script_path="olmoearth_pretrain/evals/models/clay/clay_launch.py",
        module_prefix="olmoearth_pretrain.evals.models.clay",
    ),
)

BASELINE_MODEL_SPECS_BY_NAME: dict[BaselineModelName, BaselineModelSpec] = {
    spec.name: spec for spec in BASELINE_MODEL_SPECS
}

MODELS_WITH_MULTIPLE_SIZES: dict[BaselineModelName, list[str]] = {
    spec.name: list(spec.size_names) for spec in BASELINE_MODEL_SPECS if spec.size_names
}


def get_lazy_model_exports() -> dict[str, str]:
    """Return public lazy exports for concrete baseline adapters."""
    exports: dict[str, str] = {}
    for spec in BASELINE_MODEL_SPECS:
        exports[spec.adapter_class_name] = spec.adapter_module
        exports[spec.config_class_name] = spec.adapter_module
        exports.update(spec.extra_lazy_exports)
    return exports


def get_launch_script_path(model_name: str) -> str:
    """Get the launch script path for a model."""
    try:
        return BASELINE_MODEL_SPECS_BY_NAME[
            BaselineModelName(model_name)
        ].launch_script_path
    except ValueError:
        raise ValueError(f"Invalid model name: {model_name}") from None


def build_registered_model_config(model_name: str) -> Any:
    """Build the default launch config for a registered eval baseline."""
    try:
        spec = BASELINE_MODEL_SPECS_BY_NAME[BaselineModelName(model_name)]
    except ValueError:
        raise ValueError(f"Invalid model name: {model_name}") from None

    module = import_module(spec.adapter_module)
    config_cls = getattr(module, spec.config_class_name)
    return config_cls(**dict(spec.launch_config_kwargs))


def make_registered_build_model_config(
    model_name: BaselineModelName,
) -> Callable[[Any], Any]:
    """Create a launch-compatible build_model_config function for a baseline."""

    def build_model_config(common: Any) -> Any:
        """Build the model config for an experiment."""
        return build_registered_model_config(model_name)

    return build_model_config
