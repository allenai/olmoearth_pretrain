"""Tests for lightweight eval model registry imports."""

import subprocess
import sys
from pathlib import Path

from pytest import MonkeyPatch
from torch import nn

from olmoearth_pretrain.evals.eval_wrapper import (
    _ADAPTER_WRAPPERS,
    _model_inherits_from,
    get_eval_wrapper,
)
from olmoearth_pretrain.evals.models import (
    BASELINE_MODEL_SPECS,
    BASELINE_MODEL_SPECS_BY_NAME,
    BaselineModelName,
    BaselineModelSpec,
    build_registered_model_config,
    get_launch_script_path,
    make_registered_build_model_config,
)
from olmoearth_pretrain.evals.task_types import TaskType
from olmoearth_pretrain.nn.pooling import PoolingType


class _FakeBase(nn.Module):
    pass


class _FakeChild(_FakeBase):
    pass


def test_model_metadata_import_does_not_load_concrete_adapters() -> None:
    """Metadata imports should not eagerly import optional model adapters."""
    script = """
import sys

from olmoearth_pretrain.evals.models import (
    MODELS_WITH_MULTIPLE_SIZES,
    BaselineModelName,
    get_launch_script_path,
)

assert BaselineModelName.DINO_V3 in MODELS_WITH_MULTIPLE_SIZES
assert get_launch_script_path(BaselineModelName.DINO_V3).endswith(
    "olmoearth_pretrain/evals/models/dinov3/dino_v3_launch.py"
)
assert "olmoearth_pretrain.evals.models.clay.clay" not in sys.modules
assert "olmoearth_pretrain.evals.models.terramind.terramind" not in sys.modules
assert "olmoearth_pretrain.evals.models.satlas.satlas" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_model_star_import_is_lightweight() -> None:
    """Star imports should expose lightweight metadata without optional adapters."""
    script = """
import sys

namespace = {}
exec("from olmoearth_pretrain.evals.models import *", namespace)

assert "BaselineModelName" in namespace
assert "DinoV3Models" in namespace
assert "PrithviV2Models" in namespace
assert "AnySat" not in namespace
assert "olmoearth_pretrain.evals.models.prithviv2.prithviv2" not in sys.modules
assert "olmoearth_pretrain.evals.models.clay.clay" not in sys.modules
assert "olmoearth_pretrain.evals.models.terramind.terramind" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_eval_wrapper_import_does_not_load_concrete_adapters() -> None:
    """Common eval wrapper infrastructure should not import optional adapters."""
    script = """
import sys

import olmoearth_pretrain.evals.eval_wrapper  # noqa: F401

assert "olmoearth_pretrain.evals.models.clay.clay" not in sys.modules
assert "olmoearth_pretrain.evals.models.terramind.terramind" not in sys.modules
assert "olmoearth_pretrain.evals.models.satlas.satlas" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_baseline_model_specs_cover_every_model_name() -> None:
    """Each baseline enum value should have one metadata spec."""
    assert {spec.name for spec in BASELINE_MODEL_SPECS} == set(BaselineModelName)


def test_baseline_model_launch_paths_exist() -> None:
    """Registry launch paths should point at checked-in launch scripts."""
    repo_root = Path(__file__).parents[4]

    for model_name in BaselineModelName:
        assert (repo_root / get_launch_script_path(model_name)).is_file()


def test_launch_modules_import_without_concrete_adapters() -> None:
    """Launch shims should delegate through registry without importing adapters."""
    script = """
import importlib
import sys

from olmoearth_pretrain.evals.models import BASELINE_MODEL_SPECS

for spec in BASELINE_MODEL_SPECS:
    module_name = spec.launch_script_path[:-3].replace("/", ".")
    module = importlib.import_module(module_name)
    assert callable(module.build_model_config)

loaded = [
    spec.adapter_module
    for spec in BASELINE_MODEL_SPECS
    if spec.adapter_module in sys.modules
    and spec.adapter_module != spec.launch_script_path[:-3].replace("/", ".").rsplit(".", 1)[0]
]
assert loaded == [], loaded
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_build_registered_model_config_uses_spec_metadata(
    monkeypatch: MonkeyPatch,
) -> None:
    """The shared launch helper should import the registered config class lazily."""
    module_name = "tests.fake_eval_model"

    class FakeConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    fake_module = type(sys)(module_name)
    fake_module.FakeConfig = FakeConfig

    monkeypatch.setitem(sys.modules, module_name, fake_module)
    monkeypatch.setitem(
        BASELINE_MODEL_SPECS_BY_NAME,
        BaselineModelName.CLAY,
        BaselineModelSpec(
            name=BaselineModelName.CLAY,
            adapter_class_name="FakeModel",
            adapter_module=module_name,
            config_class_name="FakeConfig",
            wrapper_class_name="FakeWrapper",
            launch_script_path="fake.py",
            module_prefix=module_name,
            launch_config_kwargs=(("size", "base"), ("apply_normalization", True)),
        ),
    )

    config = build_registered_model_config(BaselineModelName.CLAY)

    assert isinstance(config, FakeConfig)
    assert config.kwargs == {"size": "base", "apply_normalization": True}


def test_make_registered_build_model_config_delegates_lazily(
    monkeypatch: MonkeyPatch,
) -> None:
    """Launch shim factory should preserve lazy config construction."""
    module_name = "tests.fake_launch_model"

    class FakeConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    fake_module = type(sys)(module_name)
    fake_module.FakeConfig = FakeConfig

    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setitem(
        BASELINE_MODEL_SPECS_BY_NAME,
        BaselineModelName.ANYSAT,
        BaselineModelSpec(
            name=BaselineModelName.ANYSAT,
            adapter_class_name="FakeModel",
            adapter_module=module_name,
            config_class_name="FakeConfig",
            wrapper_class_name="FakeWrapper",
            launch_script_path="fake.py",
            module_prefix=module_name,
            launch_config_kwargs=(("checkpoint", "fake.pt"),),
        ),
    )

    build_model_config = make_registered_build_model_config(BaselineModelName.ANYSAT)
    assert module_name not in sys.modules

    monkeypatch.setitem(sys.modules, module_name, fake_module)
    config = build_model_config(common=object())

    assert isinstance(config, FakeConfig)
    assert config.kwargs == {"checkpoint": "fake.pt"}


def test_eval_wrapper_registrations_are_derived_from_specs() -> None:
    """Adapter wrapper registrations should follow the lightweight model specs."""
    registrations = {
        (registration.class_name, registration.module_prefix)
        for registration in _ADAPTER_WRAPPERS
    }
    specs = {
        (spec.adapter_class_name, spec.module_prefix) for spec in BASELINE_MODEL_SPECS
    }

    assert registrations == specs


def test_model_inherits_from_matches_exact_package_or_submodule() -> None:
    """Lazy wrapper detection should not match similarly prefixed packages."""
    _FakeBase.__name__ = "Clay"
    _FakeBase.__module__ = "olmoearth_pretrain.evals.models.clay.clay"

    assert _model_inherits_from(
        _FakeChild(), "Clay", "olmoearth_pretrain.evals.models.clay"
    )

    _FakeBase.__module__ = "olmoearth_pretrain.evals.models.claymore"

    assert not _model_inherits_from(
        _FakeChild(), "Clay", "olmoearth_pretrain.evals.models.clay"
    )


def test_get_eval_wrapper_uses_lazy_adapter_registry() -> None:
    """Adapter dispatch should work for every registered adapter identity."""
    for registration in _ADAPTER_WRAPPERS:
        fake_base = type(
            registration.class_name,
            (nn.Module,),
            {"__module__": registration.module_prefix},
        )
        fake_model = type(f"_Fake{registration.class_name}", (fake_base,), {})()

        wrapper = get_eval_wrapper(
            fake_model,
            task_type=TaskType.CLASSIFICATION,
            patch_size=4,
            pooling_type=PoolingType.MEAN,
        )

        assert isinstance(wrapper, registration.wrapper_class)
