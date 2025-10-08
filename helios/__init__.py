"""Backward compatibility shim for the deprecated ``helios`` package name."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types
import warnings
from collections.abc import Sequence
from typing import cast

_TARGET_PACKAGE = "olmoearth_pretrain"
_DEPRECATION_MESSAGE = (
    "The 'helios' package has been renamed to 'olmoearth_pretrain'. "
    "Please update your imports; this compatibility shim will be removed in a future release."
)

warnings.warn(_DEPRECATION_MESSAGE, FutureWarning, stacklevel=2)

_target_pkg = importlib.import_module(_TARGET_PACKAGE)

__all__: list[str] = list(getattr(_target_pkg, "__all__", []))
_doc: str | None = cast(str | None, getattr(_target_pkg, "__doc__", None))
_path: list[str] = list(getattr(_target_pkg, "__path__", []))
__package__ = __name__

globals()["__doc__"] = _doc
globals()["__path__"] = _path

globals().update(
    {name: getattr(_target_pkg, name) for name in getattr(_target_pkg, "__all__", [])}
)


class _HeliosAliasLoader(importlib.abc.Loader):
    """Loader that aliases ``helios.*`` modules to ``olmoearth_pretrain.*`` modules."""

    def __init__(self, target_name: str) -> None:
        self._target_name = target_name

    def create_module(
        self, spec: importlib.machinery.ModuleSpec
    ) -> types.ModuleType | None:
        module = importlib.import_module(self._target_name)
        return module

    def exec_module(self, module: types.ModuleType) -> None:
        # Module is already initialised by create_module; nothing else to do.
        sys.modules.setdefault(self._target_name, module)


class _HeliosAliasFinder(importlib.abc.MetaPathFinder):
    """Meta path finder that maps ``helios`` imports to ``olmoearth_pretrain``."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: types.ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname == __name__:
            return None
        if not fullname.startswith(__name__ + "."):
            return None

        target_name = _TARGET_PACKAGE + fullname[len(__name__) :]
        target_spec = importlib.util.find_spec(target_name)
        if target_spec is None:
            return None

        spec = importlib.machinery.ModuleSpec(
            name=fullname,
            loader=_HeliosAliasLoader(target_name),
            is_package=target_spec.submodule_search_locations is not None,
        )
        spec.origin = target_spec.origin
        spec.submodule_search_locations = target_spec.submodule_search_locations
        return spec


if not any(isinstance(finder, _HeliosAliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _HeliosAliasFinder())


def __getattr__(name: str) -> object:
    warnings.warn(_DEPRECATION_MESSAGE, FutureWarning, stacklevel=2)
    value = getattr(_target_pkg, name)
    if isinstance(value, types.ModuleType):
        sys.modules.setdefault(f"{__name__}.{name}", value)
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(vars(_target_pkg)))
