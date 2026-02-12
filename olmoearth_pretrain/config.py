"""Centralized config handling with optional olmo-core dependency.

This module provides a unified Config class that works with or without olmo-core:
- If olmo-core is installed: uses olmo-core's full-featured Config
- If olmo-core is not installed: uses a minimal standalone Config for inference

Usage:
    from olmoearth_pretrain.config import Config, OLMO_CORE_AVAILABLE, require_olmo_core

    @dataclass
    class MyConfig(Config):
        ...

For training code, add a guard at module level:
    from olmoearth_pretrain.config import require_olmo_core
    require_olmo_core("Training")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields, is_dataclass
from importlib import import_module
from typing import Any, TypeVar

# === Single source of truth for olmo-core availability ===
try:
    from olmo_core.config import Config as _OlmoCoreConfig

    OLMO_CORE_AVAILABLE = True
except ImportError:
    OLMO_CORE_AVAILABLE = False
    _OlmoCoreConfig = None  # type: ignore[assignment, misc]


def require_olmo_core(operation: str = "This operation") -> None:
    """Guard for training code - raises ImportError if olmo-core not available.

    Use this at the entry points of training modules to provide a clear error message.

    Args:
        operation: Description of what requires olmo-core (used in error message).

    Raises:
        ImportError: If olmo-core is not installed.

    Example:
        from olmoearth_pretrain.config import require_olmo_core
        require_olmo_core("Training")  # Raises if olmo-core not available
    """
    if not OLMO_CORE_AVAILABLE:
        raise ImportError(
            f"{operation} requires olmo-core. "
            "Install with: pip install olmoearth-pretrain[training]"
        )


C = TypeVar("C", bound="_StandaloneConfig")


@dataclass
class _StandaloneConfig:
    """Minimal Config for inference-only mode without olmo-core.

    This provides just enough functionality to deserialize model configs from JSON
    and build models. It intentionally does NOT support:
    - OmegaConf-based merging
    - CLI overrides via dotlist
    - YAML loading
    - Validation beyond what dataclasses provide

    For full functionality, install olmo-core.
    """

    CLASS_NAME_FIELD = "_CLASS_"

    @classmethod
    def _resolve_class(cls, class_name: str) -> type:
        """Resolve a fully-qualified class name to a class object."""
        if "." not in class_name:
            raise ValueError(f"Class name must be fully qualified (got '{class_name}')")
        *modules, cls_name = class_name.split(".")
        module_name = ".".join(modules)
        module = import_module(module_name)
        return getattr(module, cls_name)

    @classmethod
    def _clean_data(cls, data: Any) -> Any:
        """Recursively clean data, resolving _CLASS_ fields to actual instances."""
        if isinstance(data, dict):
            # Check if this dict represents a config class
            class_name = data.get(cls.CLASS_NAME_FIELD)
            cleaned = {
                k: cls._clean_data(v)
                for k, v in data.items()
                if k != cls.CLASS_NAME_FIELD
            }

            if class_name is not None:
                resolved_cls = cls._resolve_class(class_name)
                if not is_dataclass(resolved_cls):
                    raise TypeError(f"Class '{class_name}' is not a dataclass")
                # Get the field names for this dataclass
                field_names = {f.name for f in fields(resolved_cls)}
                # Filter to only include valid fields
                valid_kwargs = {k: v for k, v in cleaned.items() if k in field_names}
                try:
                    return resolved_cls(**valid_kwargs)
                except TypeError as e:
                    raise TypeError(f"Failed to instantiate {class_name}: {e}") from e
            return cleaned

        elif isinstance(data, list | tuple):
            cleaned_items = [cls._clean_data(item) for item in data]
            return type(data)(cleaned_items)

        else:
            return data

    @classmethod
    def from_dict(
        cls: type[C], data: dict[str, Any], overrides: list[str] | None = None
    ) -> C:
        """Deserialize from a dictionary, handling nested _CLASS_ fields.

        Args:
            data: Dictionary representation of the config.
            overrides: Ignored in standalone mode (requires olmo-core for dotlist support).

        Returns:
            An instance of the config class.

        Note:
            The `overrides` parameter is accepted for API compatibility but ignored.
            Install olmo-core for full override support.
        """
        if overrides:
            warnings.warn(
                "Config overrides are not supported in standalone mode. "
                "Install olmo-core for full functionality.",
                UserWarning,
                stacklevel=2,
            )

        cleaned = cls._clean_data(data)

        if isinstance(cleaned, cls):
            return cleaned
        elif isinstance(cleaned, dict):
            # Get field names for this class
            field_names = {f.name for f in fields(cls)}
            valid_kwargs = {k: v for k, v in cleaned.items() if k in field_names}
            return cls(**valid_kwargs)
        else:
            raise TypeError(f"Expected dict, got {type(cleaned)}")

    def as_dict(
        self,
        *,
        exclude_none: bool = False,
        exclude_private_fields: bool = False,
        include_class_name: bool = False,
        json_safe: bool = False,
        recurse: bool = True,
    ) -> dict[str, Any]:
        """Convert to a dictionary.

        Args:
            exclude_none: Don't include values that are None.
            exclude_private_fields: Don't include private fields (starting with _).
            include_class_name: Include _CLASS_ field with fully-qualified class name.
            json_safe: Convert non-JSON-safe types to strings.
            recurse: Recursively convert nested dataclasses.

        Returns:
            Dictionary representation of this config.
        """

        def convert(obj: Any) -> Any:
            if is_dataclass(obj) and not isinstance(obj, type):
                result = {}
                if include_class_name:
                    result[self.CLASS_NAME_FIELD] = (
                        f"{obj.__class__.__module__}.{obj.__class__.__name__}"
                    )
                for field in fields(obj):
                    if exclude_private_fields and field.name.startswith("_"):
                        continue
                    value = getattr(obj, field.name)
                    if exclude_none and value is None:
                        continue
                    if recurse:
                        value = convert(value)
                    result[field.name] = value
                return result
            elif isinstance(obj, dict):
                return {k: convert(v) if recurse else v for k, v in obj.items()}
            elif isinstance(obj, list | tuple | set):
                converted = [convert(item) if recurse else item for item in obj]
                if json_safe:
                    return converted
                return type(obj)(converted)
            elif obj is None or isinstance(obj, float | int | bool | str):
                return obj
            elif json_safe:
                return str(obj)
            else:
                return obj

        return convert(self)

    def as_config_dict(self) -> dict[str, Any]:
        """Convert to a JSON-safe dictionary suitable for serialization.

        This is a convenience wrapper around as_dict() with settings appropriate
        for saving configs to JSON files.
        """
        return self.as_dict(
            exclude_none=True,
            exclude_private_fields=True,
            include_class_name=True,
            json_safe=True,
            recurse=True,
        )

    def validate(self) -> None:
        """Validate the config. Override in subclasses."""
        pass

    def build(self) -> Any:
        """Build the object this config represents.

        Subclasses must implement this method.

        Raises:
            NotImplementedError: Always, unless overridden by subclass.
        """
        raise NotImplementedError("Subclasses must implement build()")


# === The unified export ===
# Use olmo-core Config if available, otherwise use standalone
if OLMO_CORE_AVAILABLE:
    Config = _OlmoCoreConfig  # type: ignore[assignment,misc]
else:
    Config = _StandaloneConfig
    # Emit warning once when module is first imported
    warnings.warn(
        "olmo-core not installed. Running in inference-only mode. "
        "For training: pip install olmoearth-pretrain[training]",
        UserWarning,
        stacklevel=2,
    )


__all__ = ["Config", "OLMO_CORE_AVAILABLE", "require_olmo_core"]
