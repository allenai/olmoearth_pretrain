"""Compatibility helpers for deprecated ``helios`` symbols."""

from __future__ import annotations

import functools
import types
import warnings
from collections.abc import Callable
from typing import Any, TypeVar, no_type_check

T = TypeVar("T", bound=type)
F = TypeVar("F", bound=Callable[..., Any])


@no_type_check
def deprecated_class_alias(new_class: T, old_qualname: str) -> T:
    """Create a deprecated alias for ``new_class`` that warns on instantiation.

    Args:
        new_class: The new class that should be used.
        old_qualname: The dotted path of the legacy class (e.g. ``helios.foo.Bar``).

    Returns:
        A subclass of ``new_class`` that emits a :class:`DeprecationWarning` when
        instantiated and carries the legacy metadata (``__name__``/``__module__``).
    """
    module_name, _, class_name = old_qualname.rpartition(".")
    if not class_name:
        class_name = old_qualname

    warning_message = (
        f"'{old_qualname}' is deprecated and will be removed in a future release. "
        f"Please update your code to use '{new_class.__module__}.{new_class.__name__}'."
    )

    alias: type = types.new_class(class_name, (new_class,))
    alias.__doc__ = new_class.__doc__
    alias.__module__ = module_name or new_class.__module__.replace(
        "olmoearth_pretrain", "helios"
    )
    alias.__qualname__ = class_name
    alias.__deprecated_target__ = new_class

    if issubclass(new_class, tuple):
        original_new = new_class.__new__

        def __new__(cls, *args, **kwargs):  # type: ignore[override]
            warnings.warn(warning_message, DeprecationWarning, stacklevel=2)
            return original_new(cls, *args, **kwargs)

        alias.__new__ = __new__  # type: ignore[assignment]
    else:
        original_init = new_class.__init__

        def __init__(self, *args, **kwargs):  # type: ignore[override]
            warnings.warn(warning_message, DeprecationWarning, stacklevel=2)
            original_init(self, *args, **kwargs)

        alias.__init__ = __init__  # type: ignore[assignment]

    return alias  # type: ignore[return-value]


@no_type_check
def deprecated_function_alias(new_function: F, old_qualname: str) -> F:
    """Create a deprecated alias for ``new_function`` that warns on call."""
    module_name, _, func_name = old_qualname.rpartition(".")
    if not func_name:
        func_name = old_qualname

    warning_message = (
        f"'{old_qualname}' is deprecated and will be removed in a future release. "
        f"Please update your code to use '{new_function.__module__}.{new_function.__name__}'."
    )

    @functools.wraps(new_function)
    def wrapper(*args: Any, **kwargs: Any):
        warnings.warn(warning_message, DeprecationWarning, stacklevel=2)
        return new_function(*args, **kwargs)

    wrapper.__name__ = func_name
    wrapper.__qualname__ = func_name
    wrapper.__module__ = module_name or new_function.__module__.replace(
        "olmoearth_pretrain", "helios"
    )
    wrapper.__deprecated_target__ = new_function  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


__all__ = ["deprecated_class_alias", "deprecated_function_alias"]
