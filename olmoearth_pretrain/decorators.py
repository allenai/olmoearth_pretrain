"""Decorators for marking experimental, deprecated, or internal features."""

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar, cast

__all__ = ["experimental", "deprecated", "internal"]

F = TypeVar("F", bound=Callable[..., Any])
C = TypeVar("C", bound=type)


def experimental(reason: str = "") -> Callable[[F], F]:
    """Mark a function or class as experimental.

    Experimental features may not be fully tested, may change without notice,
    and may not be maintained in future versions.

    Args:
        reason: Optional explanation of why this is experimental or what limitations exist.

    Example:
        >>> @experimental("This feature is still under development")
        >>> def my_function():
        ...     pass

        >>> @experimental()
        >>> class MyClass:
        ...     pass
    """

    def decorator(obj: F) -> F:
        # Add marker attribute
        setattr(obj, "__experimental__", True)

        # Build warning message
        obj_name = getattr(obj, "__qualname__", getattr(obj, "__name__", str(obj)))
        msg = f"'{obj_name}' is experimental and may change or be removed in future versions."
        if reason:
            msg += f" {reason}"

        # Handle classes differently from functions
        if isinstance(obj, type):
            # For classes, wrap __init__
            original_init = obj.__init__  # type: ignore[misc]

            @functools.wraps(original_init)
            def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
                warnings.warn(msg, FutureWarning, stacklevel=2)
                return original_init(self, *args, **kwargs)

            obj.__init__ = wrapped_init  # type: ignore[method-assign, misc]

            # Update docstring
            if obj.__doc__:
                obj.__doc__ = f"**EXPERIMENTAL**: {msg}\n\n{obj.__doc__}"
            else:
                obj.__doc__ = f"**EXPERIMENTAL**: {msg}"
            return cast(F, obj)
        else:
            # For functions, wrap the function itself
            @functools.wraps(obj)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                warnings.warn(msg, FutureWarning, stacklevel=2)
                return obj(*args, **kwargs)

            # Update docstring
            if obj.__doc__:
                wrapper.__doc__ = f"**EXPERIMENTAL**: {msg}\n\n{obj.__doc__}"
            else:
                wrapper.__doc__ = f"**EXPERIMENTAL**: {msg}"

            setattr(wrapper, "__experimental__", True)
            return cast(F, wrapper)

    return decorator


def deprecated(reason: str = "", alternative: str = "") -> Callable[[F], F]:
    """Mark a function or class as deprecated.

    Args:
        reason: Explanation of why this is deprecated.
        alternative: Suggested alternative to use instead.

    Example:
        >>> @deprecated(reason="Use new_function instead", alternative="new_function")
        >>> def old_function():
        ...     pass
    """

    def decorator(obj: F) -> F:
        setattr(obj, "__deprecated__", True)

        obj_name = getattr(obj, "__qualname__", getattr(obj, "__name__", str(obj)))
        msg = f"'{obj_name}' is deprecated and will be removed in a future version."
        if reason:
            msg += f" {reason}"
        if alternative:
            msg += f" Use '{alternative}' instead."

        if isinstance(obj, type):
            original_init = obj.__init__  # type: ignore[misc]

            @functools.wraps(original_init)
            def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
                return original_init(self, *args, **kwargs)

            obj.__init__ = wrapped_init  # type: ignore[method-assign, misc]

            if obj.__doc__:
                obj.__doc__ = f"**DEPRECATED**: {msg}\n\n{obj.__doc__}"
            else:
                obj.__doc__ = f"**DEPRECATED**: {msg}"
            return cast(F, obj)
        else:

            @functools.wraps(obj)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
                return obj(*args, **kwargs)

            if obj.__doc__:
                wrapper.__doc__ = f"**DEPRECATED**: {msg}\n\n{obj.__doc__}"
            else:
                wrapper.__doc__ = f"**DEPRECATED**: {msg}"

            setattr(wrapper, "__deprecated__", True)
            return cast(F, wrapper)

    return decorator


def internal(obj: F) -> F:
    """Mark a function or class as internal/private API.

    Internal APIs are not part of the public API and may change without notice.

    Example:
        >>> @internal
        >>> def _internal_helper():
        ...     pass
    """
    setattr(obj, "__internal__", True)

    obj_name = getattr(obj, "__qualname__", getattr(obj, "__name__", str(obj)))
    msg = f"'{obj_name}' is an internal API and may change without notice."

    if isinstance(obj, type):
        if obj.__doc__:
            obj.__doc__ = f"**INTERNAL API**: {msg}\n\n{obj.__doc__}"
        else:
            obj.__doc__ = f"**INTERNAL API**: {msg}"
    else:
        if obj.__doc__:
            obj.__doc__ = f"**INTERNAL API**: {msg}\n\n{obj.__doc__}"
        else:
            obj.__doc__ = f"**INTERNAL API**: {msg}"

    return obj
