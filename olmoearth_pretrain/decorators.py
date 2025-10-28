"""Decorators for marking experimental, deprecated, or internal features."""

import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])
C = TypeVar("C", bound=type)


def experimental(reason: str = "This is an experimental feature") -> Callable[[F], F]:
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
