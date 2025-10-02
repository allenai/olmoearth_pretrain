"""Backward compatibility package for helios -> olmo_earth rename.

This package provides backward compatibility for old checkpoints that reference
"helios" module paths in their configurations. All helios imports will be
redirected to olmo_earth with a deprecation warning.

DEPRECATED: This package is deprecated and will be removed in a future version.
Please use olmo_earth instead of helios.
"""

import sys
import warnings
from importlib import import_module

# Create a deprecation warning
_DEPRECATION_MESSAGE = (
    "The 'helios' package has been renamed to 'olmo_earth'. "
    "Please update your imports from 'helios' to 'olmo_earth'. "
    "The 'helios' compatibility shim will be removed in a future version."
)

_warned = False


def _warn_once():
    """Issue deprecation warning only once."""
    global _warned
    if not _warned:
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=4)
        _warned = True


def __getattr__(name):
    """Redirect attribute access to olmo_earth with warning."""
    _warn_once()
    try:
        # Import the corresponding olmo_earth module
        olmo_earth_module = import_module(f"olmo_earth.{name}")
        return olmo_earth_module
    except ImportError:
        # If it's not a module, try to get it as an attribute from olmo_earth
        import olmo_earth

        return getattr(olmo_earth, name)


def __dir__():
    """Return directory of olmo_earth."""
    import olmo_earth

    return dir(olmo_earth)


# Also provide __version__ for compatibility
_warn_once()
import olmo_earth

__version__ = getattr(olmo_earth, "__version__", "0.0.1")
