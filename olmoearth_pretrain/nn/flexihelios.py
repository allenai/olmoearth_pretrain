"""Deprecated module. Please import from olmoearth_pretrain.nn.flexivit instead.

Maintained for backwards compatibility with old checkpoints.
"""

import sys
import warnings

import olmoearth_pretrain.nn.flexivit as flexivit

from .flexivit import *  # noqa: F403

warnings.warn(
    "olmoearth_pretrain.nn.flexihelios is deprecated. "
    "Please import from olmoearth_pretrain.nn.flexivit instead.",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = flexivit
