"""Deprecated module. Please import from olmoearth_pretrain.nn.flexivit instead.

Maintained for backwards compatibility with old checkpoints.
"""

import sys
import warnings

import olmoearth_pretrain.nn.flexi_vit as flexivit

from .flexi_vit import *  # noqa: F403

warnings.warn(
    "olmoearth_pretrain.nn.flexi_vit is deprecated. "
    "Please import from olmoearth_pretrain.nn.flexivit instead.",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = flexivit
