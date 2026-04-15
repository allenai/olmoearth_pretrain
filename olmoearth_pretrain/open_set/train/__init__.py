"""olmo-core training module for open-set segmentation.

The launchable entrypoint lives at ``scripts/open_set/script.py`` and is
invoked via ``scripts/open_set/launch_open_set.sh``.
"""

from olmoearth_pretrain.open_set.train.train_module import (
    OpenSetTrainModule,
    OpenSetTrainModuleConfig,
)

__all__ = [
    "OpenSetTrainModule",
    "OpenSetTrainModuleConfig",
]
