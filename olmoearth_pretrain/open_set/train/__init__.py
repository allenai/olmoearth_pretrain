"""olmo-core training module + entrypoint for open-set segmentation."""

from olmoearth_pretrain.open_set.train.train_module import (
    OpenSetTrainModule,
    OpenSetTrainModuleConfig,
)

__all__ = [
    "OpenSetTrainModule",
    "OpenSetTrainModuleConfig",
]
