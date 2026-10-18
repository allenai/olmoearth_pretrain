"""CONTROL + whole-modality dropout (p=0.3 per input modality per sample).

Same shared encoder as ``control.py``; only the masking changes: on top of the v1.2
``random_time_with_decode`` masking, each of S2 / S1 / Landsat has all its visible tokens
turned into decoder targets with probability 0.3 (never all three at once). Pairs with
``towers_t4_f8_moddrop.py`` to separate the masking effect from the architecture effect.

W&B project ``20261018_multi_encoder``.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import (  # noqa: E402
    build_common_components,
    build_dataset_config,
    build_trainer_config,
    build_v1_3_model_config,  # noqa: E402
    build_visualize_config,
)
from base import build_dataloader_config as _build_dataloader_config  # noqa: E402
from base import build_train_module_config as _build_train_module_config  # noqa: E402

from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from olmoearth_pretrain.train.train_module.latent_mim import (  # noqa: E402
    LatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/vnext/20261018_multi_encoder/control_moddrop.py"
MODALITY_DROPOUT = True


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """v1.3: one shared 12-block encoder + the AttentionPooler bottleneck."""
    return build_v1_3_model_config(common)


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """The shared dataloader, with this arm's masking."""
    return _build_dataloader_config(common, modality_dropout=MODALITY_DROPOUT)


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """The shared train module, with this arm's masking."""
    return _build_train_module_config(common, modality_dropout=MODALITY_DROPOUT)


def run() -> None:
    """Run the experiment."""
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )


if __name__ == "__main__":
    run()
