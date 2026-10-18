"""TOWERS t4/f8 + whole-modality dropout (p=0.3 per input modality per sample).

``towers_t4_f8.py`` with the ``control_moddrop.py`` masking: the structured information
asymmetry (predict a dropped modality from the others) that a multi-encoder relies on.

W&B project ``20261018_multi_encoder``.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import (  # noqa: E402
    build_common_components,
    build_dataset_config,
    build_multi_encoder_model_config,  # noqa: E402
    build_trainer_config,
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

MODULE_PATH = "scripts/vnext/20261018_multi_encoder/towers_t4_f8_moddrop.py"
MODALITY_DROPOUT = True

TOWER_DEPTH = 4
FUSION_DEPTH = 8


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """v1.3 with per-modality towers (4) + shared blocks (8)."""
    return build_multi_encoder_model_config(
        common, tower_depth=TOWER_DEPTH, fusion_depth=FUSION_DEPTH
    )


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
