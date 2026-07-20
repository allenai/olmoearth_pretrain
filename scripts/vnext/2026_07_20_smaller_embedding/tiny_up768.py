"""Approach 2: native tiny (192) encoder, loss target projected up to 768.

The ViT-tiny transformer runs at its native 192-d width and emits a 192-d
deliverable embedding. The decoder predicts at 768, and the frozen exit-0 target
is the 192-d patch-embedding expanded to 768 by a frozen MLP, so the
patch-discrimination loss runs at 768. Compare against base_dim192 (Approach 1).
"""

import logging

from base import (
    PATCH_EMBED_HIDDEN_SIZES,
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_smaller_embedding_model_config,
    build_visualize_config,
)
from base_faster import (
    build_train_module_config as _build_train_module_config,
)
from base_faster import make_build_trainer_config
from olmo_core.optim import AdamWConfig

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/vnext/2026_07_20_smaller_embedding/tiny_up768.py"
LEARNING_RATE = 0.0002
WEIGHT_DECAY = 0.002


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Faster train module with the tiny-size optimizer settings."""
    config = _build_train_module_config(common)
    config.optim_config = AdamWConfig(
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fused=False,
    )
    return config


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Native tiny (192) encoder, 768-d loss/target via frozen expander."""
    return build_smaller_embedding_model_config(
        common,
        "tiny_shallow_decoder",
        PATCH_EMBED_HIDDEN_SIZES,
        output_embedding_size=None,
    )


def run() -> None:
    """Run the experiment."""
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=make_build_trainer_config(MODULE_PATH),
        visualize_config_builder=build_visualize_config,
    )


if __name__ == "__main__":
    run()
