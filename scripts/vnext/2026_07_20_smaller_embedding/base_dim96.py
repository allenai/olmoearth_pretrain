"""Approach 1: base (768) encoder bottlenecked to a 96-d deliverable embedding.

The full ViT-base transformer runs at 768 and emits a very compact 96-d
embedding for downstream evals, while the decoder/target and patch-discrimination
loss operate at 768 (raw patch-embedding).
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
from base_faster import build_train_module_config, make_build_trainer_config

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/vnext/2026_07_20_smaller_embedding/base_dim96.py"
DELIVERABLE_EMBEDDING_SIZE = 96


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Base encoder, 96-d deliverable embedding, 768-d loss/target."""
    return build_smaller_embedding_model_config(
        common,
        "base_shallow_decoder",
        PATCH_EMBED_HIDDEN_SIZES,
        output_embedding_size=DELIVERABLE_EMBEDDING_SIZE,
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
