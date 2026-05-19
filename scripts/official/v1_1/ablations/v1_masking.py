"""v1 masking ablation: use the original modality_cross_random masking from scripts/official."""

import logging

from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.tokenization import TokenizationConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

from ..base import (
    ONLY_DECODE_MODALITIES,
    build_common_components,
    build_dataset_config,
    build_model_config,
    build_trainer_config,
    build_visualize_config,
)
from ..base import (
    build_dataloader_config as build_dataloader_config_base,
)
from ..base import (
    build_train_module_config as build_train_module_config_base,
)

logger = logging.getLogger(__name__)


def _orig_masking_config(
    tokenization_config: TokenizationConfig | None = None,
) -> MaskingConfig:
    return MaskingConfig(
        strategy_config={
            "type": "modality_cross_random",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "allow_encoding_decoding_same_bandset": True,
            "only_decode_modalities": ONLY_DECODE_MODALITIES,
        },
        tokenization_config=tokenization_config,
    )


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build the train module config for an experiment."""
    config = build_train_module_config_base(common)
    config.masking_config = _orig_masking_config(common.tokenization_config)
    return config


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Build the dataloader config for an experiment."""
    config = build_dataloader_config_base(common)
    config.masking_config = _orig_masking_config(common.tokenization_config)
    return config


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
