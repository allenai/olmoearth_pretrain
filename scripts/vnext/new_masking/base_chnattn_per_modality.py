"""Base model with per-modality channel attention aggregation.

Each modality's bandset tokens are fused into a single token per (h,w,t)
via cross-attention, yielding N_modality tokens per spatial-temporal position.
The AggregatedPredictor decoder reconstructs per-modality bandset structure.

Change from baseline:
- Encoder: ChnAttn aggregates each modality's bandsets into 1 token (per_modality mode)
- Decoder: AggregatedPredictor replaces standard Predictor
- Target encoder: per-modality (skips ChnAttn)
"""

import logging

from new_masking_script import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.channel_attention import (
    AggregatedPredictorConfig,
    ChnAttnConfig,
)
from olmoearth_pretrain.nn.flexihelios import EncoderConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config with per-modality channel attention."""
    model_size = MODEL_SIZE_ARGS["base_shallow_decoder"]

    encoder_config = EncoderConfig(
        embedding_size=model_size["encoder_embedding_size"],
        num_heads=model_size["encoder_num_heads"],
        depth=model_size["encoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        supported_modality_names=common.training_modalities,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=12,
        chnattn_config=ChnAttnConfig(
            enabled=True,
            aggregation_mode="per_modality",
            num_heads=8,
        ),
    )
    decoder_config = AggregatedPredictorConfig(
        encoder_embedding_size=model_size["encoder_embedding_size"],
        decoder_embedding_size=model_size["decoder_embedding_size"],
        depth=model_size["decoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        num_heads=model_size["decoder_num_heads"],
        supported_modality_names=common.training_modalities,
        max_sequence_length=12,
    )
    model_config = LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )
    return model_config


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )
