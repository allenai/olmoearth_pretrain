"""Perceiver with a baseline-style cross-attention decoder (xdec).

Same as ``perceiver_base.py`` except the training decoder is
``PerceiverCrossPredictor``: the baseline ``Predictor``'s mask-token queries
and 4 cross-attention blocks, attending to the encoder's fused dense map at
ALL grid positions (instead of a pointwise MLP head on the broadcast dense
map). Restores decode-time gathering and modality-specific querying while
keeping the latent-bottleneck encoder.
"""

import logging

from perceiver_base import (
    HEAD_DEPTH as _UNUSED_HEAD_DEPTH,  # noqa: F401  (documents the delta)
)
from perceiver_base import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)
from perceiver_base import (
    build_model_config as build_model_config_base,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.perceiver import PerceiverCrossPredictorConfig

logger = logging.getLogger(__name__)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Perceiver base encoder + cross-attention decoder over the dense map."""
    config = build_model_config_base(common)
    base_dec = config.decoder_config
    config.decoder_config = PerceiverCrossPredictorConfig(
        supported_modality_names=base_dec.supported_modality_names,
        encoder_embedding_size=base_dec.encoder_embedding_size,
        decoder_embedding_size=base_dec.decoder_embedding_size,
        depth=4,
        mlp_ratio=base_dec.mlp_ratio,
        num_heads=base_dec.num_heads,
        max_sequence_length=base_dec.max_sequence_length,
        tokenization_config=base_dec.tokenization_config,
        position_encoding=base_dec.position_encoding,
        rope_mixed_base=base_dec.rope_mixed_base,
        rope_temporal_coordinate_scale=base_dec.rope_temporal_coordinate_scale,
    )
    return config


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
