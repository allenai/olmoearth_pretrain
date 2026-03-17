"""NOBLE experiment: Base training with nonlinear low-rank branches.

NOBLE (Nonlinear lOw-rank Branch for Linear Enhancement) augments transformer
linear layers with nonlinear low-rank branches that can accelerate training by
up to 1.47x according to the paper (arXiv:2603.06492).

This experiment applies NOBLE to the base OlmoEarth training setup. Key points:
- NOBLE branches are applied to Q, K, V projections, output projection, and MLP layers
- Uses CosNet activation (two-layer cosine nonlinearity with learnable frequency/phase)
- Adds ~4% parameters with ~7% step time overhead
- Can achieve up to 1.22x net wallclock speedup

Note: The paper found that Mixup/CutMix and stochastic augmentations may interfere
with NOBLE's benefits. Consider adjusting augmentation if results differ from expected.
"""

import logging

from script import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexi_vit import (
    EncoderConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.noble import NobleConfig

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config with NOBLE branches enabled."""
    model_size = MODEL_SIZE_ARGS["base_shallow_decoder"]

    # NOBLE configuration - applied to encoder only by default
    # Decoder is lightweight, so NOBLE overhead might not be worth it there
    noble_config = NobleConfig(
        enabled=True,
        rank_ratio=0.25,  # 25% of layer dimension
        activation="cosnet",  # CosNet performs best per the paper
        init_scale=0.01,  # Small init for near-identity at start
        apply_to_qkv=True,
        apply_to_proj=True,
        apply_to_mlp=True,
    )

    encoder_config = EncoderConfig(
        embedding_size=model_size["encoder_embedding_size"],
        num_heads=model_size["encoder_num_heads"],
        depth=model_size["encoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        supported_modality_names=common.training_modalities,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=12,
        use_linear_patch_embed=False,
        noble_config=noble_config,
    )

    # Decoder without NOBLE (it's shallow and lightweight)
    decoder_config = PredictorConfig(
        encoder_embedding_size=model_size["encoder_embedding_size"],
        decoder_embedding_size=model_size["decoder_embedding_size"],
        depth=model_size["decoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        num_heads=model_size["decoder_num_heads"],
        supported_modality_names=common.training_modalities,
        max_sequence_length=12,
        # noble_config=None  # Disabled for decoder by default
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
