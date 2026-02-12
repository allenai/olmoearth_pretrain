"""LatentMIMLITE experiment: base encoder with a smaller independent target encoder."""

import logging

from script import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_trainer_config,
    build_visualize_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexihelios import (
    EncoderConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim_lite import LatentMIMLITEConfig
from olmoearth_pretrain.train.train_module.latent_mim_lite_contrastive import (
    LatentMIMLITEContrastiveTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1


def build_model_config(common: CommonComponents) -> LatentMIMLITEConfig:
    """Build the model config for a LatentMIMLITE experiment.

    Online encoder uses base_shallow_decoder size.
    Target encoder uses a smaller architecture (nano) to demonstrate
    independent configuration.
    """
    model_size = MODEL_SIZE_ARGS["base_shallow_decoder"]

    # -- Online encoder (same as base.py) --
    encoder_config = EncoderConfig(
        embedding_size=model_size["encoder_embedding_size"],
        num_heads=model_size["encoder_num_heads"],
        depth=model_size["encoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        supported_modality_names=common.training_modalities,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=12,
    )

    # -- Target encoder (independently configured, e.g. smaller) --
    target_size = MODEL_SIZE_ARGS["nano"]
    target_encoder_config = EncoderConfig(
        embedding_size=target_size["encoder_embedding_size"],
        num_heads=target_size["encoder_num_heads"],
        depth=target_size["encoder_depth"],
        mlp_ratio=target_size["mlp_ratio"],
        supported_modality_names=common.training_modalities,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.0,  # no drop path for frozen target
        max_sequence_length=12,
    )

    # -- Decoder/predictor: output_embedding_size must match target encoder --
    decoder_config = PredictorConfig(
        encoder_embedding_size=model_size["encoder_embedding_size"],
        decoder_embedding_size=model_size["decoder_embedding_size"],
        depth=model_size["decoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        num_heads=model_size["decoder_num_heads"],
        supported_modality_names=common.training_modalities,
        max_sequence_length=12,
        output_embedding_size=target_size["encoder_embedding_size"],
    )

    model_config = LatentMIMLITEConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        target_encoder_config=target_encoder_config,
    )
    return model_config


def build_train_module_config(
    common: CommonComponents,
) -> LatentMIMLITEContrastiveTrainModuleConfig:
    """Build the train module config for LatentMIMLITE."""
    from olmo_core.config import DType
    from olmo_core.distributed.parallel.data_parallel import (
        DataParallelConfig,
        DataParallelType,
    )
    from olmo_core.optim import AdamWConfig
    from olmo_core.optim.scheduler import CosWithWarmup

    from olmoearth_pretrain.train.loss import LossConfig
    from olmoearth_pretrain.train.masking import MaskingConfig

    from script import get_masking_config

    return LatentMIMLITEContrastiveTrainModuleConfig(
        optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02, fused=False),
        rank_microbatch_size=32,
        masking_config=get_masking_config(common),
        loss_config=LossConfig(
            loss_config={
                "type": "modality_patch_discrimination_new",
                "tau": 0.1,
            }
        ),
        contrastive_config=LossConfig(
            loss_config={
                "type": "InfoNCE",
                "weight": 0.1,
            }
        ),
        token_exit_cfg={modality: 0 for modality in common.training_modalities},
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=8000),
        dp_config=DataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )


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
