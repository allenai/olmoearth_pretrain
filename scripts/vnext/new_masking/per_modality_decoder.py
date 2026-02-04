"""Experiment with per-modality decoders for decode-only modalities.

This experiment uses:
1. DecodeOnlyModalitiesMaskingStrategy: Only decode_only modalities get decoded,
   all other modalities are encode-only.
2. PerModalityPredictor: Each decode-only modality has its own dedicated decoder.
"""

import logging
import sys
from pathlib import Path

# Add official directory to path for script imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "official"))

from olmo_core.config import DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import CosWithWarmup
from script import (
    build_common_components,
    build_dataset_config,
    build_trainer_config,
    build_visualize_config,
)

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, PerModalityPredictorConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1

# Modalities that will be decoded (each gets its own predictor)
DECODE_ONLY_MODALITIES = [
    Modality.WORLDCOVER.name,
    Modality.SRTM.name,
    Modality.OPENSTREETMAP_RASTER.name,
    Modality.WRI_CANOPY_HEIGHT_MAP.name,
    Modality.CDL.name,
    Modality.WORLDCEREAL.name,
]


def get_masking_config(common: CommonComponents) -> MaskingConfig:
    """Get the masking configuration for the experiment.

    Uses RandomWithDecodeMaskingStrategy with allow_decoding_other_modalities=False:
    - Decodes only the decode_only modalities
    - All other modalities are encode-only (no DECODER tokens)
    """
    return MaskingConfig(
        strategy_config={
            "type": "random_with_decode",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,  # Only applies to only_decode_modalities
            "only_decode_modalities": DECODE_ONLY_MODALITIES,
            "allow_decoding_other_modalities": False,
        },
        tokenization_config=common.tokenization_config,
    )


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for an experiment.

    Uses PerModalityPredictorConfig which creates a separate predictor
    for each decode-only modality.
    """
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
    )

    # Per-modality predictor: one decoder per decode-only modality
    decoder_config = PerModalityPredictorConfig(
        decode_modality_names=DECODE_ONLY_MODALITIES,
        supported_modality_names=common.training_modalities,
        encoder_embedding_size=model_size["encoder_embedding_size"],
        decoder_embedding_size=model_size["decoder_embedding_size"],
        depth=model_size["decoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        num_heads=model_size["decoder_num_heads"],
        max_sequence_length=12,
    )

    model_config = LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )
    return model_config


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build the train module config for an experiment."""
    return ContrastiveLatentMIMTrainModuleConfig(
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
        ema_decay=(1.0, 1.0),
        dp_config=DataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )


def build_dataloader_config(
    common: CommonComponents,
) -> OlmoEarthDataLoaderConfig:
    """Build the dataloader config for an experiment."""
    return OlmoEarthDataLoaderConfig(
        num_workers=12,
        global_batch_size=512,
        token_budget=2250,
        prefetch_factor=2,
        sampled_hw_p_list=list(range(1, 13)),
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        work_dir=common.save_folder,
        seed=3622,
        num_masked_views=2,  # ContrastiveLatentMIM needs 2 views
        masking_config=get_masking_config(common),
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
