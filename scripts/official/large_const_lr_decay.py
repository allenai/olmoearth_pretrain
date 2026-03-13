"""Continue the large const-LR recipe from a checkpoint with cosine decay.

This matches the historical `large_const_lr_with_maps` model recipe closely so a
checkpoint from that run can be loaded safely, while enabling the current speed
knobs plus the vectorized latent-MIM loss.

- loads model weights from `trainer.load_path`
- loads optimizer state
- loads trainer/global-step state
- resumes from step 520k and runs to step 820k by default
- starts cosine decay at step 520k and finishes it at step 820k
- enables fused AdamW
- enables torch.compile
- uses the vectorized patch discrimination loss

Example:
    python scripts/official/large_const_lr_decay.py launch large_const_lr_decay_300k ai2/jupiter \
        --launch.num_gpus=8 \
        --launch.clusters=[ai2/jupiter,ai2/titan] \
        --trainer.callbacks.wandb.project=2025_01_28_large_undertraining \
        --trainer.load_path=/weka/dfive-default/olmoearth_pretrain/checkpoints/henryh/large_const_lr_with_maps/step520000
"""

import logging

from olmo_core.config import DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import CosWithWarmup
from olmo_core.train.common import Duration, LoadStrategy
from script import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_visualize_config,
)
from script import (
    build_trainer_config as _build_trainer_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexihelios import (
    EncoderConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
RESUME_STEP = 520_000
EXTRA_STEPS = 300_000
FINAL_STEP = RESUME_STEP + EXTRA_STEPS


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the large const-LR-compatible model config."""
    model_size = MODEL_SIZE_ARGS["large_shallow_decoder"]

    encoder_config = EncoderConfig(
        embedding_size=model_size["encoder_embedding_size"],
        num_heads=model_size["encoder_num_heads"],
        depth=model_size["encoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        supported_modality_names=common.training_modalities,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=12,
        use_flash_attn=True,
        use_linear_patch_embed=True,
    )
    decoder_config = PredictorConfig(
        encoder_embedding_size=model_size["encoder_embedding_size"],
        decoder_embedding_size=model_size["decoder_embedding_size"],
        depth=model_size["decoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        num_heads=model_size["decoder_num_heads"],
        supported_modality_names=common.training_modalities,
        max_sequence_length=12,
        use_flash_attn=True,
    )
    return LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build the train module config for the resumed cosine-decay phase."""
    return ContrastiveLatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02, fused=True),
        rank_microbatch_size=32,
        masking_config=MaskingConfig(
            strategy_config={
                "type": "modality_cross_random",
                "encode_ratio": 0.5,
                "decode_ratio": 0.5,
                "allow_encoding_decoding_same_bandset": True,
                "only_decode_modalities": [
                    "worldcover",
                    "srtm",
                    "openstreetmap_raster",
                    "wri_canopy_height_map",
                    "cdl",
                    "worldcereal",
                ],
            }
        ),
        loss_config=LossConfig(
            loss_config={
                "type": "modality_patch_discrimination_vec",
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
        # When trainer state is restored at step 520k, this makes the resumed run
        # start at the top of the cosine and decay over the next 300k steps.
        scheduler=CosWithWarmup(warmup=RESUME_STEP, t_max=FINAL_STEP),
        ema_decay=(1.0, 1.0),
        compile_model=True,
        dp_config=DataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )


def build_trainer_config(common: CommonComponents):
    """Build trainer config for a resumed 520k->820k continuation phase."""
    cfg = _build_trainer_config(common)
    cfg.max_duration = Duration.steps(FINAL_STEP)
    cfg.load_strategy = LoadStrategy.always
    cfg.load_optim_state = True
    cfg.load_trainer_state = True
    if "wandb" in cfg.callbacks:
        cfg.callbacks["wandb"].project = "2025_01_28_large_undertraining"
    return cfg


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
