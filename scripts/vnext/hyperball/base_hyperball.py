"""Base model with the Hyperball optimizer wrapping Adam.

Hypothesis (per Wen et al., 2025): Replacing weight decay with explicit
Frobenius-norm projection on weight matrices speeds up pretraining and makes
the optimal learning rate transfer across model scales. We expect ~20-30%
speedup over the AdamW baseline at matched compute on Helios Base.

Change from baseline:
- Optimizer: AdamWConfig(lr=1e-4, weight_decay=0.02) -> HyperballConfig(lr=5e-3, weight_decay=0.0)
- Hyperball is applied to all 2D+ weight matrices that aren't embeddings.
- Embeddings, norms, biases keep AdamW behavior with weight_decay=0.02.

Reference: "Fantastic Pretraining Optimizers and Where to Find Them 2.1:
Hyperball Optimization" - https://tinyurl.com/muonh
"""

import logging
import sys
from pathlib import Path

# Add official directory to path for script imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "official"))

from olmo_core.config import DType  # noqa: E402
from olmo_core.distributed.parallel.data_parallel import (  # noqa: E402
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.optim import OptimGroupOverride  # noqa: E402
from olmo_core.optim.scheduler import CosWithWarmup  # noqa: E402
from script import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_trainer_config,
    build_visualize_config,
)
from script import (  # noqa: E402
    build_train_module_config as build_train_module_config_base,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS  # noqa: E402
from olmoearth_pretrain.nn.flexihelios import (  # noqa: E402
    EncoderConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from olmoearth_pretrain.train.optim import HyperballConfig  # noqa: E402
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (  # noqa: E402
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1

# Hyperball recommended LR range is 2.5e-3 to 1e-2 (paper, section 2). Start
# at the geometric mean and tune from there.
HYPERBALL_LR = 5e-3
ADAMW_FALLBACK_LR = 5e-3
ADAMW_FALLBACK_WD = 0.02


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build the train module config with the Hyperball optimizer.

    Hyperball is applied to non-embedding 2D weight matrices. Embeddings, the
    LM head, biases, and norms fall back to AdamW with weight decay (per the
    paper's empirical tips). 1D parameters fall back automatically inside the
    optimizer regardless of group setting.
    """
    # Inherit everything else from the baseline so this run only varies the
    # optimizer.
    base_cfg = build_train_module_config_base(common)

    optim_config = HyperballConfig(
        lr=HYPERBALL_LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,  # ignored when apply_hyperball=True
        apply_hyperball=True,
        # Carve out all embedding-like params into AdamW with weight decay
        # (per Hyperball paper, section 2). In Helios, `*embed*` covers
        # patch_embeddings.*.proj.{weight,bias}, pos_embed, month_embed.*,
        # per_modality_channel_embeddings.*, encoder_to_decoder_embed, and
        # to_output_embed. Verified against base_shallow_decoder param names
        # on 2026-05-01.
        group_overrides=[
            OptimGroupOverride(
                params=["*embed*"],
                opts={
                    "apply_hyperball": False,
                    "weight_decay": ADAMW_FALLBACK_WD,
                    "lr": ADAMW_FALLBACK_LR,
                },
            ),
        ],
    )

    return ContrastiveLatentMIMTrainModuleConfig(
        optim_config=optim_config,
        rank_microbatch_size=base_cfg.rank_microbatch_size,
        masking_config=base_cfg.masking_config,
        loss_config=base_cfg.loss_config,
        contrastive_config=base_cfg.contrastive_config,
        token_exit_cfg=base_cfg.token_exit_cfg,
        max_grad_norm=base_cfg.max_grad_norm,
        scheduler=CosWithWarmup(warmup_steps=8000),
        ema_decay=base_cfg.ema_decay,
        dp_config=DataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Helios Base model -- unchanged from the official base.py."""
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
        use_linear_patch_embed=False,
    )
    decoder_config = PredictorConfig(
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
