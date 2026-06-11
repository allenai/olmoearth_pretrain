"""ViT-base v1.1 combined branch: CLIP token loss + RoPE + simple encodings.

Combines, on top of the v1.1 base recipe:

- **CLIP token loss** (port of PR #376): symmetric CE between decoder
  predictions and the static randomly-initialized target encoder's patch
  embeddings, with a learned temperature (``LatentMIM.logit_scale``, clamped
  at 4.6 in log space, excluded from weight decay). Same-target negative
  masking (cosine > 0.999) is kept for the map modalities. Negatives stay
  microbatch-local — no cross-GPU gathering.
- **CLIP instance loss** on the projected class token: symmetric InfoNCE
  with its own learned temperature (``instance_logit_scale`` — separate
  parameter, since within-sample token negatives and cross-sample instance
  negatives have very different similarity distributions).
- **Spatial RoPE** (axial, base 10000, coordinate scale 0.25 — the winning
  sweep setting) with ``encoding_mode='separate'``.
- **Simple temporal encoding** (4 numbers): ``[frac_year/10, sin, cos,
  year_valid]`` with epoch 2020. ``year == 0`` is the "year unknown"
  sentinel; batch-level year dropout (collator) zeroes the year for a whole
  rank batch at a time, and eval tasks with fabricated dates present year 0.
- **Latlon token**: latlon joins the training modalities and is embedded via
  unit-sphere xyz (degrees in). Masking strategies are overridden by the
  collator (pin_latlon_to_encoder), so the token is always encoder-visible
  when present and absent for whole batches under latlon dropout — matching
  eval tasks, none of which provide latlon. The per-token latlon slot from
  rope_separate is disabled (``latlon_encoding_dim=0``): location enters
  through the token only, so dropping the token actually removes it.
- **Class token**: a single learnable token per sample is the instance
  embedding. The instance InfoNCE applies to its projection; evals consume
  the raw token. It is decoder-visible as cross-attention context, so it
  also receives gradient from the token loss.
- **Channel ("modality") encoding**: decoder-only. Each (modality, bandset)
  already has its own patch-embed module with a bias, so the encoder-side
  channel constant is absorbable and adds nothing; decoder queries are a
  single shared mask token and REQUIRE it to know what to predict. Set
  ``ENCODER_CHANNEL_ENCODING_DIM = 128`` to ablate the symmetric variant.

Metadata dropout is drawn once per rank batch ('shared' across both
contrastive views — beware: with latlon kept in both views, the instance
loss can be solved by latlon matching; 'decorrelated' kills that shortcut
at the cost of metadata-invariance pressure. See MetadataDropout).
"""

import logging

from base import (
    ONLY_DECODE_MODALITIES,
    _masking_config,
    build_dataset_config,
    build_trainer_config,
    build_visualize_config,
)
from base import (
    build_common_components as build_common_components_base,
)
from base import (
    build_dataloader_config as build_dataloader_config_base,
)
from base import (
    build_model_config as build_model_config_base,
)
from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelConfig, DataParallelType
from olmo_core.optim import AdamWConfig, OptimGroupOverride
from olmo_core.optim.scheduler import CosWithWarmup

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.internal.experiment import CommonComponents, SubCmd, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

# --- Spatial RoPE (matches rope_base10k_scale0.25 W&B run) ---
SPATIAL_POS_ENCODING = "rope"
ROPE_BASE = 10000.0
ROPE_COORDINATE_SCALE = 0.25

# --- Separate-path encodings ---
ENCODING_MODE = "separate"
ENCODER_CHANNEL_ENCODING_DIM = 0  # redundant given per-bandset embed biases
DECODER_CHANNEL_ENCODING_DIM = 128  # required: queries share one mask token
TEMPORAL_ENCODING_DIM = 4  # [frac_year/10, sin, cos, year_valid]
TEMPORAL_ENCODING_TYPE = "simple"
LATLON_ENCODING_DIM = 0  # location enters via the latlon TOKEN only

# --- Batch-level metadata dropout (collator; one draw per rank batch) ---
YEAR_DROPOUT_RATE = 0.5
LATLON_DROPOUT_RATE = 0.5
METADATA_DROPOUT_VIEW_MODE = "shared"  # see module docstring re 'decorrelated'

MAX_LOGIT_SCALE = 4.6  # exp(4.6) ~ 99.5, CLIP's clamp


def _apply_shared(cfg) -> None:
    cfg.spatial_pos_encoding = SPATIAL_POS_ENCODING
    cfg.rope_base = ROPE_BASE
    cfg.rope_coordinate_scale = ROPE_COORDINATE_SCALE
    cfg.encoding_mode = ENCODING_MODE
    cfg.temporal_encoding_dim = TEMPORAL_ENCODING_DIM
    cfg.temporal_encoding_type = TEMPORAL_ENCODING_TYPE
    cfg.latlon_encoding_dim = LATLON_ENCODING_DIM
    cfg.latlon_dropout_rate = 0.0  # dropout moved to the collator


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """v1.1 base modalities plus the latlon token."""
    config = build_common_components_base(script, cmd, run_name, cluster, overrides)
    config.training_modalities = config.training_modalities + [Modality.LATLON.name]
    return config


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """v1.1 base model + RoPE + simple encodings + class/latlon tokens."""
    config = build_model_config_base(common)
    _apply_shared(config.encoder_config)
    _apply_shared(config.decoder_config)
    config.encoder_config.channel_encoding_dim = ENCODER_CHANNEL_ENCODING_DIM
    config.decoder_config.channel_encoding_dim = DECODER_CHANNEL_ENCODING_DIM
    config.encoder_config.use_class_token = True
    return config


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """CLIP token loss + InfoNCE on the projected class token."""
    return ContrastiveLatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(
            lr=0.0001,
            weight_decay=0.02,
            betas=(0.9, 0.95),
            fused=False,
            # CLIP convention: weight decay must not pull the learned
            # log-temperatures toward 0 (temperature toward 1).
            group_overrides=[
                OptimGroupOverride(
                    params=["logit_scale", "instance_logit_scale"],
                    opts=dict(weight_decay=0.0),
                )
            ],
        ),
        rank_microbatch_size=64,
        masking_config=_masking_config(common.tokenization_config),
        loss_config=LossConfig(
            loss_config={
                "type": "clip_patch_discrimination",
                # L2-normalized scores (defaults), learned temperature via
                # logit_scale threaded from the train module.
                "same_target_threshold": 0.999,
                "mask_negatives_for_modalities": ONLY_DECODE_MODALITIES,
            }
        ),
        # CLIP-style instance loss on the projected class token: symmetric,
        # learnable temperature (instance_logit_scale, separate from the token
        # loss's). Weight 0.05 was tuned for the fixed-tau mean-pool setup;
        # candidate for retuning.
        contrastive_config=LossConfig(
            loss_config={
                "type": "clip_infonce",
                "weight": 0.05,
            }
        ),
        token_exit_cfg={modality: 0 for modality in common.training_modalities},
        max_grad_norm=1.0,
        max_logit_scale=MAX_LOGIT_SCALE,
        scheduler=CosWithWarmup(warmup_steps=8000),
        # Static randomly-initialized target encoder (no EMA), per design.
        ema_decay=(1.0, 1.0),
        dp_config=DataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """v1.1 dataloader + batch-level year/latlon dropout."""
    config = build_dataloader_config_base(common)
    config.year_dropout_rate = YEAR_DROPOUT_RATE
    config.latlon_dropout_rate = LATLON_DROPOUT_RATE
    config.metadata_dropout_view_mode = METADATA_DROPOUT_VIEW_MODE
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
