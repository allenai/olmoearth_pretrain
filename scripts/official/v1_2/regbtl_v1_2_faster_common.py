"""Shared pieces for the ``_faster`` register-bottleneck runs.

``faster`` stacks the production outcome of the July 2026 speedup program
(``base_faster.py``, W&B ``2026_07_08_fused_adamw_e2e``) on top of the
``noic_lsa_1fwd_fusedadamw`` recipe:

* single forward pass per batch (plain train module, ``num_masked_views=1``) and
  the fused AdamW kernel -- inherited from the 1fwd_fusedadamw builders;
* ``projection_only_target=True`` -- the frozen exit-0 target encoder is just the
  initial projection; valid because the v1.2 base uses all-zero token exits and
  ``ema_decay=(1.0, 1.0)``, so the full target-encoder copy is dead weight;
* ``dp_config.name=ddp`` + bf16 autocast -- replicated params with one coalesced
  fp32 gradient all-reduce per step instead of FSDP's per-layer collectives.

torch.compile is deliberately EXCLUDED: it degrades training quality on both the
FSDP and DDP stacks (see ``base_faster.py`` and the speedup-program archive). Do
not enable ``compile_model`` here without a full-run quality revalidation.

These runs sweep the register bottleneck width (``register_dim``), so each run
script bakes its own dim into ``build_model_config`` -- the in-loop eval Beaker
jobs rebuild the model from the launching script and must match the checkpoint.
"""

import logging

from olmo_core.config import DType
from olmo_core.distributed.parallel import DataParallelConfig, DataParallelType
from regbtl_v1_2_common import build_regbtl_model_config
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd_fusedadamw import (
    build_train_module_config as _build_fusedadamw_train_module_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

# Bottleneck attention head dim, fixed across the width sweep. The d768 frontier runs
# the encoder's default 12 heads = head dim 64; narrower registers scale the head
# count down (d512 -> 8, d256 -> 4, d128 -> 2) instead of shrinking the head dim.
REGISTER_HEAD_DIM = 64


def build_faster_regbtl_model_config(
    common: CommonComponents, *, latent_self_attn: bool, register_dim: int
) -> LatentMIMConfig:
    """Register-bottleneck model config with the projection-only target encoder.

    ``register_dim`` must be a multiple of ``REGISTER_HEAD_DIM``: the bottleneck head
    count is derived as ``register_dim // REGISTER_HEAD_DIM`` (the encoder default of
    12 heads only divides d768). The decoder needs no head override -- it projects the
    register grid to its own width before cross-attending.
    """
    if register_dim % REGISTER_HEAD_DIM != 0:
        raise ValueError(
            f"register_dim ({register_dim}) must be a multiple of {REGISTER_HEAD_DIM}"
        )
    config = build_regbtl_model_config(
        common, latent_self_attn=latent_self_attn, register_dim=register_dim
    )
    config.encoder_config.register_num_heads = register_dim // REGISTER_HEAD_DIM
    config.projection_only_target = True
    return config


def build_faster_train_module_config(
    common: CommonComponents,
) -> LatentMIMTrainModuleConfig:
    """1fwd + fused AdamW train module, switched to replicated DP + bf16 autocast."""
    config = _build_fusedadamw_train_module_config(common)
    config.dp_config = DataParallelConfig(name=DataParallelType.ddp)
    config.autocast_precision = DType.bfloat16
    return config
