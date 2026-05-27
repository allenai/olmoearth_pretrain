"""ViT-base v1.1 + 2D RoPE + a modern-transformer / training recipe.

Builds directly on ``scripts/official/v1_1/rope.py`` (which is itself the
``base.py`` hidden-projection baseline with additive spatial encodings replaced
by axial 2D RoPE). On top of RoPE this turns on a bundle of changes that bring
the block and optimizer in line with current best practice. Each is toggled by
a constant below, so individual ablations are a one-line edit:

Model (encoder + predictor):
  2. QK_NORM          - normalize Q/K before attention (RMSNorm here).
                        ViT-22B (arXiv:2302.05442), OLMo 2 (arXiv:2501.00656).
  4. NORM_TYPE        - RMSNorm instead of LayerNorm. OLMo 2; LLaMA.
  5. FFN_TYPE         - SwiGLU instead of GELU MLP. Shazeer (arXiv:2002.05202).
  6. USE_BIAS=False   - drop biases on attention/FFN linears + norms. ViT-22B.
  7. INIT_SCHEME + LAYER_SCALE_INIT - trunc-normal init paired with LayerScale,
                        the residual-stability recipe used by DINOv2 / DINOv3
                        (CaiT, arXiv:2103.17239). The GPT-2 1/sqrt(2*depth)
                        variant is available via INIT_SCHEME="scaled" instead.

Training (optimizer):
  1. NO_DECAY_PATTERNS - exclude norms / biases / channel embeddings from weight
                         decay (olmo-core ``group_overrides``).
  3. ADAMW_BETAS       - beta2=0.95 for masked pretraining. MAE (arXiv:2111.06377).

All of these default to OFF in the model/optimizer code, so this script is the
only place the new recipe is enabled; existing checkpoints are unaffected.
"""

import logging

from base import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_trainer_config,
    build_visualize_config,
)
from base import (
    build_train_module_config as build_train_module_config_base,
)
from olmo_core.optim import AdamWConfig
from olmo_core.optim.config import OptimGroupOverride
from rope import build_model_config as build_model_config_rope

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

# --- Model recipe (applied to both encoder and predictor) ----------------------
NORM_TYPE = "rmsnorm"  # (#4) "layernorm" | "rmsnorm"
FFN_TYPE = "swiglu"  # (#5) "mlp" | "swiglu"
USE_BIAS = False  # (#6) drop biases on attn/FFN linears + norms
# (#7) Residual-stability recipe. DINO-style: trunc-normal weights + LayerScale.
# Set INIT_SCHEME="scaled" and LAYER_SCALE_INIT=None for the GPT-2 variant.
INIT_SCHEME = "trunc_normal"  # "xavier" | "trunc_normal" | "scaled"
LAYER_SCALE_INIT = 1e-5  # LayerScale init (CaiT/DINO); None disables it
QK_NORM = True  # (#2) normalize Q/K before attention

# --- Optimizer recipe ----------------------------------------------------------
LR = 0.0001
WEIGHT_DECAY = 0.02
ADAMW_BETAS = (0.9, 0.95)  # (#3) beta2=0.95 for masked pretraining
# (#1) Parameters excluded from weight decay. With RMSNorm + USE_BIAS=False the
# norm modules carry no bias, so these globs do not overlap (a parameter would
# otherwise land in two param groups). If you flip to LayerNorm + biases, drop
# the "*.bias" pattern or the norm-bias would be matched twice.
NO_DECAY_PATTERNS = ["*.bias", "*norm*", "*channel_embeddings*"]
if LAYER_SCALE_INIT is not None:
    # LayerScale gammas are 1-D residual gains, like norm scales -> no decay.
    # (Only added when LayerScale is on, else the glob would match nothing and
    # olmo-core's strict group matching would raise.)
    NO_DECAY_PATTERNS = [*NO_DECAY_PATTERNS, "*.gamma"]


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """v1.1 base + 2D RoPE + modern-block options (RMSNorm/SwiGLU/no-bias/init)."""
    config = build_model_config_rope(common)
    for sub_config in (config.encoder_config, config.decoder_config):
        sub_config.norm_type = NORM_TYPE
        sub_config.ffn_type = FFN_TYPE
        sub_config.use_bias = USE_BIAS
        sub_config.init_scheme = INIT_SCHEME
        sub_config.layer_scale_init = LAYER_SCALE_INIT
        sub_config.qk_norm = QK_NORM
    return config


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Reuse the base train module but swap in the modern optimizer recipe."""
    config = build_train_module_config_base(common)
    config.optim_config = AdamWConfig(
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=ADAMW_BETAS,
        fused=False,
        group_overrides=[
            OptimGroupOverride(
                params=NO_DECAY_PATTERNS,
                opts={"weight_decay": 0.0},
            )
        ],
    )
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
