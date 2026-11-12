r"""The third method: a text head trained on the published v1.3, kept frozen.

The joint arms (text.py, text_mix.py) train the whole model with the text loss from
scratch. Their frozen twins (text_frozen.py, text_mix_frozen.py) keep everything about
them (data, captions and labels, the head, its loss, masking, batch) except that the
model starts from the released v1.3 weights and only the text head trains. What the
joint arms add over their twin is then what learning the backbone with text adds.

Head-only training gets a short schedule of its own: STEPS steps at LR (ClipEarth's head
learning rate) after WARMUP warmup steps, instead of v1.3's 667,200 at 1e-4.
"""

from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import CosWithWarmup
from olmo_core.train.common import Duration
from olmo_core.train.config import TrainerConfig
from v1_3 import build_train_module_config as _v1_3_build_train_module_config

from olmoearth_pretrain.internal.experiment import CommonComponents
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import (
    LatentMIMTrainModuleConfig,
)

# HF allenai/OlmoEarth-v1_3-Base: gabrielt/v1_3_rc_ld1_pixtgt_pix512 at step 667,200.
PUBLISHED_WEIGHTS = (
    "/weka/dfive-default/agent/piperbots/2026-11-10-piperw-terramesh-eval/hf_v1_3_base"
)
STEPS = 20_000
LR = 1e-3
WARMUP = 1_000


def freeze(config: LatentMIMConfig) -> LatentMIMConfig:
    """The model config, started from the published weights with only the head training."""
    config.frozen_backbone_weights = PUBLISHED_WEIGHTS
    return config


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """v1_3.py's train module with the head-only schedule."""
    config = _v1_3_build_train_module_config(common)
    config.optim_config = AdamWConfig(
        lr=LR, weight_decay=config.optim_config.weight_decay, fused=True
    )
    config.scheduler = CosWithWarmup(warmup=WARMUP)
    return config


def shorten(config: TrainerConfig, steps: int = STEPS) -> TrainerConfig:
    """The trainer, run for ``steps`` steps."""
    config.max_duration = Duration.steps(steps)
    return config
