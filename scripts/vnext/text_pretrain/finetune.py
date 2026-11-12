r"""The fourth method: the published v1.3 fine-tuned with the text loss.

Between the frozen twins (frozen.py: only the head trains) and the joint arms (the whole
model from scratch): the fine-tune arms start every weight but the new text head from the
released v1.3 and train them all, on the joint arm's data and objectives.

The backbone continues at BACKBONE_LR, a fifth of v1.3's peak, so it moves without
forgetting; the fresh head gets HEAD_LR, as in the frozen twins. Both warm up over WARMUP
steps (the optimizer state starts empty) and decay over STEPS.
"""

from frozen import PUBLISHED_WEIGHTS
from olmo_core.optim import AdamWConfig, OptimGroupOverride
from olmo_core.optim.scheduler import CosWithWarmup
from v1_3 import build_train_module_config as _v1_3_build_train_module_config

from olmoearth_pretrain.internal.experiment import CommonComponents
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import (
    LatentMIMTrainModuleConfig,
)

STEPS = 50_000
BACKBONE_LR = 2e-5
HEAD_LR = 1e-3
WARMUP = 2_000


def from_published(config: LatentMIMConfig) -> LatentMIMConfig:
    """The model config, started from the published weights with everything training."""
    config.init_weights = PUBLISHED_WEIGHTS
    return config


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """v1_3.py's train module with the fine-tuning schedule."""
    config = _v1_3_build_train_module_config(common)
    config.optim_config = AdamWConfig(
        lr=BACKBONE_LR,
        weight_decay=config.optim_config.weight_decay,
        fused=True,
        group_overrides=[
            OptimGroupOverride(params=["text_head.*"], opts={"lr": HEAD_LR})
        ],
    )
    config.scheduler = CosWithWarmup(warmup=WARMUP)
    return config
