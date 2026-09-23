"""v1.2 base with weight decay removed from the learnable RoPE-Mixed frequencies.

Ablation of the OlmoEarth v1.2 paper run (``trope_mixed_tscale_months``, W&B
``nd3xh7py`` in ``2026_04_22_add_hidden_layer_to_initial_projection``). That run
uses ``AdamWConfig(weight_decay=0.02)`` with no ``group_overrides``, so olmo-core
puts every trainable parameter -- including the per-head learnable 3D RoPE-Mixed
frequencies (``*.attn.rope_mixed_freqs``, encoder and decoder) -- in the decayed
group. Decaying a frequency pulls it toward zero, i.e. toward *no* positional
signal, on every step. This run is the paper recipe with ``weight_decay=0.0``
for exactly those tensors and nothing else changed.

Differences from ``scripts/official/v1_2/base.py``:

* ``rope_mixed_base`` pinned to 10.0 on encoder and decoder. That is the value
  the paper run and the ``v1_2_{nano,tiny,small,large}`` sweep actually used
  (W&B config, checkpoint ``config.json``, ``launch_v1_2_urgent_sweep.sh``);
  ``base.py`` has said 10000.0 since 84478b8ae, which does not match them.
* ``optim_config.group_overrides`` puts ``*rope_mixed_freqs`` at weight decay 0.
  olmo-core builds groups in strict mode, so the run fails at startup if the
  glob matches nothing.
* W&B project set to the paper run's project so the two runs overlay directly.

Everything else (fused AdamW, FSDP, masking, losses, evals, schedule) is
``base.py`` verbatim.
"""

import logging
import sys
from pathlib import Path

# ``base.py`` in scripts/official/v1_2 is a plain script (its size variants do
# ``from base import ...`` from the same directory), so put that directory on
# the path before importing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "official" / "v1_2"))

from base import (  # noqa: E402
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_visualize_config,
)
from base import build_model_config as _base_build_model_config  # noqa: E402
from base import (
    build_train_module_config as _base_build_train_module_config,  # noqa: E402
)
from base import build_trainer_config as _base_build_trainer_config  # noqa: E402
from olmo_core.optim import OptimGroupOverride  # noqa: E402
from olmo_core.train.config import TrainerConfig  # noqa: E402

from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (  # noqa: E402
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

# Value used by the paper run nd3xh7py (see module docstring).
ROPE_MIXED_BASE = 10.0
# FQN glob for the learnable RoPE-Mixed frequencies in every attention layer.
ROPE_FREQ_PARAM_GLOB = "*rope_mixed_freqs"
# Same project as the paper run so both curves can be overlaid in one workspace.
WANDB_PROJECT = "2026_04_22_add_hidden_layer_to_initial_projection"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """v1.2 base model with the paper run's RoPE-Mixed init base."""
    config = _base_build_model_config(common)
    config.encoder_config.rope_mixed_base = ROPE_MIXED_BASE
    config.decoder_config.rope_mixed_base = ROPE_MIXED_BASE
    return config


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """v1.2 base optimizer with the RoPE-Mixed frequencies excluded from weight decay."""
    config = _base_build_train_module_config(common)
    config.optim_config.group_overrides = [
        OptimGroupOverride(params=[ROPE_FREQ_PARAM_GLOB], opts={"weight_decay": 0.0}),
    ]
    return config


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """v1.2 base trainer logging to the paper run's W&B project."""
    trainer_config = _base_build_trainer_config(common)
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
    return trainer_config


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
