"""regbtl_base10k_scale0.25_gdyn_d768_il, WITHOUT the bottleneck's latent self-attention.

Same as ``regbtl_base10k_scale0.25_gdyn_d768_il`` (dynamic grid, d768, interleaved
``[read -> self] x4`` schedule) but with ``register_latent_self_attn=False``, so the
latent self-attention blocks are dropped: the registers come from the 4 cross-attention
reads alone, with no register-to-register mixing. A/Bs against the il run to isolate the
latent transformer's contribution.

In-loop evals run as separate Beaker jobs (run_as_beaker_job=True) and include the
fifty_cities random-split S2 + S1+S2 probes (see ``regbtl_nolsa_common``).
"""

import logging

from hidden1_supervision import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_visualize_config,
)
from hidden1_supervision import (
    build_trainer_config as _base_build_trainer_config,
)
from regbtl_nolsa_common import add_loop_eval_beaker_job, build_nolsa_model_config

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

# This script's repo-relative path, used as the eval job's TRAIN_SCRIPT_PATH so the rebuilt
# eval-job model reproduces this exact architecture (must stay in sync with the filename).
MODULE_PATH = (
    "scripts/archived/2026_04_22_add_hidden_layer_to_initial_projection/"
    "regbtl_gdyn_d768_il_nolsa.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Nolsa base + interleaved reads ([read -> self] schedule, but with self-attn off)."""
    config = build_nolsa_model_config(common)
    config.encoder_config.register_interleave = True
    return config


def build_trainer_config(common: CommonComponents):
    """Base trainer config + fifty_cities evals routed through a Beaker job."""
    return add_loop_eval_beaker_job(_base_build_trainer_config(common), MODULE_PATH)


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
