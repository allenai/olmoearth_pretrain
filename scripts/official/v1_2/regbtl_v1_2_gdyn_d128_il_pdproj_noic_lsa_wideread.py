"""v1.2 register bottleneck noic_lsa, register_dim=128, bottleneck attention at ENCODER width.

Same recipe as the ``_faster`` runs (single forward pass, fused AdamW, projection-only
target, replicated DP + bf16 autocast) but with ``register_attn_dim=768``: the
bottleneck's read + latent attention runs with the encoder's own 12x64 head shape and
reads the K/V source at full encoder width, while the stored register stream stays at
128. ``register_dim`` is therefore purely the storage bottleneck -- see
``regbtl_v1_2_faster_common.build_wideread_regbtl_model_config`` for the rationale
(the 2026-07 width sweep showed narrow registers cannot fund both throughput-adequate
head counts and RoPE-anchoring head dims on their own).

In-loop evals run as separate Beaker jobs (run_as_beaker_job=True) and include the
fifty_cities random-split S2 + S1+S2 probes.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_common import add_loop_eval_beaker_job
from regbtl_v1_2_faster_common import (
    build_faster_train_module_config,
    build_wideread_regbtl_model_config,
)
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_visualize_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

REGISTER_DIM = 128
MODULE_PATH = (
    "scripts/official/v1_2/regbtl_v1_2_gdyn_d128_il_pdproj_noic_lsa_wideread.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d128 storage bottleneck with encoder-width (12x64) bottleneck attention."""
    return build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """1fwd + fused AdamW + replicated DP + bf16 autocast."""
    return build_faster_train_module_config(common)


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
