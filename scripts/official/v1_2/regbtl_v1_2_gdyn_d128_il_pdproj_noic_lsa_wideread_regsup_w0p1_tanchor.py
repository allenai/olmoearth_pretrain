"""d128 wideread + regsup (w0p1) with the temporally-anchored register read, OLD sampling.

Identical to ``regbtl_v1_2_gdyn_d128_il_pdproj_noic_lsa_wideread_regsup_w0p1``
(the pre-newsampling recipe: base 1fwd dataloader/masking, full rank microbatch)
except ``register_temporal_anchor="year_start"``: the bottleneck's cross-attention
reads run axial 3D RoPE with each register anchored at Jan 1 of the sample's first
observation year and the patch keys at anchor-relative days (~day-of-year), instead
of slicing the temporal coordinate off. The register grid itself stays a time-free
2D map (decoder, regsup, and eval contracts unchanged); only the READ gains temporal
geometry, so heads can learn season-selective read patterns instead of routing purely
on content salience. Motivated by the frozen ps=1 PASTIS embedding evals, where the
time-blind read is the suspected bottleneck for phenology-defined classes.

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
from regbtl_v1_2_regsup_common import add_register_supervision

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

REGISTER_DIM = 128
# 10x the regsup_common SUPERVISION_WEIGHT nudge (0.01 -> 0.1).
SUPERVISION_BASE_WEIGHT = 0.1
REGISTER_TEMPORAL_ANCHOR = "year_start"
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d128_il_pdproj_noic_lsa_wideread_regsup_w0p1_tanchor.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d128 wideread + regsup (w0p1) with the year-start-anchored register read."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    config.encoder_config.register_temporal_anchor = REGISTER_TEMPORAL_ANCHOR
    return add_register_supervision(
        config, include_latlon=False, base_weight=SUPERVISION_BASE_WEIGHT
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
