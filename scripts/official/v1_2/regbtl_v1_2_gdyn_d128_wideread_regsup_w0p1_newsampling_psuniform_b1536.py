"""d128 wideread + regsup (w0p1, newsampling, uniform ps) at token_budget 1536.

``regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_newsampling_psuniform`` at half the
committed 3072 budget, with ``temporal_bias`` left at the default 2.75. Roughly half the
wall-clock per step, so ~45h for the full 300 epochs (662,700 steps -- the duration is
deliberately unchanged, see ``apply_shape_sweep``).

WHY: isolates the budget axis at the default bias, which is what makes the
``..._psuniform_b1536_tb6`` corner interpretable. Without this arm, a good result from
that corner could not be attributed -- you could not tell whether the cheap budget was
simply harmless or whether the aggressive bias was compensating for it.

Note this arm is expected to be the weakest of the 2x2 on the frozen ps=1 probes: at
budget 1536 the ws16 ps=1 eval shape (hw=16) is only reachable at t<=2, versus t<=4 at
3072 and t<=8 at 6144. It earns its place by decomposing the corner, not by winning.

Part of the budget x bias 2x2; see the directory README for the full grid.
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
    build_dataset_config,
    build_visualize_config,
)
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_dataloader_config as _base_build_dataloader_config,
)
from regbtl_v1_2_newsampling_common import (
    SUPERVISION_BASE_WEIGHT,
    apply_microbatch,
    apply_new_sampling,
    apply_shape_sweep,
    apply_uniform_patch_sizes,
)
from regbtl_v1_2_regsup_common import add_register_supervision

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

REGISTER_DIM = 128
TOKEN_BUDGET = 1536
TEMPORAL_BIAS = 2.75
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_newsampling_psuniform_b1536.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d128 wideread + register-grid supervision at w0p1 (base_weight 0.1)."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    return add_register_supervision(
        config, include_latlon=False, base_weight=SUPERVISION_BASE_WEIGHT
    )


def build_dataloader_config(common: CommonComponents):
    """Uniform-patch-size newsampling dataloader at budget 1536, temporal_bias 2.75."""
    return apply_shape_sweep(
        apply_uniform_patch_sizes(
            apply_new_sampling(_base_build_dataloader_config(common))
        ),
        token_budget=TOKEN_BUDGET,
        temporal_bias=TEMPORAL_BIAS,
    )


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """1fwd + fused AdamW train module with a halved rank microbatch size."""
    return apply_microbatch(build_faster_train_module_config(common))


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
