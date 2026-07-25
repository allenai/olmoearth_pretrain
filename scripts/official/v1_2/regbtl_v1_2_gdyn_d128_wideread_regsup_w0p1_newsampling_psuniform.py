"""d128 wideread + regsup (w0p1, newsampling) with UNIFORM patch-size sampling.

``regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_newsampling`` with ``patch_size_probs``
reverted to uniform over ps=1..8 (0.125 each, the dataloader default) while every
other newsampling knob (decorrelated time/grid sampling, temporal_bias, token floor,
budget 3072, decode-only maps excluded) is held fixed.

WHY: the newsampling gain observed on the frozen ps=1 PASTIS probes (~+0.025 mIoU vs
old sampling, consistently across the tanchor/ndvi arms) coincides with a 3.2x
oversampling of ps=1 (0.125 -> 0.40), and it does NOT transfer to the ps=4 evals --
including the in-loop ps=4 PASTIS probe on the SAME dataset and labels, which moved
only ~+0.008. That pattern points at patch-size reallocation rather than the
full-sequence temporal exposure as the driver. This run is the direct test: if the
ps=1 gain largely disappears here, the ps=1 bias owns it and the temporal knobs are
second-order. A/B partner: ``..._w0p1_newsampling`` (ps=1 at 0.40) and
``..._w0p1_newsampling_ps1heavy`` (ps=1 at 0.70) complete the three-point sweep.
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
)
from regbtl_v1_2_regsup_common import add_register_supervision

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

REGISTER_DIM = 128
# Uniform over ps=1..8 -- the dataloader default, i.e. what patch_size_probs=None does.
# Spelled out explicitly so this run's distribution is legible next to the ps1heavy one.
PATCH_SIZE_PROBS = [0.125] * 8
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_newsampling_psuniform.py"
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
    """Newsampling dataloader with the patch-size distribution forced to uniform."""
    config = apply_new_sampling(_base_build_dataloader_config(common))
    config.patch_size_probs = PATCH_SIZE_PROBS
    return config


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
