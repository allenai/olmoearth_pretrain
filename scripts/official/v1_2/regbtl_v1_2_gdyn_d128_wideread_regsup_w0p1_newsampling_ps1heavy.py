"""d128 wideread + regsup (w0p1, newsampling) with a HEAVIER ps=1 sampling bias.

``regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_newsampling`` with ``patch_size_probs``
pushed from 0.40 to 0.70 on ps=1 (the coarse tail squeezed to match) while every other
newsampling knob is held fixed.

WHY: the newsampling gain on the frozen ps=1 PASTIS probes tracks patch-size
reallocation rather than temporal exposure (see the ``_psuniform`` sibling for the
evidence). If that read is right, ps=1 performance should keep climbing as ps=1 mass
increases -- this run asks whether 0.40 was already saturating or whether there is
real headroom. It also prices the trade: the ps=4 evals lost ~0.01-0.02 going to
0.40, so 0.70 should show a correspondingly larger ps=4 regression. Together with
``_psuniform`` (0.125) and the committed ``_newsampling`` (0.40) this is a three-point
sweep in ps=1 mass with everything else fixed, which is what makes the trend
interpretable.
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
# P(patch_size = k) for k in 1..8. ps=1 goes 0.40 -> 0.70; the remaining 0.30 keeps the
# same decreasing shape over ps=2..8 so flexi-ViT still sees every patch size. Must sum
# to exactly 1.0 (the dataloader validates this).
PATCH_SIZE_PROBS = [0.70, 0.09, 0.07, 0.05, 0.04, 0.02, 0.02, 0.01]
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_newsampling_ps1heavy.py"
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
    """Newsampling dataloader with ps=1 oversampled to 0.70."""
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
