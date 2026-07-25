"""d128 wideread + regsup (w0p1, newsampling) trained at patch_size=1 ONLY.

``regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_newsampling`` with ``patch_size_probs``
collapsed to all mass on ps=1, so flexi-ViT never sees any other patch size. Every other
newsampling knob (decorrelated time/grid sampling, temporal_bias, token floor, budget
3072, decode-only maps excluded) is held fixed.

WHY: the upper limit of the patch-size sweep. The newsampling gain on the frozen ps=1
PASTIS probes tracks P(ps=1) rather than the temporal knobs (see the ``_psuniform``
sibling for the evidence), so this asks how far that trend goes when the entire sampling
budget is spent on the deployment resolution. Together with ``_psuniform`` (0.125), the
committed ``_newsampling`` (0.40) and ``_ps1heavy`` (0.70) this is the P(ps=1) -> 1.0
endpoint.

EVALS ARE RESTRICTED. This run uses ``set_ps1_only_loop_evals`` instead of
``add_loop_eval_beaker_job``: the in-loop set becomes the PASTIS ps=1 exports plus the
AEF supplemental ps=1 probes (linear-probe and kNN), and the shared catalog is dropped.
Every catalog eval except the ws16 ps=1 exports runs at ``patch_size=4``, which this
model is never trained at, so those numbers would be measuring an untrained resolution
rather than a regression. It also means this run is NOT comparable to the rest of the
sweep on the ps=4 evals -- only on the ps=1 ones, which is the axis under study.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_common import set_ps1_only_loop_evals
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
# All mass on ps=1; ps=2..8 are never sampled. The shape sampler still has plenty of
# feasible grids here -- at ps=1 the token cost is 3*hw^2*t, so the token floor (228) and
# budget (3072) leave hw=3..9 paired with long sequences and hw up to 32 at t=1.
PATCH_SIZE_PROBS = [1.0] + [0.0] * 7
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_newsampling_ps1only.py"
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
    """Newsampling dataloader restricted to ps=1."""
    config = apply_new_sampling(_base_build_dataloader_config(common))
    config.patch_size_probs = PATCH_SIZE_PROBS
    return config


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """1fwd + fused AdamW train module with a halved rank microbatch size."""
    return apply_microbatch(build_faster_train_module_config(common))


def build_trainer_config(common: CommonComponents):
    """Base trainer config with ONLY the ps=1 evals, routed through Beaker jobs."""
    return set_ps1_only_loop_evals(_base_build_trainer_config(common), MODULE_PATH)


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
