"""d128 wideread + regsup + latlon (w0p1, newsampling, UNIFORM ps), anchored register read.

``regbtl_v1_2_gdyn_d128_wideread_regsup_latlon_w0p1_tanchor_newsampling`` with
``patch_size_probs`` reverted to uniform over ps=1..8. Every other newsampling knob,
``register_temporal_anchor="year_start"``, and the latlon regression arm (a supervised
TARGET, never a model input) are unchanged.

WHY: see ``regbtl_v1_2_gdyn_d128_wideread_regsup_w0p1_tanchor_newsampling_psuniform`` --
the tanchor and latlon arms were only measured under the ps=1-oversampled sampler, which
the patch-size sweep showed to be a confound that degrades the ps=4 evals for no ps=1
gain. This re-tests them on the clean uniform-ps baseline.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_common import add_loop_eval_beaker_job
from regbtl_v1_2_faster_common import build_wideread_regbtl_model_config
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_visualize_config,
)
from regbtl_v1_2_newsampling_common import (
    SUPERVISION_BASE_WEIGHT,
    apply_microbatch,
    apply_new_sampling,
    apply_uniform_patch_sizes,
)
from regbtl_v1_2_regsup_common import (
    add_register_supervision,
    build_latlon_dataloader_config,
    build_latlon_dataset_config,
    build_latlon_train_module_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

REGISTER_DIM = 128
REGISTER_TEMPORAL_ANCHOR = "year_start"
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d128_wideread_regsup_latlon_w0p1_tanchor_newsampling_psuniform.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d128 wideread + regsup incl. latlon (w0p1) with the anchored register read."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    config.encoder_config.register_temporal_anchor = REGISTER_TEMPORAL_ANCHOR
    return add_register_supervision(
        config, include_latlon=True, base_weight=SUPERVISION_BASE_WEIGHT
    )


def build_dataloader_config(common: CommonComponents):
    """Latlon-aware newsampling dataloader at uniform patch sizes."""
    return apply_uniform_patch_sizes(
        apply_new_sampling(build_latlon_dataloader_config(common))
    )


def build_train_module_config(common: CommonComponents):
    """Latlon-aware faster train module with a halved rank microbatch size."""
    return apply_microbatch(build_latlon_train_module_config(common))


def build_trainer_config(common: CommonComponents):
    """Base trainer config + fifty_cities evals routed through a Beaker job."""
    return add_loop_eval_beaker_job(_base_build_trainer_config(common), MODULE_PATH)


def run() -> None:
    """Run the experiment."""
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_latlon_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )


if __name__ == "__main__":
    run()
