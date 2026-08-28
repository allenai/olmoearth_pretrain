"""``lin_sup768_w1`` with band dropout OFF and a LINEAR patch projection.

Two flags changed against
``regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform`` -- the distilled
release candidate -- both in the pixel -> token stem and both carried unmeasured
since v1.1; see ``regbtl_v1_2_nobdlinpe_common`` for what each one is and why they
are cut together. Teacher, student, supervision (``supervision_source="registers"``
at w1), distillation objective, sampler and data are all byte-identical to that run.

IN-LOOP EVALS: ``set_proj_aeftrial_loop_evals`` -- the AEF balanced trials (eight
year-aligned datasets, kNN twins, so ``aeftrial_{ridge,knn5,knn20}`` come for free)
plus year-aligned PASTIS, unmasked S1+S2+Landsat, at BOTH student widths (d128, d64).
NOTE this is NOT the eval set the release candidate ran (that arm uses
``add_proj_loop_eval_beaker_job``); it matches the stunorm sibling, so the whole 2x2
compares on one readout.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform import (
    build_dataloader_config,
    build_train_module_config,
)
from regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform import (
    build_model_config as _base_build_model_config,
)
from regbtl_v1_2_nobdlinpe_common import apply_nobd_linpe
from regbtl_v1_2_proj_common import set_proj_aeftrial_loop_evals

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

MODULE_PATH = (
    "scripts/official/v1_2/regbtl_v1_2_nobdlinpe_gdyn_d768_proj128lin_"
    "sup768_w1_newsampling_psuniform.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The sup768 arm with no band dropout and a linear patch projection."""
    return apply_nobd_linpe(_base_build_model_config(common))


def build_trainer_config(common: CommonComponents):
    """Base trainer + AEF trials and PASTIS on both student widths."""
    return set_proj_aeftrial_loop_evals(_base_build_trainer_config(common), MODULE_PATH)


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
