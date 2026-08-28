"""``lin_sup768_w1`` -- the distilled release candidate -- on the AEF-trial eval set.

Model, data, sampler, supervision and distillation are imported unchanged from
``regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform``. The ONLY
difference is the in-loop eval set.

That arm uses ``add_proj_loop_eval_beaker_job`` (fifty_cities + PASTIS at 40k, on
BOTH the d768 registers and the student). This one uses
``set_proj_aeftrial_loop_evals`` -- the eight AEF-supplemental datasets' kNN twins
plus year-aligned PASTIS, unmasked S1+S2+Landsat, at both student widths (d128, d64),
80k -- which is what its ``stunorm`` sibling already runs. Without this the 2x2 over
{stunorm} x {nobdlinpe} would have three arms on one readout and its origin on
another.

Consequence worth knowing: this arm does NOT overlay on the release candidate's own
historical curves, which were logged on the fifty_cities set.
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
    build_model_config,
    build_train_module_config,
)
from regbtl_v1_2_proj_common import set_proj_aeftrial_loop_evals

from olmoearth_pretrain.internal.experiment import CommonComponents, main

logger = logging.getLogger(__name__)

MODULE_PATH = (
    "scripts/official/v1_2/regbtl_v1_2_gdyn_d768_proj128lin_"
    "sup768_w1_newsampling_psuniform_aeftrial.py"
)


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
