"""``lin_supstu0p1_w1_stunorm`` with band dropout OFF and a LINEAR patch projection.

Two flags changed against
``regbtl_v1_2_gdyn_d768_proj128lin_supstu0p1_w1_newsampling_psuniform_stunorm``, both
in the pixel -> token stem -- see ``regbtl_v1_2_nobdlinpe_common`` for what each one
is and why they are cut together. Everything else, ``register_projection_output_norm``
(the LayerNorm on the linear student's output) included, is byte-identical to that
run, so it is the control.

This is the stunorm half of the pair; its non-stunorm twin is
``regbtl_v1_2_nobdlinpe_gdyn_d768_proj128lin_supstu0p1_w1_newsampling_psuniform``.
Both run the same eval set as the 2026-08-26 student-norm arms, so the 2x2 (stem
ablation x student norm) reads off one table.

IN-LOOP EVALS: the embedding evals -- AEF balanced trials + year-aligned PASTIS on
unmasked S1+S2+Landsat at BOTH student widths (d128 and d64), teacher not scored --
``set_proj_aeftrial_loop_evals``, unchanged from the control.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from regbtl_v1_2_gdyn_d768_proj128lin_supstu0p1_w1_newsampling_psuniform_stunorm import (
    build_dataloader_config,
    build_train_module_config,
)
from regbtl_v1_2_gdyn_d768_proj128lin_supstu0p1_w1_newsampling_psuniform_stunorm import (
    build_model_config as _base_build_model_config,
)
from regbtl_v1_2_nobdlinpe_common import apply_nobd_linpe
from regbtl_v1_2_proj_common import set_proj_aeftrial_loop_evals

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

MODULE_PATH = (
    "scripts/official/v1_2/regbtl_v1_2_nobdlinpe_gdyn_d768_proj128lin_"
    "supstu0p1_w1_newsampling_psuniform_stunorm.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The stunorm supstu0p1 arm with no band dropout and a linear patch projection."""
    config = apply_nobd_linpe(_base_build_model_config(common))
    # The control's own flag, restated as an assertion: this arm must differ from it
    # in the stem only.
    assert config.encoder_config.register_projection_output_norm, (
        "expected the student-output LayerNorm from the stunorm base"
    )
    return config


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
