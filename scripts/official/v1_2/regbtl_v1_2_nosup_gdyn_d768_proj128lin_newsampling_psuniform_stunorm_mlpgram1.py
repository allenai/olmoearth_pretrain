"""No-supervision arm: the v1.3 RC with the map supervision heads deleted.

The direct-supervision ablation for the v1.3 report. The Aggregation section
claims direct supervision on the query tokens is one of the mechanisms ensuring
spatial locality; this arm removes it and nothing else.

THE CHANGE, AND ONLY IT: ``supervision_head_config = None``. The teacher then
trains on the LatentMIM loss alone, and the student on the distillation losses
alone. Everything else -- d768 wideread registers, the linear+LayerNorm student,
MLP back-projections, Gram weight, sampler, data -- is byte-identical to the RC
(``regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform_stunorm_mlpgram1``),
so the pair is a one-flag contrast.

Adjacent evidence, for the writeup: sup128 (supervision on the student only)
collapses the teacher -- settled -- so supervision PLACEMENT matters; this arm
answers whether supervision is needed at all on the current recipe.

IN-LOOP EVALS: identical to the RC (``set_proj_aeftrial_loop_evals``, both
student widths, 80k interval).

EVAL-ARM NAMING: the ``nosup`` token sits directly after ``regbtl_v1_2`` so no
existing arm name is a prefix of this one (the CSV export merges arms by
longest prefix).
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd import (
    build_common_components,
    build_dataset_config,
    build_visualize_config,
)
from regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_stunorm_mlpgram1 import (
    build_dataloader_config,
    build_train_module_config,
)
from regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_stunorm_mlpgram1 import (
    build_model_config as _base_build_model_config,
)
from regbtl_v1_2_proj_common import set_proj_aeftrial_loop_evals

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_2/regbtl_v1_2_nosup_gdyn_d768_proj128lin_newsampling_psuniform_stunorm_mlpgram1.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """The RC's model config with the map supervision heads removed."""
    config = _base_build_model_config(common)
    config.supervision_head_config = None
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
