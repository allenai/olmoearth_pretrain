"""v1.2 register bottleneck noic_lsa d768 (full width) + register-grid supervision.

The ``gdyn_d768_il_pdproj_noic_lsa`` frontier on the ``_faster`` recipe (single
forward pass, fused AdamW, projection-only target, replicated DP + bf16 autocast),
plus a low-weight (0.01) supervision head that reads the ENCODER's register grid
(``register_supervision=True``) and predicts the decode-only map modalities
(worldcover, srtm, openstreetmap_raster, wri_canopy_height_map, cdl, worldcereal).
See ``regbtl_v1_2_regsup_common`` for the weights and rationale.

At d768 the wideread builder's ``register_attn_dim = embedding_size`` is a no-op
(register_dim == encoder width already), so this model matches the d768 frontier
exactly and differs from the ``regsup_latlon`` twin only by the latlon head.

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

REGISTER_DIM = 768
MODULE_PATH = "scripts/official/v1_2/regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_regsup.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d768 frontier + register-grid supervision of the decode-only modalities."""
    config = build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )
    return add_register_supervision(config, include_latlon=False)


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
