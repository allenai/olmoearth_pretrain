"""v1.2 register bottleneck noic_lsa, register_dim=768, bottleneck attention at ENCODER width.

The no-supervision twin of ``regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_regsup``:
identical model (``build_wideread_regbtl_model_config`` at register_dim=768) and
train recipe (``build_faster_train_module_config`` -- single forward pass, fused
AdamW, projection-only target, replicated DP + bf16 autocast), but WITHOUT the
register-grid supervision head. At d768 the wideread builder's
``register_attn_dim = embedding_size`` is a no-op (register_dim == encoder width
already), so this matches the d768 frontier exactly. Exists so the 600-epoch
longer-training test has a supervision-free d768 baseline built on the same recipe
as the regsup runs.

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

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

logger = logging.getLogger(__name__)

REGISTER_DIM = 768
MODULE_PATH = (
    "scripts/official/v1_2/regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_wideread.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d768 frontier (wideread is a no-op at full width), no supervision head."""
    return build_wideread_regbtl_model_config(
        common, latent_self_attn=True, register_dim=REGISTER_DIM
    )


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
