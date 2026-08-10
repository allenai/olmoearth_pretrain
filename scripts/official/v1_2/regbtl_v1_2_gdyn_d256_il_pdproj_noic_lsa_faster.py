"""v1.2 register bottleneck noic_lsa with register_dim=256 and all validated speedups.

Same architecture as ``regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd_fusedadamw``
(gdyn + il + pdproj, no instance contrastive, latent self-attention ON) but with a
NARROWER register bottleneck (``register_dim=256`` instead of 768), and with the
remaining ``base_faster.py`` speedups stacked on top of the single-forward-pass +
fused-AdamW recipe: projection-only target encoder, replicated DP (ddp), and bf16
autocast. See ``regbtl_v1_2_faster_common`` for the full rationale.

In-loop evals run as separate Beaker jobs (run_as_beaker_job=True) and include the
fifty_cities random-split S2 + S1+S2 probes.
"""

import logging

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_common import add_loop_eval_beaker_job
from regbtl_v1_2_faster_common import (
    build_faster_regbtl_model_config,
    build_faster_train_module_config,
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

REGISTER_DIM = 256
MODULE_PATH = "scripts/official/v1_2/regbtl_v1_2_gdyn_d256_il_pdproj_noic_lsa_faster.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """d256 register bottleneck (latent self-attn ON) with projection-only target."""
    return build_faster_regbtl_model_config(
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
