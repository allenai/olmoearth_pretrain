"""v1.2 register bottleneck: gdyn + il + pdproj, instance contrastive OFF, latent self-attn OFF.

Same as ``regbtl_v1_2_gdyn_d768_il_pdproj_ic_lsa`` but drops BOTH the InfoNCE instance
contrastive loss (``noic``: ``contrastive_config=None``) and the bottleneck's latent
self-attention (``nolsa``: ``register_latent_self_attn=False``). The registers come from
the 4 cross-attention reads alone, trained by the JEPA / patch-discrimination loss only.

In-loop evals run as separate Beaker jobs (run_as_beaker_job=True) and include the
fifty_cities random-split S2 + S1+S2 probes.
"""

import logging

from base import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_visualize_config,
)
from base import (
    build_train_module_config as _base_build_train_module_config,
)
from base import (
    build_trainer_config as _base_build_trainer_config,
)
from regbtl_v1_2_common import add_loop_eval_beaker_job, build_regbtl_model_config

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/official/v1_2/regbtl_v1_2_gdyn_d768_il_pdproj_noic_nolsa.py"


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Gdyn + il + pdproj register bottleneck, latent self-attention OFF."""
    return build_regbtl_model_config(common, latent_self_attn=False)


def build_train_module_config(common: CommonComponents):
    """Drop the instance (InfoNCE) contrastive loss (the 'noic' arm)."""
    config = _base_build_train_module_config(common)
    config.contrastive_config = None
    return config


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
