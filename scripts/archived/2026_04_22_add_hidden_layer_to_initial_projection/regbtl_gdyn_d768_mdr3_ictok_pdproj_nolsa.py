"""regbtl_base10k_scale0.25_gdyn_d768_mdr3_ictok_pdproj, WITHOUT latent self-attention.

Same as ``regbtl_base10k_scale0.25_gdyn_d768_mdr3_ictok_pdproj`` -- multi-depth reads from
encoder layers [3, 6, 9, 12], per-depth read projections, and the instance contrastive head
pooling the encoder patch tokens (``register_contrastive_source="encoder_tokens"``) -- but
with ``register_latent_self_attn=False``, dropping the latent self-attention blocks so each
depth's read feeds the registers with no register-to-register mixing between reads.

In-loop evals run as separate Beaker jobs (run_as_beaker_job=True) and include the
fifty_cities random-split S2 + S1+S2 probes (see ``regbtl_nolsa_common``).
"""

import logging

from hidden1_supervision import (
    build_common_components,
    build_dataloader_config,
    build_dataset_config,
    build_train_module_config,
    build_visualize_config,
)
from hidden1_supervision import (
    build_trainer_config as _base_build_trainer_config,
)
from regbtl_nolsa_common import add_loop_eval_beaker_job, build_nolsa_model_config

from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig

logger = logging.getLogger(__name__)

MODULE_PATH = (
    "scripts/archived/2026_04_22_add_hidden_layer_to_initial_projection/"
    "regbtl_gdyn_d768_mdr3_ictok_pdproj_nolsa.py"
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Nolsa base + multi-depth reads ([3,6,9,12]), per-depth proj, encoder-token contrastive."""
    config = build_nolsa_model_config(common)
    encoder_config = config.encoder_config
    encoder_config.register_read_layers = [3, 6, 9, 12]
    encoder_config.register_contrastive_source = "encoder_tokens"
    encoder_config.register_per_depth_read_proj = True
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
