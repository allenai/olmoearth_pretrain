"""cand_ndvi with the random-count / contiguous-time / retained-modality masking.

``regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform``
(the cand_ndvi baseline, W&B run sb2yr2pe in 2026_07_02_perceiver) with ONE change:
the masking strategy is swapped from ``random_time_with_decode`` to
``random_count_time_with_decode``. Everything else -- model (d128 wideread regsup +
time-conditioned NDVI head, anchored register read), dataset, sampler (newsampling,
uniform patch sizes), optimizer, schedule -- is inherited by importing the base
script's own builders, so this cannot drift from its A/B partner.

THE CHANGE (see ``RandomCountTimeWithDecodeMaskingStrategy`` in train/masking.py):

1. The encode/decode bandset split draws ``num_encode ~ Uniform{1..N}`` over the N
   present sensor bandsets instead of ``ceil(N * encode_ratio)``.
2. The masked portion of encode-side bandsets becomes DECODER (a prediction target)
   instead of TARGET_ENCODER_ONLY, so even all-encode draws still supply sensor
   targets alongside the decode-only maps/NDVI.
3. Time masking drops a CONTIGUOUS block of timesteps (e.g. of timesteps 1-12, drop
   6-12 or 4-8) instead of a random subset.
4. Time masking retains ONE image modality out of the timestep drop (randomly
   masked instead); with a single present image modality it falls back to random
   masking.

The ``encode_ratio``/``decode_ratio``/``random_ratio`` values and the only-decode
modality list (including the derived ndvi) are inherited unchanged from the base's
masking config -- only the strategy type moves. The strategy swap is applied to BOTH
masking-config copies: the dataloader's is the operative one (masking runs inside
the collate fn) and the train module's copy is kept consistent, mirroring
``regbtl_v1_2_cloudmask_common.apply_cloud_skip_threshold``.
"""

import logging
from typing import TypeVar

from base import build_trainer_config as _base_build_trainer_config
from regbtl_v1_2_common import add_loop_eval_beaker_job
from regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform import (
    build_common_components,
    build_dataset_config,
    build_model_config,
    build_visualize_config,
)
from regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform import (
    build_dataloader_config as _base_build_dataloader_config,
)
from regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform import (
    build_train_module_config as _base_build_train_module_config,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, main

logger = logging.getLogger(__name__)

MASKING_STRATEGY_TYPE = "random_count_time_with_decode"
MODULE_PATH = (
    "scripts/official/v1_2/"
    "regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsampling_psuniform_cntmask.py"
)

# Both OlmoEarthDataLoaderConfig and the train module configs carry a masking_config.
HasMaskingConfig = TypeVar("HasMaskingConfig")


def _apply_cntmask_strategy(config: HasMaskingConfig) -> HasMaskingConfig:
    """Swap the masking strategy type, keeping every other masking knob inherited."""
    config.masking_config.strategy_config["type"] = MASKING_STRATEGY_TYPE  # type: ignore[attr-defined]
    return config


def build_dataloader_config(common: CommonComponents):
    """The base's ndvi-aware newsampling/psuniform dataloader with the new masking."""
    return _apply_cntmask_strategy(_base_build_dataloader_config(common))


def build_train_module_config(common: CommonComponents):
    """The base's ndvi-aware train module with the new masking."""
    return _apply_cntmask_strategy(_base_build_train_module_config(common))


def build_trainer_config(common: CommonComponents):
    """Base trainer config + the in-loop evals routed through a Beaker job.

    Same eval set as the A/B partner (fifty_cities + the PASTIS embedding export),
    but pointed at THIS module path so the eval job reconstructs the matching config.
    """
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
