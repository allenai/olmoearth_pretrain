"""trope_simple_temporal + 50% single-timestep collapse.

Identical to `trope_simple_temporal.py` (3D RoPE-Mixed over (t, row, col) +
separate-path simple temporal encoding + year dropout, no latlon/class
token/CLIP, standard v1.1 loss) but with a data augmentation: with prob 0.5
a batch is collapsed so every sample keeps only a single randomly-chosen
present timestep, regardless of how many timesteps are available or fit in the
token budget.

Rationale: many downstream tasks are single-timestep (one image, no time
series). Training half the batches as single-timestep should make the model's
representation more robust when only one observation is available, without
giving up the multi-timestep signal on the other half. Controlled by the new
`single_timestep_prob` config option on OlmoEarthDataLoaderConfig.

The decision is made per batch (not per sample) so the time dimension stays
uniform within a batch and the collator can stack — patch_size/sampled_hw_p
are already drawn per batch for the same reason.
"""

import logging

from base import (
    build_common_components,
    build_dataset_config,
    build_train_module_config,
    build_trainer_config,
    build_visualize_config,
)

# Reuse the trope_simple_temporal builders unchanged; only the dataloader differs.
from trope_simple_temporal import (
    build_dataloader_config as build_dataloader_config_trope,
)
from trope_simple_temporal import (
    build_model_config,
)

from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.internal.experiment import CommonComponents, main

logger = logging.getLogger(__name__)

# Fraction of batches collapsed to a single (random, present) timestep per sample.
SINGLE_TIMESTEP_PROB = 0.5


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """trope_simple_temporal dataloader (year dropout) + single-timestep collapse."""
    config = build_dataloader_config_trope(common)
    config.single_timestep_prob = SINGLE_TIMESTEP_PROB
    return config


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
