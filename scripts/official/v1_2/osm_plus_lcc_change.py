r"""Launch script: osm_sampling + lcc_change self-supervised pretraining.

Trains the v1.2-faster model on a concatenation of the global ``osm_sampling``
dataset and the ``lcc_change`` dataset. The ``lcc_change`` samples are 128x128
windows where the LCC change model predicts a real land-cover change, materialized
as merged pre/post series (six ~30-day mosaics before the change and six after; see
``olmoearth_pretrain/open_set_segmentation_data/sample_lcc_change.py``). Because
nearly every sample contains a real change between the two halves of the series,
the temporal part of the ``random_time_with_decode`` masking objective should be
substantially harder than on ``osm_sampling`` alone.

This is purely self-supervised: no open-set probe / label layers are loaded (the
``lcc_change`` dataset has no open-set labels; it only reuses the open-set
pipeline so that the pre/post windows are materialized and merged). The model,
train module, dataloader, and masking are identical to ``base_faster.py``.

Modalities: ``common.training_modalities`` is the v1.2 base list. The
``lcc_change`` H5s were built without CDL, so CDL is missing-filled for those
samples (as with any modality absent from an H5 dir) and contributes nothing for
them; ``osm_sampling`` samples still carry CDL.

Usage (from the repo root)::

    python scripts/official/v1_2/osm_plus_lcc_change.py launch osm_plus_lcc_change \\
        ai2/jupiter --launch.num_gpus=8
"""

import logging

from base import build_dataset_config as build_osm_dataset_config
from base_faster import (
    LOOP_EVAL_CLUSTERS,
    build_common_components,
    build_dataloader_config,
    build_model_config,
    build_visualize_config,
)
from base_faster import (
    build_train_module_config as _base_faster_build_train_module_config,
)
from base_faster import build_trainer_config as _base_faster_build_trainer_config

from olmoearth_pretrain.data.concat import OlmoEarthConcatDatasetConfig
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.internal.experiment import CommonComponents, main

logger = logging.getLogger(__name__)

# Path (relative to the repo root) used by the in-loop Beaker eval jobs to rebuild
# this exact model config when loading a checkpoint.
MODULE_PATH = "scripts/official/v1_2/osm_plus_lcc_change.py"

WANDB_PROJECT = "2026_09_19_osm_plus_lcc_change"

# H5 directory of the lcc_change dataset: 100k merged pre/post change samples, one
# H5 per 128x128 window (the ..._128_x_1 layout).
LCC_CHANGE_H5_DIR = "/weka/dfive-default/helios/dataset/lcc_change/h5py_data_w_missing_timesteps_zstd_3_128_x_1/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/100000"

# base_faster runs at ~90% VRAM with rank_microbatch_size=64; this variant OOMed at 64,
# so halve it. The global batch size (512) is unchanged, so this only adds grad
# accumulation steps and does not change the optimization.
RANK_MICROBATCH_SIZE = 32


def build_train_module_config(common: CommonComponents):
    """base_faster train module config with a smaller per-rank microbatch."""
    config = _base_faster_build_train_module_config(common)
    config.rank_microbatch_size = RANK_MICROBATCH_SIZE
    return config


def build_lcc_change_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Dataset config for the lcc_change H5s (same modality list as osm_sampling)."""
    return OlmoEarthDatasetConfig(
        h5py_dir=LCC_CHANGE_H5_DIR,
        training_modalities=common.training_modalities,
    )


def build_dataset_config(common: CommonComponents) -> OlmoEarthConcatDatasetConfig:
    """Concatenated osm_sampling + lcc_change dataset.

    Plain concatenation, so samples are drawn in proportion to dataset size
    (~1.14M osm_sampling vs 100k lcc_change).
    """
    return OlmoEarthConcatDatasetConfig(
        dataset_configs=[
            build_osm_dataset_config(common),
            build_lcc_change_dataset_config(common),
            build_lcc_change_dataset_config(common),
            build_lcc_change_dataset_config(common),
            build_lcc_change_dataset_config(common),
        ],
    )


def build_trainer_config(common: CommonComponents):
    """base_faster trainer config with the eval jobs pointed at this script.

    The eval job rebuilds the model from ``beaker_eval_module_path``; the model is
    identical to base_faster's, but pointing it here keeps the checkpoint <-> config
    association explicit.
    """
    trainer_config = _base_faster_build_trainer_config(common)
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    evaluator.beaker_eval_module_path = MODULE_PATH
    evaluator.beaker_eval_clusters = list(LOOP_EVAL_CLUSTERS)
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
    return trainer_config


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )
