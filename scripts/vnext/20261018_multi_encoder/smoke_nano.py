"""SMOKE TEST: nano-size towers arm (t2/f2 + modality dropout) for a 1-GPU functional check.

Not an experiment arm. It exercises every new piece at once (MultiEncoder towers, fusion
blocks, the AttentionPooler read over tower outputs, the ``modality_dropout`` masking,
the supervision + distillation heads) at nano size so a single GPU gets through a few
hundred steps quickly. Point it at the 1k subset and disable the in-loop evals, e.g.::

    python scripts/vnext/20261018_multi_encoder/smoke_nano.py train_single smoke local \
        --dataset.h5py_dir=/weka/dfive-default/helios/dataset/osm_sampling_subset_1k/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/3996 \
        --data_loader.global_batch_size=32 --train_module.rank_microbatch_size=8 \
        --trainer.max_duration.value=200 --trainer.max_duration.unit=steps \
        --trainer.callbacks.downstream_evaluator.enabled=False \
        --trainer.callbacks.wandb.enabled=False
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from base import (  # noqa: E402
    build_common_components,
    build_dataset_config,
    build_multi_encoder_model_config,  # noqa: E402
    build_trainer_config,
    build_visualize_config,
)
from base import build_dataloader_config as _build_dataloader_config  # noqa: E402
from base import build_train_module_config as _build_train_module_config  # noqa: E402

from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from olmoearth_pretrain.train.train_module.latent_mim import (  # noqa: E402
    LatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/vnext/20261018_multi_encoder/smoke_nano.py"
MODALITY_DROPOUT = True

SIZE_NAME = "nano"
TOWER_DEPTH = 2
FUSION_DEPTH = 2


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Nano v1.3 with per-modality towers (2) + shared blocks (2)."""
    return build_multi_encoder_model_config(
        common, tower_depth=TOWER_DEPTH, fusion_depth=FUSION_DEPTH, size_name=SIZE_NAME
    )


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """The shared dataloader, with this arm's masking."""
    return _build_dataloader_config(common, modality_dropout=MODALITY_DROPOUT)


def build_train_module_config(common: CommonComponents) -> LatentMIMTrainModuleConfig:
    """The shared train module, with this arm's masking."""
    return _build_train_module_config(common, modality_dropout=MODALITY_DROPOUT)


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
