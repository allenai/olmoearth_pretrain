"""base_faster + cloud-aware patch discrimination, instance-contrastive OFF.

Same 192-d smaller-embedding model/data as ``base_faster``, with two deltas:
  * The instance-contrastive (InfoNCE) loss is disabled (``contrastive_config=None``).
  * The patch-discrimination loss skips target (decoder) tokens that are mostly
    cloud, using precomputed OmniCloudMask cloud masks. The dataset is pointed at
    the cloud sidecar via ``cloud_cache_dir`` (see
    ``olmoearth_pretrain.data.cloud_mask_cache``); the masking strategy consumes it
    to drop cloudy decoder tokens (threshold ``CLOUD_SKIP_THRESHOLD``).
"""

import logging

from base import (
    build_common_components,
    build_dataloader_config,
    build_model_config,
    build_visualize_config,
)
from base import build_dataset_config as _base_build_dataset_config
from base_faster import build_train_module_config as _faster_build_train_module_config
from base_faster import make_build_trainer_config

from olmoearth_pretrain.data.cloud_mask_cache import default_cache_dir
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.internal.experiment import CommonComponents, main
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/vnext/2026_07_22_cloud_mask/base_faster_cloud.py"
# The per-token cloud-skip fraction lives on the masking strategy
# (RandomTimeWithDecodeMaskingStrategy.cloud_skip_threshold, default 0.5); to
# change it, add "cloud_skip_threshold": <x> to the masking strategy_config.


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """base_faster train module with the instance-contrastive loss removed."""
    config = _faster_build_train_module_config(common)
    config.contrastive_config = None
    return config


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Base dataset pointed at the precomputed cloud-mask sidecar."""
    config = _base_build_dataset_config(common)
    config.cloud_cache_dir = default_cache_dir(config.h5py_dir)
    return config


def build_trainer_config(common: CommonComponents):
    """Shared 4 in-loop evals, run as beaker jobs (via base_faster)."""
    return make_build_trainer_config(MODULE_PATH)(common)


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
