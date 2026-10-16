"""E5 arm C: arm B plus a second date-draw per sample and an instance-contrastive term.

Arm B (``e5_allcap_arm_b.py``: ``random_clear`` with tau = ``ARM_B_CLOUD_TAU``, real
capture dates, 12 slots, S2 + S1, no Landsat, the ``rc_ld1_pixtgt_pix512`` recipe, 60k
steps; see ``e5_allcap_common.py``) with ONE change: every sample yields TWO
independent draws of its captures (views a and b) from one H5 read, sharing the shape
(patch size, grid, number of timesteps), the spatial crop and the static/decode-only
maps. Each view gets the RC's own pixel-target latent-MIM loss (plus supervision heads
and the detached student), the two are averaged, and v1.2's instance-contrastive loss
(InfoNCE, weight 0.05, in-microbatch negatives) is added between the two views' pooled
embeddings (``TwoViewLatentMIMTrainModule``). The question: does TESSERA-style
invariance across date subsets of the same pixel-year help?

Cost: two forward/backward passes per sample, so ~2x arm B's per-step compute and
activation memory at the same batch size (512) and step count (60k).

Run name ``agent-gandalf-e5-armC-20261016`` (W&B ``eai-ai2/20261016_e5_allcap_ab``).
"""

import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import build_dataloader_config as _v1_3_build_dataloader_config  # noqa: E402
from base import build_visualize_config  # noqa: E402
from e5_allcap_common import (  # noqa: E402
    ALLCAP_H5PY_DIR,
    ALLCAP_KEEP_LIST,
    ALLCAP_RANDOM_CLEAR,
    ARM_B_CLOUD_TAU,
    build_common_components,
    build_model_config,
)
from e5_allcap_common import (
    build_trainer_config as _e5_build_trainer_config,
)
from rc_pixtgt_pix512 import (  # noqa: E402
    build_train_module_config as _rc_build_train_module_config,
)
from v1_2.base import (  # noqa: E402
    build_train_module_config as _v1_2_build_train_module_config,
)

from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig  # noqa: E402
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402
from olmoearth_pretrain.train.train_module.two_view_latent_mim import (  # noqa: E402
    TwoViewLatentMIMTrainModuleConfig,
)


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Arm B's dataset, yielding two independent date-draws per sample."""
    return OlmoEarthDatasetConfig(
        h5py_dir=ALLCAP_H5PY_DIR,
        training_modalities=common.training_modalities,
        filter_idx_file=ALLCAP_KEEP_LIST,
        allcap_selection=ALLCAP_RANDOM_CLEAR,
        allcap_cloud_tau=ARM_B_CLOUD_TAU,
        allcap_two_draws=True,
    )


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """The v1.3 dataloader with two masked views (the two date-draws)."""
    config = _v1_3_build_dataloader_config(common)
    config.num_masked_views = 2
    return config


def build_train_module_config(
    common: CommonComponents,
) -> TwoViewLatentMIMTrainModuleConfig:
    """The RC's pixel-target train module, run per view, + v1.2's contrastive loss."""
    rc = _rc_build_train_module_config(common)
    config = TwoViewLatentMIMTrainModuleConfig(
        **{f.name: getattr(rc, f.name) for f in fields(rc)}
    )
    # v1.2 base: InfoNCE (tau 0.1) with weight 0.05.
    config.contrastive_config = _v1_2_build_train_module_config(
        common
    ).contrastive_config
    assert config.contrastive_config is not None
    return config


# Arm C does ~2x arm B's per-step compute (two forward/backward passes), so keep a
# permanent checkpoint every 10k (not 20k): C@30k is the compute-matched point for B@60k.
ARM_C_PERMANENT_SAVE_INTERVAL = 10_000


def build_trainer_config(common: CommonComponents):
    """E5 trainer (60k steps, no in-loop evals) with permanent saves every 10k."""
    config = _e5_build_trainer_config(common)
    config.callbacks["checkpointer"].save_interval = ARM_C_PERMANENT_SAVE_INTERVAL
    return config


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
