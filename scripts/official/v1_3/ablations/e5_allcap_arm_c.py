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

from base import build_visualize_config  # noqa: E402
from e5_allcap_common import (  # noqa: E402
    ALLCAP_H5PY_DIR,
    ALLCAP_KEEP_LIST,
    ALLCAP_RANDOM_CLEAR,
    ARM_B_CLOUD_TAU,
    build_common_components,
    build_model_config,
    build_trainer_config,
)
from e5_allcap_common import (  # noqa: E402
    build_dataloader_config as _e5_build_dataloader_config,
)
from e5_allcap_common import (  # noqa: E402
    build_train_module_config as _e5_build_train_module_config,
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
    """Arms A/B's E5 dataloader (same masking choice) with two masked views (the two date-draws)."""
    config = _e5_build_dataloader_config(common)
    config.num_masked_views = 2
    return config


def build_train_module_config(
    common: CommonComponents,
) -> TwoViewLatentMIMTrainModuleConfig:
    """Arms A/B's E5 train module, run per view, plus v1.2's contrastive loss.

    Inherits the RC pixel targets and E5's warmup and masking choice from e5_allcap_common.
    """
    rc = _e5_build_train_module_config(common)
    config = TwoViewLatentMIMTrainModuleConfig(
        **{f.name: getattr(rc, f.name) for f in fields(rc)}
    )
    # v1.2 base: InfoNCE (tau 0.1) with weight 0.05.
    config.contrastive_config = _v1_2_build_train_module_config(
        common
    ).contrastive_config
    assert config.contrastive_config is not None
    return config


# Trainer: arms A/B's E5 trainer unchanged (same steps; permanent checkpoints every 10k,
# latest every 1k). Arm C does ~2x B's per-step compute; the headline comparison is C vs B at
# matched steps (Ilse's review), with the 10k/20k checkpoints as checks.


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
