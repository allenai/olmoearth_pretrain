"""Shared config for experiment E5: v1.3 RC recipe on the every-capture (allcap) corpus.

E5 asks *which 12 captures, with which dates* at a fixed 12-slot token shape: both arms
read the SAME allcap H5 files (``osm_allcaptures``, eval-clean keep-list) through
``olmoearth_pretrain.data.allcap``, S2 + S1 only (Landsat dropped in both, since allcap
has only Level-2 and v1.3 uses Level-1), with 12 slots of 30 days:

* arm A (``e5_allcap_arm_a.py``): per slot the least-cloudy S2 capture and the S1
  capture nearest the slot midpoint, stamped at slot start (v1.3's mosaic rule on
  single captures);
* arm B (``e5_allcap_arm_b.py``): per read, up to 12 random S2 captures with cloud
  fraction <= ``ARM_B_CLOUD_TAU`` with their real dates (tau matched so the mean input
  cloud fraction equals arm A's), S1 paired within +/-15 d, S1-only slots if fewer than
  12 clear S2.

Everything else is ``rc_ld1_pixtgt_pix512.py`` (``v1_3_rc_ld1_pixtgt_pix512``): model
(1-layer register Perceiver, 512-latent budget, d768 supervision, [128, 64] student),
pixel MIM targets, the decorrelated shape sampler with TOKEN_BUDGET=3072, batch 512,
seed 3622, lr 1e-4 + 8k warmup cosine. Differences from the RC, all shared by both arms:
60k steps (cosine anneals at 60k), no Landsat, ~387k samples instead of 1.14M, and no
in-loop evals (those run as Beaker jobs that clone the pushed git ref; this branch is
local, so checkpoints are scored offline at 20k/40k/60k).
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    build_common_components as _v1_3_build_common_components,
)
from base import (  # noqa: E402
    build_dataloader_config,
    build_visualize_config,
)
from olmo_core.train.callbacks import CheckpointerCallback  # noqa: E402
from olmo_core.train.common import Duration  # noqa: E402
from rc_ld1_pix512 import build_model_config  # noqa: E402
from rc_pixtgt_pix512 import build_train_module_config  # noqa: E402
from v1_2.base import build_trainer_config as _v1_2_build_trainer_config  # noqa: E402

from olmoearth_pretrain.data.allcap import (  # noqa: E402
    ALLCAP_LEAST_CLOUDY_SLOT,
    ALLCAP_RANDOM_CLEAR,
)
from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, main  # noqa: E402

logger = logging.getLogger(__name__)

ALLCAP_H5PY_DIR = (
    "/weka/dfive-default/helios/dataset/osm_allcaptures/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_landsat_l2_openstreetmap_raster_sentinel1_sentinel2_l2a_sentinel2_scl_srtm_"
    "worldcereal_worldcover_wri_canopy_height_map/395660"
)
# J.F.'s eval-overlap keep-list (387,084 of 395,660 samples).
ALLCAP_KEEP_LIST = (
    "/weka/dfive-default/favyen/strata/filters/allcap_eval_clean_buf0_395660.npy"
)
# Calibrated on 3,000 random keep-list samples (e5_allcap_ab/results/tau.json) so arm
# B's mean input S2 cloud fraction matches arm A's.
ARM_B_CLOUD_TAU = 0.52
# Measured there: arm A mean S2 cloud 0.0738 (median 0.0, 6.2% of slots > 0.5),
# arm B at 0.52 0.0742 (median 0.003); S2 slots/sample 11.91 vs 11.94; S1 slots
# 11.24 vs 11.03; mean |S1 date - slot stamp| 14.6 d vs 2.0 d.

MAX_STEPS = 60_000
PERMANENT_SAVE_INTERVAL = 20_000
# ~26 min at the RC's 0.52 s/step.
EPHEMERAL_SAVE_INTERVAL = 3_000
WANDB_PROJECT = "20261016_e5_allcap_ab"


def build_common_components(
    script: str, cmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """v1.3 modalities minus Landsat (S2 + S1 + the six decode-only maps)."""
    config = _v1_3_build_common_components(script, cmd, run_name, cluster, overrides)
    config.training_modalities = [
        m for m in config.training_modalities if m != Modality.LANDSAT.name
    ]
    return config


def build_dataset_config_for(selection: str, tau: float = 1.0):
    """Dataset builder for one arm."""

    def _build(common: CommonComponents) -> OlmoEarthDatasetConfig:
        return OlmoEarthDatasetConfig(
            h5py_dir=ALLCAP_H5PY_DIR,
            training_modalities=common.training_modalities,
            filter_idx_file=ALLCAP_KEEP_LIST,
            allcap_selection=selection,
            allcap_cloud_tau=tau,
        )

    return _build


def build_trainer_config(common: CommonComponents):
    """v1.2 trainer: 60k steps, saves every 20k + latest every 3k, no in-loop evals."""
    config = _v1_2_build_trainer_config(common)
    config.max_duration = Duration.steps(MAX_STEPS)
    config.callbacks.pop("downstream_evaluator")
    config.callbacks["wandb"].project = WANDB_PROJECT
    config.callbacks["checkpointer"] = CheckpointerCallback(
        save_interval=PERMANENT_SAVE_INTERVAL,
        ephemeral_save_interval=EPHEMERAL_SAVE_INTERVAL,
        # Keep every permanent (20k/40k/60k) checkpoint, as the RC launch did.
        max_checkpoints=None,
    )
    return config


def run_arm(selection: str, tau: float = 1.0) -> None:
    """Run one arm."""
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config_for(selection, tau),
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )


__all__ = [
    "ALLCAP_LEAST_CLOUDY_SLOT",
    "ALLCAP_RANDOM_CLEAR",
    "ARM_B_CLOUD_TAU",
    "run_arm",
]
