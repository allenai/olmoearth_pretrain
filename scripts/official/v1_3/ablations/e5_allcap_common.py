"""Shared config for experiment E5: v1.3 RC recipe on the every-capture (allcap) corpus.

E5 asks *which 12 captures, with which dates* at a fixed 12-slot token shape: both arms
read the SAME allcap H5 files (``osm_allcaptures``, eval-clean keep-list) through
``olmoearth_pretrain.data.allcap``, with 12 slots of 30 days.

Decision 2026-10-16 13:55Z (Hollis): **S1 + S2 with the unmodified v1.3 objective** (Landsat
dropped in both arms: allcap has only Level-2 and v1.3 uses Level-1), 30k steps, 4k warmup.

* arm A (``e5_allcap_arm_a.py``): per slot the least-cloudy S2 capture and the S1
  capture nearest the slot midpoint, stamped at slot start (v1.3's mosaic rule on
  single captures);
* arm B (``e5_allcap_arm_b.py``): per read, up to 12 random S2 captures with cloud
  fraction <= ``ARM_B_CLOUD_TAU`` with their real dates (tau matched so the mean input
  cloud fraction equals arm A's), S1 paired within +/-15 d, S1-only slots if fewer than
  12 clear S2.

**Fallback switches (env vars; both unset = the decision above).**

* ``E5_S2_ONLY=1``: drop Sentinel-1 too (the S1 stack is never read: ~39% fewer bytes per
  sample). With S2 as the only (single-bandset) sensor, v1.3's ``random_time_with_decode``
  makes S2 encode-only and decodes only the maps, i.e. no sensor latent target.
* ``E5_MIM``: unset / ``v1_3`` (= ``maps_only``, the same code) is v1.3's masking exactly;
  ``within_s2`` sets ``single_sensor_split=True`` (a lone sensor's tokens are split 50/50
  into encoder inputs and latent targets: random tokens, or encode vs decode timesteps).
  ``within_s2`` requires ``E5_S2_ONLY=1``: with S1 + S2 it would change only the
  samples that happen to have a single sensor, which is not a meaningful variant.
  Hollis's canary fallback is ``E5_S2_ONLY=1 E5_MIM=within_s2``.

Everything else is ``rc_ld1_pixtgt_pix512.py`` (``v1_3_rc_ld1_pixtgt_pix512``): model
(1-layer register Perceiver, 512-latent budget, d768 supervision, [128, 64] student),
pixel MIM targets, the decorrelated shape sampler with TOKEN_BUDGET=3072, batch 512,
seed 3622, lr 1e-4 cosine (alpha_f 0.1). Differences from the RC, all shared by both
arms: 30k steps with a 4k warmup (cosine anneals at 30k), no Landsat, ~387k samples
instead of 1.14M, and no in-loop evals (those run as Beaker jobs that clone the pushed git
ref; this branch is local, so checkpoints are scored offline at 10k/20k/30k).
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    build_common_components as _v1_3_build_common_components,
)
from base import (  # noqa: E402
    build_dataloader_config as _v1_3_build_dataloader_config,
)
from base import build_visualize_config  # noqa: E402
from olmo_core.train.callbacks import CheckpointerCallback  # noqa: E402
from olmo_core.train.common import Duration  # noqa: E402
from rc_ld1_pix512 import build_model_config as _rc_ld1_build_model_config  # noqa: E402
from rc_pixtgt_pix512 import (  # noqa: E402
    build_train_module_config as _rc_pixtgt_build_train_module_config,
)
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

MAX_STEPS = 30_000
# The RC's 8k warmup was set for a ~667k-step run (1.2%). The approved 60k-step E5 kept
# 8k (13%); 4k keeps that fraction at 30k, so the 10k checkpoint is past warmup by 6k
# steps instead of 2k. 4k x 512 = 2M samples is ample for AdamW at lr 1e-4.
WARMUP_STEPS = 4_000
PERMANENT_SAVE_INTERVAL = 10_000
# ~30-60 min at the expected 2-4 s/step of a loader-bound 4-GPU arm.
EPHEMERAL_SAVE_INTERVAL = 1_000
WANDB_PROJECT = "20261016_e5_allcap_ab"

MIM_MODES = ("v1_3", "maps_only", "within_s2")


def s2_only() -> bool:
    """``E5_S2_ONLY=1`` drops Sentinel-1 (fallback; default S1 + S2)."""
    v = os.environ.get("E5_S2_ONLY", "")
    if v not in ("", "0", "1"):
        raise ValueError(f"E5_S2_ONLY must be unset, 0 or 1 (got {v!r})")
    return v == "1"


def mim_mode() -> str:
    """``E5_MIM``: unset / v1_3 / maps_only (all v1.3's masking) or within_s2."""
    mode = os.environ.get("E5_MIM", "") or "v1_3"
    if mode not in MIM_MODES:
        raise ValueError(
            f"E5_MIM must be unset or one of {MIM_MODES} (got {mode!r}); see e5_allcap_common.py"
        )
    if mode == "within_s2" and not s2_only():
        raise ValueError("E5_MIM=within_s2 requires E5_S2_ONLY=1")
    return mode


def _apply_mim_mode(masking_config) -> None:  # noqa: ANN001
    if mim_mode() == "within_s2":
        masking_config.strategy_config["single_sensor_split"] = True


def build_common_components(
    script: str, cmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """v1.3 modalities minus Landsat (S2 + S1 + the six maps); minus S1 too if E5_S2_ONLY=1."""
    config = _v1_3_build_common_components(script, cmd, run_name, cluster, overrides)
    dropped = {Modality.LANDSAT.name}
    if s2_only():
        dropped.add(Modality.SENTINEL1.name)
    config.training_modalities = [
        m for m in config.training_modalities if m not in dropped
    ]
    mim_mode()  # validate the env vars early
    return config


def build_train_module_config(common: CommonComponents):
    """``rc_pixtgt_pix512``'s train module; 4k warmup; the E5_MIM masking choice."""
    config = _rc_pixtgt_build_train_module_config(common)
    config.scheduler.warmup = WARMUP_STEPS
    _apply_mim_mode(config.masking_config)
    return config


def build_dataloader_config(common: CommonComponents):
    """v1.3 dataloader (masking runs here) with the E5_MIM masking choice."""
    config = _v1_3_build_dataloader_config(common)
    _apply_mim_mode(config.masking_config)
    return config


def build_model_config(common: CommonComponents):
    """``rc_ld1_pixtgt_pix512``'s model; band dropout only on modalities that exist."""
    config = _rc_ld1_build_model_config(common)
    enc = config.encoder_config
    enc.band_dropout_modalities = [
        m for m in enc.band_dropout_modalities if m in common.training_modalities
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
    """v1.2 trainer: 30k steps, saves every 10k + latest every 1k, no in-loop evals."""
    config = _v1_2_build_trainer_config(common)
    config.max_duration = Duration.steps(MAX_STEPS)
    config.callbacks.pop("downstream_evaluator")
    config.callbacks["wandb"].project = WANDB_PROJECT
    config.callbacks["checkpointer"] = CheckpointerCallback(
        save_interval=PERMANENT_SAVE_INTERVAL,
        ephemeral_save_interval=EPHEMERAL_SAVE_INTERVAL,
        # Keep every permanent (10k/20k/30k) checkpoint, as the RC launch did.
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
