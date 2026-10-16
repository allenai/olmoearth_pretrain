"""Shared builders: v1.3 (``rc_ld1_pixtgt_pix512``) on AllCaptures with a narrow stage.

Model: the released v1.3 model with a narrow within-cell stage in front of its 12 ViT
blocks (``SlotMergeConfig`` with ``narrow_depth`` / ``narrow_dim``): tokens are
projected down, attend only within their patch cell over every timestep and
modality, are merged per cell into ``max(ceil(n / 3), min(n, 4))`` slots capped at
24, and projected back up (plus a slot-pooled encoder-width skip) before the 12
encoder-width blocks and the 1-layer Perceiver. Built so every-capture inputs
(100-300 tokens per cell) cost about what v1.3 costs on monthly ones.

Data: joer's AllCaptures corpus (every S2 / S1 / Landsat C2 L2 capture over 360 days,
per-modality timestamps, 9000 real-token budget, random time-range lengths), and per
rank batch a composite mode drawn from ``COMPOSITE_DAYS``: every capture, biweekly
or monthly per-pixel medians (S2 over SCL-clear pixels).

Evals (in-loop, every ``STUDENT_LOOP_EVAL_INTERVAL_STEPS``, d128 student, patch size
2, one latent per pixel): m-eurosat; PASTIS monthly and every-capture (S1+S2); and
africa_crop_mask / ethiopia_crops (AEF trials) monthly S1+S2, every-capture S1+S2,
and monthly S1+S2+Landsat-L2 (monthly medians of the twins' every L2 capture).
"""

import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

# Every rank batch has a different (grid, timeline) shape; expandable segments keep
# the caching allocator from fragmenting (joer's allcap runs). Set before CUDA init.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (
    STUDENT_LOOP_EVAL_INTERVAL_STEPS,  # noqa: E402
    route_loop_evals_through_beaker,  # noqa: E402
)
from base import build_common_components as _v1_3_common  # noqa: E402
from base import build_dataloader_config as _v1_3_dataloader  # noqa: E402
from olmo_core.train.common import Duration  # noqa: E402
from rc_ld1_pix512 import build_model_config as _rc_ld1_model_config  # noqa: E402
from rc_pixtgt_pix512 import (  # noqa: E402
    build_train_module_config as _pixtgt_train_module_config,
)
from v1_2.base import build_trainer_config as _v1_2_build_trainer_config  # noqa: E402

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig  # noqa: E402
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig  # noqa: E402
from olmoearth_pretrain.internal.all_evals import EMBEDDING_EVAL_TASKS  # noqa: E402
from olmoearth_pretrain.internal.experiment import (  # noqa: E402
    CommonComponents,
    SubCmd,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from olmoearth_pretrain.nn.slot_merge import SlotMergeConfig  # noqa: E402

logger = logging.getLogger(__name__)

WANDB_PROJECT = "20261016_allcap_slot_narrow"

ALLCAP_H5PY_DIR = (
    "/weka/dfive-default/helios/dataset/osm_allcaptures/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_landsat_l2_openstreetmap_raster_sentinel1_sentinel2_l2a_sentinel2_scl_"
    "srtm_worldcereal_worldcover_wri_canopy_height_map/395660"
)
# joer's real-token budget and time-range menu (the corpus spans 360 days, so the
# last three entries take the whole span).
TOKEN_BUDGET = 9000
TIME_RANGE_DAYS = [7.0, 30.0, 90.0, 180.0, 365.0, 730.0, 1825.0]
# Composite bin length per rank batch: 0 = every capture, 14 = biweekly, 30 = monthly.
COMPOSITE_DAYS = [0.0, 14.0, 30.0]
RANK_MICROBATCH_SIZE = 32

SLOT_RATIO = 3
SLOT_MIN = 4
SLOT_CAP = 24

S1_S2 = [Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name]
S1_S2_L2 = [*S1_S2, Modality.LANDSAT_L2.name]
AEF_DATASETS = ("africa_crop_mask", "ethiopia_crops")


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """v1.3's components with Landsat C2 Level-2 (the corpus's Landsat)."""
    config = _v1_3_common(script, cmd, run_name, cluster, overrides)
    config.training_modalities = [
        Modality.LANDSAT_L2.name if m == Modality.LANDSAT.name else m
        for m in config.training_modalities
    ]
    return config


def build_narrow_model_config(
    common: CommonComponents, narrow_depth: int, narrow_dim: int
) -> LatentMIMConfig:
    """rc_ld1_pixtgt_pix512's model with the narrow stage + capped slot merge."""
    config = _rc_ld1_model_config(common)
    config.encoder_config.slot_merge_config = SlotMergeConfig(
        merge_after=0,
        ratio=SLOT_RATIO,
        min_slots=SLOT_MIN,
        max_slots_per_cell=SLOT_CAP,
        iters=3,
        narrow_depth=narrow_depth,
        narrow_dim=narrow_dim,
    )
    return config


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """The AllCaptures corpus with per-modality timestamps and SCL (for composites)."""
    return OlmoEarthDatasetConfig(
        h5py_dir=ALLCAP_H5PY_DIR,
        training_modalities=common.training_modalities,
        per_modality_timestamps=True,
        load_s2_cloud_mask=True,
    )


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """v1.3's sampler with the real-token budget, time ranges and composite modes."""
    config = _v1_3_dataloader(common)
    config.token_budget = TOKEN_BUDGET
    config.time_range_days_choices = list(TIME_RANGE_DAYS)
    config.composite_days_choices = list(COMPOSITE_DAYS)
    config.tokenization_config = common.tokenization_config
    # Rank batches of every-capture samples are large; joer's settings that kept
    # /dev/shm from filling (worker SIGBUS) and avoided per-batch cudaHostAlloc.
    config.num_workers = 4
    config.prefetch_factor = 1
    config.pin_memory = False
    return config


def build_train_module_config(common: CommonComponents):
    """rc_pixtgt_pix512's train module (shared pixel targets) at a smaller microbatch."""
    config = _pixtgt_train_module_config(common)
    config.rank_microbatch_size = RANK_MICROBATCH_SIZE
    return config


def _ps2_student(task, **changes):
    return replace(
        task,
        patch_size=2,
        eval_on_student_registers=True,
        eval_student_dim=128,
        eval_interval=Duration.steps(STUDENT_LOOP_EVAL_INTERVAL_STEPS),
        **changes,
    )


def allcap_eval_tasks(catalog: dict) -> dict:
    """The eval set, PASTIS first (the trailing tasks lose metrics on preemption)."""
    tasks = {}
    pastis = EMBEDDING_EVAL_TASKS[
        "pastis_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat"
    ]
    tasks["pastis_year_aligned_ws16_ps2_sentinel1_sentinel2_proj128"] = _ps2_student(
        pastis, input_modalities=S1_S2
    )
    tasks["pastis_year_aligned_ws16_ps2_sentinel1_sentinel2_everycapture_proj128"] = (
        _ps2_student(pastis, input_modalities=S1_S2, every_capture=True)
    )
    for ds in AEF_DATASETS:
        base = EMBEDDING_EVAL_TASKS[
            f"{ds}_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat_knn"
        ]
        prefix = f"{ds}_year_aligned_ws16_ps2"
        tasks[f"{prefix}_sentinel1_sentinel2_knn_proj128"] = _ps2_student(
            base, input_modalities=S1_S2
        )
        tasks[f"{prefix}_sentinel1_sentinel2_everycapture_knn_proj128"] = _ps2_student(
            base, input_modalities=S1_S2, every_capture=True
        )
        tasks[f"{prefix}_sentinel1_sentinel2_landsatl2_knn_proj128"] = _ps2_student(
            base, input_modalities=S1_S2_L2, twin_landsat_l2=True
        )
    tasks["m-eurosat_ps2_proj128"] = _ps2_student(catalog["m-eurosat"])
    return tasks


def build_allcap_trainer_config(common: CommonComponents, module_path: str):
    """v1.2 trainer, this eval set routed through Beaker, own W&B project."""
    trainer_config = _v1_2_build_trainer_config(common)
    catalog = trainer_config.callbacks["downstream_evaluator"].tasks
    trainer_config = route_loop_evals_through_beaker(
        trainer_config, module_path, allcap_eval_tasks(catalog)
    )
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
    return trainer_config
