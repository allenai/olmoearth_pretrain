"""Shared builders: v1.3 (``rc_ld1_pixtgt_pix512``) on AllCaptures with a narrow stage.

Model: the released v1.3 model with a narrow within-cell stage in front of its 12 ViT
blocks (``SlotMergeConfig`` with ``narrow_depth`` / ``narrow_dim``): tokens are
projected down, attend only within their patch cell over every timestep and
modality, are merged per cell into one slot per calendar month (at least
``min(n, 12)``, at most 24; cells of <= 12 tokens bypass), and projected back up (plus a slot-pooled encoder-width skip) before the 12
encoder-width blocks and the 1-layer Perceiver. Built so every-capture inputs
(100-300 tokens per cell) cost about what v1.3 costs on monthly ones.

Data: joer's AllCaptures corpus (every S2 / S1 / Landsat C2 L2 capture over 360 days,
per-modality timestamps, v1.3's real-token budget, random time ranges of 180 / 365
days), and per rank batch a mode drawn from ``COMPOSITE_DAYS``: a contiguous run of
every capture, biweekly or monthly composites (one clear capture per bin), or a random
set of captures over the whole year (``THIN_CAPTURES``).

Evals (in-loop, every ``STUDENT_LOOP_EVAL_INTERVAL_STEPS``, d128 student, patch size
2, one latent per pixel): m-eurosat; PASTIS monthly and every-capture (S1+S2); and
africa_crop_mask / ethiopia_crops (AEF trials) monthly S1+S2, every-capture S1+S2,
monthly S1+S2+Landsat-L2 (monthly medians of the twins' every L2 capture) and
every-capture S1+S2+Landsat-L2; PASTIS also monthly and every-capture S1+S2+L2.
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
from v1_2.base import S2_SINGLE_BANDSET  # noqa: E402
from v1_2.base import build_trainer_config as _v1_2_build_trainer_config  # noqa: E402

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig  # noqa: E402
from olmoearth_pretrain.data.dataset import (  # noqa: E402
    THIN_CAPTURES,
    OlmoEarthDatasetConfig,
)
from olmoearth_pretrain.internal.all_evals import EMBEDDING_EVAL_TASKS  # noqa: E402
from olmoearth_pretrain.internal.experiment import (  # noqa: E402
    CommonComponents,
    SubCmd,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from olmoearth_pretrain.nn.slot_merge import SlotMergeConfig  # noqa: E402
from olmoearth_pretrain.nn.tokenization import TokenizationConfig  # noqa: E402
from olmoearth_pretrain.train.callbacks.rank_timing import (  # noqa: E402
    RankTimingCallback,
)

logger = logging.getLogger(__name__)

WANDB_PROJECT = "20261016_allcap_slot_narrow"

# The 32 x 32 x 16-chunked copy of osm_allcaptures (same arrays; the original stores one
# chunk per array, so a crop decompressed the whole tile; ~4x less h5 time, 2026-10-19).
ALLCAP_H5PY_DIR = (
    "/weka/dfive-default/helios/dataset/osm_allcaptures_rechunk32x32x16/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_landsat_l2_openstreetmap_raster_sentinel1_sentinel2_l2a_sentinel2_scl_"
    "srtm_worldcereal_worldcover_wri_canopy_height_map/395660"
)
# Real-token budget per sample = v1.3's (3072). joer's runs use 9000; at microbatch 32
# that OOMed (the patch embedding, composite encodings and frozen target run over every
# input token at width 768), and 4500 OOMed on one rank of eight at step 33 (Gabi,
# 2026-10-16). Time ranges: >= 180 days. joer's menu (7 / 30 / 90 / 180 / 365 / 730 /
# 1825 days) left the median sample spanning 2 calendar months in every mode (v1.3: 9;
# measure_time_coverage.py, 2026-10-22). Every tile spans ~360 days.
TOKEN_BUDGET = 3072
# v1.3's training length in steps: 300 epochs of v1.3's corpus. AllCaptures is ~35% the
# size, so v1.2's Duration.epochs(300) would stop at 231,600 steps; pinning the steps
# keeps the samples seen (and the cosine schedule) the same as v1.3.
MAX_STEPS = 667_200
TIME_RANGE_DAYS = [180.0, 365.0]
# Mode per rank batch: 0 = every capture, 14 = biweekly, 30 = monthly, THIN_CAPTURES =
# random captures over the whole year (median 12 months; the others' runs stop when the
# budget fills).
COMPOSITE_DAYS = [0.0, 14.0, 30.0, THIN_CAPTURES]
RANK_MICROBATCH_SIZE = 32
# Decoder targets per sample are capped near v1.3's 90th percentile (simulated v1.3
# sampler + masking: mean ~1.1k, median ~0.9k, p90 ~2.1k incl. maps), so the decoder
# costs what it does in v1.3 while the encoder sees every capture.
MAX_DECODE_TOKENS = 2048

# Slots per cell: max(min(months, SLOT_CAP), min(n, SLOT_MIN)) -- one per calendar
# month, never fewer than min(n, 12) (short dense spans), so the trunk sees one slot
# per month on full-year inputs. SLOT_RATIO is unused with month_length set.
SLOT_RATIO = 3
SLOT_MIN = 12
SLOT_CAP = 24
# Each worker reads whole h5 files into a pool and uses each one TILE_REUSE times
# (fresh crop / time range / mode draws): few large reads instead of ~100 small ones
# per sample, which capped a full node at ~600 samples/s (2026-10-21/22 benches).
TILE_REUSE = 10

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
    # landsat_l2 is one band set already; only S2 needs the single-band-set override.
    config.tokenization_config = TokenizationConfig(
        overrides={Modality.SENTINEL2_L2A.name: S2_SINGLE_BANDSET}
    )
    return config


def build_narrow_model_config(
    common: CommonComponents, narrow_depth: int, narrow_dim: int
) -> LatentMIMConfig:
    """rc_ld1_pixtgt_pix512's model with the narrow stage + capped slot merge."""
    config = _rc_ld1_model_config(common)
    # v1.2's band-dropout list names Landsat L1; the corpus carries L2.
    config.encoder_config.band_dropout_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.LANDSAT_L2.name,
    ]
    config.encoder_config.slot_merge_config = SlotMergeConfig(
        merge_after=0,
        ratio=SLOT_RATIO,
        min_slots=SLOT_MIN,
        max_slots_per_cell=SLOT_CAP,
        month_length=30.4375 * config.encoder_config.rope_temporal_coordinate_scale,
        iters=3,
        narrow_depth=narrow_depth,
        narrow_dim=narrow_dim,
    )
    return config


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """The AllCaptures corpus with per-modality timestamps and SCL (for composites).

    Composite bins are one real capture each (a random one under 20% cloud over the
    crop, S2), not per-pixel medians: on a full 8-GPU node the median compositing made
    the loaders CPU-bound, so every step waited on the slowest rank's data (~0.4-0.7 s,
    2026-10-20).
    """
    return OlmoEarthDatasetConfig(
        h5py_dir=ALLCAP_H5PY_DIR,
        training_modalities=common.training_modalities,
        per_modality_timestamps=True,
        load_s2_cloud_mask=True,
        composite_selection="random_clear",
        # Composite samples choose their budget run of bins before reading, so they
        # read ~2-3 MB instead of ~5 (loader ~30% faster on a full node, 2026-10-20).
        budget_before_read=True,
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
    # A worker builds a 64-sample every-capture batch in ~15 s (v1.3: <= ~6.6 s), and
    # per-batch cost varies 7-19 s with shape / composite mode. In-order delivery with
    # prefetch 1 left workers blocked behind slow batches (~21-30 s between a worker's
    # batches, 2026-10-19 bench); out of order + v1.3's 16 workers keeps the loader
    # ahead of the model.
    config.num_workers = 16
    config.in_order = False
    # A thread per worker reads the next samples from WEKA while the worker computes
    # the current one: on the full corpus most reads are first touches (~15% data
    # wait at 16 workers, 1-GPU profile 2026-10-20).
    config.read_ahead = 2
    # Workers fork from a server that preloaded the heavy imports instead of each
    # spawned worker importing them, every epoch: epoch-boundary stall 141 -> 66 s
    # (1-GPU test, 2026-10-21). An epoch here is only ~770 steps.
    config.multiprocessing_context = "forkserver"
    config.tile_reuse = TILE_REUSE
    config.prefetch_factor = 1
    config.pin_memory = False
    config.uint8_masks = True
    config.max_decode_tokens = MAX_DECODE_TOKENS
    # SCL is loaded for the cloud-masked composites only. Dropping cloudy S2 decode
    # targets was a measured null on v1.3-era runs (2026-08-21) and would be a
    # second difference from v1.3, so it stays off.
    config.drop_cloudy_s2_targets = False
    return config


def build_train_module_config(common: CommonComponents):
    """rc_pixtgt_pix512's train module (shared pixel targets) at a smaller microbatch.

    Plus the sub-patch offset loss (PR #630) at weight 1.0, on for new runs: it keeps the
    per-offset stripe out of registers finer than the patch tokens.
    """
    config = _pixtgt_train_module_config(common)
    config.rank_microbatch_size = RANK_MICROBATCH_SIZE
    config.subpatch_offset_loss_weight = 1.0
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
    # Landsat C2 L2 from the twin (the corpus's Landsat; the monthly layers carry L1).
    tasks["pastis_year_aligned_ws16_ps2_sentinel1_sentinel2_landsatl2_proj128"] = (
        _ps2_student(pastis, input_modalities=S1_S2_L2, twin_landsat_l2=True)
    )
    tasks[
        "pastis_year_aligned_ws16_ps2_sentinel1_sentinel2_landsatl2_everycapture_proj128"
    ] = _ps2_student(
        pastis, input_modalities=S1_S2_L2, every_capture=True, twin_landsat_l2=True
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
        tasks[f"{prefix}_sentinel1_sentinel2_landsatl2_everycapture_knn_proj128"] = (
            _ps2_student(
                base,
                input_modalities=S1_S2_L2,
                every_capture=True,
                twin_landsat_l2=True,
            )
        )
    tasks["m-eurosat_ps2_proj128"] = _ps2_student(catalog["m-eurosat"])
    return tasks


def build_allcap_trainer_config(common: CommonComponents, module_path: str):
    """v1.2 trainer, this eval set routed through Beaker, own W&B project."""
    trainer_config = _v1_2_build_trainer_config(common)
    trainer_config.max_duration = Duration.steps(MAX_STEPS)
    catalog = trainer_config.callbacks["downstream_evaluator"].tasks
    trainer_config = route_loop_evals_through_beaker(
        trainer_config, module_path, allcap_eval_tasks(catalog)
    )
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
    # 1-GPU eval jobs also fit saturn, too slow for training but scheduled far sooner.
    trainer_config.callbacks["downstream_evaluator"].beaker_eval_clusters = [
        "ai2/jupiter",
        "ai2/ceres",
        "ai2/saturn",
    ]
    # The eval jobs rebuild the model from MODULE_PATH: pass on this run's own
    # --model.* overrides (e.g. a sweep arm's depth / narrow width), or they build the
    # script's default model and fail to load the checkpoint (2026-10-21).
    if common.launch is not None:
        trainer_config.callbacks["downstream_evaluator"].beaker_eval_extra_overrides = [
            arg for arg in common.launch.cmd if arg.startswith("--model.")
        ]
    # Every rank's data wait and step time (max / mean / min across ranks): the speed
    # monitor only sees rank 0, where other ranks' stalls look like model time.
    trainer_config.callbacks["rank_timing"] = RankTimingCallback()
    return trainer_config
