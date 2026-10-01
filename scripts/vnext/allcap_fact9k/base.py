"""v1.2 Base + factorized encoder attention, trained on the every-capture corpus.

The model and recipe are ``scripts/official/v1_2/base_faster.py`` (v1.2 Base:
hidden patch-embed projection, learnable mixed 3D RoPE, latent MIM with
patch discrimination + InfoNCE, projection-only target, DDP) with three changes
taken from earthy's best long-sequence arm (``earthy-oe-4xh100-ts48-fact9k-v1``):

* ``attention_mode="factorized"``: even encoder blocks attend within each
  location (across its modality / timestep tokens), odd blocks within each
  (modality, timestep) slot (across locations), so cost is linear in the number
  of timesteps.
* Data: the allcap corpus (every S2 / S1 / Landsat-L2 capture over 360 days, no
  cloud filtering, per-modality timestamps) instead of 12 monthly composites.
  Each microbatch draws a time-range length from TIME_RANGE_DAYS; a sample takes
  every capture in a random range of that length (ranges >= the 360-day span
  take the whole span).
* Token budget 9000 counting real tokens only, applied as a contiguous run of
  timesteps (v1.2's 2250 counts the default bandsets, ~1,125 real tokens).

Everything else is v1.2, for comparability with the other OlmoEarth models:
single-bandset S2 tokenization, decode-only map targets, AdamW lr 1e-4 wd 0.02,
CosWithWarmup(8000) with the cosine horizon pinned to v1.2's 667,200 steps (so
the learning rate matches v1.2 at every step of a shorter run), the v1.2
in-loop eval suite, and ``rope_mixed_base=10`` (the value the released v1.2 runs
used; ``v1_2/base.py`` says 10000).
"""

import logging
import os
import sys
from pathlib import Path

# Every microbatch has a different (grid, timeline) shape; without expandable
# segments the caching allocator fragments (smoke test: ~48 GiB active, 76 of
# 80 GiB reserved at microbatch 8). Must be set before the first CUDA allocation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "official" / "v1_2"))

import base as v1_2_base  # noqa: E402
import base_faster as v1_2_faster  # noqa: E402
from olmo_core.optim.scheduler import CosWithWarmup  # noqa: E402
from olmo_core.train.common import Duration  # noqa: E402

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig  # noqa: E402
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import (  # noqa: E402
    CommonComponents,
    SubCmd,
    main,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from olmoearth_pretrain.nn.tokenization import TokenizationConfig  # noqa: E402

logger = logging.getLogger(__name__)

MODULE_PATH = "scripts/vnext/allcap_fact9k/base.py"
WANDB_PROJECT = "2026_09_30_allcap_fact9k"

ALLCAP_H5PY_DIR = (
    "/weka/dfive-default/helios/dataset/osm_allcaptures/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_landsat_l2_openstreetmap_raster_sentinel1_sentinel2_l2a_sentinel2_scl_"
    "srtm_worldcereal_worldcover_wri_canopy_height_map/395660"
)

TOKEN_BUDGET = 9000
# earthy arm B's range menu; this corpus spans 360 days, so the last three
# entries take the whole span.
TIME_RANGE_DAYS = [7.0, 30.0, 90.0, 180.0, 365.0, 730.0, 1825.0]
# The released v1.2 runs' rope_mixed_base (the v1_2/base.py constant is 10000).
ROPE_MIXED_BASE = 10.0
# v1.2's schedule horizon: 300 epochs x floor(1,138,828 / 512) steps.
V1_2_TOTAL_STEPS = 667_200
MAX_STEPS = 100_000
RANK_MICROBATCH_SIZE = 8


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """v1.2 modalities with Landsat Level-2 in place of Level-1."""
    config = v1_2_base.build_common_components(
        script, cmd, run_name, cluster, overrides
    )
    config.training_modalities = [
        Modality.LANDSAT_L2.name if m == Modality.LANDSAT.name else m
        for m in config.training_modalities
    ]
    # landsat_l2 has a single bandset already.
    config.tokenization_config = TokenizationConfig(
        overrides={"sentinel2_l2a": v1_2_base.S2_SINGLE_BANDSET}
    )
    return config


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """base_faster's model with factorized encoder attention."""
    config = v1_2_faster.build_model_config(common)
    config.encoder_config.attention_mode = "factorized"
    config.encoder_config.band_dropout_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.LANDSAT_L2.name,
    ]
    config.encoder_config.rope_mixed_base = ROPE_MIXED_BASE
    config.decoder_config.rope_mixed_base = ROPE_MIXED_BASE
    return config


def build_train_module_config(common: CommonComponents):
    """base_faster's train module; cosine horizon pinned to v1.2's."""
    config = v1_2_faster.build_train_module_config(common)
    config.rank_microbatch_size = RANK_MICROBATCH_SIZE
    config.scheduler = CosWithWarmup(warmup=8000, t_max=V1_2_TOTAL_STEPS)
    return config


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """v1.2 shape sampling with the real-token budget and time-range menu."""
    config = v1_2_base.build_dataloader_config(common)
    config.token_budget = TOKEN_BUDGET
    config.time_range_days_choices = list(TIME_RANGE_DAYS)
    config.tokenization_config = common.tokenization_config
    config.prefetch_factor = 2
    # Batch shapes vary per microbatch, so pinned buffers are never reused: each
    # batch was a fresh ~33 ms cudaHostAlloc that stalled kernel launches.
    config.pin_memory = False
    return config


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """The allcap corpus (per-modality timestamps)."""
    return OlmoEarthDatasetConfig(
        h5py_dir=ALLCAP_H5PY_DIR,
        training_modalities=common.training_modalities,
        per_modality_timestamps=True,
    )


def build_trainer_config(common: CommonComponents):
    """base_faster's trainer (evals as Beaker jobs), 100k steps, own W&B project."""
    config = v1_2_faster.build_trainer_config(common)
    config.max_duration = Duration.steps(MAX_STEPS)
    config.callbacks["wandb"].project = WANDB_PROJECT
    config.callbacks["downstream_evaluator"].beaker_eval_module_path = MODULE_PATH
    return config


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=v1_2_base.build_visualize_config,
    )
