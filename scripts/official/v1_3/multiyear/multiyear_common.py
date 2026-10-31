"""Shared builders for the multi-year (4-year quarterly) pretraining arms (card t-nxpbum).

Both arms train the v1.3 control recipe (``claude_findings/control_pixtgt.py``: single
masked view, frozen random-projection target, 3D mixed RoPE encoder, AttentionPooler +
[128, 64] student, map supervision, pixel targets, DDP + bf16, rank microbatch 64) from
scratch on a 2 x 100k-window mix:

* one-year: 100k windows of the osm_sampling H5 (12 monthly steps, 2016-2024 dates),
  picked with ``OSM_RANDOM_TIME_FILTER_IDX_FILE`` (400k sample indices = 4 sub-tiles
  each); identical to the NISAR card's ``no_nisar`` control (C0).
* four-year: the osm_recent 100k locations re-materialized as 16 quarterly mosaics over
  a random 4-year span per window ending 2021-01..2026-10 (``OSM_QUARTERLY4Y_H5_DIR``).

On top of the v1.3 losses each arm adds the time-hidden temporal aux loss
(``olmoearth_pretrain/train/temporal_aux.py``): every year chunk of a multi-year sample
is encoded as its own batch element by the online encoder, and

* ``ordering.py`` (A2): antisymmetric pairwise year-ordering head + BCE (OPTIMUS-style);
* ``gated_contrast.py`` (A3): two token-dropout views, weighted InfoNCE with other-year
  negatives gated by a CaCo-style change gate.

The C0 reference is ``scripts/official/v1_3/nisar/no_nisar.py`` (exp 01M5WZZCAGWN0TA7GW8HVW640M).
"""

import sys
from pathlib import Path

# The v1.3 control builders, one directory over.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "claude_findings"))

from common import build_common_components as _build_common_components  # noqa: E402
from common import build_dataloader_config as _build_dataloader_config  # noqa: E402
from common import build_model_config as _build_model_config  # noqa: E402
from common import (  # noqa: E402
    build_train_module_config as _build_train_module_config,
)
from common import build_trainer_config as _build_trainer_config  # noqa: E402
from olmo_core.train.common import Duration  # noqa: E402

from olmoearth_pretrain.data.concat import OlmoEarthConcatDatasetConfig  # noqa: E402
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig  # noqa: E402
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from olmoearth_pretrain.nn.temporal_aux import TemporalAuxHeadConfig  # noqa: E402
from olmoearth_pretrain.train.olmo_core_patches import (  # noqa: E402
    drop_async_checkpoint_duration_metric,
)
from olmoearth_pretrain.train.temporal_aux import TemporalAuxConfig  # noqa: E402
from olmoearth_pretrain.train.train_module.latent_mim import (  # noqa: E402
    LatentMIMTrainModuleConfig,
)

# The async-save duration metric is recorded at a rank-dependent step and hangs the gloo
# bookkeeping all-reduce; see olmo_core_patches.
drop_async_checkpoint_duration_metric()

OSM_SAMPLING_H5_DIR = "/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828"
OSM_RANDOM_TIME_FILTER_IDX_FILE = "/weka/dfive-default/helios/dataset/osm_recent/filters/osm_sampling_random_time_100k_sample_idx.npy"
# TODO(multiyear data build): replace with the real 16-step quarterly H5 dir (the last
# path component is the sample count) once osm_quarterly4y is converted.
OSM_QUARTERLY4Y_H5_DIR = "/weka/dfive-default/helios/dataset/osm_quarterly4y/h5py_data_w_missing_timesteps_zstd_3_128_x_4/TODO_MODALITIES/TODO_NUM_SAMPLES"
# Both corpora are padded to 16 steps (the one-year samples get 4 MISSING slots) so
# rank batches stack.
MAX_SEQUENCE_LENGTH = 16

WANDB_PROJECT = "2026_10_31_multiyear_pretrain"
# Same step count as the v1.3 control (667,200 steps).
MAX_STEPS = 667_200

# --- temporal aux loss -------------------------------------------------------------------
AUX_WEIGHT = {"ordering": 0.1, "gated_contrast": 0.1}
AUX_MAX_GRID = 4
AUX_KEEP_RATIO = 0.75
AUX_PROJECTOR_DIMS = [512, 128]
# [h_hidden, h_out, g_hidden] of the antisymmetric ordering head.
AUX_ORDERING_DIMS = [128, 64, 128]

# The v1.3 control's modalities (S2, S1, Landsat + decode-only maps); no NISAR.
build_common_components = _build_common_components


def build_dataset_config(common: CommonComponents) -> OlmoEarthConcatDatasetConfig:
    """random-time osm_sampling subset (12 monthly) + osm_quarterly4y (16 quarterly)."""
    # TODO(multiyear data branch): pass max_sequence_length=MAX_SEQUENCE_LENGTH to BOTH
    # sub-configs once OlmoEarthDatasetConfig exposes it (the other agent's change in
    # data/dataset.py); until then the 16-step H5 fails the 12-step length assert.
    return OlmoEarthConcatDatasetConfig(
        dataset_configs=[
            OlmoEarthDatasetConfig(
                h5py_dir=OSM_SAMPLING_H5_DIR,
                training_modalities=common.training_modalities,
                filter_idx_file=OSM_RANDOM_TIME_FILTER_IDX_FILE,
            ),
            OlmoEarthDatasetConfig(
                h5py_dir=OSM_QUARTERLY4Y_H5_DIR,
                training_modalities=common.training_modalities,
            ),
        ],
    )


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """The v1.3 control dataloader."""
    config = _build_dataloader_config(common)
    # TODO(multiyear data branch): config.max_timesteps = MAX_SEQUENCE_LENGTH and the
    # "years x quarters" timestep sampler for multi-year samples.
    return config


def build_model_config_fn(variant: str):
    """``build_model_config`` for one arm: the v1.3 control model + the aux head."""

    def build_model_config(common: CommonComponents) -> LatentMIMConfig:
        config = _build_model_config(common)
        config.temporal_aux_head_config = TemporalAuxHeadConfig(
            embedding_source="location_tokens",
            projector_dims=list(AUX_PROJECTOR_DIMS),
            ordering_dims=list(AUX_ORDERING_DIMS) if variant == "ordering" else None,
        )
        return config

    return build_model_config


def build_train_module_config_fn(variant: str):
    """``build_train_module_config`` for one arm: v1.3 + pixel targets + the aux loss."""

    def build_train_module_config(
        common: CommonComponents,
    ) -> LatentMIMTrainModuleConfig:
        config = _build_train_module_config(common, pixel_targets=True)
        config.temporal_aux_config = TemporalAuxConfig(
            variant=variant,
            weight=AUX_WEIGHT[variant],
            max_grid=AUX_MAX_GRID,
            keep_ratio=AUX_KEEP_RATIO,
        )
        return config

    return build_train_module_config


def build_trainer_config(common: CommonComponents, module_path: str):
    """The v1.3 control trainer (in-loop evals re-import ``module_path``), own W&B project, 667,200 steps."""
    trainer_config = _build_trainer_config(common, module_path)
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
    trainer_config.max_duration = Duration.steps(MAX_STEPS)
    return trainer_config
