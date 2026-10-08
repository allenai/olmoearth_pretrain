"""Shared config builders for open-set *supervised* training on the v1.3 recipe.

Builds on ``base.py`` (the v1.3 release recipe: Perceiver register bottleneck with
map supervision heads, distilled ``[128, 64]`` student, single-forward LatentMIM
train module, DDP + bf16, decorrelated shape sampler) and adds a supervised
open-set probe head on the register grid:

* The dataset loads the imagery modalities **plus** the open-set label layers
  (``open_set`` classification, ``open_set_regression``, and the paired change
  samples' ``open_set_change_boundary``), so every sample carries the labels
  (missing-filled where absent, e.g. for ``osm_sampling`` samples).
* The **encoder / decoder** only ever see the imagery modalities -- the label layers
  are never tokenized -- so there is no label leakage. The labels are listed as
  decode-only so they also stay out of the encoder token budget.
* An :class:`OpenSetLatentMIM` model owns an :class:`OpenSetProbe` that reads the
  d768 teacher register grid (the same grid the map supervision heads read); the
  :class:`OpenSetLatentMIMTrainModule` adds the weighted supervised loss.
* Paired pre/post change samples are guaranteed at least one "before" and one
  "after" timestep in the encoder input (temporal crop + time masking, keyed on
  ``open_set_change_boundary``); their labels are only supervised when both sides
  were visible.
* The v1.3 per-pixel latent updates (``v1_3_rc_pixtgt_pix512``): the dataloader
  draws a Perceiver latent patch size per rank batch under a ``MAX_LATENTS`` budget
  (so the register grid, and with it the open-set probe's label blocks, can be finer
  than the token grid), and the MIM targets are one random pixel per token cell
  (``pixel_targets``).

The concrete launch scripts (``open_set_only.py``, ``open_set_osm.py`` and
``open_set_post_train.py``) import these builders and only supply the dataset,
schedule and module path.
"""

import dataclasses
import logging
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

# v1.3 builds on the v1.2 config, which lives one directory over.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import (  # noqa: E402
    build_common_components as base_build_common_components,
)
from base import build_dataloader_config as base_build_dataloader_config  # noqa: E402
from base import build_model_config as base_build_model_config  # noqa: E402
from base import (
    build_train_module_config as base_build_train_module_config,  # noqa: E402
)
from base import build_trainer_config as base_build_trainer_config  # noqa: E402
from base import set_student_loop_evals  # noqa: E402
from v1_2.base import ONLY_DECODE_MODALITIES  # noqa: E402
from v1_2.base import build_dataset_config as build_osm_dataset_config  # noqa: E402

from olmoearth_pretrain.data.concat import OlmoEarthConcatDatasetConfig  # noqa: E402
from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig  # noqa: E402
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import (  # noqa: E402
    CommonComponents,
    SubCmd,
)
from olmoearth_pretrain.nn.open_set_latent_mim import (
    OpenSetLatentMIMConfig,  # noqa: E402
)
from olmoearth_pretrain.train.open_set_probe import OpenSetProbeConfig  # noqa: E402
from olmoearth_pretrain.train.train_module.open_set_latentmim import (  # noqa: E402
    OpenSetLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

WANDB_PROJECT = "2026_09_22_open_set_v1_3"

# Imagery modalities the encoder is trained on (identical to the v1.2/v1.3 base).
IMAGERY_MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
    Modality.LANDSAT.name,
    Modality.WORLDCOVER.name,
    Modality.SRTM.name,
    Modality.OPENSTREETMAP_RASTER.name,
    Modality.WRI_CANOPY_HEIGHT_MAP.name,
    Modality.CDL.name,
    Modality.WORLDCEREAL.name,
]

# Supervision label layers: loaded by the dataset, never encoded or decoded.
LABEL_MODALITIES = [
    Modality.OPEN_SET.name,
    Modality.OPEN_SET_REGRESSION.name,
    Modality.OPEN_SET_CHANGE_BOUNDARY.name,
]

# Version-controlled global class mapping (data/open_set_segmentation_data/).
# scripts/official/v1_3/open_set_base.py -> parents[3] is the repo root.
CLASS_MAPPING_PATH = str(
    Path(__file__).resolve().parents[3]
    / "data"
    / "open_set_segmentation_data"
    / "class_mapping.json"
)
CLASS_MAPPING_SHA256_PATH = Path(CLASS_MAPPING_PATH).with_suffix(".sha256")
CLASS_MAPPING_SHA256 = CLASS_MAPPING_SHA256_PATH.read_text().split()[0]
# Frozen class text embeddings for the ``text`` probe head, generated once by
# ``open_set_segmentation_data.embed_class_names`` (all-mpnet-base-v2, 768-d) and kept
# on weka next to the label bank; the ``.json`` sidecar beside it pins the class
# mapping hash, which the probe verifies at build time.
CLASS_TEXT_EMBEDDINGS_PATH = (
    "/weka/dfive-default/helios/dataset_creation/open_set_segmentation/"
    "class_text_embeddings/class_text_embeddings.npy"
)

# Half the v1.3 base microbatch (64). The label layers make the open-set batches
# variable-shape enough that per-step active memory swung 5-67 GiB at 64 and the
# runs OOMed (~10 GiB lost to fragmentation) on 80 GiB GPUs; 32 runs as two
# microbatches per step with the same global batch.
RANK_MICROBATCH_SIZE = 32

# Weight on the combined supervised (CE + MSE) loss relative to the SSL objective.
# 1.0 inflated the total grad norm under the fixed clip and slowed SSL learning in the
# v1.2 runs; the probe converges fine at 0.1 (per-sample-balanced loss).
SUP_LOSS_WEIGHT = 0.1

# Per-sample Perceiver latent budget (v1.3 "pix512"): each rank batch draws a latent
# patch size uniformly among the divisors of its patch size whose latent grid fits this
# budget (one latent per token is always allowed). Evals at patch size 1 are already
# one latent per pixel.
MAX_LATENTS = 512

# H5 directory of the open-set supervised dataset (the ..._128_x_1 layout: one H5
# sample per 128x128 window, zstd level 3). The layout is
# h5py_data_w_missing_timesteps_zstd_3_128_x_1/<sorted modality names>/<count>. This
# build (1,448,494 samples = every example with at least one multitemporal modality)
# includes the open_set_change_boundary modality on the ~133.8k (9.2%) paired pre/post
# change samples; it is missing-filled elsewhere. Anything derived from H5 indices
# (e.g. open_set_hq/select_h5_indices.py's filter file) is specific to this build and
# must be regenerated whenever this path changes.
OPEN_SET_H5_DIR = "/weka/dfive-default/helios/dataset/open_set_dataset/h5py_data_w_missing_timesteps_zstd_3_128_x_1/cdl_landsat_open_set_open_set_change_boundary_open_set_regression_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_wri_canopy_height_map/1448494"

# Keep-list of the open-set H5 samples that do not overlap any AlphaEarth supplemental
# eval window (any split), produced by
# ``open_set_aef_clean/select_aef_clean_indices.py`` for this build. The label bank
# ingests those evals (and many of their sources), so only runs trained on this subset
# give valid AEF metrics. ``allsplits`` rather than ``valtest`` because the in-loop
# AEF kNN evals use AEF's balanced-trial protocol, which pools all splits.
OPEN_SET_FILTERS_DIR = "/weka/dfive-default/helios/dataset/open_set_dataset/filters"
AEF_CLEAN_FILTER_IDX_FILE = (
    f"{OPEN_SET_FILTERS_DIR}/open_set_aef_clean_allsplits_"
    f"{Path(OPEN_SET_H5_DIR).name}.npy"
)


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Common components with imagery + label modalities in the dataset load list."""
    config = base_build_common_components(script, cmd, run_name, cluster, overrides)
    # The dataset loads imagery AND the label layers; the encoder sees imagery only
    # (see build_model_config, which passes IMAGERY_MODALITIES to the encoder).
    config.training_modalities = IMAGERY_MODALITIES + LABEL_MODALITIES
    return config


def _imagery_common(common: CommonComponents) -> CommonComponents:
    """A shallow copy of ``common`` whose modalities are imagery-only.

    Used to build the encoder / decoder so the label layers are never tokenized.
    """
    return dataclasses.replace(common, training_modalities=list(IMAGERY_MODALITIES))


def _label_aware_only_decode() -> list[str]:
    """The v1.2 decode-only maps plus the label layers (never encoded)."""
    return list(ONLY_DECODE_MODALITIES) + list(LABEL_MODALITIES)


def build_model_config(
    common: CommonComponents, **probe_overrides: Any
) -> OpenSetLatentMIMConfig:
    """The v1.3 model (imagery-only encoder) plus the open-set probe on the registers.

    ``probe_overrides`` are forwarded to :class:`OpenSetProbeConfig` so the variant
    launch scripts (``open_set_only_{mlp,text,dsbal}.py``) only name what differs.
    """
    base_config = base_build_model_config(_imagery_common(common))
    return OpenSetLatentMIMConfig(
        encoder_config=base_config.encoder_config,
        decoder_config=base_config.decoder_config,
        reconstructor_config=base_config.reconstructor_config,
        supervision_head_config=base_config.supervision_head_config,
        register_distillation_head_config=base_config.register_distillation_head_config,
        projection_only_target=base_config.projection_only_target,
        open_set_probe_config=OpenSetProbeConfig(
            class_mapping_path=CLASS_MAPPING_PATH,
            expected_class_mapping_sha256=CLASS_MAPPING_SHA256,
            **probe_overrides,
        ),
    )


def build_train_module_config(
    common: CommonComponents,
) -> OpenSetLatentMIMTrainModuleConfig:
    """The v1.3 train module config with the supervised probe loss."""
    base_config = base_build_train_module_config(_imagery_common(common))
    config = OpenSetLatentMIMTrainModuleConfig(
        **{f.name: getattr(base_config, f.name) for f in fields(base_config)},
        sup_loss_weight=SUP_LOSS_WEIGHT,
    )
    config.rank_microbatch_size = RANK_MICROBATCH_SIZE
    # v1.3 "pixtgt": score each masked token on one random pixel of its cell.
    config.pixel_targets = True
    # token_exit_cfg is only meaningful for encoded modalities; keep it imagery-only.
    config.token_exit_cfg = {modality: 0 for modality in IMAGERY_MODALITIES}
    # The masking strategy must know the labels are decode-only (never encoded).
    config.masking_config.strategy_config["only_decode_modalities"] = (
        _label_aware_only_decode()
    )
    return config


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """The v1.3 dataloader, carrying labels without treating them as model tokens."""
    config = base_build_dataloader_config(common)
    config.max_latents = MAX_LATENTS
    # The dataloader excludes only_decode_modalities from the token budget, so
    # listing the label modalities here also keeps them from consuming budget.
    config.masking_config.strategy_config["only_decode_modalities"] = (
        _label_aware_only_decode()
    )
    return config


def build_trainer_config(common: CommonComponents, module_path: str):
    """The v1.3 trainer (student in-loop evals) pointed at ``module_path``.

    The eval job rebuilds the model from ``module_path`` to load the checkpoint, so
    it must point at the open-set launch script (whose model config includes the
    probe) rather than ``base.py``.
    """
    trainer_config = base_build_trainer_config(common)
    set_student_loop_evals(trainer_config, module_path)
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
    return trainer_config


def build_open_set_dataset_config(
    common: CommonComponents, filter_idx_file: str | None = None
) -> OlmoEarthDatasetConfig:
    """Dataset config for the open-set supervised H5s.

    ``filter_idx_file`` optionally restricts the build to a keep-list of H5 indices
    (e.g. ``AEF_CLEAN_FILTER_IDX_FILE``).
    """
    return OlmoEarthDatasetConfig(
        h5py_dir=OPEN_SET_H5_DIR,
        training_modalities=common.training_modalities,
        filter_idx_file=filter_idx_file,
    )


def build_osm_plus_open_set_dataset_config(
    common: CommonComponents, open_set_filter_idx_file: str | None = None
) -> OlmoEarthConcatDatasetConfig:
    """Concatenated dataset: osm_sampling (SSL only) + open-set (SSL + supervised).

    Both sub-datasets share ``common.training_modalities`` (imagery + label layers);
    ``osm_sampling`` H5s lack the label layers, so they are missing-filled and
    contribute only the self-supervised (+ map supervision + student) losses.
    ``open_set_filter_idx_file`` filters the open-set half only.
    """
    return OlmoEarthConcatDatasetConfig(
        dataset_configs=[
            build_osm_dataset_config(common),
            build_open_set_dataset_config(common, open_set_filter_idx_file),
        ],
    )
