"""Shared config builders for open-set *supervised* pretraining.

Builds on ``base_faster.py`` (the merged production baseline: projection-only
target + replicated DDP + bf16 autocast + in-loop evals as separate Beaker jobs)
and adds a supervised open-set probe head:

* The dataset loads the imagery modalities **plus** the ``open_set`` /
  ``open_set_regression`` label layers, so every sample carries the labels
  (missing-filled where absent, e.g. for ``osm_sampling`` samples).
* The **encoder** only ever sees the imagery modalities -- the label layers are
  never tokenized -- so there is no label leakage.
* An :class:`OpenSetLatentMIM` model owns an :class:`OpenSetProbe`; the
  :class:`OpenSetLatentMIMTrainModule` adds the weighted supervised loss.

The two concrete launch scripts (``open_set_only.py`` and ``open_set_osm.py``)
import these builders and only supply ``build_dataset_config`` /
``build_trainer_config``.
"""

import dataclasses
import logging
from dataclasses import fields
from pathlib import Path

# Import the v1.2 helpers we reuse verbatim.
from base import _tokenization_config  # noqa: E402
from base import build_dataset_config as build_osm_dataset_config  # noqa: E402

# base_faster re-exports base's builders and layers the validated speedups on top.
from base_faster import build_model_config as base_faster_build_model_config
from base_faster import (
    build_train_module_config as base_faster_build_train_module_config,
)
from base_faster import build_trainer_config as base_faster_build_trainer_config

from olmoearth_pretrain.data.concat import OlmoEarthConcatDatasetConfig
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.internal.common import (
    build_common_components as build_common_components_default,
)
from olmoearth_pretrain.internal.experiment import CommonComponents, SubCmd
from olmoearth_pretrain.nn.open_set_latent_mim import OpenSetLatentMIMConfig
from olmoearth_pretrain.train.open_set_probe import OpenSetProbeConfig
from olmoearth_pretrain.train.train_module.open_set_latentmim import (
    OpenSetLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

# Imagery modalities the encoder is trained on (identical to v1.2 base).
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

# Supervision label layers: loaded by the dataset, never encoded.
LABEL_MODALITIES = [
    Modality.OPEN_SET.name,
    Modality.OPEN_SET_REGRESSION.name,
]

# Version-controlled global class mapping (data/open_set_segmentation_data/).
# scripts/official/v1_2/open_set_base.py -> parents[3] is the repo root.
CLASS_MAPPING_PATH = str(
    Path(__file__).resolve().parents[3]
    / "data"
    / "open_set_segmentation_data"
    / "class_mapping.json"
)
CLASS_MAPPING_SHA256_PATH = Path(CLASS_MAPPING_PATH).with_suffix(".sha256")
CLASS_MAPPING_SHA256 = CLASS_MAPPING_SHA256_PATH.read_text().split()[0]

# Weight on the combined supervised (CE + MSE) loss relative to the SSL objective.
SUP_LOSS_WEIGHT = 1.0

# H5 directory of the open-set supervised dataset. Set once the H5s are built; it will
# live under /weka/dfive-default/helios/dataset/open_set_dataset/... The layout mirrors
# the grid pipeline but with the ..._128_x_1 suffix (one H5 sample per 128x128 window)
# and must include the open_set / open_set_regression label datasets.
OPEN_SET_H5_DIR = (
    "/weka/dfive-default/helios/dataset/open_set_dataset/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_1/TODO_MODALITIES/TODO_COUNT"
)


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Common components with imagery + label modalities in the dataset load list."""
    config = build_common_components_default(script, cmd, run_name, cluster, overrides)
    # The dataset loads imagery AND the label layers; the encoder sees imagery only
    # (see build_model_config, which passes IMAGERY_MODALITIES to the encoder).
    config.training_modalities = IMAGERY_MODALITIES + LABEL_MODALITIES
    config.tokenization_config = _tokenization_config()
    return config


def _imagery_common(common: CommonComponents) -> CommonComponents:
    """A shallow copy of ``common`` whose modalities are imagery-only.

    Used to build the encoder / decoder so the label layers are never tokenized.
    """
    return dataclasses.replace(common, training_modalities=list(IMAGERY_MODALITIES))


def build_model_config(common: CommonComponents) -> OpenSetLatentMIMConfig:
    """Build the v1.2-faster model (imagery-only encoder) plus the open-set probe."""
    base_config = base_faster_build_model_config(_imagery_common(common))
    return OpenSetLatentMIMConfig(
        encoder_config=base_config.encoder_config,
        decoder_config=base_config.decoder_config,
        reconstructor_config=base_config.reconstructor_config,
        projection_only_target=base_config.projection_only_target,
        open_set_probe_config=OpenSetProbeConfig(
            class_mapping_path=CLASS_MAPPING_PATH,
            expected_class_mapping_sha256=CLASS_MAPPING_SHA256,
        ),
    )


def build_train_module_config(
    common: CommonComponents,
) -> OpenSetLatentMIMTrainModuleConfig:
    """Build the DDP+bf16 train module config with the supervised probe loss."""
    base_config = base_faster_build_train_module_config(common)
    open_config = OpenSetLatentMIMTrainModuleConfig(
        **{f.name: getattr(base_config, f.name) for f in fields(base_config)},
        sup_loss_weight=SUP_LOSS_WEIGHT,
    )
    # token_exit_cfg is only meaningful for encoded modalities; keep it imagery-only.
    open_config.token_exit_cfg = {modality: 0 for modality in IMAGERY_MODALITIES}
    return open_config


def build_trainer_config(common: CommonComponents, module_path: str):
    """Trainer config; point the in-loop Beaker eval jobs at ``module_path``.

    The eval job rebuilds the model from ``module_path`` to load the checkpoint, so
    it must point at the open-set launch script (whose model config includes the
    probe) rather than base_faster.
    """
    trainer_config = base_faster_build_trainer_config(common)
    evaluator = trainer_config.callbacks["downstream_evaluator"]
    evaluator.beaker_eval_module_path = module_path
    return trainer_config


def build_open_set_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Dataset config for the open-set supervised H5s."""
    return OlmoEarthDatasetConfig(
        h5py_dir=OPEN_SET_H5_DIR,
        training_modalities=common.training_modalities,
    )


def build_osm_plus_open_set_dataset_config(
    common: CommonComponents,
) -> OlmoEarthConcatDatasetConfig:
    """Concatenated dataset: osm_sampling (SSL only) + open-set (SSL + supervised).

    Both sub-datasets share ``common.training_modalities`` (imagery + label layers);
    ``osm_sampling`` H5s lack the label layers, so they are missing-filled and
    contribute only the self-supervised loss.
    """
    return OlmoEarthConcatDatasetConfig(
        dataset_configs=[
            build_osm_dataset_config(common),
            build_open_set_dataset_config(common),
        ],
    )
