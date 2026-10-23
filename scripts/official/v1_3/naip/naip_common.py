"""Shared builders for the v1.3 NAIP arms (card t-pgudhr, 2026-10-23).

Every arm is the v1.3 control ``ablations/rc_ld1_pixtgt_pix512.py``
(``v1_3_rc_ld1_pixtgt_pix512``) trained from scratch on the sibling H5 that carries
``naip_10``. That H5 has the same 1,138,828 samples; it only lacks gse/worldpop,
which v1.3 does not train on, so the S1/S2/Landsat inputs and the decode-only maps
are the control's. About 17.5% of the samples have NAIP (512 x 512 RGBIR at 2.5 m
over the 128 x 128 S2 grid). Runs are compared to the control curve at matched
steps; the schedule is unchanged so the LR matches.

The arms differ in how NAIP enters:

* N2 (``n2_naip_input.py``): ``naip_10`` is an ordinary masked encoder input and
  MIM target (pixel targets take the 4 x 4 NAIP block under the drawn S2 pixel).
* N3 (``n3_naip_subpixel.py``): ``naip_10`` is a target only. A register-grid
  supervision head predicts its sub-pixel statistics (5 m means + per-10 m-pixel std).
* N4 (``n4_naip_dino.py``): frozen DINOv3-sat features of 0.625 m NAIP
  (``naip_dino``, 64-d whitened PCA per 10 m pixel) are a target only, regressed
  from the register grid with a cosine loss.

Target-only modalities are loaded by the dataset but left out of the encoder and
decoder, and listed as decode-only so they stay out of the token budget.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ablations"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base import build_common_components as _base_build_common_components  # noqa: E402
from base import build_dataloader_config as _base_build_dataloader_config  # noqa: E402
from pure_perceiver_mix import build_mix_trainer_config  # noqa: E402
from rc_ld1_pix512 import build_model_config as _control_model_config  # noqa: E402
from rc_pixtgt_pix512 import (  # noqa: E402
    build_train_module_config as _control_train_module_config,
)

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig  # noqa: E402
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents  # noqa: E402
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig  # noqa: E402
from olmoearth_pretrain.train.train_module.latent_mim import (  # noqa: E402
    LatentMIMTrainModuleConfig,
)

WANDB_PROJECT = "2026_10_23_naip_v13"

# The control's H5 plus naip_10 (minus gse/worldpop, unused by v1.3).
NAIP_H5_DIR = (
    "/weka/dfive-default/helios/dataset/osm_sampling/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_landsat_naip_10_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_"
    "worldcereal_worldcover_wri_canopy_height_map/1138828"
)
# DINOv3-sat features of 0.625 m NAIP, sample_{i}.npy (128, 128, 1, 64) float16.
NAIP_DINO_DIR = "/weka/dfive-default/favyen/naip_dino_features_20261023"

# The control's training modalities (encoder/decoder inputs + decode-only maps).
CONTROL_MODALITIES = [
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


def build_common_components(
    extra_modalities: list[str],
    script: str,
    cmd,
    run_name: str,
    cluster: str,
    overrides: list[str],
) -> CommonComponents:
    """The control's common components with ``extra_modalities`` loaded too."""
    config = _base_build_common_components(script, cmd, run_name, cluster, overrides)
    assert list(config.training_modalities) == CONTROL_MODALITIES, (
        config.training_modalities
    )
    config.training_modalities = CONTROL_MODALITIES + list(extra_modalities)
    return config


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """The naip_10 sibling H5 (plus the naip_dino side directory when trained on)."""
    aux = None
    if Modality.NAIP_DINO.name in common.training_modalities:
        aux = {Modality.NAIP_DINO.name: NAIP_DINO_DIR}
    return OlmoEarthDatasetConfig(
        h5py_dir=NAIP_H5_DIR,
        training_modalities=common.training_modalities,
        aux_npy_dirs=aux,
    )


def build_model_config(
    common: CommonComponents, encoded_modalities: list[str]
) -> LatentMIMConfig:
    """The control's model, with the encoder/decoder restricted to ``encoded_modalities``."""
    config = _control_model_config(common)
    config.encoder_config.supported_modality_names = list(encoded_modalities)
    config.decoder_config.supported_modality_names = list(encoded_modalities)
    return config


def _add_decode_only(masking_config, names: list[str]) -> None:
    strategy = masking_config.strategy_config
    only_decode = list(strategy.get("only_decode_modalities") or [])
    strategy["only_decode_modalities"] = only_decode + [
        n for n in names if n not in only_decode
    ]


def build_train_module_config(
    common: CommonComponents, target_only_modalities: list[str]
) -> LatentMIMTrainModuleConfig:
    """The control's train module (pixel targets); target-only modalities decode-only."""
    config = _control_train_module_config(common)
    _add_decode_only(config.masking_config, target_only_modalities)
    return config


def build_dataloader_config(
    common: CommonComponents, target_only_modalities: list[str]
) -> OlmoEarthDataLoaderConfig:
    """The control's dataloader; target-only modalities decode-only (out of the budget)."""
    config = _base_build_dataloader_config(common)
    _add_decode_only(config.masking_config, target_only_modalities)
    return config


def build_trainer_config(common: CommonComponents, module_path: str):
    """The control's evals (ps1 + ps2 + ps4 student, d768 registers), our W&B project."""
    trainer_config = build_mix_trainer_config(
        common, module_path, ps4_student_evals=True, ps2_student_evals=True
    )
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
    return trainer_config
