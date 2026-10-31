"""H1: the v1.3 control plus hydrology, GHSL and national-lidar targets (card t-v3sdsf).

The v1.3 release recipe (``ablations/rc_ld1_pixtgt_pix512.py``, trained as
``v1_3_rc_ld1_pixtgt_pix512``) on the SAME H5 the release trained on, plus four
target-only modalities read from a side directory of ``sample_{i}.h5`` files aligned
with that H5 (``aux_h5_dir``), each supervised from the register grid like the
control's map heads:

* ``hand``: GLO-30 HAND (log1p m), L1.
* ``hydro``: HydroSHEDS v1 log10 upstream area + log1p distance to rivers >= 10 km^2, L1.
* ``ghsl``: GHS-BUILT-S 2018 built-up fraction + log1p GHS-BUILT-H ANBH, L1.
* ``lidar_height``: national lidar (US 3DEP, NL AHN, FR LiDAR HD, England EA) height
  above ground at 2.5 m: mean, max, fraction > 2 m. L1 with the Laplacian split
  (MSE on 10 m block means + L1 on the residual), as open_buildings in N5. Present
  for ~15-20% of samples; pixels without lidar are missing.

None of them is encoded or a MIM target, and they stay out of the token budget. The
encoder, decoder, masking, evals and schedule are the control's, so the run is compared
to the released v1.3 curve at matched steps (no new control).
"""

import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "naip"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import naip_common  # noqa: E402
from base import build_visualize_config  # noqa: E402

from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import main  # noqa: E402
from olmoearth_pretrain.nn.supervision_head import (  # noqa: E402
    SupervisionModalityConfig,
    SupervisionTaskType,
)

MODULE_PATH = "scripts/official/v1_3/hydro/h1_hydro_ghsl_lidar.py"
WANDB_PROJECT = "2026_10_31_hydro_ghsl_lidar"

# The H5 the released v1.3 trained on (its config.json dataset.h5py_dir).
RELEASE_H5_DIR = (
    "/weka/dfive-default/helios/dataset/osm_sampling/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_"
    "worldcover_worldpop_wri_canopy_height_map/1138828"
)
# sample_{i}.h5 with hand/hydro/ghsl/lidar_height for sample i of RELEASE_H5_DIR.
AUX_H5_DIR = "/weka/dfive-default/helios/dataset/osm_sampling/aux_hydro_ghsl_lidar_20261031/1138828"

TARGET_ONLY = [
    Modality.HAND.name,
    Modality.HYDRO.name,
    Modality.GHSL.name,
    Modality.LIDAR_HEIGHT.name,
]
# Same weight as the control's other regression heads (base.py: 1.0 x 1.0).
TARGET_WEIGHT = 1.0

build_common_components = partial(naip_common.build_common_components, TARGET_ONLY)


def build_dataset_config(common) -> OlmoEarthDatasetConfig:
    """The release H5 plus the aligned side H5s."""
    return OlmoEarthDatasetConfig(
        h5py_dir=RELEASE_H5_DIR,
        training_modalities=common.training_modalities,
        aux_h5_dir=AUX_H5_DIR,
    )


def build_model_config(common):
    """The control's model plus the four regression heads."""
    config = naip_common.build_model_config(common, naip_common.CONTROL_MODALITIES)
    assert config.supervision_head_config is not None
    heads = config.supervision_head_config.modality_configs
    for name in [Modality.HAND.name, Modality.HYDRO.name, Modality.GHSL.name]:
        heads[name] = SupervisionModalityConfig(
            task_type=SupervisionTaskType.REGRESSION,
            num_output_channels=Modality.get(name).num_bands,
            weight=TARGET_WEIGHT,
            regression_loss_type="l1",
        )
    heads[Modality.LIDAR_HEIGHT.name] = SupervisionModalityConfig(
        task_type=SupervisionTaskType.REGRESSION,
        num_output_channels=Modality.LIDAR_HEIGHT.num_bands,
        weight=TARGET_WEIGHT,
        regression_loss_type="l1",
        laplacian_loss=True,
    )
    return config


def build_train_module_config(common):
    """The control's train module; the targets decode-only (never encoded)."""
    return naip_common.build_train_module_config(common, TARGET_ONLY)


def build_dataloader_config(common):
    """The control's dataloader; the targets out of the token budget."""
    return naip_common.build_dataloader_config(common, TARGET_ONLY)


def build_trainer_config(common):
    """The control's evals, re-importing THIS module; our W&B project."""
    trainer_config = naip_common.build_trainer_config(common, MODULE_PATH)
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
    return trainer_config


def run() -> None:
    """Run the experiment."""
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )


if __name__ == "__main__":
    run()
