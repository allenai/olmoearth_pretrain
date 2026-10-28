"""Shared builders for the high-res target arms N5 and N6 (2026-10-28).

Both are N3 (``n3_naip_subpixel.py``: the v1.3 control plus target-only NAIP) with
three target-only modalities, all supervised from the register grid:

* ``naip_10``: the raw 2.5 m RGBIR (N3 regressed only 5 m means + per-10 m-pixel stds).
* ``open_buildings``: Open Buildings 2.5D Temporal at 2.5 m (presence, height, count).
* ``viirs``: annual VIIRS night lights (log1p radiance) on the 10 m grid.

The 2.5 m targets use ``laplacian_loss`` (MSE on 10 m block means + L1 on the
within-block residual), following Scale-MAE (arXiv:2212.14532), whose separate
low/high-frequency losses beat one combined loss and whose high-frequency target
mattered most (Table 9). N5 keeps the linear per-cell head + bilinear resize; N6 swaps
in ``UpsampleHead`` for the two 2.5 m targets, after Scale-MAE's upsampling decoder
(Table 6: +1.8 kNN over a vanilla MAE decoder). N6 vs N5 isolates the decoder.

Possible follow-ups if N6 helps: Scale-MAE's few transformer layers before upsampling
and its per-frequency output blocks; pooling VIIRS predictions over ~320 m blocks
instead of scoring every 10 m pixel (a VIIRS pixel is ~460 m, stored nearest-upsampled).

Trained on the osm_sampling H5 rebuilt with naip_10 + viirs + open_buildings, so the
samples match the control's tiles but the H5 indices are not aligned with the older
builds. W&B project ``2026_10_23_naip_v13`` (next to N2-N4).
"""

from functools import partial

import naip_common

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.nn.supervision_head import (
    SupervisionModalityConfig,
    SupervisionTaskType,
)

TARGET_ONLY = [
    Modality.NAIP_10.name,
    Modality.OPEN_BUILDINGS.name,
    Modality.VIIRS.name,
]
# Same weight as the other regression heads (base.py: base_weight 1.0 x 1.0), as in N3.
TARGET_WEIGHT = 1.0

# The osm_sampling H5 rebuilt on 2026-10-28 with naip_10, viirs and open_buildings.
HIRES_H5_DIR = (
    "/weka/dfive-default/helios/dataset/osm_sampling/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_gse_landsat_naip_10_open_buildings_openstreetmap_raster_sentinel1_"
    "sentinel2_l2a_srtm_viirs_worldcereal_worldcover_worldpop_wri_canopy_height_map/"
    "TODO_NUM_SAMPLES"
)

build_common_components = partial(naip_common.build_common_components, TARGET_ONLY)


def build_dataset_config(common) -> OlmoEarthDatasetConfig:
    """The rebuilt H5 with all three target-only modalities."""
    return OlmoEarthDatasetConfig(
        h5py_dir=HIRES_H5_DIR,
        training_modalities=common.training_modalities,
    )


def build_model_config(common, hires_decoder: str):
    """The control's model plus naip_10 / open_buildings / viirs regression heads."""
    config = naip_common.build_model_config(common, naip_common.CONTROL_MODALITIES)
    assert config.supervision_head_config is not None
    heads = config.supervision_head_config.modality_configs
    for modality in (Modality.NAIP_10, Modality.OPEN_BUILDINGS):
        heads[modality.name] = SupervisionModalityConfig(
            task_type=SupervisionTaskType.REGRESSION,
            num_output_channels=modality.num_bands,
            weight=TARGET_WEIGHT,
            regression_loss_type="l1",
            laplacian_loss=True,
            decoder=hires_decoder,
        )
    heads[Modality.VIIRS.name] = SupervisionModalityConfig(
        task_type=SupervisionTaskType.REGRESSION,
        num_output_channels=Modality.VIIRS.num_bands,
        weight=TARGET_WEIGHT,
        regression_loss_type="l1",
    )
    return config


def build_train_module_config(common):
    """The control's train module; the targets decode-only (never encoded)."""
    return naip_common.build_train_module_config(common, TARGET_ONLY)


def build_dataloader_config(common):
    """The control's dataloader; the targets out of the token budget."""
    return naip_common.build_dataloader_config(common, TARGET_ONLY)
