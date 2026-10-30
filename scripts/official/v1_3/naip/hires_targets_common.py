"""Shared builders for the high-res target arms N5 and N6 (2026-10-28).

Both are N2 (``n2_naip_input.py``: the v1.3 control with ``naip_10`` as an ordinary
masked encoder input and MIM target, the only NAIP arm with a hint of a gain) plus two
target-only modalities supervised from the register grid:

* ``open_buildings``: Open Buildings 2.5D Temporal at 2.5 m (presence, height, count).
* ``viirs``: annual VIIRS night lights (log1p radiance) on the 10 m grid.

NAIP needs no supervision head: as in N2 its masked tokens are MIM targets at pixel
resolution (the 4 x 4 NAIP block under each drawn S2 pixel).

The 2.5 m Open Buildings target uses ``laplacian_loss`` (MSE on 10 m block means + L1
on the within-block residual), following Scale-MAE (arXiv:2212.14532), whose separate
low/high-frequency losses beat one combined loss and whose high-frequency target
mattered most (Table 9). N5 keeps the linear per-cell head + bilinear resize; N6 swaps
in ``UpsampleHead`` for Open Buildings, after Scale-MAE's upsampling decoder (Table 6:
+1.8 kNN over a vanilla MAE decoder). N6 vs N5 isolates the decoder; N5 vs N2 isolates
the Open Buildings + VIIRS targets.

Possible follow-ups if N6 helps: Scale-MAE's few transformer layers before upsampling
and its per-frequency output blocks; pooling VIIRS predictions over ~320 m blocks
instead of scoring every 10 m pixel (a VIIRS pixel is ~460 m, stored nearest-upsampled).

Trained on the osm_sampling H5 rebuilt with naip_10 + viirs + open_buildings (+ glo30 and
meta_canopy_height, unused here), so the samples match the control's tiles but the H5
indices are not aligned with the older builds. W&B project ``2026_10_23_naip_v13``
(next to N2-N4).
"""

from functools import partial

import naip_common

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.nn.supervision_head import (
    SupervisionModalityConfig,
    SupervisionTaskType,
)

# As in N2: naip_10 is encoded and decoded (and counts toward the token budget).
ENCODED_MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
    Modality.LANDSAT.name,
    Modality.NAIP_10.name,
    *naip_common.CONTROL_MODALITIES[3:],
]
TARGET_ONLY = [Modality.OPEN_BUILDINGS.name, Modality.VIIRS.name]
# Same weight as the other regression heads (base.py: base_weight 1.0 x 1.0), as in N3.
TARGET_WEIGHT = 1.0

# The osm_sampling H5 rebuilt on 2026-10-28 with naip_10, viirs, open_buildings, glo30
# and meta_canopy_height (the last two unused by these arms).
HIRES_H5_DIR = (
    "/weka/dfive-default/helios/dataset/osm_sampling/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_glo30_gse_landsat_meta_canopy_height_naip_10_open_buildings_"
    "openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_viirs_worldcereal_worldcover_"
    "worldpop_wri_canopy_height_map/"
    "1138828"
)

build_common_components = partial(
    naip_common.build_common_components, [Modality.NAIP_10.name, *TARGET_ONLY]
)


def build_dataset_config(common) -> OlmoEarthDatasetConfig:
    """The rebuilt H5 with all three target-only modalities."""
    return OlmoEarthDatasetConfig(
        h5py_dir=HIRES_H5_DIR,
        training_modalities=common.training_modalities,
    )


def build_model_config(
    common,
    hires_decoder: str,
    viirs_footprint_loss: str | None = None,
    mlp_heads: bool = False,
):
    """N2's model (naip_10 encoded) plus open_buildings / viirs regression heads.

    ``viirs_footprint_loss`` (N7: ``"log1p"``) scores the VIIRS head on footprint
    means of its 10 m predictions instead of per pixel; see
    ``SupervisionModalityConfig.footprint_loss``. None keeps N5/N6's per-pixel L1.
    ``mlp_heads`` (N7) turns every per-cell linear head (the six map targets and
    VIIRS) into a 2-layer MLP (``decoder="mlp"``); Open Buildings keeps
    ``hires_decoder``.
    """
    config = naip_common.build_model_config(common, ENCODED_MODALITIES)
    assert config.supervision_head_config is not None
    heads = config.supervision_head_config.modality_configs
    heads[Modality.OPEN_BUILDINGS.name] = SupervisionModalityConfig(
        task_type=SupervisionTaskType.REGRESSION,
        num_output_channels=Modality.OPEN_BUILDINGS.num_bands,
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
        footprint_loss=viirs_footprint_loss,
    )
    if mlp_heads:
        for cfg in heads.values():
            if cfg.decoder == "linear":
                cfg.decoder = "mlp"
    return config


def build_train_module_config(common):
    """The control's train module; the targets decode-only (never encoded)."""
    return naip_common.build_train_module_config(common, TARGET_ONLY)


def build_dataloader_config(common):
    """The control's dataloader; the targets out of the token budget."""
    return naip_common.build_dataloader_config(common, TARGET_ONLY)
