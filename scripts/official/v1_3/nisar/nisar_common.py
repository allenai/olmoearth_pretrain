"""Shared builders for the NISAR from-scratch runs (card t-7xgg82).

Both arms train the v1.3 control recipe (``claude_findings/control_pixtgt.py``) from
scratch on the same 2 x 100k-window mix:

* random-time: 100k windows of the osm_sampling H5 (2016-2024 dates, no NISAR), picked
  with ``OSM_RANDOM_TIME_FILTER_IDX_FILE`` (400k sample indices = 4 sub-tiles each).
* recent: 100k other osm_sampling locations re-materialized with 12 months spanning
  2025-10-20..2026-10-15, so S2 / S1 / Landsat / NISAR are co-temporal
  (``OSM_RECENT_H5_DIR``).

The arms differ only in whether ``nisar_asc`` / ``nisar_desc`` are training
modalities. NISAR gets no band dropout (like S1) and is an encoded modality (not
decode-only).
"""

import sys
from pathlib import Path

# The v1.3 control builders, one directory over.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "claude_findings"))

from common import build_common_components as _build_common_components  # noqa: E402
from common import build_trainer_config as _build_trainer_config  # noqa: E402

from olmo_core.train.common import Duration  # noqa: E402

from olmoearth_pretrain.data.concat import OlmoEarthConcatDatasetConfig  # noqa: E402
from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import (  # noqa: E402
    CommonComponents,
    SubCmd,
)

OSM_SAMPLING_H5_DIR = "/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828"
OSM_RANDOM_TIME_FILTER_IDX_FILE = "/weka/dfive-default/agent/favyen/2026-10-23-favyen-nisar-pretrain/selection/osm_sampling_random_time_100k_sample_idx.npy"
# Set once the osm_recent H5 is built.
OSM_RECENT_H5_DIR = "/weka/dfive-default/helios/dataset/osm_recent/TODO"

NISAR_MODALITIES = [Modality.NISAR_ASC.name, Modality.NISAR_DESC.name]
WANDB_PROJECT = "2026_10_23_nisar_pretrain"
# Same step count as the v1.3 control (667,200 steps). The default 300 epochs would be
# only ~470k steps on this smaller ~800k-sample mix.
MAX_STEPS = 667_200


def build_common_components_fn(with_nisar: bool):
    """``build_common_components`` for one arm (v1.2 modalities, + NISAR if asked)."""

    def build_common_components(
        script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
    ) -> CommonComponents:
        config = _build_common_components(script, cmd, run_name, cluster, overrides)
        if with_nisar:
            config.training_modalities = [
                *config.training_modalities,
                *NISAR_MODALITIES,
            ]
        return config

    return build_common_components


def build_dataset_config(common: CommonComponents) -> OlmoEarthConcatDatasetConfig:
    """random-time osm_sampling subset + osm_recent, on the arm's modalities."""
    return OlmoEarthConcatDatasetConfig(
        dataset_configs=[
            OlmoEarthDatasetConfig(
                h5py_dir=OSM_SAMPLING_H5_DIR,
                training_modalities=common.training_modalities,
                filter_idx_file=OSM_RANDOM_TIME_FILTER_IDX_FILE,
            ),
            OlmoEarthDatasetConfig(
                h5py_dir=OSM_RECENT_H5_DIR,
                training_modalities=common.training_modalities,
            ),
        ],
    )


def build_trainer_config(common: CommonComponents, module_path: str):
    """The v1.3 control trainer (in-loop evals re-import ``module_path``), own W&B project, 667,200 steps."""
    trainer_config = _build_trainer_config(common, module_path)
    trainer_config.callbacks["wandb"].project = WANDB_PROJECT
    trainer_config.max_duration = Duration.steps(MAX_STEPS)
    return trainer_config
