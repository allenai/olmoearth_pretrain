"""Shared builders for the NISAR + NAIP from-scratch runs (card t-grl7q6, 2026-10-27).

Both runs are ``with_nisar_tb5120.py`` (v1.3 control recipe, NISAR asc/desc encoded,
token budget 5120 / floor 380) plus:

* ``naip_10`` (2.5 m NAIP RGBIR on the S2 token grid) as an ordinary encoded modality,
  as in the NAIP N2 arm (``favyen/20261023-naip-v13``).
* Data with NAIP in both halves:
  - random-time: the osm_sampling naip_10 H5 (same 1,138,828 samples as the gse H5
    the NISAR arms use, in a different order) restricted to the same 100k tiles
    (filter remapped by tile lat/lon; 400,000 samples, 70,068 with NAIP);
  - recent: osm_recent (2025-10..2026-10, NISAR) rebuilt with 2024/2025 NAIP from AWS
    naip-analytic for its CONUS tiles.
* Masking: per sample, ``e ~ U{1..n-1}`` of the ``n`` present encode-decode band sets
  are visible (``num_visible="uniform"``) and the tokens of a visible band set that
  are not encoded are predicted (``decode_visible_remainder``); hidden band sets keep
  half predicted, half unused. Decode-only maps are unchanged.

Run A (``nisar_naip.py``) keeps the all-token budget. Run B
(``nisar_naip_encbudget.py``) budgets the online-encoder tokens instead, with one
visible count per rank batch and a per-sample predicted-token cap.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "claude_findings"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import build_dataloader_config as _build_dataloader_config  # noqa: E402
from common import build_train_module_config as _build_train_module_config  # noqa: E402
from nisar_common import NISAR_MODALITIES  # noqa: E402
from nisar_common import build_common_components_fn as _nisar_components_fn  # noqa: E402
from nisar_common import build_trainer_config as _build_trainer_config  # noqa: E402
from with_nisar_tb5120 import SCALED_MIN_TOKENS, SCALED_TOKEN_BUDGET  # noqa: E402

from olmoearth_pretrain.data.concat import OlmoEarthConcatDatasetConfig  # noqa: E402
from olmoearth_pretrain.data.constants import Modality  # noqa: E402
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig  # noqa: E402
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig  # noqa: E402
from olmoearth_pretrain.internal.experiment import CommonComponents, SubCmd  # noqa: E402
from olmoearth_pretrain.train.masking import MaskingConfig  # noqa: E402
from olmoearth_pretrain.train.train_module.latent_mim import (  # noqa: E402
    LatentMIMTrainModuleConfig,
)

OSM_SAMPLING_NAIP_H5_DIR = (
    "/weka/dfive-default/helios/dataset/osm_sampling/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_landsat_naip_10_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_"
    "worldcereal_worldcover_wri_canopy_height_map/1138828"
)
# The NISAR arms' random-time tiles, as indices into the naip_10 H5 above
# (bin/remap_random_time_filter.py in the t-grl7q6 session folder).
OSM_RANDOM_TIME_NAIP_FILTER_IDX_FILE = (
    "/weka/dfive-default/helios/dataset/osm_recent/filters/"
    "osm_sampling_naip10_random_time_100k_sample_idx.npy"
)
OSM_RECENT_NAIP_H5_DIR = (
    "/weka/dfive-default/helios/dataset/osm_recent/"
    "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
    "cdl_gse_landsat_naip_10_nisar_asc_nisar_desc_openstreetmap_raster_sentinel1_"
    "sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/399848"
)

# Encode-decode modalities: S2 + S1 + Landsat + NISAR asc/desc + NAIP, one band set each.
NUM_ENCODE_DECODE_BANDSETS = 6
# Like tb5120: rank microbatch 32 (patch-discrimination memory); the smoke tests decide
# whether predicting the visible remainder too needs 16.
RANK_MICROBATCH_SIZE = 32


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """The v1.3 control modalities + NISAR asc/desc + naip_10."""
    config = _nisar_components_fn(True)(script, cmd, run_name, cluster, overrides)
    assert all(m in config.training_modalities for m in NISAR_MODALITIES)
    config.training_modalities = [
        *config.training_modalities,
        Modality.NAIP_10.name,
    ]
    return config


def build_dataset_config(common: CommonComponents) -> OlmoEarthConcatDatasetConfig:
    """random-time osm_sampling 100k (with naip_10) + osm_recent (with NISAR + naip_10)."""
    return OlmoEarthConcatDatasetConfig(
        dataset_configs=[
            OlmoEarthDatasetConfig(
                h5py_dir=OSM_SAMPLING_NAIP_H5_DIR,
                training_modalities=common.training_modalities,
                filter_idx_file=OSM_RANDOM_TIME_NAIP_FILTER_IDX_FILE,
            ),
            OlmoEarthDatasetConfig(
                h5py_dir=OSM_RECENT_NAIP_H5_DIR,
                training_modalities=common.training_modalities,
            ),
        ],
    )


def set_masking(
    masking_config: MaskingConfig, max_decode_tokens: int | None = None
) -> None:
    """Uniform visible count + predicted visible remainder (+ optional predicted-token cap)."""
    strategy = masking_config.strategy_config
    assert strategy["type"] == "random_time_with_decode", strategy
    strategy["num_visible"] = "uniform"
    strategy["decode_visible_remainder"] = True
    if max_decode_tokens is not None:
        strategy["max_decode_tokens"] = max_decode_tokens


def build_train_module_config(
    common: CommonComponents, max_decode_tokens: int | None = None
) -> LatentMIMTrainModuleConfig:
    """The v1.3 train module (pixel targets) with the new masking, rank microbatch 32."""
    config = _build_train_module_config(common, pixel_targets=True)
    set_masking(config.masking_config, max_decode_tokens)
    config.rank_microbatch_size = RANK_MICROBATCH_SIZE
    return config


def build_dataloader_config(
    common: CommonComponents, max_decode_tokens: int | None = None
) -> OlmoEarthDataLoaderConfig:
    """tb5120's dataloader (budget 5120 / floor 380) with the new masking."""
    config = _build_dataloader_config(common)
    config.token_budget = SCALED_TOKEN_BUDGET
    config.min_tokens_per_instance = SCALED_MIN_TOKENS
    assert config.masking_config is not None
    set_masking(config.masking_config, max_decode_tokens)
    return config


def build_trainer_config(common: CommonComponents, module_path: str):
    """The NISAR arms' trainer (v1.3 in-loop evals re-importing ``module_path``)."""
    return _build_trainer_config(common, module_path)
