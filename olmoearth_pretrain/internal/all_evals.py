"""Launch script for evaluation allowing you to easily run all the evals for your model by just pointing at your training script."""

import importlib.util
import json
import os
import sys
from logging import getLogger
from typing import Any

from olmo_core.config import Config
from olmo_core.train.callbacks import (
    BeakerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.common import Duration, LoadStrategy
from olmo_core.train.config import TrainerConfig
from upath import UPath

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.datasets.normalize import NormMethod
from olmoearth_pretrain.evals.metrics import EvalMetric
from olmoearth_pretrain.internal.constants import EVAL_WANDB_PROJECT, WANDB_ENTITY
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
    main,
)
from olmoearth_pretrain.model_loader import patch_legacy_encoder_config
from olmoearth_pretrain.nn.pooling import PoolingType
from olmoearth_pretrain.train.callbacks import (
    DownstreamEvaluatorCallbackConfig,
    OlmoEarthWandBCallback,
)
from olmoearth_pretrain.train.callbacks.evaluator_callback import (
    DownstreamTaskConfig,
    EvalMode,
)
from olmoearth_pretrain.train.train_module.train_module import _strip_unknown_fields

logger = getLogger(__name__)


def load_user_module(path: str) -> Any:
    """Load the user module from the given path."""
    logger.info(f"Loading user module from {path}")

    # Add the script's directory to sys.path so relative imports work
    script_dir = os.path.dirname(os.path.abspath(path))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    # Ensure helios shim is available for dynamic module loading
    # The helios shim's meta path finder needs to be active when the module executes
    try:
        import helios  # noqa: F401 # This ensures the helios shim is loaded and meta path finder is active
    except ImportError:
        pass  # If helios is not available, continue without it

    spec = importlib.util.spec_from_file_location("user_module", path)
    assert spec is not None
    user_mod = importlib.util.module_from_spec(spec)
    sys.modules["user_module"] = user_mod
    loader = spec.loader
    assert loader is not None
    loader.exec_module(user_mod)
    return user_mod


def _load_path_from_argv() -> str | None:
    """Extract the checkpoint path from a ``--trainer.load_path=...`` CLI override."""
    prefix = "--trainer.load_path="
    for arg in sys.argv:
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def build_model_config_from_checkpoint(fallback_builder: Any) -> Any:
    """Wrap a model-config builder to reconstruct the architecture from a checkpoint.

    When ``LOAD_ARCH_FROM_CHECKPOINT`` is set, the returned builder reads
    ``{load_path}/config.json`` -- the fully-resolved config that ConfigSaverCallback
    writes alongside every checkpoint -- and deserializes its ``model`` block. This
    rebuilds the EXACT architecture the checkpoint weights expect, so train-time
    architecture overrides (e.g. ``--model.encoder_config.register_dim=768``) do NOT
    need to be re-passed at eval time.

    Falls back to ``fallback_builder`` (the training module's ``build_model_config``)
    when no ``--trainer.load_path`` is given or the ``config.json`` is missing -- e.g.
    older checkpoints or baseline models -- so existing flows are unaffected.
    """

    def builder(common: Any) -> Any:
        load_path = _load_path_from_argv()
        if load_path is None:
            logger.warning(
                "LOAD_ARCH_FROM_CHECKPOINT is set but no --trainer.load_path was "
                "provided; falling back to the module's build_model_config."
            )
            return fallback_builder(common)
        config_path = UPath(load_path) / "config.json"
        if not config_path.exists():
            logger.warning(
                "LOAD_ARCH_FROM_CHECKPOINT is set but %s does not exist; falling back "
                "to the module's build_model_config.",
                config_path,
            )
            return fallback_builder(common)
        logger.info("Reconstructing model architecture from %s", config_path)
        # Use the same reconstruction pipeline as the train-module compatibility check
        # (patch legacy fields, strip fields unknown to the current schema) so the eval
        # model and that check agree exactly.
        config_dict = patch_legacy_encoder_config(json.loads(config_path.read_text()))
        return Config.from_dict(_strip_unknown_fields(config_dict["model"]))

    return builder


EVAL_TASKS = {
    "m_eurosat": DownstreamTaskConfig(
        dataset="m-eurosat",
        embedding_batch_size=128,
        num_workers=0,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        eval_interval=Duration.epochs(5),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        eval_mode=EvalMode.KNN,
        primary_metric=EvalMetric.ACCURACY,
    ),
    "m_forestnet": DownstreamTaskConfig(
        dataset="m-forestnet",
        embedding_batch_size=64,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=False,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        eval_interval=Duration.epochs(5),
        input_modalities=[Modality.LANDSAT.name],
        eval_mode=EvalMode.KNN,
        primary_metric=EvalMetric.ACCURACY,
    ),
    "m_bigearthnet": DownstreamTaskConfig(
        dataset="m-bigearthnet",
        embedding_batch_size=64,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.epochs(5),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        eval_mode=EvalMode.KNN,
        primary_metric=EvalMetric.MACRO_F1,
    ),
    "m_so2sat": DownstreamTaskConfig(
        dataset="m-so2sat",
        embedding_batch_size=128,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.epochs(5),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        eval_mode=EvalMode.KNN,
        primary_metric=EvalMetric.ACCURACY,
    ),
    "m_brick_kiln": DownstreamTaskConfig(
        dataset="m-brick-kiln",
        embedding_batch_size=128,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.epochs(5),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        eval_mode=EvalMode.KNN,
        primary_metric=EvalMetric.ACCURACY,
    ),
    "m_sa_crop_type": DownstreamTaskConfig(
        dataset="m-sa-crop-type",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=False,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.1,
        eval_interval=Duration.epochs(10),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MIOU,
    ),
    "m_cashew_plant": DownstreamTaskConfig(
        dataset="m-cashew-plant",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=False,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.1,
        eval_interval=Duration.epochs(10),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MIOU,
    ),
    "mados": DownstreamTaskConfig(
        dataset="mados",
        embedding_batch_size=128,
        probe_batch_size=128,
        num_workers=8,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=False,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.01,
        eval_interval=Duration.epochs(10),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MICRO_F1,
    ),
    "sen1floods11": DownstreamTaskConfig(
        dataset="sen1floods11",
        embedding_batch_size=128,
        probe_batch_size=128,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        probe_lr=0.1,
        eval_interval=Duration.epochs(10),
        input_modalities=[Modality.SENTINEL1.name],
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MIOU,
    ),
    "pastis_sentinel2": DownstreamTaskConfig(
        dataset="pastis",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        probe_lr=0.1,
        eval_interval=Duration.epochs(50),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MIOU,
    ),
    "pastis_sentinel1": DownstreamTaskConfig(
        dataset="pastis",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        probe_lr=0.1,
        eval_interval=Duration.epochs(50),
        input_modalities=[Modality.SENTINEL1.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MIOU,
    ),
    "pastis_sentinel1_sentinel2": DownstreamTaskConfig(
        dataset="pastis",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        probe_lr=0.1,
        eval_interval=Duration.epochs(20),
        input_modalities=[Modality.SENTINEL1.name, Modality.SENTINEL2_L2A.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MIOU,
    ),
    "pastis128_sentinel2": DownstreamTaskConfig(
        dataset="pastis128",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=2,
        pooling_type=PoolingType.MAX,
        norm_stats_from_pretrained=True,
        probe_lr=0.1,
        eval_interval=Duration.epochs(50),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MIOU,
    ),
    "pastis128_sentinel1": DownstreamTaskConfig(
        dataset="pastis128",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        probe_lr=0.1,
        eval_interval=Duration.epochs(50),
        input_modalities=[Modality.SENTINEL1.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MIOU,
    ),
    "pastis128_sentinel1_sentinel2": DownstreamTaskConfig(
        dataset="pastis128",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        probe_lr=0.1,
        eval_interval=Duration.epochs(20),
        input_modalities=[Modality.SENTINEL1.name, Modality.SENTINEL2_L2A.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MIOU,
    ),
    # TODO: Auto-generate EVAL_TASKS from registry entries. Most of this config
    # (dataset name, task_type -> eval_mode, modalities) is not task-specific and
    # can be derived from EvalDatasetEntry. Only batch sizes and learning rates
    # need manual tuning. See: olmoearth_pretrain.evals.studio_ingest.registry
    "tolbi_crop": DownstreamTaskConfig(
        dataset="tolbi_crop",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=16,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.1,
        eval_interval=Duration.epochs(10),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
    ),
    # TODO: commenting out for now to avoid the errors.
    # "burnrisk_8d_nbac": DownstreamTaskConfig(
    #     dataset="burnrisk_8d_nbac",
    #     embedding_batch_size=32,
    #     probe_batch_size=16,
    #     patch_size=5,
    #     num_workers=4,
    #     pooling_type=PoolingType.MEAN,
    #     norm_stats_from_pretrained=True,
    #     norm_method=NormMethod.NORM_NO_CLIP_2_STD,
    #     probe_lr=0.0001,
    #     eval_interval=Duration.epochs(10),
    #     input_modalities=[Modality.SENTINEL2_L2A.name],
    #     epochs=50,
    #     eval_mode=EvalMode.LINEAR_PROBE,
    #     use_dice_loss=True,
    #     primary_metric=EvalMetric.CLASS_F1,
    #     primary_metric_class=1,
    # ),
    "yemen_crop": DownstreamTaskConfig(
        dataset="yemen_crop",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        eval_interval=Duration.epochs(10),
        probe_lr=0.001,
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
    ),
    "geo_ecosystem_annual_test": DownstreamTaskConfig(
        dataset="geo_ecosystem_annual_test",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=8,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.01,
        eval_interval=Duration.epochs(10),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.OVERALL_ACC,
    ),
    "forest_loss_driver": DownstreamTaskConfig(
        dataset="forest_loss_driver",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=8,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.01,
        eval_interval=Duration.epochs(10),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
    ),
    "nigeria_settlement": DownstreamTaskConfig(
        dataset="nigeria_settlement",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=8,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.01,
        eval_interval=Duration.epochs(10),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
    ),
    "nandi_crop_map": DownstreamTaskConfig(
        dataset="nandi_crop_map",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=8,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.01,
        eval_interval=Duration.epochs(10),
        input_modalities=[
            Modality.SENTINEL2_L2A.name,
        ],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
    ),
    "awf_lulc_map": DownstreamTaskConfig(
        dataset="awf_lulc_map",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=8,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.01,
        eval_interval=Duration.epochs(10),
        input_modalities=[
            Modality.SENTINEL2_L2A.name,
        ],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
    ),
    # this eval is very large and can lead to
    # OOM errors. Skipping for now.
    # "oil_spill_detection": DownstreamTaskConfig(
    #     dataset="oil_spill_detection",
    #     embedding_batch_size=128,
    #     probe_batch_size=8,
    #     num_workers=8,
    #     pooling_type=PoolingType.MEAN,
    #     norm_stats_from_pretrained=True,
    #     norm_method=NormMethod.NORM_NO_CLIP_2_STD,
    #     probe_lr=0.01,
    #     eval_interval=Duration.epochs(10),
    #     input_modalities=[Modality.SENTINEL1.name],
    #     epochs=50,
    #     eval_mode=EvalMode.LINEAR_PROBE,
    # ),
}

PRETRAIN_SUBSET_H5PY_DIR = "/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828"

# Auxiliary probe eval set: drawn from the osmbig corpus, which is disjoint
# from the osm_sampling pretraining corpus used in scripts/official/*. Using
# osmbig keeps WorldCover/OSM/SRTM probes out-of-sample. The other map
# modalities (CDL, WORLDCEREAL, WRI canopy) aren't present in osmbig, so their
# probes fall back to PRETRAIN_SUBSET_H5PY_DIR (in-distribution).
PRETRAIN_AUX_EVAL_H5PY_DIR = "/weka/dfive-default/helios/dataset/osmbig/h5py_data_w_missing_timesteps_zstd_3_128_x_4/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/1297928"

MAP_MODALITY_PROBE_INPUTS = [
    Modality.SENTINEL2_L2A.name,
]
MAP_MODALITY_PROBE_INPUT_SUFFIX = "_".join(MAP_MODALITY_PROBE_INPUTS)

# Additional input-modality combinations used only for the SRTM probe so we can
# compare elevation regression quality from S1, S2, and S1+S2 inputs.
SRTM_PROBE_INPUT_VARIANTS: list[list[str]] = [
    [Modality.SENTINEL1.name],
    [Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name],
]


def _map_modality_probe(
    *,
    dataset: str,
    target_modality: str,
    primary_metric: EvalMetric,
    h5py_dir: str,
    split_strategy: str = "random",
    input_modalities: list[str] | None = None,
) -> DownstreamTaskConfig:
    """Build a uniform DownstreamTaskConfig for a decode-only map modality probe."""
    return DownstreamTaskConfig(
        dataset=dataset,
        embedding_batch_size=16,
        probe_batch_size=4,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=False,
        eval_interval=Duration.epochs(10),
        input_modalities=input_modalities
        if input_modalities is not None
        else MAP_MODALITY_PROBE_INPUTS,
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        probe_lr=0.01,
        primary_metric=primary_metric,
        h5py_dir=h5py_dir,
        pretrain_target_modality=target_modality,
        pretrain_train_samples=6144,
        pretrain_valid_samples=3072,
        pretrain_test_samples=3072,
        pretrain_split_strategy=split_strategy,
    )


EVAL_TASKS.update(
    {
        # Out-of-sample probes (osmbig).
        f"pretrain_worldcover_probe_{MAP_MODALITY_PROBE_INPUT_SUFFIX}": _map_modality_probe(
            dataset="pretrain_subset_worldcover",
            target_modality=Modality.WORLDCOVER.name,
            primary_metric=EvalMetric.MIOU,
            h5py_dir=PRETRAIN_AUX_EVAL_H5PY_DIR,
        ),
        f"pretrain_osm_probe_{MAP_MODALITY_PROBE_INPUT_SUFFIX}": _map_modality_probe(
            dataset="pretrain_subset_osm",
            target_modality=Modality.OPENSTREETMAP_RASTER.name,
            primary_metric=EvalMetric.MIOU,
            h5py_dir=PRETRAIN_AUX_EVAL_H5PY_DIR,
        ),
        f"pretrain_srtm_regression_{MAP_MODALITY_PROBE_INPUT_SUFFIX}": _map_modality_probe(
            dataset="pretrain_subset_srtm",
            target_modality=Modality.SRTM.name,
            primary_metric=EvalMetric.NEG_RMSE,
            h5py_dir=PRETRAIN_AUX_EVAL_H5PY_DIR,
        ),
        # In-distribution probes (osm_sampling) for map modalities absent from osmbig.
        f"pretrain_canopy_regression_{MAP_MODALITY_PROBE_INPUT_SUFFIX}": _map_modality_probe(
            dataset="pretrain_subset_canopy",
            target_modality=Modality.WRI_CANOPY_HEIGHT_MAP.name,
            primary_metric=EvalMetric.NEG_RMSE,
            h5py_dir=PRETRAIN_SUBSET_H5PY_DIR,
        ),
        f"pretrain_cdl_probe_{MAP_MODALITY_PROBE_INPUT_SUFFIX}": _map_modality_probe(
            dataset="pretrain_subset_cdl",
            target_modality=Modality.CDL.name,
            primary_metric=EvalMetric.MIOU,
            h5py_dir=PRETRAIN_SUBSET_H5PY_DIR,
        ),
        f"pretrain_worldcereal_probe_{MAP_MODALITY_PROBE_INPUT_SUFFIX}": _map_modality_probe(
            dataset="pretrain_subset_worldcereal",
            target_modality=Modality.WORLDCEREAL.name,
            primary_metric=EvalMetric.MIOU,
            h5py_dir=PRETRAIN_SUBSET_H5PY_DIR,
        ),
        # Geographic-holdout variants: train/val/test split by spatial bins
        # so the test set is geographically disjoint from train.
        f"pretrain_worldcover_probe_geo_{MAP_MODALITY_PROBE_INPUT_SUFFIX}": _map_modality_probe(
            dataset="pretrain_subset_worldcover",
            target_modality=Modality.WORLDCOVER.name,
            primary_metric=EvalMetric.MIOU,
            h5py_dir=PRETRAIN_AUX_EVAL_H5PY_DIR,
            split_strategy="geographic",
        ),
        f"pretrain_osm_probe_geo_{MAP_MODALITY_PROBE_INPUT_SUFFIX}": _map_modality_probe(
            dataset="pretrain_subset_osm",
            target_modality=Modality.OPENSTREETMAP_RASTER.name,
            primary_metric=EvalMetric.MIOU,
            h5py_dir=PRETRAIN_AUX_EVAL_H5PY_DIR,
            split_strategy="geographic",
        ),
        f"pretrain_srtm_regression_geo_{MAP_MODALITY_PROBE_INPUT_SUFFIX}": _map_modality_probe(
            dataset="pretrain_subset_srtm",
            target_modality=Modality.SRTM.name,
            primary_metric=EvalMetric.NEG_RMSE,
            h5py_dir=PRETRAIN_AUX_EVAL_H5PY_DIR,
            split_strategy="geographic",
        ),
        f"pretrain_canopy_regression_geo_{MAP_MODALITY_PROBE_INPUT_SUFFIX}": _map_modality_probe(
            dataset="pretrain_subset_canopy",
            target_modality=Modality.WRI_CANOPY_HEIGHT_MAP.name,
            primary_metric=EvalMetric.NEG_RMSE,
            h5py_dir=PRETRAIN_SUBSET_H5PY_DIR,
            split_strategy="geographic",
        ),
        f"pretrain_cdl_probe_geo_{MAP_MODALITY_PROBE_INPUT_SUFFIX}": _map_modality_probe(
            dataset="pretrain_subset_cdl",
            target_modality=Modality.CDL.name,
            primary_metric=EvalMetric.MIOU,
            h5py_dir=PRETRAIN_SUBSET_H5PY_DIR,
            split_strategy="geographic",
        ),
        f"pretrain_worldcereal_probe_geo_{MAP_MODALITY_PROBE_INPUT_SUFFIX}": _map_modality_probe(
            dataset="pretrain_subset_worldcereal",
            target_modality=Modality.WORLDCEREAL.name,
            primary_metric=EvalMetric.MIOU,
            h5py_dir=PRETRAIN_SUBSET_H5PY_DIR,
            split_strategy="geographic",
        ),
        # SRTM elevation regression from S1-only and S2+S1 inputs, so we can
        # compare elevation signal across modality combinations.
        **{
            f"pretrain_srtm_regression_{'_'.join(inputs)}": _map_modality_probe(
                dataset="pretrain_subset_srtm",
                target_modality=Modality.SRTM.name,
                primary_metric=EvalMetric.NEG_RMSE,
                h5py_dir=PRETRAIN_AUX_EVAL_H5PY_DIR,
                input_modalities=inputs,
            )
            for inputs in SRTM_PROBE_INPUT_VARIANTS
        },
        **{
            f"pretrain_srtm_regression_geo_{'_'.join(inputs)}": _map_modality_probe(
                dataset="pretrain_subset_srtm",
                target_modality=Modality.SRTM.name,
                primary_metric=EvalMetric.NEG_RMSE,
                h5py_dir=PRETRAIN_AUX_EVAL_H5PY_DIR,
                split_strategy="geographic",
                input_modalities=inputs,
            )
            for inputs in SRTM_PROBE_INPUT_VARIANTS
        },
        # Embedding diagnostics on standard downstream datasets, so we can track
        # representation quality (effective rank / norm / cosine stats) on real
        # eval distributions alongside the probe metrics.
        "m_eurosat_embed_diag": DownstreamTaskConfig(
            dataset="m-eurosat",
            embedding_batch_size=128,
            num_workers=0,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            norm_method=NormMethod.NORM_NO_CLIP_2_STD,
            eval_interval=Duration.epochs(5),
            input_modalities=[Modality.SENTINEL2_L2A.name],
            eval_mode=EvalMode.EMBEDDING_DIAGNOSTICS,
        ),
        "pastis_sentinel2_embed_diag": DownstreamTaskConfig(
            dataset="pastis",
            embedding_batch_size=32,
            num_workers=2,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_interval=Duration.epochs(50),
            input_modalities=[Modality.SENTINEL2_L2A.name],
            eval_mode=EvalMode.EMBEDDING_DIAGNOSTICS,
        ),
    }
)

EMBED_DIAG_TASKS = {
    "pretrain_subset": DownstreamTaskConfig(
        dataset="pretrain_subset",
        embedding_batch_size=4,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=False,
        eval_interval=Duration.epochs(1),
        input_modalities=[
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.LANDSAT.name,
        ],
        eval_mode=EvalMode.EMBEDDING_DIAGNOSTICS,
        h5py_dir=PRETRAIN_SUBSET_H5PY_DIR,
        pretrain_max_samples=256,
    ),
}

FT_EVAL_TASKS = {
    "m_eurosat": DownstreamTaskConfig(
        dataset="m-eurosat",
        ft_batch_size=64,
        num_workers=0,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        epochs=50,
        primary_metric=EvalMetric.ACCURACY,
    ),
    "m_bigearthnet": DownstreamTaskConfig(
        dataset="m-bigearthnet",
        ft_batch_size=16,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        epochs=50,
        primary_metric=EvalMetric.MACRO_F1,
    ),
    "m_so2sat": DownstreamTaskConfig(
        dataset="m-so2sat",
        ft_batch_size=16,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        epochs=50,
        primary_metric=EvalMetric.ACCURACY,
    ),
    "m_sa_crop_type": DownstreamTaskConfig(
        dataset="m-sa-crop-type",
        ft_batch_size=8,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=False,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        epochs=50,
        primary_metric=EvalMetric.MIOU,
    ),
    "mados": DownstreamTaskConfig(
        dataset="mados",
        ft_batch_size=16,
        num_workers=8,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=False,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        epochs=50,
        primary_metric=EvalMetric.MICRO_F1,
    ),
    "pastis_sentinel2": DownstreamTaskConfig(
        dataset="pastis",
        ft_batch_size=16,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        primary_metric=EvalMetric.MIOU,
    ),
    "m_brick_kiln": DownstreamTaskConfig(
        dataset="m-brick-kiln",
        ft_batch_size=64,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        epochs=50,
        primary_metric=EvalMetric.ACCURACY,
    ),
    "sen1floods11": DownstreamTaskConfig(
        dataset="sen1floods11",
        ft_batch_size=32,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        epochs=50,
        primary_metric=EvalMetric.MIOU,
    ),
    # Cashew plant requires a larger patch size; 16 performed best.
    "m_cashew_plant": DownstreamTaskConfig(
        dataset="m-cashew-plant",
        ft_batch_size=4,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=False,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        epochs=50,
        patch_size=16,
        primary_metric=EvalMetric.MIOU,
    ),
    "m_forestnet": DownstreamTaskConfig(
        dataset="m-forestnet",
        ft_batch_size=4,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=False,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        epochs=50,
        primary_metric=EvalMetric.ACCURACY,
    ),
}


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build the trainer config for an experiment."""
    MAX_DURATION = Duration.epochs(300)
    METRICS_COLLECT_INTERVAL = 10
    CANCEL_CHECK_INTERVAL = 1
    LOAD_STRATEGY = LoadStrategy.if_available
    checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)
    wandb_callback = OlmoEarthWandBCallback(
        name=common.run_name,
        project=EVAL_WANDB_PROJECT,
        entity=WANDB_ENTITY,
        enabled=True,  # set to False to avoid wandb errors
        upload_dataset_distribution_pre_train=False,
        upload_modality_data_band_distribution_pre_train=False,
    )
    # Safe to collect everys tep for now
    garbage_collector_callback = GarbageCollectorCallback(gc_interval=1)
    trainer_config = (
        TrainerConfig(
            work_dir=common.save_folder,
            load_strategy=LOAD_STRATEGY,
            save_folder=common.save_folder,
            cancel_check_interval=CANCEL_CHECK_INTERVAL,
            metrics_collect_interval=METRICS_COLLECT_INTERVAL,
            max_duration=MAX_DURATION,
            checkpointer=checkpointer_config,
        )
        .with_callback("wandb", wandb_callback)
        .with_callback("gpu_memory_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=(
                    EMBED_DIAG_TASKS
                    if os.environ.get("EMBEDDING_DIAGNOSTICS_ONLY")
                    else FT_EVAL_TASKS
                    if os.environ.get("FINETUNE")
                    else EVAL_TASKS
                ),
                eval_on_startup=True,
                cancel_after_first_eval=True,
                run_on_test=True,
            ),
        )
        .with_callback("garbage_collector", garbage_collector_callback)
        .with_callback("beaker", BeakerCallback())
    )
    return trainer_config


if __name__ == "__main__":
    module_path = os.environ.get("TRAIN_SCRIPT_PATH")
    if module_path is None:
        raise ValueError("TRAIN_SCRIPT_PATH environment variable must be set")
    user_mod = load_user_module(module_path)

    try:
        build_common_components = user_mod.build_common_components
    except AttributeError:
        from olmoearth_pretrain.internal.common import build_common_components

    # if the user module has no train module config builder, because it is an external model, we can just pass None
    # If the model is an olmoearth model, we need to build the train module config to load the checkpoint
    try:
        build_train_module_config = user_mod.build_train_module_config
    except AttributeError:
        build_train_module_config = None

    build_model_config = user_mod.build_model_config
    # Optionally reconstruct the architecture from the checkpoint's saved config.json,
    # so train-time architecture overrides don't need to be re-passed at eval time.
    if os.environ.get("LOAD_ARCH_FROM_CHECKPOINT"):
        build_model_config = build_model_config_from_checkpoint(build_model_config)
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        trainer_config_builder=build_trainer_config,
        train_module_config_builder=build_train_module_config,
    )
