"""Launch script for evaluation allowing you to easily run all the evals for your model by just pointing at your training script."""

import importlib.util
import json
import os
import sys
from dataclasses import replace
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
from olmoearth_pretrain.evals.balanced_trial import BalancedTrialConfig
from olmoearth_pretrain.evals.datasets.normalize import NormMethod
from olmoearth_pretrain.evals.datasets.rslearn_dataset import (
    L8QA_CLOUD_ONLY_BITS_MASK,
    SCL_CLOUDLESS_CLASSES,
)
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
    # 64x64-tiled variants of the two 256px segmentation tasks: each native
    # 256x256 image becomes 16 non-overlapping 64x64 tiles, shrinking the token
    # grid the register read sees (64/patch vs 256/patch). Used to test whether
    # the large-grid read dilution drives the register regressions on these tasks.
    # Not directly comparable in absolute terms to the 256px versions (less spatial
    # context per window); the signal is the rope-vs-latents gap at 64 vs 256.
    "m_sa_crop_type_64": DownstreamTaskConfig(
        dataset="m-sa-crop-type",
        tile_size=64,
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
    "m_cashew_plant_64": DownstreamTaskConfig(
        dataset="m-cashew-plant",
        tile_size=64,
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
    # 50Cities: single-timestep S2+S1 land-cover segmentation, 64x64 tiles.
    # Three split modes (random / by_city / by_continent), each with an S2-only,
    # an S1-only, and an S1+S2 task. The split mode is carried by the dataset
    # name; the modality choice is the per-task input_modalities here.
    "fifty_cities_sentinel2": DownstreamTaskConfig(
        dataset="fifty_cities",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        probe_lr=0.1,
        eval_interval=Duration.epochs(50),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MIOU,
    ),
    "fifty_cities_sentinel1": DownstreamTaskConfig(
        dataset="fifty_cities",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        probe_lr=0.1,
        eval_interval=Duration.epochs(50),
        input_modalities=[Modality.SENTINEL1.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MIOU,
    ),
    "fifty_cities_sentinel1_sentinel2": DownstreamTaskConfig(
        dataset="fifty_cities",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        probe_lr=0.1,
        eval_interval=Duration.epochs(20),
        input_modalities=[Modality.SENTINEL1.name, Modality.SENTINEL2_L2A.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MIOU,
    ),
    "fifty_cities_by_city_sentinel2": DownstreamTaskConfig(
        dataset="fifty_cities_by_city",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        probe_lr=0.1,
        eval_interval=Duration.epochs(50),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MIOU,
    ),
    "fifty_cities_by_city_sentinel1": DownstreamTaskConfig(
        dataset="fifty_cities_by_city",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        probe_lr=0.1,
        eval_interval=Duration.epochs(50),
        input_modalities=[Modality.SENTINEL1.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MIOU,
    ),
    "fifty_cities_by_city_sentinel1_sentinel2": DownstreamTaskConfig(
        dataset="fifty_cities_by_city",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        probe_lr=0.1,
        eval_interval=Duration.epochs(20),
        input_modalities=[Modality.SENTINEL1.name, Modality.SENTINEL2_L2A.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MIOU,
    ),
    "fifty_cities_by_continent_sentinel2": DownstreamTaskConfig(
        dataset="fifty_cities_by_continent",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        probe_lr=0.1,
        eval_interval=Duration.epochs(50),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MIOU,
    ),
    "fifty_cities_by_continent_sentinel1": DownstreamTaskConfig(
        dataset="fifty_cities_by_continent",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        probe_lr=0.1,
        eval_interval=Duration.epochs(50),
        input_modalities=[Modality.SENTINEL1.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MIOU,
    ),
    "fifty_cities_by_continent_sentinel1_sentinel2": DownstreamTaskConfig(
        dataset="fifty_cities_by_continent",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=4,
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
    # AEF-style single-labeled-center-pixel S2 timeseries datasets, cropped to
    # 32x32 and ingested via the registry. Modeled as segmentation over
    # label_raster (nodata=255 elsewhere) -> linear probe, overall accuracy.
    "africa_crop_mask": DownstreamTaskConfig(
        dataset="africa_crop_mask",
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
    "canada_crops_coarse": DownstreamTaskConfig(
        dataset="canada_crops_coarse",
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
    "canada_crops_fine": DownstreamTaskConfig(
        dataset="canada_crops_fine",
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
    "descals": DownstreamTaskConfig(
        dataset="descals",
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
    "ethiopia_crops": DownstreamTaskConfig(
        dataset="ethiopia_crops",
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
    "glance": DownstreamTaskConfig(
        dataset="glance",
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
    "lcmap_lu": DownstreamTaskConfig(
        dataset="lcmap_lu",
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
    "us_trees": DownstreamTaskConfig(
        dataset="us_trees",
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
    "surface_fuels": DownstreamTaskConfig(
        dataset="surface_fuels",
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
    "kenya_intercropping": DownstreamTaskConfig(
        dataset="kenya_intercropping",
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
    # Vessel attribute datasets (single-timestep, centered vessel). Length is a
    # regression task (RMSE); type is a 9-class classification task. We enable
    # use_center_token because the crops are centered at the vessel; this way, we
    # use the token at the center spatial patch instead of pooling over all spatial
    # patches.
    "small_landsat_vessel_type": DownstreamTaskConfig(
        dataset="small_landsat_vessel_type",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=8,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.01,
        eval_interval=Duration.epochs(10),
        input_modalities=[Modality.LANDSAT.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        use_center_token=True,
    ),
    "small_landsat_vessel_length": DownstreamTaskConfig(
        dataset="small_landsat_vessel_length",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=8,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.01,
        eval_interval=Duration.epochs(10),
        input_modalities=[Modality.LANDSAT.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.RMSE,
        use_center_token=True,
    ),
    "small_sentinel1_vessel_type": DownstreamTaskConfig(
        dataset="small_sentinel1_vessel_type",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=8,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.01,
        eval_interval=Duration.epochs(10),
        input_modalities=[Modality.SENTINEL1.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        use_center_token=True,
    ),
    "small_sentinel1_vessel_length": DownstreamTaskConfig(
        dataset="small_sentinel1_vessel_length",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=8,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.01,
        eval_interval=Duration.epochs(10),
        input_modalities=[Modality.SENTINEL1.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.RMSE,
        use_center_token=True,
    ),
    "small_sentinel2_vessel_type": DownstreamTaskConfig(
        dataset="small_sentinel2_vessel_type",
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
        use_center_token=True,
    ),
    "small_sentinel2_vessel_length": DownstreamTaskConfig(
        dataset="small_sentinel2_vessel_length",
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
        primary_metric=EvalMetric.RMSE,
        use_center_token=True,
    ),
    # GeoBench v2 (`gb2-*` datasets; keys use underscores for Hydra)
    "gb2_benv2": DownstreamTaskConfig(
        dataset="gb2-benv2",
        embedding_batch_size=16,
        probe_batch_size=16,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        eval_interval=Duration.epochs(50),
        input_modalities=[Modality.SENTINEL1.name, Modality.SENTINEL2_L2A.name],
        eval_mode=EvalMode.KNN,
        # Multilabel: GeoBench-2 reports micro-averaged mAP (threshold-free).
        primary_metric=EvalMetric.MICRO_MAP,
        epochs=50,
    ),
    "gb2_biomassters": DownstreamTaskConfig(
        dataset="gb2-biomassters",
        embedding_batch_size=8,
        probe_batch_size=8,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=1e-3,
        eval_interval=Duration.epochs(50),
        input_modalities=[Modality.SENTINEL1.name, Modality.SENTINEL2_L2A.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        # RMSE (on z-scored targets) to match what GeoBench-2 reports/ranks on.
        primary_metric=EvalMetric.RMSE,
    ),
    "gb2_burn_scars": DownstreamTaskConfig(
        dataset="gb2-burn_scars",
        embedding_batch_size=8,
        probe_batch_size=8,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.1,
        eval_interval=Duration.epochs(10),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MIOU,
        patch_size=4,
    ),
    # Only 1 SAR amplitude band is provided, so we pass it in as a Sentinel1
    # modality but only the "vv" band is used.
    "gb2_caffe": DownstreamTaskConfig(
        dataset="gb2-caffe",
        embedding_batch_size=8,
        probe_batch_size=8,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.1,
        eval_interval=Duration.epochs(10),
        input_modalities=[Modality.SENTINEL1.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MIOU,
        patch_size=4,
    ),
    "gb2_cloudsen12": DownstreamTaskConfig(
        dataset="gb2-cloudsen12",
        embedding_batch_size=8,
        probe_batch_size=8,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.1,
        eval_interval=Duration.epochs(50),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MIOU,
        patch_size=4,
    ),
    "gb2_kuro_siwo": DownstreamTaskConfig(
        dataset="gb2-kuro_siwo",
        embedding_batch_size=16,
        probe_batch_size=16,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.1,
        eval_interval=Duration.epochs(50),
        input_modalities=[Modality.SENTINEL1.name, Modality.SRTM.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MIOU,
        patch_size=4,
    ),
    "gb2_spacenet2": DownstreamTaskConfig(
        dataset="gb2-spacenet2",
        embedding_batch_size=8,
        probe_batch_size=8,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.1,
        eval_interval=Duration.epochs(50),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MIOU,
        patch_size=4,
    ),
    "gb2_spacenet7": DownstreamTaskConfig(
        dataset="gb2-spacenet7",
        embedding_batch_size=8,
        probe_batch_size=8,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.1,
        eval_interval=Duration.epochs(10),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MIOU,
        patch_size=4,
    ),
    "gb2_flair2": DownstreamTaskConfig(
        dataset="gb2-flair2",
        embedding_batch_size=8,
        probe_batch_size=8,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.1,
        eval_interval=Duration.epochs(10),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MIOU,
        patch_size=4,
    ),
    "gb2_fotw": DownstreamTaskConfig(
        dataset="gb2-fotw",
        embedding_batch_size=16,
        probe_batch_size=16,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.1,
        eval_interval=Duration.epochs(10),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MIOU,
        patch_size=4,
    ),
    "gb2_treesatai": DownstreamTaskConfig(
        dataset="gb2-treesatai",
        embedding_batch_size=16,
        probe_batch_size=16,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        eval_interval=Duration.epochs(10),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        eval_mode=EvalMode.KNN,
        # Multilabel: GeoBench-2 reports micro-averaged mAP (threshold-free).
        primary_metric=EvalMetric.MICRO_MAP,
        epochs=50,
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
    "lfmc_woody_3k": DownstreamTaskConfig(
        dataset="lfmc_woody_3k",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=8,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.00005,
        eval_interval=Duration.epochs(10),
        input_modalities=[
            Modality.SENTINEL2_L2A.name,
        ],
        patch_size=4,
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
    ),
    "lfmc_woody_3k_s1_s2": DownstreamTaskConfig(
        dataset="lfmc_woody_3k",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=8,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.00005,
        eval_interval=Duration.epochs(10),
        input_modalities=[
            Modality.SENTINEL1.name,
            Modality.SENTINEL2_L2A.name,
        ],
        patch_size=4,
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
    ),
    "mapbiomas_3k_dense": DownstreamTaskConfig(
        dataset="mapbiomas_3k_dense",
        embedding_batch_size=32,
        probe_batch_size=8,
        num_workers=8,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.001,
        eval_interval=Duration.epochs(10),
        input_modalities=[
            Modality.SENTINEL2_L2A.name,
        ],
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MACRO_F1,
    ),
    # "mapbiomas_3k_sparse": DownstreamTaskConfig(
    #     dataset="mapbiomas_3k_sparse",
    #     embedding_batch_size=32,
    #     probe_batch_size=8,
    #     num_workers=8,
    #     pooling_type=PoolingType.MEAN,
    #     norm_stats_from_pretrained=True,
    #     norm_method=NormMethod.NORM_NO_CLIP_2_STD,
    #     probe_lr=0.0001,
    #     eval_interval=Duration.epochs(10),
    #     input_modalities=[
    #         Modality.SENTINEL2_L2A.name,
    #     ],
    #     epochs=50,
    #     eval_mode=EvalMode.LINEAR_PROBE,
    #     primary_metric=EvalMetric.MACRO_F1,
    # ),
}

# Pretrain-subset evals read from frozen snapshots under presto_eval_sets, NOT
# from the live pretraining datasets: the live datasets are periodically
# cleaned up and are not reproducible to the sample when recreated, which both
# breaks these paths (the trailing dir name must equal the exact sample count)
# and silently changes the seed-derived eval splits. Snapshots are created with
# scripts/tools/20260611_snapshot_pretrain_eval_subset.py; the trailing count
# here must match the --total the snapshot was built with.
PRETRAIN_SUBSET_H5PY_DIR = "/weka/dfive-default/presto_eval_sets/pretrain_subset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/98304"

# Auxiliary probe eval set: drawn from the osmbig corpus, which is disjoint
# from the osm_sampling pretraining corpus used in scripts/official/*. Using
# osmbig keeps WorldCover/OSM/SRTM probes out-of-sample. The other map
# modalities (CDL, WORLDCEREAL, WRI canopy) aren't present in osmbig, so their
# probes fall back to PRETRAIN_SUBSET_H5PY_DIR (in-distribution).
PRETRAIN_AUX_EVAL_H5PY_DIR = "/weka/dfive-default/presto_eval_sets/pretrain_subset/osmbig/h5py_data_w_missing_timesteps_zstd_3_128_x_4/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/65536"

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

# The AEF supplemental evaluation datasets (arXiv:2507.22291): S2 timeseries
# crops carrying a single labeled center pixel each, ingested via the registry
# (their plain 32x32 segmentation variants are defined above).
AEF_SUPPLEMENTAL_DATASETS = (
    "africa_crop_mask",
    "canada_crops_coarse",
    "canada_crops_fine",
    "descals",
    "ethiopia_crops",
    "glance",
    "lcmap_lu",
    "us_trees",
)

# Year-aligned re-exports (2026-08-04): the same labels and windows, but the
# imagery is twelve ASCENDING 30-day Sentinel-1 + Sentinel-2 layers spanning the
# calendar year of the label, matching what AEF and Tessera are built over. The
# parents feed OlmoEarth a trailing year from the observation date (canada,
# ethiopia, us_trees) or a fixed Sep-Aug year (pastis), so the published
# comparisons were not input-matched. See
# scripts/tools/reanchor_year_aligned_dataset.py.
#
# Registered at ws16 only, like MATCHED_SUBSET_DATASETS below and for the same
# reason: the point is the three-way comparison against the precomputed
# products, which are ws16-only. Add the smaller context sizes if the
# spatial-context ablation is wanted here too.
#
# All eight AEF supplemental datasets are now re-exported and registered.
#
# lcmap_lu and us_trees additionally carry tessera as a *required* input, so on
# those two the resolved window set is intersected with Tessera's coverage
# (lcmap 26 409/26 513, us_trees 44 886/45 382) rather than being the
# S1+S2+gse intersection the other six use. That keeps AEF / Tessera /
# OlmoEarth on one identical window set per dataset -- which is the point --
# but it does mean their window counts are not comparable to the other six.
AEF_SUPPLEMENTAL_YEAR_ALIGNED = (
    # Ordered by window count, smallest first. Task registration order is
    # execution order inside every eval job (the evaluator walks the registry
    # dict), so this makes each job report its cheap datasets within minutes
    # instead of queueing them behind us_trees's ~75-minute tasks.
    "ethiopia_crops_year_aligned",  # 2 530 windows
    "africa_crop_mask_year_aligned",  # 2 556
    "canada_crops_fine_year_aligned",  # 14 566
    "canada_crops_coarse_year_aligned",  # 16 079
    "descals_year_aligned",  # 17 477
    "lcmap_lu_year_aligned",  # 26 513
    "glance_year_aligned",  # 34 885
    "us_trees_year_aligned",  # 45 382
)

# Matched-subset siblings: the same windows as their parent dataset, but with
# every embedding product marked required, so rslearn resolves ONE window set
# and OlmoEarth / AEF / Tessera are scored on exactly those windows.
#
# These exist for datasets where a product's coverage sits below the
# --min_coverage gate in wire_embedding_modalities.py. Enabling the product on
# the parent entry would drop its coverage-gap windows from every eval on that
# dataset — silently re-baselining numbers already recorded — so the stricter
# input set gets its own entry instead. us_trees_tessera: Tessera covers
# 44 894/45 382 (98.92%) vs the 99% gate; see docs/PrecomputedEmbeddingCoverage.md.
#
# Deliberately NOT part of AEF_SUPPLEMENTAL_DATASETS: that tuple is also the
# default --datasets for materialize_aef_supplemental_embeddings.py, and these
# names share their parent's weka_path, so including them would re-walk the same
# 45k windows under a second name. Registered at ws16 only — the point is the
# three-way comparison, and the precomputed baselines are ws16-only.
MATCHED_SUBSET_DATASETS = ("us_trees_tessera",)


# Window sizes the embedding evals run at by default for OlmoEarth
# checkpoints: the ws16 embedding-product convention plus smaller spatial
# contexts (8, 4, 1) to measure how much surrounding context the per-pixel
# embeddings rely on (and what a cheaper eval would cost in accuracy). ws16
# is registered first so the precomputed baselines — which keep one task per
# dataset — stay on the ws16 convention.
EMBEDDING_EVAL_WINDOW_SIZES = (16, 8, 4, 1)


def _embedding_eval_batch_scale(window_size: int) -> int:
    """Batch-size multiplier keeping tokens per batch constant across ws.

    Each window carries (window_size/patch_size)^2 spatial tokens, so halving
    the window quarters the tokens per window; scaling the batch by
    (16/ws)^2 keeps the token throughput (and for PASTIS the
    one-stored-sample-per-batch tiling property) identical to ws16.
    """
    return (16 // window_size) ** 2


# AEF's per-dataset "Max Trial Size (n)" column (their Table 1, read per class),
# used directly as our per-class draw size. Keyed by dataset-name prefix so the
# _year_aligned re-exports inherit their parent's value.
#
# Taking the whole column -- including the odd entries -- rather than only the
# round ones is what gives exact training-budget parity with AEF on all eight
# datasets. It also sidesteps a question we could not settle: whether 49/75/68
# are caps they chose or their least-populated classes binding. It does not
# matter, because every one of OUR least classes exceeds the corresponding
# value (ethiopia 96>49, canada fine 87>75, coarse 106>68, africa 318>200,
# descals 290>200, lcmap 588>300, glance 467>300, us_trees 393>300), so the
# value binds first everywhere and the draw is theirs by construction.
#
# The earlier reading -- that only the round values were caps and the odd ones
# were least classes -- implied AEF draws its rarest class in full on those
# three datasets, leaving ZERO of it in the remainder they evaluate on, in every
# fold. That would make their published ethiopia/canada figures K-1-class
# balanced accuracies with a wrong 1/K chance line, which a paper about
# rare-class performance under sparse labels is unlikely to be doing. Reading
# the column as a budget avoids attributing that to them AND leaves 12-288
# rarest-class rows in our eval set per dataset.
AEF_MAX_TRIAL_CAPS = {
    "ethiopia_crops": 49,
    "canada_crops_fine": 75,
    "canada_crops_coarse": 68,
    "africa_crop_mask": 200,
    "descals": 200,
    "lcmap_lu": 300,
    "glance": 300,
    "us_trees": 300,
}
DEFAULT_AEF_MAX_TRIAL_CAP = 300


def _aef_max_trial_cap(dataset: str) -> int:
    """AEF's per-class draw cap for a dataset (300 unless Table 1 says otherwise)."""
    for prefix, cap in AEF_MAX_TRIAL_CAPS.items():
        if dataset.startswith(prefix):
            return cap
    return DEFAULT_AEF_MAX_TRIAL_CAP


def _aef_ps1_task(
    name: str,
    eval_mode: EvalMode,
    window_size: int = 16,
    input_modalities: list[str] | None = None,
    scl_cloud_mask: bool = False,
    scl_cloud_classes: tuple[int, ...] | None = None,
    landsat_cloud_cover_max: float | None = None,
    l8_pixel_cloud_mask: bool = False,
    l8_pixel_cloud_bits: int | None = None,
) -> DownstreamTaskConfig:
    """AEF supplemental task under the per-pixel embedding-product convention.

    Each sample is center-cropped to a window_size x window_size window around
    its labeled pixel, OlmoEarth emits per-pixel (patch_size=1) embeddings
    int8 round-tripped like an embedding product, and only the labeled pixel's
    token is kept — the task runs as center-pixel classification
    (label_at_center_pixel + use_center_token). Balanced accuracy is the AEF
    paper's protocol metric.

    The KNN twin additionally runs AEF's balanced-trial protocol (their S4) on
    the embeddings it already materializes: a class-balanced draw from the
    pooled splits, scored on the remainder, repeated over AEF's k draws. It is
    hosted here rather than on the LP tasks because the KNN twin is the only
    single-instance job (embedding_eval_sweep.py emits one KNN job but eight LP
    jobs, one per swept LR), so the trials compute once instead of eight
    redundant times, and a neighbor lookup is the cheapest job to hang
    millisecond-scale closed-form fits off. The precomputed baselines (AEF,
    Tessera) run these same task objects, so they inherit the trials and stay
    directly comparable.
    """
    scale = _embedding_eval_batch_scale(window_size)
    return DownstreamTaskConfig(
        dataset=name,
        embedding_batch_size=32 * scale,
        probe_batch_size=8 * scale,
        num_workers=8,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        probe_lr=0.01,
        eval_interval=Duration.epochs(10),
        input_modalities=input_modalities or [Modality.SENTINEL2_L2A.name],
        epochs=50,
        eval_mode=eval_mode,
        primary_metric=EvalMetric.BALANCED_ACCURACY,
        window_size=window_size,
        patch_size=1,
        quantize_embeddings=True,
        use_center_token=True,
        label_at_center_pixel=True,
        scl_cloud_mask=scl_cloud_mask,
        scl_cloud_classes=scl_cloud_classes,
        landsat_cloud_cover_max=landsat_cloud_cover_max,
        l8_pixel_cloud_mask=l8_pixel_cloud_mask,
        l8_pixel_cloud_bits=l8_pixel_cloud_bits,
        balanced_trial=(
            BalancedTrialConfig(cap=_aef_max_trial_cap(name))
            if eval_mode == EvalMode.KNN
            else None
        ),
    )


# Embedding-product evals: OlmoEarth scored under the same conventions as the
# precomputed embedding products (AEF/Tessera) — per-pixel (patch_size=1)
# embeddings from fixed windows, int8 round-tripped. OlmoEarth checkpoints
# run every window size in EMBEDDING_EVAL_WINDOW_SIZES by default (ws16 is
# the product-parity convention; ws8/ws4/ws1 ablate the spatial context the
# embeddings are computed from). Kept separate from EVAL_TASKS and swept by
# embedding_eval_sweep.py (EMBEDDING_EVALS=1), which holds normalization
# fixed to pretraining stats and sweeps only the probe LR for olmoearth /
# aef / tessera_precomputed. The precomputed baselines run these same tasks
# with input_modalities overridden to the embedding modality and
# quantize_embeddings=False (they are already int8 at source); they keep one
# task per dataset, so they stay ws16-only.
#
# The AEF supplemental tasks are effectively pixel-wise classification, so each
# gets a KNN twin (`_knn`). The PASTIS tasks stay LP-only: their dense labels
# flatten to millions of train pixels, and KNN keeps every one as a reference
# point (cost scales with train x query pixels), unlike the LP which compresses
# them into a single weight matrix.
#
# The PASTIS tasks run on `pastis_rslearn`, an rslearn export that mirrors the
# pretraining dataset (12 monthly Planetary Computer mosaics per sensor on the
# native PASTIS patch grid; see
# olmoearth_pretrain/evals/datasets/pastis_rslearn_export.py) rather than the
# imagery shipped with the PASTIS benchmark. Each 128x128 patch is tiled into
# 16x16 windows (tile_samples). The gse/tessera layers were converted from the
# embeddings previously fetched by pastis_processor.py --embedding_products.


def _pastis_ps1_task(
    input_modalities: list[str], window_size: int = 16
) -> DownstreamTaskConfig:
    """PASTIS (rslearn export) under the per-pixel embedding-product convention."""
    scale = _embedding_eval_batch_scale(window_size)
    return DownstreamTaskConfig(
        dataset="pastis_rslearn",
        # At ws16, 64 = one full 128x128 stored sample (8x8 tiles of 16x16)
        # per batch, so each DataLoader worker's batch maps to exactly one
        # base-sample load with the tiled-__getitem__ cache; the (16/ws)^2
        # scaling preserves both that mapping and the tokens per batch at
        # smaller window sizes. Peak GPU memory at ws16 batch 32 was ~7.6GB,
        # so 64 stays far from OOM.
        embedding_batch_size=64 * scale,
        probe_batch_size=8 * scale,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        probe_lr=0.1,
        eval_interval=Duration.epochs(50),
        input_modalities=input_modalities,
        epochs=50,
        eval_mode=EvalMode.LINEAR_PROBE,
        primary_metric=EvalMetric.MIOU,
        window_size=window_size,
        patch_size=1,
        tile_samples=True,
        quantize_embeddings=True,
    )


# The _pretrain_export suffix marks that the PASTIS tasks read the
# pastis_rslearn pretraining-mirror export, distinguishing their metrics from
# earlier pastis_ws16_ps1_* runs on the benchmark-shipped imagery. One task
# set per window size in EMBEDDING_EVAL_WINDOW_SIZES, ws16 first.
EMBEDDING_EVAL_TASKS = {}
for _ws in EMBEDDING_EVAL_WINDOW_SIZES:
    EMBEDDING_EVAL_TASKS.update(
        {
            f"pastis_ws{_ws}_ps1_sentinel2_pretrain_export": _pastis_ps1_task(
                [Modality.SENTINEL2_L2A.name], window_size=_ws
            ),
            f"pastis_ws{_ws}_ps1_sentinel1_sentinel2_pretrain_export": (
                _pastis_ps1_task(
                    [Modality.SENTINEL1.name, Modality.SENTINEL2_L2A.name],
                    window_size=_ws,
                )
            ),
            **{
                f"{name}_ws{_ws}_ps1": _aef_ps1_task(
                    name, EvalMode.LINEAR_PROBE, window_size=_ws
                )
                for name in AEF_SUPPLEMENTAL_DATASETS
            },
            **{
                f"{name}_ws{_ws}_ps1_knn": _aef_ps1_task(
                    name, EvalMode.KNN, window_size=_ws
                )
                for name in AEF_SUPPLEMENTAL_DATASETS
            },
            # Matched-subset siblings at ws16 only (see MATCHED_SUBSET_DATASETS).
            **(
                {
                    f"{name}_ws16_ps1": _aef_ps1_task(
                        name, EvalMode.LINEAR_PROBE, window_size=16
                    )
                    for name in MATCHED_SUBSET_DATASETS
                }
                if _ws == 16
                else {}
            ),
            **(
                {
                    f"{name}_ws16_ps1_knn": _aef_ps1_task(
                        name, EvalMode.KNN, window_size=16
                    )
                    for name in MATCHED_SUBSET_DATASETS
                }
                if _ws == 16
                else {}
            ),
        }
    )

# Year-aligned tasks, ws16 only. Both a Sentinel-2-only and a Sentinel-1 +
# Sentinel-2 variant: the S1+S2 pair is the point of the re-export, while the
# S2-only pair isolates the year/ordering/cloud-filter change from the effect of
# adding a sensor -- without it, a delta against the parent task confounds the
# two. Same naming convention as the pastis embedding tasks.
#
# Each gets a linear-probe and a kNN variant, like its parent task above: the
# AEF paper scores every dataset as best-of-{kNN-1, kNN-3, linear}, so dropping
# kNN here would compare our linear-probe number against their best-of-three.
_YEAR_ALIGNED_MODALITIES = {
    "sentinel2": [Modality.SENTINEL2_L2A.name],
    "sentinel1_sentinel2": [Modality.SENTINEL1.name, Modality.SENTINEL2_L2A.name],
    # The Landsat pairs. Both require the landsat_moNN layers on weka
    # (setup_extra_layers.py, layer set `landsat`); the input is optional in
    # every model.yaml, so windows the Landsat prepare/materialize has not
    # reached run without it rather than failing.
    #
    # sentinel2_landsat isolates Landsat's optical-only contribution (vs the
    # sentinel2 pair); sentinel1_sentinel2_landsat is the everything-config
    # and the sensor-fair match to AEF, which fuses Landsat internally.
    # Together with the S1 pairs this completes the sensor half-lattice —
    # every single-sensor addition to S2 is measurable in isolation and in
    # combination.
    "sentinel2_landsat": [
        Modality.SENTINEL2_L2A.name,
        Modality.LANDSAT.name,
    ],
    "sentinel1_sentinel2_landsat": [
        Modality.SENTINEL1.name,
        Modality.SENTINEL2_L2A.name,
        Modality.LANDSAT.name,
    ],
}
for _suffix, _modalities in _YEAR_ALIGNED_MODALITIES.items():
    EMBEDDING_EVAL_TASKS.update(
        {
            f"{name}_ws16_ps1_{_suffix}": _aef_ps1_task(
                name,
                EvalMode.LINEAR_PROBE,
                window_size=16,
                input_modalities=_modalities,
            )
            for name in AEF_SUPPLEMENTAL_YEAR_ALIGNED
        }
    )
    EMBEDDING_EVAL_TASKS.update(
        {
            f"{name}_ws16_ps1_{_suffix}_knn": _aef_ps1_task(
                name,
                EvalMode.KNN,
                window_size=16,
                input_modalities=_modalities,
            )
            for name in AEF_SUPPLEMENTAL_YEAR_ALIGNED
        }
    )
    # SCL cloud-masked siblings: identical except cloud-contaminated S2
    # pixel-timesteps are masked MISSING at load time (scl_cloud_mask), which
    # reproduces the pre-year-aligned exports' eo:cloud_cover scene filter at
    # pixel granularity. Requires the SCL layers on weka
    # (setup_extra_layers.py); without them the tasks run unmasked and
    # match the plain variants. The window set is unchanged, so plain vs
    # _sclmask deltas isolate the cloud effect (descals is the motivation).
    EMBEDDING_EVAL_TASKS.update(
        {
            f"{name}_ws16_ps1_{_suffix}_sclmask": _aef_ps1_task(
                name,
                EvalMode.LINEAR_PROBE,
                window_size=16,
                input_modalities=_modalities,
                scl_cloud_mask=True,
            )
            for name in AEF_SUPPLEMENTAL_YEAR_ALIGNED
        }
    )
    EMBEDDING_EVAL_TASKS.update(
        {
            f"{name}_ws16_ps1_{_suffix}_sclmask_knn": _aef_ps1_task(
                name,
                EvalMode.KNN,
                window_size=16,
                input_modalities=_modalities,
                scl_cloud_mask=True,
            )
            for name in AEF_SUPPLEMENTAL_YEAR_ALIGNED
        }
    )
    # "Cloudless" siblings: same mechanism, narrower policy -- only
    # unambiguous cloud (SCL 8 medium / 9 high probability) is masked, leaving
    # shadow/cirrus/nodata in place. With the plain and _sclmask variants this
    # gives a three-point masking-aggressiveness ladder per task.
    EMBEDDING_EVAL_TASKS.update(
        {
            f"{name}_ws16_ps1_{_suffix}_cloudless": _aef_ps1_task(
                name,
                EvalMode.LINEAR_PROBE,
                window_size=16,
                input_modalities=_modalities,
                scl_cloud_mask=True,
                scl_cloud_classes=SCL_CLOUDLESS_CLASSES,
            )
            for name in AEF_SUPPLEMENTAL_YEAR_ALIGNED
        }
    )
    EMBEDDING_EVAL_TASKS.update(
        {
            f"{name}_ws16_ps1_{_suffix}_cloudless_knn": _aef_ps1_task(
                name,
                EvalMode.KNN,
                window_size=16,
                input_modalities=_modalities,
                scl_cloud_mask=True,
                scl_cloud_classes=SCL_CLOUDLESS_CLASSES,
            )
            for name in AEF_SUPPLEMENTAL_YEAR_ALIGNED
        }
    )
# Scene-level Landsat cloud-mask siblings of the landsat tasks: months whose
# chosen Landsat scene reports cloud_cover >= this are masked MISSING (the
# same threshold convention as the original exports' S2 scene filter).
# Requires the landsat_cloud_cover.json sidecar at each dataset root
# (build_landsat_cloud_cover_sidecar.py); without it the tasks run unmasked
# with a warning, like _sclmask without SCL layers.
L8MASK_CLOUD_COVER_MAX = 50.0
for _suffix in ("sentinel2_landsat", "sentinel1_sentinel2_landsat"):
    _modalities = _YEAR_ALIGNED_MODALITIES[_suffix]
    for _mode, _knn in ((EvalMode.LINEAR_PROBE, ""), (EvalMode.KNN, "_knn")):
        EMBEDDING_EVAL_TASKS.update(
            {
                f"{name}_ws16_ps1_{_suffix}_l8mask{_knn}": _aef_ps1_task(
                    name,
                    _mode,
                    window_size=16,
                    input_modalities=_modalities,
                    landsat_cloud_cover_max=L8MASK_CLOUD_COVER_MAX,
                )
                for name in AEF_SUPPLEMENTAL_YEAR_ALIGNED
            }
        )
        EMBEDDING_EVAL_TASKS.update(
            {
                f"{name}_ws16_ps1_{_suffix}_sclmask_l8mask{_knn}": _aef_ps1_task(
                    name,
                    _mode,
                    window_size=16,
                    input_modalities=_modalities,
                    scl_cloud_mask=True,
                    landsat_cloud_cover_max=L8MASK_CLOUD_COVER_MAX,
                )
                for name in AEF_SUPPLEMENTAL_YEAR_ALIGNED
            }
        )

# Per-pixel Landsat cloud-mask siblings (QA_PIXEL / CFMask): the Landsat
# analogue of _sclmask. Cloud/shadow/cirrus/dilated pixel-timesteps are
# masked MISSING via the optional landsat_qa input (setup_extra_layers.py,
# layer set `landsat_qa`); without the layers the tasks run unmasked with a
# warning, like _sclmask without SCL. _l8pixmask isolates the Landsat-side
# pixel mask against the plain landsat pair; _sclmask_l8pixmask masks both
# optical sensors -- the S1+S2+L8 + SCL interaction (the one masking config
# that ever won) is the motivating comparison.
#
# _cloudless_l8pixmask is the same both-optical-sensors combination against the
# NARROWER S2 policy (SCL 8/9 only). Added 2026-08-13: with the Landsat ladder
# complete, cloudless overtook sclmask as the best S2-side cleaner on the full
# stack (+1.40 vs +1.33 for cand_ndvi), so the both-masked cell has to be built
# on cloudless too or the grid tests the pixel mask only against the runner-up.
for _suffix in ("sentinel2_landsat", "sentinel1_sentinel2_landsat"):
    _modalities = _YEAR_ALIGNED_MODALITIES[_suffix]
    for _mode, _knn in ((EvalMode.LINEAR_PROBE, ""), (EvalMode.KNN, "_knn")):
        EMBEDDING_EVAL_TASKS.update(
            {
                f"{name}_ws16_ps1_{_suffix}_l8pixmask{_knn}": _aef_ps1_task(
                    name,
                    _mode,
                    window_size=16,
                    input_modalities=_modalities,
                    l8_pixel_cloud_mask=True,
                )
                for name in AEF_SUPPLEMENTAL_YEAR_ALIGNED
            }
        )
        EMBEDDING_EVAL_TASKS.update(
            {
                f"{name}_ws16_ps1_{_suffix}_sclmask_l8pixmask{_knn}": _aef_ps1_task(
                    name,
                    _mode,
                    window_size=16,
                    input_modalities=_modalities,
                    scl_cloud_mask=True,
                    l8_pixel_cloud_mask=True,
                )
                for name in AEF_SUPPLEMENTAL_YEAR_ALIGNED
            }
        )
        EMBEDDING_EVAL_TASKS.update(
            {
                f"{name}_ws16_ps1_{_suffix}_cloudless_l8pixmask{_knn}": _aef_ps1_task(
                    name,
                    _mode,
                    window_size=16,
                    input_modalities=_modalities,
                    scl_cloud_mask=True,
                    scl_cloud_classes=SCL_CLOUDLESS_CLASSES,
                    l8_pixel_cloud_mask=True,
                )
                for name in AEF_SUPPLEMENTAL_YEAR_ALIGNED
            }
        )

# Per-pixel cloud-mosaic pilot on ethiopia, the dataset where we lose hardest to
# Tessera. `ccmos` replaces each 30-day S2 mosaic with a per-pixel selection of the
# least-SCL-contaminated acquisition (pixel_mosaic_export.py); measured 93.3% of
# chosen pixels clear, against the parent's scene-level `sort_by: eo:cloud_cover`.
#
# `10band` is its CONTROL and is not optional: the composite carries only ten bands
# (the tessera_v2 fetch group it was built from has no B01/B09, and rslearn cannot
# backfill a band set onto an already-materialized layer), so a ccmos-vs-parent
# delta would confound cloud selection with dropping two bands. Compare
# ccmos <-> 10band; the 12-band parent stays the untouched reference.
#
# Registered as an explicit block rather than by appending to
# AEF_SUPPLEMENTAL_YEAR_ALIGNED: that tuple drives every other sweep's task
# cross-product and is length-pinned by a test.
#
# 10band shares the parent's weka_path (only its band list differs) -- the
# us_trees_tessera pattern -- so its registry entry is hand-cloned rather than
# re-ingested, and it must stay out of AEF_SUPPLEMENTAL_DATASETS for the same
# re-walking reason documented at MATCHED_SUBSET_DATASETS.
PIXEL_MOSAIC_DATASETS = (
    "ethiopia_crops_ccmos_year_aligned",
    "ethiopia_crops_10band_year_aligned",
)
for _suffix, _modalities in _YEAR_ALIGNED_MODALITIES.items():
    for _mode, _mode_suffix in (
        (EvalMode.LINEAR_PROBE, ""),
        (EvalMode.KNN, "_knn"),
    ):
        EMBEDDING_EVAL_TASKS.update(
            {
                f"{name}_ws16_ps1_{_suffix}{_mode_suffix}": _aef_ps1_task(
                    name,
                    _mode,
                    window_size=16,
                    input_modalities=_modalities,
                )
                for name in PIXEL_MOSAIC_DATASETS
            }
        )
        # SCL-masked siblings. On the 10band control this is the old pipeline's
        # only cloud defence -- subtractive masking over an already-chosen
        # mosaic. On ccmos the SCL layers describe the pixels the composite
        # actually selected, so plain-ccmos vs sclmask-ccmos asks whether any
        # masking is still worth doing once selection has happened.
        EMBEDDING_EVAL_TASKS.update(
            {
                f"{name}_ws16_ps1_{_suffix}_sclmask{_mode_suffix}": _aef_ps1_task(
                    name,
                    _mode,
                    window_size=16,
                    input_modalities=_modalities,
                    scl_cloud_mask=True,
                )
                for name in PIXEL_MOSAIC_DATASETS
            }
        )

# The NARROW Landsat pixel policy: `_l8pixstrict` masks on the cloud bit alone,
# where `_l8pixmask` above also masks dilated cloud, cirrus and cloud shadow.
# So `_l8pixstrict` MASKS LESS, despite the name: "strict" qualifies the
# criterion for calling a pixel cloudy, not the amount of masking. Aggressiveness
# runs unmasked < _l8pixstrict < _l8pixmask, as S2's runs
# unmasked < _cloudless < _sclmask.
# Registered 2026-08-14 because the aggressive policy measurably HURTS: with the
# flag finally reaching the data, cand_ndvi lost 1.8 pts on descals and 2.8 on
# ethiopia (KNN, matched pairs), worst on the S2-only stacks and rescued by S1.
# The S2 ladder had already found the same shape -- the narrow `_cloudless`
# policy beat the aggressive `_sclmask` one -- so this tests whether the Landsat
# result is a policy-calibration problem or an argument against masking Landsat
# at all. Same three S2-side pairings as the aggressive variants, so each strict
# task has both an aggressive and an unmasked sibling to be read against.
for _suffix in ("sentinel2_landsat", "sentinel1_sentinel2_landsat"):
    _modalities = _YEAR_ALIGNED_MODALITIES[_suffix]
    for _mode, _knn in ((EvalMode.LINEAR_PROBE, ""), (EvalMode.KNN, "_knn")):
        _s2_policies: tuple[tuple[str, dict[str, Any]], ...] = (
            ("", {}),
            ("_sclmask", {"scl_cloud_mask": True}),
            (
                "_cloudless",
                {
                    "scl_cloud_mask": True,
                    "scl_cloud_classes": SCL_CLOUDLESS_CLASSES,
                },
            ),
        )
        for _s2_tag, _s2_kwargs in _s2_policies:
            EMBEDDING_EVAL_TASKS.update(
                {
                    f"{name}_ws16_ps1_{_suffix}{_s2_tag}_l8pixstrict{_knn}": _aef_ps1_task(
                        name,
                        _mode,
                        window_size=16,
                        input_modalities=_modalities,
                        l8_pixel_cloud_mask=True,
                        l8_pixel_cloud_bits=L8QA_CLOUD_ONLY_BITS_MASK,
                        **_s2_kwargs,
                    )
                    for name in AEF_SUPPLEMENTAL_YEAR_ALIGNED
                }
            )

# pastis_year_aligned keeps the pastis conventions (128x128 stored samples,
# tile_samples, mIoU) rather than the AEF center-pixel ones, so it reuses the
# pastis helper with its dataset name overridden.
EMBEDDING_EVAL_TASKS.update(
    {
        "pastis_year_aligned_ws16_ps1_sentinel2": replace(
            _pastis_ps1_task([Modality.SENTINEL2_L2A.name], window_size=16),
            dataset="pastis_year_aligned",
        ),
        "pastis_year_aligned_ws16_ps1_sentinel1_sentinel2": replace(
            _pastis_ps1_task(
                [Modality.SENTINEL1.name, Modality.SENTINEL2_L2A.name],
                window_size=16,
            ),
            dataset="pastis_year_aligned",
        ),
        # SCL cloud-masked siblings (see the _sclmask comment above).
        "pastis_year_aligned_ws16_ps1_sentinel2_sclmask": replace(
            _pastis_ps1_task([Modality.SENTINEL2_L2A.name], window_size=16),
            dataset="pastis_year_aligned",
            scl_cloud_mask=True,
        ),
        "pastis_year_aligned_ws16_ps1_sentinel1_sentinel2_sclmask": replace(
            _pastis_ps1_task(
                [Modality.SENTINEL1.name, Modality.SENTINEL2_L2A.name],
                window_size=16,
            ),
            dataset="pastis_year_aligned",
            scl_cloud_mask=True,
        ),
        # "Cloudless" siblings (see the _cloudless comment above).
        "pastis_year_aligned_ws16_ps1_sentinel2_cloudless": replace(
            _pastis_ps1_task([Modality.SENTINEL2_L2A.name], window_size=16),
            dataset="pastis_year_aligned",
            scl_cloud_mask=True,
            scl_cloud_classes=SCL_CLOUDLESS_CLASSES,
        ),
        "pastis_year_aligned_ws16_ps1_sentinel1_sentinel2_cloudless": replace(
            _pastis_ps1_task(
                [Modality.SENTINEL1.name, Modality.SENTINEL2_L2A.name],
                window_size=16,
            ),
            dataset="pastis_year_aligned",
            scl_cloud_mask=True,
            scl_cloud_classes=SCL_CLOUDLESS_CLASSES,
        ),
        # Landsat siblings (see the _YEAR_ALIGNED_MODALITIES comment above).
        "pastis_year_aligned_ws16_ps1_sentinel2_landsat": replace(
            _pastis_ps1_task(
                [Modality.SENTINEL2_L2A.name, Modality.LANDSAT.name],
                window_size=16,
            ),
            dataset="pastis_year_aligned",
        ),
        "pastis_year_aligned_ws16_ps1_sentinel2_landsat_sclmask": replace(
            _pastis_ps1_task(
                [Modality.SENTINEL2_L2A.name, Modality.LANDSAT.name],
                window_size=16,
            ),
            dataset="pastis_year_aligned",
            scl_cloud_mask=True,
        ),
        "pastis_year_aligned_ws16_ps1_sentinel2_landsat_cloudless": replace(
            _pastis_ps1_task(
                [Modality.SENTINEL2_L2A.name, Modality.LANDSAT.name],
                window_size=16,
            ),
            dataset="pastis_year_aligned",
            scl_cloud_mask=True,
            scl_cloud_classes=SCL_CLOUDLESS_CLASSES,
        ),
        "pastis_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat": replace(
            _pastis_ps1_task(
                [
                    Modality.SENTINEL1.name,
                    Modality.SENTINEL2_L2A.name,
                    Modality.LANDSAT.name,
                ],
                window_size=16,
            ),
            dataset="pastis_year_aligned",
        ),
        "pastis_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat_sclmask": replace(
            _pastis_ps1_task(
                [
                    Modality.SENTINEL1.name,
                    Modality.SENTINEL2_L2A.name,
                    Modality.LANDSAT.name,
                ],
                window_size=16,
            ),
            dataset="pastis_year_aligned",
            scl_cloud_mask=True,
        ),
        "pastis_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat_cloudless": replace(
            _pastis_ps1_task(
                [
                    Modality.SENTINEL1.name,
                    Modality.SENTINEL2_L2A.name,
                    Modality.LANDSAT.name,
                ],
                window_size=16,
            ),
            dataset="pastis_year_aligned",
            scl_cloud_mask=True,
            scl_cloud_classes=SCL_CLOUDLESS_CLASSES,
        ),
        # QA_PIXEL (per-pixel Landsat cloud) siblings, with and without the
        # S2-side SCL mask. LP-only like the rest of pastis.
        "pastis_year_aligned_ws16_ps1_sentinel2_landsat_l8pixmask": replace(
            _pastis_ps1_task(
                [Modality.SENTINEL2_L2A.name, Modality.LANDSAT.name],
                window_size=16,
            ),
            dataset="pastis_year_aligned",
            l8_pixel_cloud_mask=True,
        ),
        "pastis_year_aligned_ws16_ps1_sentinel2_landsat_sclmask_l8pixmask": replace(
            _pastis_ps1_task(
                [Modality.SENTINEL2_L2A.name, Modality.LANDSAT.name],
                window_size=16,
            ),
            dataset="pastis_year_aligned",
            scl_cloud_mask=True,
            l8_pixel_cloud_mask=True,
        ),
        "pastis_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat_l8pixmask": replace(
            _pastis_ps1_task(
                [
                    Modality.SENTINEL1.name,
                    Modality.SENTINEL2_L2A.name,
                    Modality.LANDSAT.name,
                ],
                window_size=16,
            ),
            dataset="pastis_year_aligned",
            l8_pixel_cloud_mask=True,
        ),
        "pastis_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat_sclmask_l8pixmask": replace(
            _pastis_ps1_task(
                [
                    Modality.SENTINEL1.name,
                    Modality.SENTINEL2_L2A.name,
                    Modality.LANDSAT.name,
                ],
                window_size=16,
            ),
            dataset="pastis_year_aligned",
            scl_cloud_mask=True,
            l8_pixel_cloud_mask=True,
        ),
        # Both optical sensors masked, S2 side on the narrow cloudless policy
        # (see the _cloudless_l8pixmask comment above).
        "pastis_year_aligned_ws16_ps1_sentinel2_landsat_cloudless_l8pixmask": replace(
            _pastis_ps1_task(
                [Modality.SENTINEL2_L2A.name, Modality.LANDSAT.name],
                window_size=16,
            ),
            dataset="pastis_year_aligned",
            scl_cloud_mask=True,
            scl_cloud_classes=SCL_CLOUDLESS_CLASSES,
            l8_pixel_cloud_mask=True,
        ),
        "pastis_year_aligned_ws16_ps1_sentinel1_sentinel2_landsat_cloudless_l8pixmask": replace(
            _pastis_ps1_task(
                [
                    Modality.SENTINEL1.name,
                    Modality.SENTINEL2_L2A.name,
                    Modality.LANDSAT.name,
                ],
                window_size=16,
            ),
            dataset="pastis_year_aligned",
            scl_cloud_mask=True,
            scl_cloud_classes=SCL_CLOUDLESS_CLASSES,
            l8_pixel_cloud_mask=True,
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
    # GeoBench v2 (same ``dataset=`` strings and modalities as EVAL_TASKS gb2_*).
    "gb2_benv2": DownstreamTaskConfig(
        dataset="gb2-benv2",
        ft_batch_size=2,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        input_modalities=[Modality.SENTINEL1.name, Modality.SENTINEL2_L2A.name],
        epochs=50,
        # Multilabel: GeoBench-2 reports micro-averaged mAP (threshold-free),
        # not macro-F1 at a fixed 0.5 threshold.
        primary_metric=EvalMetric.MICRO_MAP,
    ),
    "gb2_biomassters": DownstreamTaskConfig(
        dataset="gb2-biomassters",
        ft_batch_size=2,
        ft_grad_accum_steps=4,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        input_modalities=[Modality.SENTINEL1.name, Modality.SENTINEL2_L2A.name],
        epochs=50,
        # RMSE (on z-scored targets) to match what GeoBench-2 reports/ranks on.
        primary_metric=EvalMetric.RMSE,
    ),
    "gb2_burn_scars": DownstreamTaskConfig(
        dataset="gb2-burn_scars",
        ft_batch_size=2,
        ft_grad_accum_steps=4,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        patch_size=4,
        primary_metric=EvalMetric.MIOU,
    ),
    # Only 1 SAR amplitude band is provided, so we pass it in as a Sentinel1
    # modality but only the "vv" band is used.
    "gb2_caffe": DownstreamTaskConfig(
        dataset="gb2-caffe",
        ft_batch_size=2,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        input_modalities=[Modality.SENTINEL1.name],
        epochs=50,
        patch_size=4,
        primary_metric=EvalMetric.MIOU,
    ),
    "gb2_cloudsen12": DownstreamTaskConfig(
        dataset="gb2-cloudsen12",
        ft_batch_size=2,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        patch_size=4,
        primary_metric=EvalMetric.MIOU,
    ),
    "gb2_kuro_siwo": DownstreamTaskConfig(
        dataset="gb2-kuro_siwo",
        ft_batch_size=2,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        input_modalities=[Modality.SENTINEL1.name, Modality.SRTM.name],
        epochs=50,
        patch_size=4,
        primary_metric=EvalMetric.MIOU,
    ),
    "gb2_spacenet2": DownstreamTaskConfig(
        dataset="gb2-spacenet2",
        ft_batch_size=2,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        patch_size=4,
        primary_metric=EvalMetric.MIOU,
    ),
    "gb2_spacenet7": DownstreamTaskConfig(
        dataset="gb2-spacenet7",
        ft_batch_size=2,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        patch_size=4,
        primary_metric=EvalMetric.MIOU,
    ),
    "gb2_flair2": DownstreamTaskConfig(
        dataset="gb2-flair2",
        ft_batch_size=2,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        patch_size=4,
        primary_metric=EvalMetric.MIOU,
    ),
    "gb2_fotw": DownstreamTaskConfig(
        dataset="gb2-fotw",
        ft_batch_size=4,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        patch_size=4,
        primary_metric=EvalMetric.MIOU,
    ),
    "gb2_treesatai": DownstreamTaskConfig(
        dataset="gb2-treesatai",
        ft_batch_size=2,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
        # Multilabel: GeoBench-2 reports micro-averaged mAP (threshold-free),
        # not macro-F1 at a fixed 0.5 threshold.
        primary_metric=EvalMetric.MICRO_MAP,
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
                    else EMBEDDING_EVAL_TASKS
                    if os.environ.get("EMBEDDING_EVALS")
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
