"""Launch script for evaluation allowing you to easily run all the evals for your model by just pointing at your training script."""

import importlib.util
import os
import sys
from logging import getLogger
from typing import Any

from olmo_core.train.callbacks import (
    BeakerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.common import Duration, LoadStrategy
from olmo_core.train.config import TrainerConfig

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.datasets.normalize import NormMethod
from olmoearth_pretrain.internal.constants import EVAL_WANDB_PROJECT, WANDB_ENTITY
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
    main,
)
from olmoearth_pretrain.nn.flexi_vit import (
    PoolingType,
)
from olmoearth_pretrain.train.callbacks import (
    DownstreamEvaluatorCallbackConfig,
    OlmoEarthWandBCallback,
)
from olmoearth_pretrain.train.callbacks.evaluator_callback import (
    DownstreamTaskConfig,
    EvalMode,
)

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


EVAL_TASKS = {
    "m_eurosat": DownstreamTaskConfig(
        dataset="m-eurosat",
        embedding_batch_size=128,
        num_workers=0,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.epochs(5),
        eval_mode=EvalMode.KNN,
    ),
    "m_forestnet": DownstreamTaskConfig(
        dataset="m-forestnet",
        embedding_batch_size=64,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=False,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        eval_interval=Duration.epochs(5),
        eval_mode=EvalMode.KNN,
    ),
    "m_bigearthnet": DownstreamTaskConfig(
        dataset="m-bigearthnet",
        embedding_batch_size=64,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.epochs(5),
        eval_mode=EvalMode.KNN,
    ),
    "m_so2sat": DownstreamTaskConfig(
        dataset="m-so2sat",
        embedding_batch_size=128,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.epochs(5),
        eval_mode=EvalMode.KNN,
    ),
    "m_brick_kiln": DownstreamTaskConfig(
        dataset="m-brick-kiln",
        embedding_batch_size=128,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.epochs(5),
        eval_mode=EvalMode.KNN,
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
        eval_mode=EvalMode.LINEAR_PROBE,
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
        eval_mode=EvalMode.LINEAR_PROBE,
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
        eval_mode=EvalMode.LINEAR_PROBE,
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
        eval_mode=EvalMode.LINEAR_PROBE,
    ),
    "pastis_sentinel2": DownstreamTaskConfig(
        dataset="pastis",
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
    ),
    "breizhcrops": DownstreamTaskConfig(
        dataset="breizhcrops",
        embedding_batch_size=128,
        probe_batch_size=128,
        num_workers=0,
        pooling_type=PoolingType.MAX,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.epochs(50),
        patch_size=1,
        eval_mode=EvalMode.LINEAR_PROBE,
        probe_lr=0.1,
        epochs=50,
    ),
    "cropharvest_Togo_12_sentinel2": DownstreamTaskConfig(
        dataset="cropharvest_Togo_12",
        embedding_batch_size=128,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.epochs(20),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        patch_size=1,
        eval_mode=EvalMode.LINEAR_PROBE,
        probe_lr=0.1,
        epochs=50,
    ),
    "cropharvest_Togo_12_sentinel1": DownstreamTaskConfig(
        dataset="cropharvest_Togo_12",
        embedding_batch_size=128,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.epochs(20),
        input_modalities=[Modality.SENTINEL1.name],
        patch_size=1,
        eval_mode=EvalMode.LINEAR_PROBE,
        probe_lr=0.1,
        epochs=50,
    ),
    # example of "in season" cropland mapping - 6 indicates only the
    # first 6 timesteps are passed to the model
    "cropharvest_Peoples_Republic_of_China_6": DownstreamTaskConfig(
        dataset="cropharvest_People's Republic of China_6",
        embedding_batch_size=128,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.epochs(20),
        input_modalities=[Modality.SENTINEL2_L2A.name],
        patch_size=1,
        eval_mode=EvalMode.LINEAR_PROBE,
        probe_lr=0.1,
        epochs=50,
    ),
    "cropharvest_Peoples_Republic_of_China_6_sentinel1": DownstreamTaskConfig(
        dataset="cropharvest_People's Republic of China_6",
        embedding_batch_size=128,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.epochs(20),
        input_modalities=[Modality.SENTINEL1.name],
        patch_size=1,
        eval_mode=EvalMode.LINEAR_PROBE,
        probe_lr=0.1,
        epochs=50,
    ),
    "cropharvest_Togo_12_sentinel2_sentinel1": DownstreamTaskConfig(
        dataset="cropharvest_Togo_12",
        embedding_batch_size=128,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.epochs(20),
        input_modalities=[Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name],
        patch_size=1,
        eval_mode=EvalMode.LINEAR_PROBE,
        probe_lr=0.1,
        epochs=50,
    ),
    "cropharvest_Peoples_Republic_of_China_6_sentinel1_sentinel2": DownstreamTaskConfig(
        dataset="cropharvest_People's Republic of China_6",
        embedding_batch_size=128,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        eval_interval=Duration.epochs(20),
        input_modalities=[
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
        ],
        patch_size=1,
        eval_mode=EvalMode.LINEAR_PROBE,
        probe_lr=0.1,
        epochs=50,
    ),
    "nandi_sentinel2": DownstreamTaskConfig(
        dataset="nandi",
        embedding_batch_size=128,
        num_workers=0,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        input_modalities=[Modality.SENTINEL2_L2A.name],
        input_layers=["sentinel2"],
        eval_interval=Duration.epochs(20),
    ),
    "nandi_sentinel1": DownstreamTaskConfig(
        dataset="nandi",
        embedding_batch_size=128,
        num_workers=0,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        input_modalities=[Modality.SENTINEL1.name],
        input_layers=["sentinel1_ascending"],
        eval_interval=Duration.epochs(20),
        eval_mode=EvalMode.KNN,
    ),
    "nandi_landsat": DownstreamTaskConfig(
        dataset="nandi",
        embedding_batch_size=128,
        num_workers=0,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        input_modalities=[Modality.LANDSAT.name],
        input_layers=["landsat"],
        eval_interval=Duration.epochs(20),
        eval_mode=EvalMode.KNN,
    ),
    "awf_sentinel2": DownstreamTaskConfig(
        dataset="awf",
        embedding_batch_size=128,
        num_workers=0,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        input_modalities=[Modality.SENTINEL2_L2A.name],
        input_layers=["sentinel2"],
        eval_interval=Duration.epochs(20),
        eval_mode=EvalMode.KNN,
    ),
    "awf_sentinel1": DownstreamTaskConfig(
        dataset="awf",
        embedding_batch_size=128,
        num_workers=0,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        input_modalities=[Modality.SENTINEL1.name],
        input_layers=["sentinel1_ascending"],
        eval_interval=Duration.epochs(20),
        eval_mode=EvalMode.KNN,
    ),
    "awf_landsat": DownstreamTaskConfig(
        dataset="awf",
        embedding_batch_size=128,
        num_workers=0,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        input_modalities=[Modality.LANDSAT.name],
        input_layers=["landsat"],
        eval_interval=Duration.epochs(20),
        eval_mode=EvalMode.KNN,
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
    ),
    "m_bigearthnet": DownstreamTaskConfig(
        dataset="m-bigearthnet",
        ft_batch_size=16,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        epochs=50,
    ),
    "m_so2sat": DownstreamTaskConfig(
        dataset="m-so2sat",
        ft_batch_size=32,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        epochs=50,
    ),
    "m_sa_crop_type": DownstreamTaskConfig(
        dataset="m-sa-crop-type",
        ft_batch_size=32,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=False,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        epochs=50,
    ),
    "mados": DownstreamTaskConfig(
        dataset="mados",
        ft_batch_size=16,
        num_workers=8,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=False,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        epochs=50,
    ),
    "pastis_sentinel2": DownstreamTaskConfig(
        dataset="pastis",
        ft_batch_size=16,
        num_workers=2,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        input_modalities=[Modality.SENTINEL2_L2A.name],
        epochs=50,
    ),
    "m_brick_kiln": DownstreamTaskConfig(
        dataset="m-brick-kiln",
        ft_batch_size=64,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        epochs=50,
    ),
    "sen1floods11": DownstreamTaskConfig(
        dataset="sen1floods11",
        ft_batch_size=32,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=True,
        epochs=50,
    ),
    "m_cashew_plant": DownstreamTaskConfig(
        dataset="m-cashew-plant",
        ft_batch_size=32,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=False,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        epochs=50,
    ),
    "m_forestnet": DownstreamTaskConfig(
        dataset="m-forestnet",
        ft_batch_size=16,
        num_workers=4,
        pooling_type=PoolingType.MEAN,
        norm_stats_from_pretrained=False,
        norm_method=NormMethod.NORM_NO_CLIP_2_STD,
        epochs=50,
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
                tasks=FT_EVAL_TASKS if os.environ.get("FINETUNE") else EVAL_TASKS,
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

    # 3) Inject all of the builder names into your namespace
    try:
        build_common_components = user_mod.build_common_components
    except AttributeError:
        from olmoearth_pretrain.internal.common import build_common_components

    build_model_config = user_mod.build_model_config
    build_train_module_config = user_mod.build_train_module_config
    build_dataset_config = user_mod.build_dataset_config
    build_dataloader_config = user_mod.build_dataloader_config
    build_visualize_config = user_mod.build_visualize_config
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )
