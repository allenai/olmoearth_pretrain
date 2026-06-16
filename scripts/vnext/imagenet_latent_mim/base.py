"""ImageNet-only latent MIM experiment with a fixed EMA target encoder.

The training dataset expects ImageFolder-style ImageNet data:
``root_dir/<class_name>/<image>.JPEG``. Override ``--dataset.root_dir`` at launch
to point at the real train split.
"""

import logging

from olmo_core.config import DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import CosWithWarmup
from olmo_core.train.callbacks import (
    BeakerCallback,
    CheckpointerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.common import Duration, LoadStrategy
from olmo_core.train.config import TrainerConfig

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.data.dataset import ImageNetDatasetConfig
from olmoearth_pretrain.evals.datasets.normalize import NormMethod
from olmoearth_pretrain.evals.metrics import EvalMetric
from olmoearth_pretrain.internal.common import (
    build_common_components as build_common_components_default,
)
from olmoearth_pretrain.internal.experiment import CommonComponents, SubCmd, main
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexi_vit import PoolingType
from olmoearth_pretrain.nn.flexihelios import EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.callbacks import (
    DownstreamEvaluatorCallbackConfig,
    OlmoEarthSpeedMonitorCallback,
    OlmoEarthWandBCallback,
)
from olmoearth_pretrain.train.callbacks.evaluator_callback import (
    DownstreamTaskConfig,
    EvalMode,
)
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

IMAGENET_TRAIN_ROOT = "/weka/dfive-default/joer/imagenet/train"
WANDB_PROJECT = "2026_06_09_imagenet_latent_mim"

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 4
PATCH_EMBED_HIDDEN_SIZES: list[int] = [64]


def _masking_config() -> MaskingConfig:
    return MaskingConfig(
        strategy_config={
            "type": "random",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
        }
    )


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build the common components for an ImageNet-only experiment."""
    config = build_common_components_default(script, cmd, run_name, cluster, overrides)
    config.training_modalities = [Modality.IMAGENET.name]
    config.tokenization_config = None
    return config


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build the contrastive latent MIM train module with fixed EMA."""
    return ContrastiveLatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02, fused=False),
        rank_microbatch_size=64,
        masking_config=_masking_config(),
        loss_config=LossConfig(
            loss_config={
                "type": "modality_patch_discrimination_masked_negatives_vec",
                "tau": 0.1,
                "same_target_threshold": 0.999,
                "mask_negatives_for_modalities": [],
            }
        ),
        contrastive_config=LossConfig(
            loss_config={
                "type": "InfoNCE",
                "weight": 0.05,
            }
        ),
        token_exit_cfg={modality: 0 for modality in common.training_modalities},
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=8000),
        ema_decay=(1.0, 1.0),
        dp_config=DataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
    )


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Build the dataloader config for two-view contrastive latent MIM."""
    return OlmoEarthDataLoaderConfig(
        num_workers=16,
        global_batch_size=512,
        token_budget=2250,
        prefetch_factor=4,
        sampled_hw_p_list=list(range(8, 33)),
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        work_dir=common.save_folder,
        seed=3622,
        num_masked_views=2,
        masking_config=_masking_config(),
    )


def build_dataset_config(common: CommonComponents) -> ImageNetDatasetConfig:
    """Build the ImageNet ImageFolder dataset config."""
    return ImageNetDatasetConfig(
        root_dir=IMAGENET_TRAIN_ROOT,
        normalize=True,
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build the trainer config with ImageNet loop eval."""
    max_duration = Duration.epochs(300)
    checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)
    wandb_callback = OlmoEarthWandBCallback(
        name=common.run_name,
        project=WANDB_PROJECT,
        entity="eai-ai2",
        enabled=True,
    )
    image_net_eval = {
        "imagenet": DownstreamTaskConfig(
            dataset="imagenet",
            embedding_batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            norm_method=NormMethod.NORM_NO_CLIP_2_STD,
            input_modalities=[Modality.IMAGENET.name],
            eval_mode=EvalMode.KNN,
            primary_metric=EvalMetric.ACCURACY,
            eval_interval=Duration.steps(20000),
        ),
        # Transfer check: probe the ImageNet-pretrained encoder on EuroSAT RGB.
        "eurosat_rgb": DownstreamTaskConfig(
            dataset="eurosat_rgb",
            embedding_batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            norm_method=NormMethod.NORM_NO_CLIP_2_STD,
            input_modalities=[Modality.IMAGENET.name],
            eval_mode=EvalMode.KNN,
            primary_metric=EvalMetric.ACCURACY,
            eval_interval=Duration.steps(20000),
        ),
    }
    trainer_config = (
        TrainerConfig(
            work_dir=common.save_folder,
            load_strategy=LoadStrategy.if_available,
            save_folder=common.save_folder,
            cancel_check_interval=25,
            metrics_collect_interval=10,
            max_duration=max_duration,
            checkpointer=checkpointer_config,
        )
        .with_callback("wandb", wandb_callback)
        .with_callback("speed_monitor", OlmoEarthSpeedMonitorCallback())
        .with_callback("gpu_memory_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(tasks=image_net_eval),
        )
        .with_callback("garbage_collector", GarbageCollectorCallback(gc_interval=1))
        .with_callback("beaker", BeakerCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=5000,
                ephemeral_save_interval=250,
            ),
        )
    )
    return trainer_config


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the base model config for ImageNet RGB tokens."""
    model_size = MODEL_SIZE_ARGS["base_shallow_decoder"]
    encoder_config = EncoderConfig(
        embedding_size=model_size["encoder_embedding_size"],
        num_heads=model_size["encoder_num_heads"],
        depth=model_size["encoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        supported_modality_names=common.training_modalities,
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=12,
        patch_embed_hidden_sizes=PATCH_EMBED_HIDDEN_SIZES,
    )
    decoder_config = PredictorConfig(
        encoder_embedding_size=model_size["encoder_embedding_size"],
        decoder_embedding_size=model_size["decoder_embedding_size"],
        depth=model_size["decoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        num_heads=model_size["decoder_num_heads"],
        supported_modality_names=common.training_modalities,
        max_sequence_length=12,
    )
    return LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
    )
