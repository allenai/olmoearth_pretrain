"""80K step 1/4 learning rate."""

from latent_mim_128 import (
    MAX_PATCH_SIZE,
    MIN_PATCH_SIZE,
    build_common_components,
    build_dataloader_config,
    build_trainer_config,
    build_visualize_config,
)
from olmo_core.config import DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import (
    ConstantWithWarmup,
    LinearWithWarmup,
    SequentialScheduler,
)
from olmo_core.train.common import Duration

from helios.data.concat import HeliosConcatDatasetConfig
from helios.data.dataset import HeliosDatasetConfig
from helios.internal.experiment import CommonComponents, main
from helios.internal.utils import MODEL_SIZE_ARGS
from helios.nn.latent_mim import LatentMIMConfig
from helios.nn.st_model import STEncoderConfig, STPredictorConfig
from helios.train.loss import LossConfig
from helios.train.masking import MaskingConfig
from helios.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for an experiment."""
    model_size = MODEL_SIZE_ARGS["base"]
    encoder_config = STEncoderConfig(
        embedding_size=model_size["encoder_embedding_size"],
        num_heads=model_size["encoder_num_heads"],
        depth=model_size["encoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        supported_modality_names=common.training_modalities,
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=12,
    )
    decoder_config = STPredictorConfig(
        encoder_embedding_size=model_size["encoder_embedding_size"],
        decoder_embedding_size=model_size["decoder_embedding_size"],
        depth=model_size["decoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        num_heads=model_size["decoder_num_heads"],
        max_sequence_length=12,
        supported_modality_names=common.training_modalities,
    )
    model_config = LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )
    return model_config


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build the train module config for an experiment."""
    scheduler = SequentialScheduler(
        schedulers=[
            ConstantWithWarmup(warmup_steps=2000),
            LinearWithWarmup(alpha_f=0.25, warmup_steps=0),
            LinearWithWarmup(alpha_f=0.25, warmup_steps=0),
            LinearWithWarmup(alpha_f=0.25, warmup_steps=0),
            LinearWithWarmup(alpha_f=0.25, warmup_steps=0),
            LinearWithWarmup(alpha_f=0.1, warmup_steps=0),
        ],
        schedulers_max_steps=[80000, 80000, 80000, 80000, 80000],
    )
    return ContrastiveLatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02),
        masking_config=MaskingConfig(
            strategy_config={
                "type": "random",
                "encode_ratio": 0.1,
                "decode_ratio": 0.75,
            }
        ),
        warmup_duration=Duration.steps(8000),
        loss_config=LossConfig(
            loss_config={
                "type": "patch_discrimination_new",
            }
        ),
        rank_microbatch_size=16,
        token_exit_cfg={modality: 0 for modality in common.training_modalities},
        autocast_precision=DType.bfloat16,
        contrastive_config=LossConfig(
            loss_config={
                "type": "InfoNCE",
                "weight": 0.1,
            }
        ),
        ema_decay=(1.0, 1.0),
        max_grad_norm=1.0,
        dp_config=DataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
        ),
        scheduler=scheduler,
    )


def build_dataset_config(common: CommonComponents) -> HeliosDatasetConfig:
    """Build the dataset config for an experiment."""
    dataset_configs = [
        # presto
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/presto/h5py_data_w_missing_timesteps_128_x_4_zstd_3/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/469892",
            training_modalities=common.training_modalities,
        ),
        # osm_sampling
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_128_x_4_zstd_3/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/1141152",
            training_modalities=common.training_modalities,
        ),
        # osmbig
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/osmbig/h5py_data_w_missing_timesteps_zstd_3_128_x_4/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/1297928",
            training_modalities=common.training_modalities,
        ),
        # presto neighbor
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/presto_neighbor/h5py_data_w_missing_timesteps_zstd_3_128_x_4/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/3507748",
            training_modalities=common.training_modalities,
        ),
        # worldcover
        HeliosDatasetConfig(
            h5py_dir="/weka/dfive-default/helios/dataset/worldcover_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcover/6370580",
            training_modalities=common.training_modalities,
        ),
    ]
    return HeliosConcatDatasetConfig(dataset_configs=dataset_configs)


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )
