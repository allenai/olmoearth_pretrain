"""Memory leak bisect: drop callbacks + replace dataset with mock dataloader.

Same callback changes as hidden1_no_callbacks.py (W&B disabled, Beaker and
Checkpointer removed), but also replaces the h5py-backed dataloader with a
synthetic one that generates one mock batch and repeats it every step.

This eliminates h5py reads, DataLoader worker processes, and worker→main
shared-memory IPC from the equation. If the leak persists, it's in the
model/FSDP/trainer/Gloo path. If it disappears, it's in the
dataset/worker/IPC path.

NOTE: The real dataset is briefly constructed to generate a properly-shaped
mock batch through the existing collator/masking pipeline, then discarded.
The h5py directory must exist at build time (fine on Beaker).
"""

import logging
from collections.abc import Iterable
from typing import Any, cast

from olmo_core.config import DType
from olmo_core.data.data_loader import DataLoaderBase
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.distributed.utils import (
    get_fs_local_rank,
    get_rank,
    get_world_size,
)
from olmo_core.utils import get_default_device, seed_all
from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import CosWithWarmup
from olmo_core.train.callbacks import (
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    WandBCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.common import Duration, LoadStrategy
from olmo_core.train.config import TrainerConfig

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.internal.common import (
    build_common_components as build_common_components_default,
)
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
    OlmoEarthExperimentConfig,
    OlmoEarthVisualizeConfig,
    SubCmd,
    main,
)
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexihelios import (
    EncoderConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.tokenization import ModalityTokenization, TokenizationConfig
from olmoearth_pretrain.train.callbacks import (
    OlmoEarthSpeedMonitorCallback,
    OlmoEarthWandBCallback,
)
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1
RANDOM_BAND_DROPOUT_MAX_RATE = 0.2
PATCH_EMBED_HIDDEN_SIZES: list[int] = [64]

S2_SINGLE_BANDSET = ModalityTokenization(
    band_groups=[
        [
            "B02",
            "B03",
            "B04",
            "B08",
            "B05",
            "B06",
            "B07",
            "B8A",
            "B11",
            "B12",
            "B01",
            "B09",
        ],
    ]
)

LANDSAT_SINGLE_BANDSET = ModalityTokenization(
    band_groups=[
        ["B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"],
    ]
)

ONLY_DECODE_MODALITIES = [
    Modality.WORLDCOVER.name,
    Modality.SRTM.name,
    Modality.OPENSTREETMAP_RASTER.name,
    Modality.WRI_CANOPY_HEIGHT_MAP.name,
    Modality.CDL.name,
    Modality.WORLDCEREAL.name,
]

BAND_DROPOUT_MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.LANDSAT.name,
]


# ---------------------------------------------------------------------------
# Repeating mock dataloader — yields one pre-computed batch in a loop.
# No worker processes, no shared memory, no h5py reads during training.
# ---------------------------------------------------------------------------

MOCK_BATCHES_PER_EPOCH = 200


class RepeatingMockDataLoader(DataLoaderBase):
    """DataLoader that endlessly replays a single pre-generated batch."""

    def __init__(
        self,
        mock_batch: Any,
        *,
        work_dir: str,
        global_batch_size: int,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: int = 0,
        batches_per_epoch: int = MOCK_BATCHES_PER_EPOCH,
    ) -> None:
        super().__init__(
            work_dir=work_dir,
            global_batch_size=global_batch_size,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            fs_local_rank=fs_local_rank,
        )
        self._mock_batch = mock_batch
        self._batches_per_epoch = batches_per_epoch
        self._seed = 42
        self.token_budget: int | None = None

    def _iter_batches(self) -> Iterable[Any]:
        for _ in range(self._batches_per_epoch):
            yield self._mock_batch

    @property
    def total_batches(self) -> int:
        return self._batches_per_epoch

    def reshuffle(self, epoch: int | None = None, **_: Any) -> None:
        if epoch is not None:
            self._epoch = epoch

    def state_dict(self) -> dict[str, Any]:
        return {"seed": self._seed, "epoch": self._epoch}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._seed = state_dict.get("seed", self._seed)
        self._epoch = state_dict.get("epoch", self._epoch)


# ---------------------------------------------------------------------------
# Custom train function — replaces experiment.train via monkey-patch.
# ---------------------------------------------------------------------------


def train_with_mock_data(config: OlmoEarthExperimentConfig) -> None:
    """Train with a synthetic repeating dataloader instead of real h5py data."""
    seed_all(config.init_seed)

    model = config.model.build()
    device = get_default_device()
    model = model.to(device)
    train_module = config.train_module.build(model)

    # Build the real dataloader once to generate a properly-shaped mock batch
    # through the existing collator + masking pipeline.
    dataset = config.dataset.build()
    real_loader = config.data_loader.build(
        dataset, dp_process_group=train_module.dp_process_group
    )
    mock_batch = real_loader.get_mock_batch()
    del real_loader, dataset

    dp_pg = train_module.dp_process_group
    mock_loader = RepeatingMockDataLoader(
        mock_batch,
        work_dir=config.data_loader.work_dir,
        global_batch_size=config.data_loader.global_batch_size,
        dp_world_size=get_world_size(dp_pg),
        dp_rank=get_rank(dp_pg),
        fs_local_rank=get_fs_local_rank(),
    )

    trainer = config.trainer.build(train_module, mock_loader)

    config_dict = config.as_config_dict()
    cast(WandBCallback, trainer.callbacks["wandb"]).config = config_dict
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict
    trainer.fit()


# ---------------------------------------------------------------------------
# Standard config builders (identical to hidden1_no_callbacks.py)
# ---------------------------------------------------------------------------


def _tokenization_config() -> TokenizationConfig:
    return TokenizationConfig(
        overrides={
            "sentinel2_l2a": S2_SINGLE_BANDSET,
            "landsat": LANDSAT_SINGLE_BANDSET,
        }
    )


def _masking_config(
    tokenization_config: TokenizationConfig | None = None,
) -> MaskingConfig:
    return MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "random_ratio": 0.5,
            "only_decode_modalities": ONLY_DECODE_MODALITIES,
        },
        tokenization_config=tokenization_config,
    )


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build the common components for an experiment."""
    config = build_common_components_default(script, cmd, run_name, cluster, overrides)
    config.training_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.LANDSAT.name,
        Modality.WORLDCOVER.name,
        Modality.SRTM.name,
        Modality.OPENSTREETMAP_RASTER.name,
        Modality.WRI_CANOPY_HEIGHT_MAP.name,
        Modality.CDL.name,
        Modality.WORLDCEREAL.name,
    ]
    config.tokenization_config = _tokenization_config()
    return config


def build_train_module_config(
    common: CommonComponents,
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Build the train module config for an experiment."""
    return ContrastiveLatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02, fused=False),
        rank_microbatch_size=64,
        masking_config=_masking_config(common.tokenization_config),
        loss_config=LossConfig(
            loss_config={
                "type": "modality_patch_discrimination_masked_negatives_vec",
                "tau": 0.1,
                "same_target_threshold": 0.999,
                "mask_negatives_for_modalities": ONLY_DECODE_MODALITIES,
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
    """Build the dataloader config for an experiment."""
    return OlmoEarthDataLoaderConfig(
        num_workers=16,
        global_batch_size=512,
        token_budget=2250,
        prefetch_factor=4,
        sampled_hw_p_list=list(range(1, 13)),
        min_patch_size=MIN_PATCH_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        work_dir=common.save_folder,
        seed=3622,
        num_masked_views=2,
        masking_config=_masking_config(common.tokenization_config),
    )


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Build the dataset config for an experiment."""
    return OlmoEarthDatasetConfig(
        h5py_dir="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828",
        training_modalities=common.training_modalities,
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build the trainer config for an experiment."""
    MAX_DURATION = Duration.epochs(300)
    METRICS_COLLECT_INTERVAL = 10
    CANCEL_CHECK_INTERVAL = 25
    LOAD_STRATEGY = LoadStrategy.if_available
    checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)
    wandb_callback = OlmoEarthWandBCallback(
        name=common.run_name,
        project="2026_05_14_hidden1_no_evals_memleak_test",
        entity="eai-ai2",
        enabled=False,
    )
    garbage_collector_callback = GarbageCollectorCallback(enabled=False)
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
        .with_callback("speed_monitor", OlmoEarthSpeedMonitorCallback())
        .with_callback("gpu_memory_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("garbage_collector", garbage_collector_callback)
    )
    return trainer_config


def build_visualize_config(common: CommonComponents) -> OlmoEarthVisualizeConfig:
    """Build the visualize config for an experiment."""
    return OlmoEarthVisualizeConfig(
        num_samples=None,
        output_dir=str(f"{common.save_folder}/visualizations"),
        std_multiplier=2.0,
    )


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for an experiment."""
    model_size = MODEL_SIZE_ARGS["base_shallow_decoder"]

    encoder_config = EncoderConfig(
        embedding_size=model_size["encoder_embedding_size"],
        num_heads=model_size["encoder_num_heads"],
        depth=model_size["encoder_depth"],
        mlp_ratio=model_size["mlp_ratio"],
        supported_modality_names=common.training_modalities,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=12,
        tokenization_config=common.tokenization_config,
        band_dropout_rate=RANDOM_BAND_DROPOUT_MAX_RATE,
        random_band_dropout=True,
        band_dropout_modalities=BAND_DROPOUT_MODALITIES,
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
        tokenization_config=common.tokenization_config,
    )
    model_config = LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )
    return model_config


if __name__ == "__main__":
    # Monkey-patch experiment.train so main() dispatches to our mock version.
    import olmoearth_pretrain.internal.experiment as _experiment_module

    _experiment_module.train = train_with_mock_data  # type: ignore[assignment]

    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )
