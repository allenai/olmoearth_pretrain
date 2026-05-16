"""Channel-attention patch embed initialised from a pretrained checkpoint (keep decoder).

Same model architecture as channel_attn_patch_embed.py, but loads all weights
from a pretrained checkpoint.  If the checkpoint architecture differs (e.g.
old linear patch embed), mismatched keys are left at their random init while
all matching weights are loaded.  Decoder weights are kept from the checkpoint.

See channel_attn_pretrained_reinit_decoder.py for the variant that
reinitialises the decoder after loading.

Pass the checkpoint via --trainer.load_path=<path>.  For resumed (preempted)
runs the trainer loads from save_folder first, so load_path is only used on
the very first launch.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import torch.distributed.checkpoint.state_dict as dist_cp_sd
from olmo_core.config import DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import CosWithWarmup
from olmo_core.train.callbacks import (
    BeakerCallback,
    Callback,
    CheckpointerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.common import Duration, LoadStrategy
from olmo_core.train.config import TrainerConfig
from olmo_core.utils import gc_cuda
from torch.distributed.checkpoint.metadata import Metadata

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.internal.common import (
    build_common_components as build_common_components_default,
)
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
    OlmoEarthVisualizeConfig,
    SubCmd,
    main,
)
from olmoearth_pretrain.internal.utils import MODEL_SIZE_ARGS
from olmoearth_pretrain.nn.flexi_vit import (
    PoolingType,
)
from olmoearth_pretrain.nn.flexihelios import (
    EncoderConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIM, LatentMIMConfig
from olmoearth_pretrain.nn.tokenization import ModalityTokenization, TokenizationConfig
from olmoearth_pretrain.train.callbacks import (
    DownstreamEvaluatorCallbackConfig,
    OlmoEarthSpeedMonitorCallback,
    OlmoEarthWandBCallback,
)
from olmoearth_pretrain.train.callbacks.evaluator_callback import DownstreamTaskConfig
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModule,
    ContrastiveLatentMIMTrainModuleConfig,
)

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1
RANDOM_BAND_DROPOUT_MAX_RATE = 0.3
CHANNEL_ATTN_DIM = 768

BAND_DROPOUT_WARMUP_FRAC = 0.20
BAND_DROPOUT_MID_FRAC = 0.50
BAND_DROPOUT_MID_RATE = 0.15


# ---------------------------------------------------------------------------
# Partial-load train module
# ---------------------------------------------------------------------------
# The pretrained checkpoint used a flat linear patch projection, but this
# script uses ChannelAttentionPatchEmbed.  The two have entirely different
# parameter names, so we need to:
#   1. Skip the base-class architecture compatibility check in _get_state_dict.
#   2. Prune the load template so dist_cp only reads keys present in the
#      checkpoint (new patch-embed keys are left at their random init).
#   3. Skip optimizer state when doing a partial load (the optimizer entries
#      reference the old parameter names).
# When resuming from the run's own save_folder all keys match and the normal
# full-load path (including optimizer state) is taken automatically.
# ---------------------------------------------------------------------------


@dataclass
class PartialLoadTrainModuleConfig(ContrastiveLatentMIMTrainModuleConfig):
    """Like the base config but defaults to ``strict=False`` for checkpoint loading."""

    state_dict_load_opts: dict[str, Any] | None = field(
        default_factory=lambda: {"strict": False, "flatten_optimizer_state_dict": True}
    )

    def build(
        self,
        model: LatentMIM,
        device=None,
    ) -> "PartialLoadTrainModule":
        """Build the corresponding :class:`PartialLoadTrainModule`."""
        kwargs = self.prepare_kwargs()
        return PartialLoadTrainModule(model=model, device=device, **kwargs)


class PartialLoadTrainModule(ContrastiveLatentMIMTrainModule):
    """Handles checkpoint loading when the model architecture has changed."""

    # -- override 1: drop the architecture compatibility check ---------------
    def _get_state_dict(
        self, sd_options: dist_cp_sd.StateDictOptions
    ) -> dict[str, Any]:
        model_state_dict = dist_cp_sd.get_model_state_dict(
            self.model, options=sd_options
        )
        optim_state_dict = dist_cp_sd.get_optimizer_state_dict(
            self.model, self.optimizer, options=sd_options
        )
        return {"model": model_state_dict, "optim": optim_state_dict}

    # -- override 2: prune keys that don't exist in the checkpoint -----------
    def state_dict_to_load(
        self, metadata: Metadata, optim: bool | None = None
    ) -> dict[str, Any]:
        """Return a state-dict template pruned to keys present in the checkpoint."""
        load_opts = self.state_dict_load_opts
        model_state_dict = dist_cp_sd.get_model_state_dict(
            self.model, options=load_opts
        )

        checkpoint_keys = set(metadata.state_dict_metadata.keys())
        skip_keys: list[str] = []
        for k in list(model_state_dict.keys()):
            ckpt_key = f"model.{k}"
            if ckpt_key not in checkpoint_keys:
                skip_keys.append(k)
                logger.info(
                    "Key model.%s absent from checkpoint – will keep random init", k
                )
                continue
            ckpt_meta = metadata.state_dict_metadata[ckpt_key]
            if hasattr(ckpt_meta, "size"):
                current_shape = model_state_dict[k].shape
                if tuple(ckpt_meta.size) != tuple(current_shape):
                    skip_keys.append(k)
                    logger.info(
                        "Key model.%s shape mismatch: checkpoint %s vs current %s"
                        " – will keep random init",
                        k,
                        tuple(ckpt_meta.size),
                        tuple(current_shape),
                    )

        for key in skip_keys:
            del model_state_dict[key]

        if skip_keys:
            # Architecture mismatch → partial load, skip optimizer state.
            return {"model": model_state_dict}

        # All keys present → full resume, include optimizer state.
        optim_state_dict = dist_cp_sd.get_optimizer_state_dict(
            self.model, self.optimizer, options=load_opts
        )
        return {"model": model_state_dict, "optim": optim_state_dict}

    # -- override 3: conditionally load optimizer state ----------------------
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load model (and optionally optimizer) state from a checkpoint."""
        dist_cp_sd.set_model_state_dict(
            self.model,
            state_dict["model"],
            options=self.state_dict_load_opts,
        )
        gc_cuda()
        if "optim" in state_dict:
            dist_cp_sd.set_optimizer_state_dict(
                self.model,
                self.optimizer,
                state_dict["optim"],
                options=self.state_dict_load_opts,
            )
            gc_cuda()


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@dataclass
class BandDropoutCurriculumCallback(Callback):
    """Schedule the online encoder's band dropout rate over training.

    Stage 1 (frac < warmup_frac):                 rate = 0
    Stage 2 (warmup_frac <= frac < mid_frac):     ramp 0 -> mid_rate
    Stage 3 (mid_frac <= frac <= 1):              ramp mid_rate -> max_rate

    ``frac = trainer.global_step / trainer.max_steps``.
    """

    warmup_frac: float = BAND_DROPOUT_WARMUP_FRAC
    mid_frac: float = BAND_DROPOUT_MID_FRAC
    mid_rate: float = BAND_DROPOUT_MID_RATE
    max_rate: float = RANDOM_BAND_DROPOUT_MAX_RATE
    metric_name: str = "train/band_dropout_rate"

    def _scheduled_rate(self, frac: float) -> float:
        if frac < self.warmup_frac:
            return 0.0
        if frac < self.mid_frac:
            t = (frac - self.warmup_frac) / max(1e-8, self.mid_frac - self.warmup_frac)
            return self.mid_rate * t
        t = (frac - self.mid_frac) / max(1e-8, 1.0 - self.mid_frac)
        return self.mid_rate + (self.max_rate - self.mid_rate) * t

    def pre_step(self, batch: Any) -> None:
        """Update the online encoder's band dropout rate for the upcoming step."""
        max_steps = getattr(self.trainer, "max_steps", None)
        if not max_steps:
            return
        frac = max(0.0, min(1.0, self.trainer.global_step / max_steps))
        rate = self._scheduled_rate(frac)
        encoder = self.trainer.train_module.model.encoder
        patch_embeddings = getattr(encoder, "patch_embeddings", None)
        if patch_embeddings is None:
            return
        patch_embeddings.band_dropout_rate = rate
        self.trainer.record_metric(self.metric_name, rate)


@dataclass
class ReinitPatchEmbedCallback(Callback):
    """Reinitialise patch embedding weights after a checkpoint is loaded.

    The online encoder's patch embeddings are reinitialised, then the weights
    are copied to the target encoder so both start from identical random state
    (matching the deepcopy behaviour at model init).
    """

    def post_checkpoint_loaded(self, path: Any) -> None:
        """Reinitialise encoder patch embeddings and copy to target encoder."""
        save_folder = str(self.trainer.save_folder)
        if str(path).startswith(save_folder):
            logger.info("Resuming from save_folder – skipping patch embed reinit")
            return
        model = self.trainer.train_module.model
        encoder = model.encoder
        encoder.patch_embeddings.apply(encoder._init_weights)
        logger.info("Reinitialised encoder.patch_embeddings (loaded from %s)", path)
        target = model.target_encoder
        target_pe = target.patch_embeddings
        encoder_pe = encoder.patch_embeddings
        target_pe.load_state_dict(encoder_pe.state_dict())
        logger.info(
            "Copied encoder.patch_embeddings weights to target_encoder.patch_embeddings"
        )


# ---------------------------------------------------------------------------
# Tokenization / masking / data helpers (unchanged from channel_attn_patch_embed)
# ---------------------------------------------------------------------------

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
    Modality.SENTINEL1.name,
    Modality.LANDSAT.name,
]


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
            "type": "random_with_decode",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "only_decode_modalities": ONLY_DECODE_MODALITIES,
        },
        tokenization_config=tokenization_config,
    )


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------


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
) -> PartialLoadTrainModuleConfig:
    """Build the train module config for an experiment."""
    return PartialLoadTrainModuleConfig(
        optim_config=AdamWConfig(lr=0.0001, weight_decay=0.02, fused=False),
        rank_microbatch_size=32,
        masking_config=_masking_config(common.tokenization_config),
        loss_config=LossConfig(
            loss_config={
                "type": "modality_patch_discrimination_masked_negatives",
                "tau": 0.1,
                "same_target_threshold": 0.999,
                "mask_negatives_for_modalities": ONLY_DECODE_MODALITIES,
            }
        ),
        contrastive_config=LossConfig(
            loss_config={
                "type": "InfoNCE",
                "weight": 0.1,
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
    WANDB_USERNAME = "eai-ai2"  # nosec
    WANDB_PROJECT = "2026_04_15_channel_attn_patch_embed"
    PERMANENT_SAVE_INTERVAL = 5000
    EPHERMERAL_SAVE_INTERVAL = 250
    checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)
    wandb_callback = OlmoEarthWandBCallback(
        name=common.run_name,
        project=WANDB_PROJECT,
        entity=WANDB_USERNAME,
        enabled=True,
    )
    garbage_collector_callback = GarbageCollectorCallback(gc_interval=1)
    EVAL_TASKS = {
        "m-eurosat": DownstreamTaskConfig(
            dataset="m-eurosat",
            embedding_batch_size=128,
            num_workers=0,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_interval=Duration.steps(4000),
        ),
        "m_so2sat": DownstreamTaskConfig(
            dataset="m-so2sat",
            embedding_batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            eval_interval=Duration.steps(20000),
        ),
        "mados": DownstreamTaskConfig(
            dataset="mados",
            embedding_batch_size=128,
            probe_batch_size=128,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=False,
            probe_lr=0.01,
            epochs=50,
            eval_interval=Duration.steps(4000),
        ),
        "pastis": DownstreamTaskConfig(
            dataset="pastis",
            embedding_batch_size=4,
            probe_batch_size=8,
            num_workers=8,
            pooling_type=PoolingType.MEAN,
            norm_stats_from_pretrained=True,
            probe_lr=0.1,
            eval_interval=Duration.steps(20000),
            input_modalities=[Modality.SENTINEL2_L2A.name],
            epochs=50,
        ),
    }
    trainer_config = (
        TrainerConfig(
            work_dir=common.save_folder,
            load_strategy=LOAD_STRATEGY,
            save_folder=common.save_folder,
            cancel_check_interval=CANCEL_CHECK_INTERVAL,
            metrics_collect_interval=METRICS_COLLECT_INTERVAL,
            max_duration=MAX_DURATION,
            checkpointer=checkpointer_config,
            # For pretrained-backbone loading: don't restore the old run's
            # training state or optimizer state.  Resume from save_folder
            # (preemption) overrides these with True anyway.
            load_trainer_state=False,
            load_optim_state=False,
        )
        .with_callback("wandb", wandb_callback)
        .with_callback("reinit_patch_embed", ReinitPatchEmbedCallback())
        .with_callback("band_dropout_curriculum", BandDropoutCurriculumCallback())
        .with_callback("speed_monitor", OlmoEarthSpeedMonitorCallback())
        .with_callback("gpu_memory_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=EVAL_TASKS,
            ),
        )
        .with_callback("garbage_collector", garbage_collector_callback)
        .with_callback("beaker", BeakerCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=PERMANENT_SAVE_INTERVAL,
                ephemeral_save_interval=EPHERMERAL_SAVE_INTERVAL,
            ),
        )
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
        band_dropout_rate=0.0,
        random_band_dropout=True,
        band_dropout_modalities=BAND_DROPOUT_MODALITIES,
        channel_attn_dim=CHANNEL_ATTN_DIM,
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
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )
