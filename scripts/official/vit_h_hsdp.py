"""ViT-H recipe with 2-node hybrid sharding.

This recipe uses a ViT-H-style encoder with a shallow decoder, and defaults to
2-node launch settings with hybrid sharded data parallelism:

- 2 replicas across 2 nodes
- FSDP sharding within each 8-GPU node
- flash attention
- fused AdamW
- torch.compile
- vectorized patch discrimination loss

Launch example:
    python scripts/official/vit_h_hsdp.py launch vit_h_hsdp ai2/jupiter \
        --launch.num_nodes=2 \
        --launch.num_gpus=8 \
        --launch.clusters=[ai2/jupiter,ai2/ceres] \
        --trainer.callbacks.wandb.project=YYYY_MM_DD_vith_hsdp
"""

import logging

from olmo_core.config import DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.optim import AdamWConfig
from script import (
    build_common_components as _build_common_components_base,
)
from script import (
    build_dataloader_config,
    build_dataset_config,
    build_trainer_config,
    build_visualize_config,
)
from script import (
    build_train_module_config as _build_train_module_config_base,
)

from olmoearth_pretrain.internal.experiment import CommonComponents, SubCmd, main
from olmoearth_pretrain.nn.flexihelios import (
    EncoderConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.train.loss import LossConfig

logger = logging.getLogger(__name__)

MAX_PATCH_SIZE = 8
VIT_H_SIZE = {
    "decoder_depth": 4,
    "encoder_embedding_size": 1280,
    "decoder_embedding_size": 1280,
    "encoder_depth": 32,
    "encoder_num_heads": 16,
    "decoder_num_heads": 16,
    "mlp_ratio": 4.0,
}


def build_common_components(
    script: str, cmd: SubCmd, run_name: str, cluster: str, overrides: list[str]
) -> CommonComponents:
    """Build common components and default to a 2-node launch."""
    config = _build_common_components_base(script, cmd, run_name, cluster, overrides)
    if config.launch is not None:
        config.launch.num_nodes = 2
        config.launch.num_gpus = 8
    return config


def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the ViT-H model config."""
    encoder_config = EncoderConfig(
        embedding_size=VIT_H_SIZE["encoder_embedding_size"],
        num_heads=VIT_H_SIZE["encoder_num_heads"],
        depth=VIT_H_SIZE["encoder_depth"],
        mlp_ratio=VIT_H_SIZE["mlp_ratio"],
        supported_modality_names=common.training_modalities,
        max_patch_size=MAX_PATCH_SIZE,
        drop_path=0.1,
        max_sequence_length=12,
        use_flash_attn=True,
        use_linear_patch_embed=False,
    )
    decoder_config = PredictorConfig(
        encoder_embedding_size=VIT_H_SIZE["encoder_embedding_size"],
        decoder_embedding_size=VIT_H_SIZE["decoder_embedding_size"],
        depth=VIT_H_SIZE["decoder_depth"],
        mlp_ratio=VIT_H_SIZE["mlp_ratio"],
        num_heads=VIT_H_SIZE["decoder_num_heads"],
        supported_modality_names=common.training_modalities,
        max_sequence_length=12,
        use_flash_attn=True,
    )
    return LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    )


def build_train_module_config(common: CommonComponents):
    """Build the train module config with HSDP + speed knobs."""
    cfg = _build_train_module_config_base(common)
    cfg.optim_config = AdamWConfig(lr=0.0001, weight_decay=0.02, fused=True)
    cfg.loss_config = LossConfig(
        loss_config={"type": "modality_patch_discrimination_vec", "tau": 0.1}
    )
    cfg.compile_model = True
    cfg.dp_config = DataParallelConfig(
        name=DataParallelType.hsdp,
        param_dtype=DType.bfloat16,
        reduce_dtype=DType.float32,
        num_replicas=2,
        shard_degree=8,
    )
    return cfg


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
