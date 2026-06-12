"""Integration tests for the latent MIM Training Module."""

import logging

import pytest
import torch
from olmo_core.optim.adamw import AdamWConfig

from olmoearth_pretrain.data.collate import collate_single_masked_batched
from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.datatypes import OlmoEarthSample
from olmoearth_pretrain.modalities import Modality
from olmoearth_pretrain.nn.flexi_vit import (
    EncoderConfig,
    PredictorConfig,
    ReconstructorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIM, LatentMIMConfig
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

from .helper import attached_train_module, check_loss_is_a_reasonable_value

torch.set_default_device("cpu")
logger = logging.getLogger(__name__)


@pytest.fixture
def latent_mim_model(
    supported_modality_names: list[str], set_random_seeds: None
) -> LatentMIM:
    """Create a real LatentMIM model for testing."""
    # Create encoder config
    encoder_config = EncoderConfig(
        supported_modality_names=supported_modality_names,
        embedding_size=16,
        max_patch_size=8,
        num_heads=2,
        mlp_ratio=1.0,
        depth=2,
        drop_path=0.1,
        max_sequence_length=12,
    )

    # Create predictor config
    predictor_config = PredictorConfig(
        supported_modality_names=supported_modality_names,
        encoder_embedding_size=16,
        decoder_embedding_size=16,
        depth=2,
        mlp_ratio=1.0,
        num_heads=2,
        max_sequence_length=12,
        drop_path=0.0,
        output_embedding_size=None,
    )

    reconstructor_config = ReconstructorConfig(
        supported_modality_names=[
            m for m in supported_modality_names if m != Modality.LATLON.name
        ],
        max_patch_size=8,
        decoder_config=predictor_config,
    )

    # Create LatentMIM config
    latent_mim_config = LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=predictor_config,
        reconstructor_config=reconstructor_config,
    )

    # Build the model
    model = latent_mim_config.build()
    model.to(device="cpu")
    return model


@pytest.fixture
def train_module_config(
    optim_config: AdamWConfig,
) -> LatentMIMTrainModuleConfig:
    """Create a LatentMIMTrainModuleConfig for testing."""
    token_exit_cfg = {modality: 0 for modality in Modality.names()}
    loss_cfg = {"type": "patch_discrimination"}
    masking_cfg = {"type": "random"}
    transform_cfg = TransformConfig(
        transform_type="no_transform",
    )

    # Create the config with all required parameters
    config = LatentMIMTrainModuleConfig(
        optim_config=optim_config,
        rank_microbatch_size=3,
        loss_config=LossConfig(loss_config=loss_cfg),
        mae_loss_config=LossConfig(loss_config={"type": "mae"}),
        masking_config=MaskingConfig(strategy_config=masking_cfg),
        token_exit_cfg=token_exit_cfg,
        ema_decay=(0.996, 1.0),
        max_grad_norm=1.0,
        transform_config=transform_cfg,
    )
    return config


def test_train_batch_without_missing_modalities(
    samples_without_missing_modalities: list[tuple[int, OlmoEarthSample]],
    latent_mim_model: LatentMIM,
    train_module_config: LatentMIMTrainModuleConfig,
    set_random_seeds: None,
) -> None:
    """Test train batch without missing modalities."""
    # Create a fresh masking strategy for collation (MaskingConfig.build() mutates the config)
    masking_strategy = MaskingConfig(strategy_config={"type": "random"}).build()
    batch = collate_single_masked_batched(
        samples_without_missing_modalities,
        transform=None,
        masking_strategy=masking_strategy,
    )
    train_module = train_module_config.build(latent_mim_model, device="cpu")
    with attached_train_module(train_module) as mock_trainer:
        train_module.train_batch(batch)
        logger.info(mock_trainer._metrics)
        check_loss_is_a_reasonable_value(mock_trainer._metrics["train/PatchDisc+MAE"])


def test_train_batch_with_missing_modalities(
    samples_with_missing_modalities: list[tuple[int, OlmoEarthSample]],
    latent_mim_model: LatentMIM,
    train_module_config: LatentMIMTrainModuleConfig,
    set_random_seeds: None,
) -> None:
    """Test train batch with missing modalities."""
    # Create a collated batch with masking (using fresh MaskingConfig since build() mutates)
    masking_strategy = MaskingConfig(strategy_config={"type": "random"}).build()
    batch = collate_single_masked_batched(
        samples_with_missing_modalities,
        transform=None,
        masking_strategy=masking_strategy,
    )
    train_module = train_module_config.build(latent_mim_model, device="cpu")
    with attached_train_module(train_module) as mock_trainer:
        train_module.train_batch(batch)
        logger.info(mock_trainer._metrics)
        check_loss_is_a_reasonable_value(mock_trainer._metrics["train/PatchDisc+MAE"])
