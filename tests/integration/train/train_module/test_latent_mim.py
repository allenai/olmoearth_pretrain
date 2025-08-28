"""Integration tests for the latent MIM Training Module."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from olmo_core.optim.adamw import AdamWConfig
from olmo_core.train.config import TrainerConfig

from helios.data.constants import Modality
from helios.data.dataset import HeliosSample, collate_helios
from helios.data.transform import TransformConfig
from helios.nn.flexihelios import EncoderConfig, PredictorConfig
from helios.nn.latent_mim import LatentMIM, LatentMIMConfig
from helios.train.loss import LossConfig
from helios.train.masking import MaskingConfig
from helios.train.train_module.latent_mim import LatentMIMTrainModuleConfig

torch.set_default_device("cpu")
logger = logging.getLogger(__name__)


@pytest.fixture
def supported_modality_names() -> list[str]:
    """Return the supported modality names for the test."""
    return [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.LATLON.name,
    ]


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

    # Create LatentMIM config
    latent_mim_config = LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=predictor_config,
    )

    # Build the model
    model = latent_mim_config.build()
    model.to(device="cpu")
    return model


@pytest.fixture
def optim_config() -> AdamWConfig:
    """Create an AdamWConfig for testing."""
    return AdamWConfig(
        lr=1e-4,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


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
        masking_config=MaskingConfig(strategy_config=masking_cfg),
        token_exit_cfg=token_exit_cfg,
        ema_decay=(0.996, 1.0),
        max_grad_norm=1.0,
        transform_config=transform_cfg,
    )
    return config


@pytest.fixture
def trainer_config(tmp_path: Path) -> TrainerConfig:
    """Create a TrainerConfig for testing."""
    return TrainerConfig(
        work_dir=tmp_path,
        save_folder=tmp_path,
    )


class MockTrainer:
    """Mock trainer class for testing."""

    def __init__(self) -> None:
        """Initialize the mock trainer."""
        self._metrics: dict[str, float] = {}
        self.global_step = 0
        self.max_steps = 100

    def record_metric(
        self,
        name: str,
        value: float,
        reduce_type: str,
        namespace: str | None = None,
    ) -> None:
        """Record a metric in the mock trainer.

        Args:
            name: Name of the metric
            value: Value of the metric
            reduce_type: Type of reduction to apply
            namespace: Optional namespace for the metric
        """
        self._metrics[name] = value


def test_train_batch_without_missing_modalities(
    samples_without_missing_modalities: list[tuple[int, HeliosSample]],
    latent_mim_model: LatentMIM,
    train_module_config: LatentMIMTrainModuleConfig,
    set_random_seeds: None,
) -> None:
    """Test train batch without missing modalities."""
    batch = collate_helios(samples_without_missing_modalities)
    train_module = train_module_config.build(latent_mim_model, device="cpu")
    with patch("helios.train.train_module.train_module.build_world_mesh"):
        # Mock the trainer property
        mock_trainer = MockTrainer()
        # Create a MagicMock for on_attach
        on_attach_mock = MagicMock(return_value=None)
        # Patch the on_attach method
        train_module.on_attach = on_attach_mock  # type: ignore
        train_module._attach_trainer(mock_trainer)
        train_module.train_batch(batch)
        logger.info(mock_trainer._metrics)
        assert torch.allclose(
            mock_trainer._metrics["train/PatchDisc"],
            torch.tensor(2.0),
            atol=1e-1,
        )


def test_train_batch_with_missing_modalities(
    samples_with_missing_modalities: list[tuple[int, HeliosSample]],
    latent_mim_model: LatentMIM,
    train_module_config: LatentMIMTrainModuleConfig,
    set_random_seeds: None,
) -> None:
    """Test train batch with missing modalities."""
    # Create a collated batch
    batch = collate_helios(samples_with_missing_modalities)
    train_module = train_module_config.build(latent_mim_model, device="cpu")
    with patch("helios.train.train_module.train_module.build_world_mesh"):
        # Mock the trainer property
        mock_trainer = MockTrainer()
        # Create a MagicMock for on_attach
        on_attach_mock = MagicMock(return_value=None)
        # Patch the on_attach method
        train_module.on_attach = on_attach_mock  # type: ignore
        train_module._attach_trainer(mock_trainer)
        train_module.train_batch(batch)
        logger.info(mock_trainer._metrics)
        assert torch.allclose(
            mock_trainer._metrics["train/PatchDisc"],
            torch.tensor(1.89),
            atol=1e-1,
        )
