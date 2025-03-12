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
from helios.nn.flexihelios import EncoderConfig, PredictorConfig
from helios.nn.latent_mim import LatentMIM, LatentMIMConfig
from helios.train.loss import LossConfig
from helios.train.masking import MaskingConfig
from helios.train.train_module.latent_mim import LatentMIMTrainModuleConfig

torch.set_default_device("cpu")
logger = logging.getLogger(__name__)


@pytest.fixture
def samples_with_missing_modalities() -> list[HeliosSample]:
    """Samples with missing modalities."""
    s2_H, s2_W, s2_T, s2_C = 16, 16, 12, 13
    s1_H, s1_W, s1_T, s1_C = 16, 16, 12, 2
    wc_H, wc_W, wc_T, wc_C = 16, 16, 1, 10
    example_s2_data = torch.randn(
        s2_H, s2_W, s2_T, s2_C, device="cpu", dtype=torch.float32
    )
    example_s1_data = torch.randn(
        s1_H, s1_W, s1_T, s1_C, device="cpu", dtype=torch.float32
    )
    example_wc_data = torch.randn(
        wc_H, wc_W, wc_T, wc_C, device="cpu", dtype=torch.float32
    )
    example_latlon_data = torch.randn(2, device="cpu", dtype=torch.float32)
    timestamps = torch.tensor(
        [[15, 7, 2023], [15, 8, 2023], [15, 9, 2023]],
        dtype=torch.int32,
        device="cpu",
    )

    sample1 = HeliosSample(
        sentinel2_l2a=example_s2_data,
        sentinel1=example_s1_data,
        worldcover=example_wc_data,
        latlon=example_latlon_data,
        timestamps=timestamps,
    )

    sample2 = HeliosSample(
        sentinel2_l2a=example_s2_data,
        sentinel1=None,
        worldcover=example_wc_data,
        latlon=example_latlon_data,
        timestamps=timestamps,
    )

    sample_3 = HeliosSample(
        sentinel2_l2a=example_s2_data,
        sentinel1=example_s1_data,
        worldcover=None,
        latlon=example_latlon_data,
        timestamps=timestamps,
    )

    batch = [sample1, sample2, sample_3]
    return batch


@pytest.fixture
def samples_without_missing_modalities(
    set_random_seeds: None,
) -> list[HeliosSample]:
    """Samples without missing modalities."""
    s2_H, s2_W, s2_T, s2_C = 16, 16, 12, 13
    s1_H, s1_W, s1_T, s1_C = 16, 16, 12, 2
    wc_H, wc_W, wc_T, wc_C = 16, 16, 1, 10
    example_s2_data = torch.randn(
        s2_H,
        s2_W,
        s2_T,
        s2_C,
        device="cpu",
        dtype=torch.float32,
        requires_grad=True,
    )
    example_s1_data = torch.randn(
        s1_H,
        s1_W,
        s1_T,
        s1_C,
        device="cpu",
        dtype=torch.float32,
        requires_grad=True,
    )
    example_wc_data = torch.randn(
        wc_H,
        wc_W,
        wc_T,
        wc_C,
        device="cpu",
        dtype=torch.float32,
        requires_grad=True,
    )
    print(f"example_wc_data device: {example_wc_data.device}")
    example_latlon_data = torch.randn(2, device="cpu", dtype=torch.float32)
    timestamps = torch.tensor(
        [[15, 7, 2023], [15, 8, 2023], [15, 9, 2023]],
        dtype=torch.int32,
        device="cpu",
    )

    sample1 = HeliosSample(
        sentinel2_l2a=example_s2_data,
        sentinel1=example_s1_data,
        worldcover=example_wc_data,
        latlon=example_latlon_data,
        timestamps=timestamps,
    )

    sample2 = HeliosSample(
        sentinel2_l2a=example_s2_data,
        sentinel1=example_s1_data,
        worldcover=example_wc_data,
        latlon=example_latlon_data,
        timestamps=timestamps,
    )

    sample_3 = HeliosSample(
        sentinel2_l2a=example_s2_data,
        sentinel1=example_s1_data,
        worldcover=example_wc_data,
        latlon=example_latlon_data,
        timestamps=timestamps,
    )

    batch = [sample1, sample2, sample_3]
    return batch


@pytest.fixture
def supported_modalities() -> list:
    """Return the supported modalities for the test."""
    return [
        Modality.get("sentinel2_l2a"),
        Modality.get("sentinel1"),
        Modality.get("worldcover"),
        Modality.get("latlon"),
    ]


@pytest.fixture
def supported_modality_names() -> list[str]:
    """Return the supported modality names for the test."""
    return ["sentinel2_l2a", "sentinel1", "worldcover", "latlon"]


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
        use_channel_embs=True,
        random_channel_embs=False,
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
        learnable_channel_embeddings=True,
        random_channel_embeddings=False,
        output_embedding_size=None,
    )

    # Create LatentMIM config
    latent_mim_config = LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=predictor_config,
        transform_type="no_transform",
        token_budget=1500,
        h_w_to_sample_min=2,
        h_w_to_sample_max=13,
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

    # Create the config with all required parameters
    config = LatentMIMTrainModuleConfig(
        optim_config=optim_config,
        rank_microbatch_size=3,
        loss_config=LossConfig(loss_config=loss_cfg),
        masking_config=MaskingConfig(strategy_config=masking_cfg),
        token_exit_cfg=token_exit_cfg,
        ema_decay=(0.996, 1.0),
        max_grad_norm=1.0,
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
    samples_without_missing_modalities: list[HeliosSample],
    supported_modalities: list,
    latent_mim_model: LatentMIM,
    train_module_config: LatentMIMTrainModuleConfig,
    set_random_seeds: None,
) -> None:
    """Test train batch without missing modalities."""
    # Create a collated batch
    import random

    import numpy as np

    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    random.seed(42)

    torch.backends.cuda.matmul.allow_tf32 = (
        False  # Disables TensorFloat32 (TF32) on matmul ops
    )
    torch.backends.cudnn.allow_tf32 = False  # Disables TF32 on cuDNN
    torch.backends.cudnn.benchmark = False  # Disables the cuDNN auto-tuner
    torch.backends.cudnn.deterministic = (
        True  # Forces cuDNN to use deterministic algorithms
    )
    # in the worst case, use this:
    torch.backends.cudnn.enabled = False  # Disables cuDNN entirely
    batch = collate_helios(samples_without_missing_modalities, supported_modalities)
    train_module = train_module_config.build(latent_mim_model, device="cpu")
    # tokens = torch.randn((3, 53, 16), device="cpu", dtype=torch.float32)
    # normed_tokens = latent_mim_model.encoder.norm(tokens)
    # logger.info(
    #     f"normed tokens sum and std: {normed_tokens.sum()} {normed_tokens.std()}"
    # )
    with patch("helios.train.train_module.train_module.build_world_mesh"):
        # Mock the trainer property
        mock_trainer = MockTrainer()
        # Create a MagicMock for on_attach
        on_attach_mock = MagicMock(return_value=None)
        # Patch the on_attach method
        train_module.on_attach = on_attach_mock  # type: ignore
        train_module._attach_trainer(mock_trainer)

        # Patch the update_target_encoder method to avoid MagicMock issues
        with patch.object(train_module, "update_target_encoder"):
            # No need to create actual dataset and dataloader
            # Just call train_batch directly with our batch
            train_module.train_batch(batch)
            # I want to be able to have a trainer object that actually has the metrics
            logger.info(mock_trainer._metrics)
            assert torch.allclose(
                mock_trainer._metrics["train/PatchDisc"],
                torch.tensor(0.9216),
                atol=1e-2,
            )


def test_train_batch_with_missing_modalities(
    samples_with_missing_modalities: list[HeliosSample],
    supported_modalities: list,
    latent_mim_model: LatentMIM,
    train_module_config: LatentMIMTrainModuleConfig,
    set_random_seeds: None,
) -> None:
    """Test train batch with missing modalities."""
    # Create a collated batch
    batch = collate_helios(samples_with_missing_modalities, supported_modalities)
    train_module = train_module_config.build(latent_mim_model, device="cpu")
    with patch("helios.train.train_module.train_module.build_world_mesh"):
        # Mock the trainer property
        mock_trainer = MockTrainer()
        # Create a MagicMock for on_attach
        on_attach_mock = MagicMock(return_value=None)
        # Patch the on_attach method
        train_module.on_attach = on_attach_mock  # type: ignore
        train_module._attach_trainer(mock_trainer)

        # Patch the update_target_encoder method to avoid MagicMock issues
        with patch.object(train_module, "update_target_encoder"):
            # No need to create actual dataset and dataloader
            # Just call train_batch directly with our batch
            train_module.train_batch(batch)
            # I want to be able to have a trainer object that actually has the metrics
            logger.info(mock_trainer._metrics)
            assert torch.allclose(
                mock_trainer._metrics["train/PatchDisc"],
                torch.tensor(0.864),
                atol=1e-2,
            )
