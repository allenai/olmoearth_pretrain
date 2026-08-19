"""Integration tests for the latent MIM Training Module."""

import logging
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.optim.adamw import AdamWConfig
from olmo_core.train.config import TrainerConfig

from olmoearth_pretrain.data.collate import collate_single_masked_batched
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIM, LatentMIMConfig
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.latent_mim import LatentMIMTrainModuleConfig

from .helper import check_loss_is_a_reasonable_value

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
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        # Mock the trainer property
        mock_trainer = MockTrainer()
        # Create a MagicMock for on_attach
        on_attach_mock = MagicMock(return_value=None)
        # Patch the on_attach method
        train_module.on_attach = on_attach_mock  # type: ignore
        train_module._attach_trainer(mock_trainer)
        train_module.train_batch(batch)
        logger.info(mock_trainer._metrics)
        check_loss_is_a_reasonable_value(mock_trainer._metrics["train/PatchDisc"])


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
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        # Mock the trainer property
        mock_trainer = MockTrainer()
        # Create a MagicMock for on_attach
        on_attach_mock = MagicMock(return_value=None)
        # Patch the on_attach method
        train_module.on_attach = on_attach_mock  # type: ignore
        train_module._attach_trainer(mock_trainer)
        train_module.train_batch(batch)
        logger.info(mock_trainer._metrics)
        check_loss_is_a_reasonable_value(mock_trainer._metrics["train/PatchDisc"])


def test_band_dropout_enabled_by_train_module(
    supported_modality_names: list[str],
    train_module_config: LatentMIMTrainModuleConfig,
    set_random_seeds: None,
) -> None:
    """Band dropout stays off after model build and turns on after train module build.

    Building the LatentMIM directly (the path fine-tuning uses) must leave band
    dropout disabled. Wrapping it in the pretraining train module must enable it
    on the online encoder while leaving the target encoder disabled.
    """
    encoder_config = EncoderConfig(
        supported_modality_names=supported_modality_names,
        embedding_size=16,
        max_patch_size=8,
        num_heads=2,
        mlp_ratio=1.0,
        depth=2,
        drop_path=0.1,
        max_sequence_length=12,
        band_dropout_rate=0.5,
    )
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
    model = LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=predictor_config,
    ).build()

    # Configured rate is stored on the encoder, but the patch embeddings start
    # with rate 0.0 (the fine-tuning path stops here and never enables it).
    assert model.encoder.band_dropout_rate == 0.5
    assert model.encoder.patch_embeddings.band_dropout_rate == 0.0
    assert model.target_encoder.patch_embeddings.band_dropout_rate == 0.0

    train_module_config.build(model, device="cpu")

    # Online encoder is now enabled; target encoder (deepcopied before this
    # point) remains off so it always sees full spectral info.
    assert model.encoder.patch_embeddings.band_dropout_rate == 0.5
    assert model.target_encoder.patch_embeddings.band_dropout_rate == 0.0


def _small_latent_mim(
    supported_modality_names: list[str], keep_encoder_ema: bool
) -> LatentMIM:
    """A small LatentMIM, optionally with the eval-only encoder EMA copy."""
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
    model = LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=predictor_config,
        keep_encoder_ema=keep_encoder_ema,
    ).build()
    model.to(device="cpu")
    return model


def test_update_encoder_ema(
    supported_modality_names: list[str],
    train_module_config: LatentMIMTrainModuleConfig,
    set_random_seeds: None,
) -> None:
    """update_encoder_ema folds the encoder into the EMA copy and nothing else."""
    model = _small_latent_mim(supported_modality_names, keep_encoder_ema=True)
    config = replace(train_module_config, encoder_ema_decay=(0.5, 0.5))
    train_module = config.build(model, device="cpu")
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        mock_trainer = MockTrainer()
        train_module.on_attach = MagicMock(return_value=None)  # type: ignore
        train_module._attach_trainer(mock_trainer)

        # Diverge the online encoder from the copies (as an optimizer step would).
        with torch.no_grad():
            for p in model.encoder.parameters():
                p.add_(1.0)

        target_before = {
            k: v.clone() for k, v in model.target_encoder.state_dict().items()
        }
        train_module.update_encoder_ema()

    # With decay 0.5 the copy lands exactly halfway between its old value
    # (encoder - 1) and the new encoder: encoder - 0.5.
    assert model.ema_encoder is not None
    for p, ep in zip(model.encoder.parameters(), model.ema_encoder.parameters()):
        torch.testing.assert_close(ep.data, p.data - 0.5)
    # The target encoder is a different average and must not move.
    for k, v in model.target_encoder.state_dict().items():
        torch.testing.assert_close(v, target_before[k])
    assert mock_trainer._metrics["train/encoder_ema_decay"] == 0.5


def test_encoder_ema_decay_requires_ema_encoder(
    supported_modality_names: list[str],
    train_module_config: LatentMIMTrainModuleConfig,
    set_random_seeds: None,
) -> None:
    """A decay without a model-side EMA copy (and vice versa) must raise."""
    plain_model = _small_latent_mim(supported_modality_names, keep_encoder_ema=False)
    with_decay = replace(train_module_config, encoder_ema_decay=(0.999, 0.999))
    with pytest.raises(OLMoConfigurationError, match="keep_encoder_ema"):
        with_decay.build(plain_model, device="cpu")

    ema_model = _small_latent_mim(supported_modality_names, keep_encoder_ema=True)
    with pytest.raises(OLMoConfigurationError, match="keep_encoder_ema"):
        train_module_config.build(ema_model, device="cpu")
