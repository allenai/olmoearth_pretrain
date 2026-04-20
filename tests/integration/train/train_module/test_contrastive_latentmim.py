"""Integration tests for the contrastive latent MIM Training Module."""

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from olmo_core.optim.adamw import AdamWConfig
from olmo_core.train.config import TrainerConfig

from olmoearth_pretrain.data.collate import collate_double_masked_batched
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIM, LatentMIMConfig
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

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
) -> ContrastiveLatentMIMTrainModuleConfig:
    """Create a LatentMIMTrainModuleConfig for testing."""
    token_exit_cfg = {modality: 0 for modality in Modality.names()}
    loss_cfg = {"type": "patch_discrimination"}
    masking_cfg = {"type": "random"}
    transform_cfg = TransformConfig(
        transform_type="no_transform",
    )
    # Create the config with all required parameters
    config = ContrastiveLatentMIMTrainModuleConfig(
        optim_config=optim_config,
        rank_microbatch_size=3,
        loss_config=LossConfig(loss_config=loss_cfg),
        contrastive_config=LossConfig(
            loss_config={
                "type": "InfoNCE",
                "weight": 0.1,
            }
        ),
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
    train_module_config: ContrastiveLatentMIMTrainModuleConfig,
    set_random_seeds: None,
) -> None:
    """Test train batch without missing modalities."""
    # Create a fresh masking strategy for collation (MaskingConfig.build() mutates the config)
    masking_strategy = MaskingConfig(strategy_config={"type": "random"}).build()
    batch = collate_double_masked_batched(
        samples_without_missing_modalities,
        transform=None,
        masking_strategy=masking_strategy,
        masking_strategy_b=None,  # Use same strategy for both views
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
        check_loss_is_a_reasonable_value(mock_trainer._metrics["train/InfoNCE"])


def test_train_batch_with_vicreg_views(
    samples_without_missing_modalities: list[tuple[int, OlmoEarthSample]],
    latent_mim_model: LatentMIM,
    optim_config: AdamWConfig,
    set_random_seeds: None,
) -> None:
    """VICReg on two masked views (pooled_a vs pooled_b), no InfoNCE."""
    token_exit_cfg = {modality: 0 for modality in Modality.names()}
    config = ContrastiveLatentMIMTrainModuleConfig(
        optim_config=optim_config,
        rank_microbatch_size=3,
        loss_config=LossConfig(loss_config={"type": "patch_discrimination"}),
        contrastive_config=None,
        vicreg_views_config=LossConfig(
            loss_config={
                "type": "vicreg_views",
                "inv_weight": 1.0,
                "var_weight": 1.0,
                "cov_weight": 1.0,
            }
        ),
        masking_config=MaskingConfig(strategy_config={"type": "random"}),
        token_exit_cfg=token_exit_cfg,
        ema_decay=(0.996, 1.0),
        max_grad_norm=1.0,
        transform_config=TransformConfig(transform_type="no_transform"),
    )
    masking_strategy = MaskingConfig(strategy_config={"type": "random"}).build()
    batch = collate_double_masked_batched(
        samples_without_missing_modalities,
        transform=None,
        masking_strategy=masking_strategy,
        masking_strategy_b=None,
    )
    train_module = config.build(latent_mim_model, device="cpu")
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        mock_trainer = MockTrainer()
        on_attach_mock = MagicMock(return_value=None)
        train_module.on_attach = on_attach_mock  # type: ignore
        train_module._attach_trainer(mock_trainer)
        train_module.train_batch(batch)
    assert "train/VICRegViews" in mock_trainer._metrics
    check_loss_is_a_reasonable_value(mock_trainer._metrics["train/VICRegViews"])
    check_loss_is_a_reasonable_value(
        mock_trainer._metrics["train/PatchDisc+VICRegViews"]
    )


def test_train_batch_with_patch_varcov(
    samples_without_missing_modalities: list[tuple[int, OlmoEarthSample]],
    latent_mim_model: LatentMIM,
    optim_config: AdamWConfig,
    set_random_seeds: None,
) -> None:
    """Patch-level var+cov regularization on student latent patches."""
    token_exit_cfg = {modality: 0 for modality in Modality.names()}
    config = ContrastiveLatentMIMTrainModuleConfig(
        optim_config=optim_config,
        rank_microbatch_size=3,
        loss_config=LossConfig(loss_config={"type": "patch_discrimination"}),
        contrastive_config=None,
        patch_varcov_config=LossConfig(
            loss_config={
                "type": "patch_var_cov",
                "var_weight": 1.0,
                "cov_weight": 1.0,
                "gamma": 1.0,
                "modalities": [
                    Modality.SENTINEL2_L2A.name,
                    Modality.SENTINEL1.name,
                    Modality.LANDSAT.name,
                ],
            }
        ),
        masking_config=MaskingConfig(strategy_config={"type": "random"}),
        token_exit_cfg=token_exit_cfg,
        ema_decay=(0.996, 1.0),
        max_grad_norm=1.0,
        transform_config=TransformConfig(transform_type="no_transform"),
    )
    masking_strategy = MaskingConfig(strategy_config={"type": "random"}).build()
    batch = collate_double_masked_batched(
        samples_without_missing_modalities,
        transform=None,
        masking_strategy=masking_strategy,
        masking_strategy_b=None,
    )
    train_module = config.build(latent_mim_model, device="cpu")
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        mock_trainer = MockTrainer()
        on_attach_mock = MagicMock(return_value=None)
        train_module.on_attach = on_attach_mock  # type: ignore
        train_module._attach_trainer(mock_trainer)
        train_module.train_batch(batch)
    assert "train/PatchVarCov" in mock_trainer._metrics
    check_loss_is_a_reasonable_value(mock_trainer._metrics["train/PatchVarCov"])
    check_loss_is_a_reasonable_value(
        mock_trainer._metrics["train/PatchDisc+PatchVarCov"]
    )


def test_train_batch_with_missing_modalities(
    samples_with_missing_modalities: list[tuple[int, OlmoEarthSample]],
    latent_mim_model: LatentMIM,
    train_module_config: ContrastiveLatentMIMTrainModuleConfig,
    set_random_seeds: None,
) -> None:
    """Test train batch with missing modalities."""
    # Create a collated batch with masking (using fresh MaskingConfig since build() mutates)
    masking_strategy = MaskingConfig(strategy_config={"type": "random"}).build()
    batch = collate_double_masked_batched(
        samples_with_missing_modalities,
        transform=None,
        masking_strategy=masking_strategy,
        masking_strategy_b=None,  # Use same strategy for both views
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
        check_loss_is_a_reasonable_value(mock_trainer._metrics["train/InfoNCE"])


def _run_train_batch_and_get_loss(
    model: LatentMIM,
    config: ContrastiveLatentMIMTrainModuleConfig,
    batch: Any,
) -> float:
    """Run a single train_batch and return the ModalityPatchDisc loss value.

    Saves and restores model weights so successive calls see identical parameters.
    """
    import copy

    saved_state = copy.deepcopy(model.state_dict())
    torch.manual_seed(42)
    try:
        train_module = config.build(model, device="cpu")
        with patch(
            "olmoearth_pretrain.train.train_module.train_module.build_world_mesh"
        ):
            mock_trainer = MockTrainer()
            on_attach_mock = MagicMock(return_value=None)
            train_module.on_attach = on_attach_mock  # type: ignore
            train_module._attach_trainer(mock_trainer)
            train_module.train_batch(batch)
        patch_disc_key = [k for k in mock_trainer._metrics if "PatchDisc" in k]
        assert patch_disc_key, (
            f"No PatchDisc metric found in {list(mock_trainer._metrics.keys())}"
        )
        return mock_trainer._metrics[patch_disc_key[0]]
    finally:
        model.load_state_dict(saved_state)


def test_new_vs_vec_loss_through_train_module(
    samples_without_missing_modalities: list[tuple[int, OlmoEarthSample]],
    latent_mim_model: LatentMIM,
    optim_config: AdamWConfig,
    set_random_seeds: None,
) -> None:
    """End-to-end: modality_patch_discrimination_new and vec produce same loss through train module."""
    masking_strategy = MaskingConfig(strategy_config={"type": "random"}).build()
    batch = collate_double_masked_batched(
        samples_without_missing_modalities,
        transform=None,
        masking_strategy=masking_strategy,
        masking_strategy_b=None,
    )

    token_exit_cfg = {modality: 0 for modality in Modality.names()}

    def _make_config(loss_type: str) -> ContrastiveLatentMIMTrainModuleConfig:
        return ContrastiveLatentMIMTrainModuleConfig(
            optim_config=optim_config,
            rank_microbatch_size=3,
            loss_config=LossConfig(loss_config={"type": loss_type, "tau": 0.1}),
            contrastive_config=LossConfig(
                loss_config={"type": "InfoNCE", "weight": 0.1}
            ),
            masking_config=MaskingConfig(strategy_config={"type": "random"}),
            token_exit_cfg=token_exit_cfg,
            ema_decay=(0.996, 1.0),
            max_grad_norm=1.0,
        )

    loss_new = _run_train_batch_and_get_loss(
        latent_mim_model, _make_config("modality_patch_discrimination_new"), batch
    )
    loss_vec = _run_train_batch_and_get_loss(
        latent_mim_model, _make_config("modality_patch_discrimination_vec"), batch
    )

    logger.info(f"loss_new={loss_new}, loss_vec={loss_vec}")
    assert abs(loss_new - loss_vec) < 1e-4, (
        f"Loss mismatch through train module: new={loss_new}, vec={loss_vec}"
    )


def test_new_vs_vec_loss_with_missing_modalities(
    samples_with_missing_modalities: list[tuple[int, OlmoEarthSample]],
    latent_mim_model: LatentMIM,
    optim_config: AdamWConfig,
    set_random_seeds: None,
) -> None:
    """End-to-end with missing modalities: new and vec losses must match."""
    masking_strategy = MaskingConfig(strategy_config={"type": "random"}).build()
    batch = collate_double_masked_batched(
        samples_with_missing_modalities,
        transform=None,
        masking_strategy=masking_strategy,
        masking_strategy_b=None,
    )

    token_exit_cfg = {modality: 0 for modality in Modality.names()}

    def _make_config(loss_type: str) -> ContrastiveLatentMIMTrainModuleConfig:
        return ContrastiveLatentMIMTrainModuleConfig(
            optim_config=optim_config,
            rank_microbatch_size=3,
            loss_config=LossConfig(loss_config={"type": loss_type, "tau": 0.1}),
            contrastive_config=LossConfig(
                loss_config={"type": "InfoNCE", "weight": 0.1}
            ),
            masking_config=MaskingConfig(strategy_config={"type": "random"}),
            token_exit_cfg=token_exit_cfg,
            ema_decay=(0.996, 1.0),
            max_grad_norm=1.0,
        )

    loss_new = _run_train_batch_and_get_loss(
        latent_mim_model, _make_config("modality_patch_discrimination_new"), batch
    )
    loss_vec = _run_train_batch_and_get_loss(
        latent_mim_model, _make_config("modality_patch_discrimination_vec"), batch
    )

    logger.info(f"loss_new={loss_new}, loss_vec={loss_vec}")
    assert abs(loss_new - loss_vec) < 1e-4, (
        f"Loss mismatch with missing modalities: new={loss_new}, vec={loss_vec}"
    )


def test_train_batch_with_pooled_target_align(
    samples_without_missing_modalities: list[tuple[int, OlmoEarthSample]],
    latent_mim_model: LatentMIM,
    optim_config: AdamWConfig,
    set_random_seeds: None,
) -> None:
    """End-to-end: pooled_target_align_weight adds MSE loss between student and target pooled reps."""
    masking_strategy = MaskingConfig(strategy_config={"type": "random"}).build()
    batch = collate_double_masked_batched(
        samples_without_missing_modalities,
        transform=None,
        masking_strategy=masking_strategy,
        masking_strategy_b=None,
    )

    token_exit_cfg = {modality: 0 for modality in Modality.names()}

    config = ContrastiveLatentMIMTrainModuleConfig(
        optim_config=optim_config,
        rank_microbatch_size=3,
        loss_config=LossConfig(loss_config={"type": "patch_discrimination"}),
        contrastive_config=LossConfig(loss_config={"type": "InfoNCE", "weight": 0.1}),
        masking_config=MaskingConfig(strategy_config={"type": "random"}),
        token_exit_cfg=token_exit_cfg,
        ema_decay=(0.996, 1.0),
        max_grad_norm=1.0,
        pooled_target_align_weight=1.0,
    )

    train_module = config.build(latent_mim_model, device="cpu")
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        mock_trainer = MockTrainer()
        on_attach_mock = MagicMock(return_value=None)
        train_module.on_attach = on_attach_mock  # type: ignore
        train_module._attach_trainer(mock_trainer)
        train_module.train_batch(batch)
        logger.info(mock_trainer._metrics)
        check_loss_is_a_reasonable_value(mock_trainer._metrics["train/PatchDisc"])
        check_loss_is_a_reasonable_value(mock_trainer._metrics["train/InfoNCE"])
        assert "train/pooled_target_mse" in mock_trainer._metrics, (
            f"pooled_target_mse not logged. Metrics: {list(mock_trainer._metrics.keys())}"
        )
        pt_mse = mock_trainer._metrics["train/pooled_target_mse"]
        assert torch.isfinite(pt_mse), f"pooled_target_mse is not finite: {pt_mse}"
        assert pt_mse >= 0, f"pooled_target_mse should be non-negative: {pt_mse}"


def test_train_batch_with_multi_level_supervision(
    samples_without_missing_modalities: list[tuple[int, OlmoEarthSample]],
    supported_modality_names: list[str],
    optim_config: AdamWConfig,
    set_random_seeds: None,
) -> None:
    """Multi-level supervision at depths [2, 4] on a depth-6 encoder produces distinct intermediate losses."""
    deep_encoder_config = EncoderConfig(
        supported_modality_names=supported_modality_names,
        embedding_size=16,
        max_patch_size=8,
        num_heads=2,
        mlp_ratio=1.0,
        depth=6,
        drop_path=0.0,
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
    )
    model = LatentMIMConfig(
        encoder_config=deep_encoder_config,
        decoder_config=predictor_config,
    ).build()
    model.to(device="cpu")

    masking_strategy = MaskingConfig(strategy_config={"type": "random"}).build()
    batch = collate_double_masked_batched(
        samples_without_missing_modalities,
        transform=None,
        masking_strategy=masking_strategy,
        masking_strategy_b=None,
    )

    token_exit_cfg = {modality: 0 for modality in Modality.names()}

    config = ContrastiveLatentMIMTrainModuleConfig(
        optim_config=optim_config,
        rank_microbatch_size=3,
        loss_config=LossConfig(loss_config={"type": "patch_discrimination"}),
        contrastive_config=LossConfig(loss_config={"type": "InfoNCE", "weight": 0.1}),
        masking_config=MaskingConfig(strategy_config={"type": "random"}),
        token_exit_cfg=token_exit_cfg,
        ema_decay=(0.996, 1.0),
        max_grad_norm=1.0,
        multi_level_depths=[2, 4],
        multi_level_weight=1.0,
    )

    train_module = config.build(model, device="cpu")
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        mock_trainer = MockTrainer()
        on_attach_mock = MagicMock(return_value=None)
        train_module.on_attach = on_attach_mock  # type: ignore
        train_module._attach_trainer(mock_trainer)
        train_module.train_batch(batch)
        logger.info(mock_trainer._metrics)

        check_loss_is_a_reasonable_value(mock_trainer._metrics["train/PatchDisc"])
        check_loss_is_a_reasonable_value(mock_trainer._metrics["train/InfoNCE"])
        assert "train/multi_level_mse" in mock_trainer._metrics, (
            f"multi_level_mse not logged. Metrics: {list(mock_trainer._metrics.keys())}"
        )
        ml_mse = mock_trainer._metrics["train/multi_level_mse"]
        assert torch.isfinite(ml_mse), f"multi_level_mse is not finite: {ml_mse}"
        assert ml_mse > 0, f"multi_level_mse should be positive: {ml_mse}"

    # Verify intermediates were actually captured at the right depths
    student_inter = model.encoder._intermediate_pooled
    assert student_inter is not None, "Student intermediates should be captured"
    assert set(student_inter.keys()) == {2, 4}, (
        f"Expected depths {{2, 4}}, got {set(student_inter.keys())}"
    )
    for depth, pooled in student_inter.items():
        assert pooled.ndim == 2, f"Depth {depth}: expected [B, D], got {pooled.shape}"


def test_multi_level_validation_rejects_invalid_depth(
    samples_without_missing_modalities: list[tuple[int, OlmoEarthSample]],
    supported_modality_names: list[str],
    optim_config: AdamWConfig,
    set_random_seeds: None,
) -> None:
    """capture_at with out-of-range depths raises ValueError during training."""
    deep_model = LatentMIMConfig(
        encoder_config=EncoderConfig(
            supported_modality_names=supported_modality_names,
            embedding_size=16,
            max_patch_size=8,
            num_heads=2,
            mlp_ratio=1.0,
            depth=4,
            drop_path=0.0,
            max_sequence_length=12,
        ),
        decoder_config=PredictorConfig(
            supported_modality_names=supported_modality_names,
            encoder_embedding_size=16,
            decoder_embedding_size=16,
            depth=2,
            mlp_ratio=1.0,
            num_heads=2,
            max_sequence_length=12,
            drop_path=0.0,
        ),
    ).build()
    deep_model.to(device="cpu")

    masking_strategy = MaskingConfig(strategy_config={"type": "random"}).build()
    batch = collate_double_masked_batched(
        samples_without_missing_modalities,
        transform=None,
        masking_strategy=masking_strategy,
        masking_strategy_b=None,
    )

    config = ContrastiveLatentMIMTrainModuleConfig(
        optim_config=optim_config,
        rank_microbatch_size=3,
        loss_config=LossConfig(loss_config={"type": "patch_discrimination"}),
        masking_config=MaskingConfig(strategy_config={"type": "random"}),
        token_exit_cfg={modality: 0 for modality in Modality.names()},
        ema_decay=(0.996, 1.0),
        max_grad_norm=1.0,
        multi_level_depths=[5],
        multi_level_weight=1.0,
    )

    train_module = config.build(deep_model, device="cpu")
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        mock_trainer = MockTrainer()
        on_attach_mock = MagicMock(return_value=None)
        train_module.on_attach = on_attach_mock  # type: ignore
        train_module._attach_trainer(mock_trainer)
        with pytest.raises(ValueError, match="capture_at depths .* out of range"):
            train_module.train_batch(batch)
