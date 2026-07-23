"""Integration test for the open-set supervised latent-MIM train module."""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from olmo_core.config import DType
from olmo_core.optim.adamw import AdamWConfig

from olmoearth_pretrain.data.collate import collate_single_masked_batched
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.open_set_latent_mim import OpenSetLatentMIMConfig
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.open_set_probe import OpenSetProbeConfig
from olmoearth_pretrain.train.train_module.open_set_latentmim import (
    OpenSetLatentMIMTrainModuleConfig,
)

torch.set_default_device("cpu")
logger = logging.getLogger(__name__)

_CLASS_MAPPING_PATH = (
    Path(__file__).resolve().parents[4]
    / "data"
    / "open_set_segmentation_data"
    / "class_mapping.json"
)

# Imagery modalities the encoder is trained on (labels are excluded on purpose).
_IMAGERY = [
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
    Modality.WORLDCOVER.name,
    Modality.LATLON.name,
]


class MockTrainer:
    """Minimal trainer stub that records metrics."""

    def __init__(self) -> None:
        """Initialize an empty metric store and minimal step state."""
        self._metrics: dict[str, float] = {}
        self._metric_record_counts: dict[str, int] = {}
        self.global_step = 0
        self.max_steps = 100

    def record_metric(
        self, name: str, value: float, reduce_type: object = None, **kwargs: object
    ) -> None:
        """Record the latest value for a metric."""
        self._metric_record_counts[name] = self._metric_record_counts.get(name, 0) + 1
        if self._metric_record_counts[name] > 1:
            raise AssertionError(f"duplicate metric recorded: {name}")
        self._metrics[name] = value


def _make_samples() -> list[tuple[int, OlmoEarthSample]]:
    """Three 8x8 samples carrying valid open_set classification labels."""
    s2 = np.random.randn(8, 8, 12, 13).astype(np.float32)
    s1 = np.random.randn(8, 8, 12, 2).astype(np.float32)
    wc = np.random.randn(8, 8, 1, 10).astype(np.float32)
    latlon = np.random.randn(2).astype(np.float32)
    timestamps = np.tile(np.array([15, 7, 2023], dtype=np.int32), (12, 1))

    # Classification label: global class id 5 (dataset agrifieldnet_india, group 0..12).
    open_set = np.full((8, 8, 1, 1), 5.0, dtype=np.float32)
    # Regression label: no label (dataset id 0) -> exercises the zero-touch guard.
    open_set_regression = np.zeros((8, 8, 1, 2), dtype=np.float32)

    samples = []
    for _ in range(3):
        sample = OlmoEarthSample(
            sentinel2_l2a=s2,
            sentinel1=s1,
            worldcover=wc,
            latlon=latlon,
            open_set=open_set,
            open_set_regression=open_set_regression,
            timestamps=timestamps,
        )
        samples.append((1, sample))
    return samples


@pytest.fixture
def model(set_random_seeds: None) -> OpenSetLatentMIMConfig:
    """Build a small CPU open-set latent-MIM model for integration tests.

    Uses the register (Perceiver) bottleneck with a spatial-latent dim different
    from the encoder token dim, so the test also covers the probe reading a
    narrower spatial latent (the d128-style configuration).
    """
    register_dim = 8
    encoder_config = EncoderConfig(
        supported_modality_names=_IMAGERY,
        embedding_size=16,
        max_patch_size=8,
        num_heads=2,
        mlp_ratio=1.0,
        depth=2,
        drop_path=0.1,
        max_sequence_length=12,
        position_encoding="rope_3d_mixed",  # 3D encoder self-attention
        use_register_bottleneck=True,
        register_grid_size=0,  # dynamic single-latent grid (gdyn)
        register_dim=register_dim,
        register_interleave=True,
        register_per_depth_read_proj=True,
    )
    decoder_config = PredictorConfig(
        supported_modality_names=_IMAGERY,
        encoder_embedding_size=16,
        decoder_embedding_size=16,
        depth=2,
        mlp_ratio=1.0,
        num_heads=2,
        max_sequence_length=12,
        drop_path=0.0,
        output_embedding_size=None,
        position_encoding="rope",  # 2D decoder cross-attends the spatial latent
        use_register_bottleneck=True,
        register_dim=register_dim,
    )
    config = OpenSetLatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        open_set_probe_config=OpenSetProbeConfig(
            class_mapping_path=str(_CLASS_MAPPING_PATH),
        ),
    )
    built = config.build()
    built.to(device="cpu")
    return built


@pytest.mark.parametrize("autocast_precision", [None, DType.bfloat16])
@pytest.mark.parametrize("rank_microbatch_size", [1, 3])
def test_open_set_train_batch_records_supervised_loss(
    model: OpenSetLatentMIMConfig,
    set_random_seeds: None,
    autocast_precision: DType | None,
    rank_microbatch_size: int,
) -> None:
    """train_batch runs and records a finite supervised CE loss."""
    masking_strategy = MaskingConfig(strategy_config={"type": "random"}).build()
    batch = collate_single_masked_batched(
        _make_samples(),
        transform=None,
        masking_strategy=masking_strategy,
    )

    config = OpenSetLatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=1e-4, weight_decay=0.0),
        rank_microbatch_size=rank_microbatch_size,
        loss_config=LossConfig(loss_config={"type": "patch_discrimination"}),
        masking_config=MaskingConfig(strategy_config={"type": "random"}),
        token_exit_cfg={modality: 0 for modality in _IMAGERY},
        ema_decay=(0.996, 1.0),
        max_grad_norm=1.0,
        autocast_precision=autocast_precision,
        sup_loss_weight=1.0,
    )
    train_module = config.build(model, device=torch.device("cpu"))

    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        mock_trainer = MockTrainer()
        train_module.on_attach = MagicMock(return_value=None)  # type: ignore
        train_module._attach_trainer(mock_trainer)
        train_module.train_batch(batch)

    logger.info(mock_trainer._metrics)
    assert "train/open_set_ce" in mock_trainer._metrics
    ce = mock_trainer._metrics["train/open_set_ce"]
    assert torch.isfinite(torch.as_tensor(ce))
    # Every patch has a valid classification label.
    assert mock_trainer._metrics["train/open_set_ce_patches"] > 0
    # Regression had no labels, so no patches contributed.
    assert mock_trainer._metrics["train/open_set_mse_patches"] == 0.0
    for metric_name in (
        "train/open_set_ce",
        "train/open_set_ce_patches",
        "train/open_set_mse",
        "train/open_set_mse_patches",
    ):
        assert mock_trainer._metric_record_counts[metric_name] == 1


def test_supervised_loss_is_weighted_by_global_valid_patch_count(
    model: OpenSetLatentMIMConfig,
) -> None:
    """Local means are scaled so DP averaging yields a global patch mean."""
    config = OpenSetLatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=1e-4, weight_decay=0.0),
        rank_microbatch_size=1,
        loss_config=LossConfig(loss_config={"type": "patch_discrimination"}),
        masking_config=MaskingConfig(strategy_config={"type": "random"}),
        token_exit_cfg={modality: 0 for modality in _IMAGERY},
        ema_decay=(0.996, 1.0),
        max_grad_norm=1.0,
        sup_loss_weight=1.0,
    )
    train_module = config.build(model, device=torch.device("cpu"))
    losses = {
        "zero_touch": torch.tensor(0.0),
        "open_set_ce": torch.tensor(2.0),
        "open_set_mse": torch.tensor(3.0),
    }
    metrics = {
        "open_set_ce_patches": 2.0,
        "open_set_mse_patches": 1.0,
    }

    with (
        patch.object(
            train_module,
            "_global_patch_counts",
            return_value={"open_set_ce": 8.0, "open_set_mse": 4.0},
        ),
        patch(
            "olmoearth_pretrain.train.train_module.open_set_latentmim.get_world_size",
            return_value=2,
        ),
    ):
        loss = train_module._combine_supervised_losses(losses, metrics)

    assert loss.item() == pytest.approx(2.5)
