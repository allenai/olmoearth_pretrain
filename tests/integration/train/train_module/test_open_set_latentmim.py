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
from olmoearth_pretrain.nn.flexi_vit import (
    EncoderConfig,
    PerceiverConfig,
    PredictorConfig,
)
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

    Uses the Perceiver with a register dim different from the encoder token dim,
    so the test also covers the probe reading a narrower register grid.
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
        perceiver_config=PerceiverConfig(
            register_dim=register_dim,
            latent_depth=2,
            per_depth_read_proj=True,
        ),
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
        position_encoding="rope",  # 2D decoder cross-attends the register grid
        use_perceiver=True,
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
    assert train_module.total_loss_name.endswith("+open_set")
    ce = mock_trainer._metrics["open_set/ce"]
    assert torch.isfinite(torch.as_tensor(ce))
    # Every sample has valid classification labels on every patch. Counts are
    # averaged over microbatches, so this is the per-microbatch labeled count.
    assert mock_trainer._metrics["open_set/ce_samples"] == float(
        min(rank_microbatch_size, 3)
    )
    assert mock_trainer._metrics["open_set/ce_patches"] > 0
    # Regression had no labels: no value is reported, only zero counts.
    assert "open_set/mse" not in mock_trainer._metrics
    assert mock_trainer._metrics["open_set/mse_samples"] == 0.0
    assert mock_trainer._metrics["open_set/mse_patches"] == 0.0
    assert mock_trainer._metrics["open_set/backbone_frozen"] == 0.0
    for metric_name in (
        "open_set/ce",
        "open_set/ce_samples",
        "open_set/ce_patches",
        "open_set/mse_samples",
        "open_set/mse_patches",
    ):
        assert mock_trainer._metric_record_counts[metric_name] == 1


def test_init_weights_from_checkpoint_without_probe(
    model: OpenSetLatentMIMConfig, tmp_path: Path
) -> None:
    """A weights.pth lacking the probe initializes the backbone; the probe stays fresh."""
    # Simulate the converted v1.3 base checkpoint: same model minus the probe.
    source_state = {
        k: v + 1.0 if v.is_floating_point() else v
        for k, v in model.state_dict().items()
        if not k.startswith("open_set_probe.")
    }
    weights_path = tmp_path / "weights.pth"
    torch.save(source_state, weights_path)
    fresh_probe_weight = model.open_set_probe.cls_head.weight.detach().clone()

    def _config(**overrides: object) -> OpenSetLatentMIMTrainModuleConfig:
        return OpenSetLatentMIMTrainModuleConfig(
            optim_config=AdamWConfig(lr=1e-4, weight_decay=0.0),
            rank_microbatch_size=1,
            loss_config=LossConfig(loss_config={"type": "patch_discrimination"}),
            masking_config=MaskingConfig(strategy_config={"type": "random"}),
            token_exit_cfg={modality: 0 for modality in _IMAGERY},
            ema_decay=(0.996, 1.0),
            max_grad_norm=1.0,
            init_weights_path=str(weights_path),
            **overrides,  # type: ignore[arg-type]
        )

    # Without the allowance the missing probe keys are an error.
    with pytest.raises(RuntimeError, match="does not match the model"):
        _config().build(model, device=torch.device("cpu"))

    _config(init_weights_allow_missing=["open_set_probe."]).build(
        model, device=torch.device("cpu")
    )
    for key, value in source_state.items():
        assert torch.equal(model.state_dict()[key], value), key
    assert torch.equal(model.open_set_probe.cls_head.weight, fresh_probe_weight)


@pytest.mark.parametrize("freeze_probe_after_unfreeze", [True, False])
def test_freeze_schedule_trains_probe_only_then_unfreezes(
    model: OpenSetLatentMIMConfig,
    set_random_seeds: None,
    freeze_probe_after_unfreeze: bool,
) -> None:
    """Phase 1 trains only the probe; phase 2 trains the backbone (probe frozen).

    Params that were already frozen at init (e.g. the projection-only target
    copies) must never be flipped trainable by the schedule. With
    ``freeze_probe_after_unfreeze=False`` the probe keeps training in phase 2.
    """
    # Simulate an intentionally-frozen param (like FrozenTargetProjection).
    always_frozen_param = next(model.encoder.parameters())
    always_frozen_param.requires_grad_(False)

    masking_strategy = MaskingConfig(strategy_config={"type": "random"}).build()
    batch = collate_single_masked_batched(
        _make_samples(),
        transform=None,
        masking_strategy=masking_strategy,
    )

    config = OpenSetLatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=1e-4, weight_decay=0.0),
        rank_microbatch_size=3,
        loss_config=LossConfig(loss_config={"type": "patch_discrimination"}),
        masking_config=MaskingConfig(strategy_config={"type": "random"}),
        token_exit_cfg={modality: 0 for modality in _IMAGERY},
        ema_decay=(0.996, 1.0),
        max_grad_norm=1.0,
        sup_loss_weight=1.0,
        freeze_backbone_until_step=5,
        freeze_probe_after_unfreeze=freeze_probe_after_unfreeze,
    )
    train_module = config.build(model, device=torch.device("cpu"))
    probe_param_ids = {id(p) for p in model.open_set_probe.parameters()}

    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        mock_trainer = MockTrainer()
        train_module.on_attach = MagicMock(return_value=None)  # type: ignore
        train_module._attach_trainer(mock_trainer)

        # Phase 1: only the probe has requires_grad / receives gradients, and the
        # probe-only fast path still records the supervised metrics.
        mock_trainer.global_step = 0
        train_module.train_batch(batch)
        assert train_module.backbone_frozen
        assert mock_trainer._metrics["open_set/backbone_frozen"] == 1.0
        assert mock_trainer._metrics["open_set/ce_samples"] == 3.0
        for p in model.open_set_probe.parameters():
            assert p.requires_grad and p.grad is not None
        for p in model.parameters():
            if id(p) not in probe_param_ids:
                assert not p.requires_grad
                assert p.grad is None

        # Phase 2: the backbone trains again, except always-frozen params (the
        # manually frozen one plus e.g. the frozen month-embedding tables), and the
        # probe is frozen unless configured otherwise.
        train_module.zero_grads()
        mock_trainer._metric_record_counts.clear()  # second batch re-records
        mock_trainer.global_step = 5
        train_module.train_batch(batch)
        assert not train_module.backbone_frozen
        assert mock_trainer._metrics["open_set/backbone_frozen"] == 0.0
        assert id(always_frozen_param) in train_module._always_frozen_param_ids
        assert not always_frozen_param.requires_grad
        assert always_frozen_param.grad is None
        for p in model.open_set_probe.parameters():
            if freeze_probe_after_unfreeze:
                assert not p.requires_grad
                assert p.grad is None
            else:
                assert p.requires_grad
                assert p.grad is not None
        num_backbone_with_grad = 0
        for p in model.parameters():
            if id(p) in probe_param_ids:
                continue
            if id(p) in train_module._always_frozen_param_ids:
                assert not p.requires_grad
                continue
            assert p.requires_grad
            num_backbone_with_grad += int(p.grad is not None)
        assert num_backbone_with_grad > 0
        # The supervised gradient reaches the backbone through the frozen probe.
        assert any(p.grad is not None for p in model.encoder.perceiver.parameters())
