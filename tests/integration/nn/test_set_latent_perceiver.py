"""Integration test for the Set-Latent Perceiver train-module glue.

Constructs the real ``SetLatentPerceiverTrainModule`` (optimizer, microbatch
splitting, rank-synced K / per-rank mask seeds, metric aggregation) with a stub
trainer and runs a training step end-to-end on CPU.
"""

from typing import Any

import pytest
import torch
from olmo_core.optim import AdamWConfig

from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample
from olmoearth_pretrain.nn.set_latent_perceiver import SetLatentPerceiverConfig
from olmoearth_pretrain.train.train_module.set_latent_perceiver import (
    SetLatentPerceiverTrainModuleConfig,
)


class _StubTrainer:
    """Minimal trainer stub exposing what train_batch reads."""

    def __init__(self) -> None:
        self.global_step = 0
        self.metrics: dict[str, float] = {}

    def record_metric(
        self, name: str, value: Any, reduce_type: Any = None, namespace: Any = None
    ) -> None:
        """Record a metric value."""
        self.metrics[name] = float(
            value.detach() if hasattr(value, "detach") else value
        )


def _make_sample(
    b: int = 6, h: int = 16, w: int = 16, t: int = 3
) -> MaskedOlmoEarthSample:
    torch.manual_seed(0)
    ts = torch.zeros(b, t, 3, dtype=torch.float32)
    for ti in range(t):
        ts[:, ti, 0] = 1
        ts[:, ti, 1] = ti % 12
        ts[:, ti, 2] = 2021
    return MaskedOlmoEarthSample(
        timestamps=ts,
        sentinel2_l2a=torch.randn(b, h, w, t, 12),
        sentinel1=torch.randn(b, h, w, t, 2),
        latlon=torch.randn(b, 2) * 30,
    )


def test_train_module_steps_and_records_metrics() -> None:
    """A full train_batch runs, records metrics, and flows grads to all params."""
    model = SetLatentPerceiverConfig(
        supported_modality_names=["sentinel2_l2a", "sentinel1"],
        dim=64,
        heads=4,
        latents=16,
        nested_latents=(8, 16),
        self_depth_per_read=1,
        level2_depth=1,
        decoder_depth=1,
        target_dim=32,
        cond_dropout=0.0,
    ).build()

    tm_config = SetLatentPerceiverTrainModuleConfig(
        optim_config=AdamWConfig(lr=1e-3, fused=False),
        rank_microbatch_size=3,  # forces 2 microbatches for b=6
        transform_config=TransformConfig(transform_type="flip_and_rotate"),
        max_grad_norm=1.0,
    )
    train_module = tm_config.build(model, device=torch.device("cpu"))
    train_module._trainer = _StubTrainer()

    train_module.zero_grads()
    train_module.train_batch((8, _make_sample()), dry_run=False)

    metrics = train_module.trainer.metrics
    assert "train/loss" in metrics and metrics["train/loss"] > 0
    assert "train/top1" in metrics
    assert "train/target_count" in metrics and metrics["train/target_count"] > 0
    # Per-group valid fraction logged for each group.
    assert any(k.startswith("train/valid_frac/") for k in metrics)

    # Grads flowed to every trainable parameter (all modalities present).
    missing = [
        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert missing == []

    # Optimizer step runs without error.
    train_module.optim_step()


def test_k_seed_is_rank_synced_across_microbatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The nested-K seed is derived from global_step (rank-free)."""
    model = SetLatentPerceiverConfig(
        supported_modality_names=["sentinel2_l2a"],
        dim=32,
        heads=4,
        latents=16,
        nested_latents=(8, 16),
        self_depth_per_read=1,
        level2_depth=1,
        decoder_depth=1,
        target_dim=16,
    ).build()
    tm = SetLatentPerceiverTrainModuleConfig(
        optim_config=AdamWConfig(lr=1e-3, fused=False),
        rank_microbatch_size=4,
    ).build(model, device=torch.device("cpu"))
    tm._trainer = _StubTrainer()
    tm._trainer.global_step = 7

    # Capture the k_seed used per microbatch by wrapping the model forward.
    seen_k: list[int] = []
    orig_forward = model.forward

    def spy(
        sample: Any,
        patch_size: Any = None,
        *,
        mask_seed: int | None = None,
        k_seed: int | None = None,
        cloud: Any = None,
    ) -> Any:
        out = orig_forward(
            sample, patch_size, mask_seed=mask_seed, k_seed=k_seed, cloud=cloud
        )
        assert k_seed is not None
        seen_k.append(k_seed)
        return out

    monkeypatch.setattr(model, "forward", spy)
    tm.zero_grads()
    tm.train_batch((8, _make_sample(b=8)), dry_run=True)
    # Same rank-free k_seed (== global_step) across all microbatches.
    assert len(set(seen_k)) == 1 and seen_k[0] == 7
