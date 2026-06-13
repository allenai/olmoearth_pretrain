"""End-to-end test of the combined recipe: CLIP loss + RoPE + simple encodings.

Exercises the full clip_rope_simple.py configuration at toy scale through the
real collator (metadata dropout + latlon pinning), masking strategy, model
(class token, latlon token, RoPE, separate simple encodings), and train module
(CLIP token loss and CLIP instance loss, each with its own learned
temperature).
"""

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from olmo_core.optim.adamw import AdamWConfig

from olmoearth_pretrain.data.collate import (
    MetadataDropout,
    collate_double_masked_batched,
)
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.datatypes import MaskValue
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIM, LatentMIMConfig
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModuleConfig,
)

torch.set_default_device("cpu")
logger = logging.getLogger(__name__)

MODALITY_NAMES = [
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
    Modality.WORLDCOVER.name,
    Modality.LATLON.name,
]


@pytest.fixture
def combined_model(set_random_seeds: None) -> LatentMIM:
    """Toy-scale model with the clip_rope_simple.py architecture choices."""
    encoder_config = EncoderConfig(
        supported_modality_names=MODALITY_NAMES,
        embedding_size=16,
        max_patch_size=8,
        num_heads=2,
        mlp_ratio=1.0,
        depth=2,
        drop_path=0.0,
        max_sequence_length=12,
        use_class_token=True,
        spatial_pos_encoding="rope",
        rope_coordinate_scale=0.25,
        encoding_mode="separate",
        channel_encoding_dim=0,  # encoder-side channel encoding dropped
        temporal_encoding_dim=4,
        temporal_encoding_type="simple",
        latlon_encoding_dim=0,  # location enters via the latlon token only
    )
    predictor_config = PredictorConfig(
        supported_modality_names=MODALITY_NAMES,
        encoder_embedding_size=16,
        decoder_embedding_size=16,
        depth=2,
        mlp_ratio=1.0,
        num_heads=2,
        max_sequence_length=12,
        spatial_pos_encoding="rope",
        rope_coordinate_scale=0.25,
        encoding_mode="separate",
        channel_encoding_dim=8,  # decoder keeps the channel encoding
        temporal_encoding_dim=4,
        temporal_encoding_type="simple",
        latlon_encoding_dim=0,
    )
    model = LatentMIMConfig(
        encoder_config=encoder_config, decoder_config=predictor_config
    ).build()
    model.to(device="cpu")
    return model


def _train_module_config() -> ContrastiveLatentMIMTrainModuleConfig:
    return ContrastiveLatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=1e-4, weight_decay=0.0, betas=(0.9, 0.95)),
        rank_microbatch_size=3,
        loss_config=LossConfig(
            loss_config={
                "type": "clip_patch_discrimination",
                "same_target_threshold": 0.999,
                "mask_negatives_for_modalities": [Modality.WORLDCOVER.name],
            }
        ),
        contrastive_config=LossConfig(
            loss_config={"type": "clip_infonce", "weight": 0.05}
        ),
        masking_config=MaskingConfig(strategy_config={"type": "random"}),
        token_exit_cfg={modality: 0 for modality in Modality.names()},
        ema_decay=(1.0, 1.0),
        max_grad_norm=1.0,
        transform_config=TransformConfig(transform_type="no_transform"),
    )


class MockTrainer:
    """Minimal trainer stub that records metrics."""

    def __init__(self) -> None:
        """Initialize the metric store."""
        self._metrics: dict[str, Any] = {}
        self.global_step = 0
        self.max_steps = 100

    def record_metric(
        self,
        name: str,
        value: Any,
        reduce_type: Any,
        namespace: str | None = None,
    ) -> None:
        """Record a metric value."""
        self._metrics[name] = value


def _collate(
    samples: list[tuple[int, OlmoEarthSample]],
    metadata_dropout: MetadataDropout | None,
) -> Any:
    masking_strategy = MaskingConfig(strategy_config={"type": "random"}).build()
    return collate_double_masked_batched(
        samples,
        transform=None,
        masking_strategy=masking_strategy,
        masking_strategy_b=None,
        metadata_dropout=metadata_dropout,
    )


def _run_train_batch(
    model: LatentMIM,
    batch: Any,
) -> dict[str, Any]:
    train_module = _train_module_config().build(model, device="cpu")
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        mock_trainer = MockTrainer()
        train_module.on_attach = MagicMock(return_value=None)  # type: ignore[method-assign]
        train_module._attach_trainer(mock_trainer)  # type: ignore[arg-type]
        train_module.train_batch(batch)
    return mock_trainer._metrics


def _assert_finite_positive(value: torch.Tensor) -> None:
    assert torch.isfinite(value).all()
    assert value > 0


def test_combined_recipe_latlon_kept(
    samples_without_missing_modalities: list[tuple[int, OlmoEarthSample]],
    combined_model: LatentMIM,
    set_random_seeds: None,
) -> None:
    """Full train batch with the latlon token present and pinned to encoder."""
    batch = _collate(samples_without_missing_modalities, metadata_dropout=None)
    # The collator must have pinned latlon to ONLINE_ENCODER (never DECODER).
    for view in (batch[1], batch[2]):
        assert (
            (view.latlon_mask == MaskValue.ONLINE_ENCODER.value)
            | (view.latlon_mask == MaskValue.MISSING.value)
        ).all()
    metrics = _run_train_batch(combined_model, batch)
    _assert_finite_positive(metrics["train/ClipPatchDisc"])
    _assert_finite_positive(metrics["train/ClipInfoNCE"])
    # Initial learned temperatures: exp(log(1/0.07)) ~ 14.29
    assert abs(float(metrics["train/logit_scale"]) - 1.0 / 0.07) < 0.1
    assert abs(float(metrics["train/instance_logit_scale"]) - 1.0 / 0.07) < 0.1


def test_combined_recipe_metadata_dropped(
    samples_without_missing_modalities: list[tuple[int, OlmoEarthSample]],
    combined_model: LatentMIM,
    set_random_seeds: None,
) -> None:
    """Full train batch with year and latlon dropped for the whole batch."""
    dropout = MetadataDropout(
        year_dropout_rate=1.0, latlon_dropout_rate=1.0, view_mode="shared"
    )
    batch = _collate(samples_without_missing_modalities, metadata_dropout=dropout)
    for view in (batch[1], batch[2]):
        # Dropped latlon = MISSING-masked (removed pre-attention), with the
        # field kept so the embedding module participates in every backward.
        assert view.latlon is not None
        assert (view.latlon_mask == MaskValue.MISSING.value).all()
        assert (view.timestamps[..., 2] == 0).all()
    metrics = _run_train_batch(combined_model, batch)
    _assert_finite_positive(metrics["train/ClipPatchDisc"])
    _assert_finite_positive(metrics["train/ClipInfoNCE"])


def test_combined_recipe_opposite_metadata_views(
    samples_without_missing_modalities: list[tuple[int, OlmoEarthSample]],
    combined_model: LatentMIM,
    set_random_seeds: None,
) -> None:
    """Opposite mode: the two views have field-wise anti-correlated metadata."""
    dropout = MetadataDropout(
        year_dropout_rate=1.0, latlon_dropout_rate=1.0, view_mode="opposite"
    )
    _, view_a, view_b = _collate(
        samples_without_missing_modalities, metadata_dropout=dropout
    )
    # latlon: exactly one view MISSING-masked, the other encoder-visible.
    a_latlon_dropped = bool((view_a.latlon_mask == MaskValue.MISSING.value).all())
    b_latlon_dropped = bool((view_b.latlon_mask == MaskValue.MISSING.value).all())
    assert a_latlon_dropped != b_latlon_dropped
    # year: exactly one view carries the year-0 sentinel.
    a_year_dropped = bool((view_a.timestamps[..., 2] == 0).all())
    b_year_dropped = bool((view_b.timestamps[..., 2] == 0).all())
    assert a_year_dropped != b_year_dropped
    # Both views still carry the latlon field (module participates every step).
    assert view_a.latlon is not None and view_b.latlon is not None
    metrics = _run_train_batch(combined_model, (1, view_a, view_b))
    _assert_finite_positive(metrics["train/ClipPatchDisc"])
    _assert_finite_positive(metrics["train/ClipInfoNCE"])


def test_combined_recipe_missing_modalities(
    samples_with_missing_modalities: list[tuple[int, OlmoEarthSample]],
    combined_model: LatentMIM,
    set_random_seeds: None,
) -> None:
    """Full train batch with missing modalities in the samples."""
    batch = _collate(samples_with_missing_modalities, metadata_dropout=None)
    metrics = _run_train_batch(combined_model, batch)
    _assert_finite_positive(metrics["train/ClipPatchDisc"])
    _assert_finite_positive(metrics["train/ClipInfoNCE"])
