"""Tests for OlmoEarthEvalWrapper instance-embedding selection (class token vs mean pool)."""

from typing import Any

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.evals.datasets.configs import TaskType
from olmoearth_pretrain.evals.eval_wrapper import (
    INSTANCE_EMBEDDING_OPTIONS,
    OlmoEarthEvalWrapper,
)
from olmoearth_pretrain.nn.flexi_vit import Encoder
from olmoearth_pretrain.nn.pooling import PoolingType, pool_unmasked_tokens

SUPPORTED_MODALITIES = [Modality.SENTINEL2_L2A, Modality.LATLON]
EMBEDDING_SIZE = 16
PATCH_SIZE = 2
B = 2


def _build_encoder(**overrides: Any) -> Encoder:
    """Build a small Encoder (class token enabled by default)."""
    kwargs: dict[str, Any] = dict(
        supported_modalities=SUPPORTED_MODALITIES,
        embedding_size=EMBEDDING_SIZE,
        max_patch_size=8,
        min_patch_size=1,
        num_heads=2,
        mlp_ratio=2.0,
        max_sequence_length=12,
        depth=2,
        drop_path=0.0,
        use_class_token=True,
    )
    kwargs.update(overrides)
    encoder = Encoder(**kwargs)
    encoder.eval()
    return encoder


def _masked_sample() -> MaskedOlmoEarthSample:
    """Build a fully visible (no MISSING tokens) eval-style masked sample."""
    H = W = 4
    T = 2
    C = Modality.SENTINEL2_L2A.num_bands
    torch.manual_seed(0)
    sentinel2_l2a = torch.randn(B, H, W, T, C)
    sentinel2_l2a_mask = torch.full(
        (B, H, W, T, C), fill_value=MaskValue.ONLINE_ENCODER.value, dtype=torch.long
    )
    latlon = torch.tensor([[37.0, -122.0]] * B)
    latlon_mask = torch.full(
        (B, Modality.LATLON.num_bands),
        fill_value=MaskValue.ONLINE_ENCODER.value,
        dtype=torch.long,
    )
    days = torch.randint(1, 28, (B, T, 1), dtype=torch.long)
    months = torch.randint(0, 12, (B, T, 1), dtype=torch.long)
    years = torch.randint(2018, 2025, (B, T, 1), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=-1)
    return MaskedOlmoEarthSample(
        sentinel2_l2a=sentinel2_l2a,
        sentinel2_l2a_mask=sentinel2_l2a_mask,
        latlon=latlon,
        latlon_mask=latlon_mask,
        timestamps=timestamps,
    )


def _wrapper(
    encoder: Encoder, task_type: TaskType, instance_embedding: str = "auto"
) -> OlmoEarthEvalWrapper:
    """Build an OlmoEarthEvalWrapper around the test encoder."""
    return OlmoEarthEvalWrapper(
        model=encoder,
        task_type=task_type,
        patch_size=PATCH_SIZE,
        pooling_type=PoolingType.MEAN,
        instance_embedding=instance_embedding,
    )


def test_classification_auto_uses_class_token() -> None:
    """With a class-token encoder, 'auto' picks the class token as the [B, D] embedding."""
    encoder = _build_encoder()
    wrapper = _wrapper(encoder, TaskType.CLASSIFICATION)
    x = _masked_sample()
    labels = torch.zeros(B, dtype=torch.long)
    with torch.no_grad():
        embeddings, out_labels = wrapper(x, labels)
        expected = encoder(x, patch_size=PATCH_SIZE, fast_pass=True)["class_token"]
    assert embeddings.shape == (B, EMBEDDING_SIZE)
    assert torch.allclose(embeddings, expected)
    assert out_labels is labels
    # And it is not just the masked mean.
    with torch.no_grad():
        mean_pooled = pool_unmasked_tokens(
            encoder(x, patch_size=PATCH_SIZE, fast_pass=True)["tokens_and_masks"],
            PoolingType.MEAN,
            spatial_pooling=False,
        )
    assert not torch.allclose(embeddings, mean_pooled)


def test_classification_auto_without_class_token_mean_pools() -> None:
    """'auto' falls back to mean pooling when the encoder has no class token."""
    encoder = _build_encoder(use_class_token=False)
    wrapper = _wrapper(encoder, TaskType.CLASSIFICATION)
    x = _masked_sample()
    with torch.no_grad():
        embeddings, _ = wrapper(x, torch.zeros(B, dtype=torch.long))
        expected = pool_unmasked_tokens(
            encoder(x, patch_size=PATCH_SIZE, fast_pass=True)["tokens_and_masks"],
            PoolingType.MEAN,
            spatial_pooling=False,
        )
    assert embeddings.shape == (B, EMBEDDING_SIZE)
    assert torch.allclose(embeddings, expected)


def test_segmentation_keeps_spatial_pooling_with_class_token() -> None:
    """Spatial tasks keep per-patch spatial pooling even when a class token exists."""
    encoder = _build_encoder()
    wrapper = _wrapper(encoder, TaskType.SEGMENTATION)
    x = _masked_sample()
    with torch.no_grad():
        embeddings, _ = wrapper(x, torch.zeros(B, 4, 4, dtype=torch.long))
        expected = pool_unmasked_tokens(
            encoder(x, patch_size=PATCH_SIZE, fast_pass=True)["tokens_and_masks"],
            PoolingType.MEAN,
            spatial_pooling=True,
        )
    # 4x4 input at patch size 2 -> 2x2 patch grid, one embedding per patch.
    assert embeddings.shape == (B, 2, 2, EMBEDDING_SIZE)
    assert torch.allclose(embeddings, expected)


def test_forced_mean_pool_overrides_class_token() -> None:
    """instance_embedding='mean_pool' uses the masked mean despite the class token."""
    encoder = _build_encoder()
    wrapper = _wrapper(encoder, TaskType.CLASSIFICATION, instance_embedding="mean_pool")
    x = _masked_sample()
    with torch.no_grad():
        embeddings, _ = wrapper(x, torch.zeros(B, dtype=torch.long))
        output = encoder(x, patch_size=PATCH_SIZE, fast_pass=True)
        expected = pool_unmasked_tokens(
            output["tokens_and_masks"], PoolingType.MEAN, spatial_pooling=False
        )
    assert embeddings.shape == (B, EMBEDDING_SIZE)
    assert torch.allclose(embeddings, expected)
    assert not torch.allclose(embeddings, output["class_token"])


def test_forced_class_token_without_one_raises() -> None:
    """instance_embedding='class_token' errors clearly when the model emits none."""
    encoder = _build_encoder(use_class_token=False)
    wrapper = _wrapper(
        encoder, TaskType.CLASSIFICATION, instance_embedding="class_token"
    )
    x = _masked_sample()
    with pytest.raises(ValueError, match="class token"):
        with torch.no_grad():
            wrapper(x, torch.zeros(B, dtype=torch.long))


def test_forced_class_token_works() -> None:
    """instance_embedding='class_token' returns the class token when available."""
    encoder = _build_encoder()
    wrapper = _wrapper(
        encoder, TaskType.CLASSIFICATION, instance_embedding="class_token"
    )
    x = _masked_sample()
    with torch.no_grad():
        embeddings, _ = wrapper(x, torch.zeros(B, dtype=torch.long))
        expected = encoder(x, patch_size=PATCH_SIZE, fast_pass=True)["class_token"]
    assert torch.allclose(embeddings, expected)


def test_invalid_instance_embedding_rejected() -> None:
    """Unknown instance_embedding values fail fast at construction."""
    assert set(INSTANCE_EMBEDDING_OPTIONS) == {"auto", "class_token", "mean_pool"}
    with pytest.raises(ValueError, match="instance_embedding"):
        _wrapper(_build_encoder(), TaskType.CLASSIFICATION, instance_embedding="cls")
