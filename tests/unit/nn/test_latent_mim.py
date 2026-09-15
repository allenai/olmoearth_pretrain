"""Unit tests for the LatentMIM model wrapper."""

import pytest
import torch

from olmoearth_pretrain.data.collate import collate_single_masked_batched
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.latent_mim import (
    FrozenTargetProjection,
    LatentMIM,
    LatentMIMConfig,
)
from olmoearth_pretrain.train.masking import MaskingConfig

torch.set_default_device("cpu")


@pytest.fixture
def supported_modality_names() -> list[str]:
    """Return the supported modality names for the test."""
    return [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
        Modality.WORLDCOVER.name,
        Modality.LATLON.name,
    ]


def _latent_mim_config(
    supported_modality_names: list[str], projection_only_target: bool
) -> LatentMIMConfig:
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
    return LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=predictor_config,
        projection_only_target=projection_only_target,
    )


@pytest.fixture
def latent_mim_model(
    supported_modality_names: list[str], set_random_seeds: None
) -> LatentMIM:
    """A LatentMIM model with the standard full-copy target encoder."""
    model = _latent_mim_config(supported_modality_names, False).build()
    model.to(device="cpu")
    return model


def test_projection_target_matches_exit0_target_encoder(
    latent_mim_model: LatentMIM,
    samples_without_missing_modalities: list[tuple[int, OlmoEarthSample]],
) -> None:
    """FrozenTargetProjection must reproduce the full target encoder at exit 0."""
    masking_strategy = MaskingConfig(strategy_config={"type": "random"}).build()
    patch_size, masked_batch = collate_single_masked_batched(
        samples_without_missing_modalities,
        transform=None,
        masking_strategy=masking_strategy,
    )
    unmasked = masked_batch.unmask()
    token_exit_cfg = {modality: 0 for modality in Modality.names()}

    full_target = latent_mim_model.target_encoder
    projection_target = FrozenTargetProjection(full_target)
    # eval mode so stochastic band dropout is off for both paths
    full_target.eval()
    projection_target.eval()

    with torch.no_grad():
        expected = full_target.forward(
            unmasked, patch_size=patch_size, token_exit_cfg=token_exit_cfg
        )
        actual = projection_target.forward(
            unmasked, patch_size=patch_size, token_exit_cfg=token_exit_cfg
        )

    expected_tokens = expected["tokens_and_masks"]
    actual_tokens = actual["tokens_and_masks"]
    assert set(actual_tokens.modalities) == set(expected_tokens.modalities)
    for modality in expected_tokens.modalities:
        torch.testing.assert_close(
            getattr(actual_tokens, modality), getattr(expected_tokens, modality)
        )
        mask_name = f"{modality}_mask"
        torch.testing.assert_close(
            getattr(actual_tokens, mask_name), getattr(expected_tokens, mask_name)
        )


def test_projection_target_rejects_nonzero_exits(
    latent_mim_model: LatentMIM,
    samples_without_missing_modalities: list[tuple[int, OlmoEarthSample]],
) -> None:
    """Non-zero token exit depths are not representable by the projection target."""
    masking_strategy = MaskingConfig(strategy_config={"type": "random"}).build()
    patch_size, masked_batch = collate_single_masked_batched(
        samples_without_missing_modalities,
        transform=None,
        masking_strategy=masking_strategy,
    )
    projection_target = FrozenTargetProjection(latent_mim_model.target_encoder)
    token_exit_cfg = {modality: 1 for modality in Modality.names()}
    with pytest.raises(ValueError, match="exit depths 0"):
        projection_target.forward(
            masked_batch.unmask(), patch_size=patch_size, token_exit_cfg=token_exit_cfg
        )


def test_projection_only_target_build(
    supported_modality_names: list[str], set_random_seeds: None
) -> None:
    """projection_only_target=True builds a frozen projection with no blocks."""
    model = _latent_mim_config(supported_modality_names, True).build()
    assert isinstance(model.target_encoder, FrozenTargetProjection)
    assert all(not p.requires_grad for p in model.target_encoder.parameters())
    target_keys = model.target_encoder.state_dict().keys()
    assert not any("blocks" in key for key in target_keys)
    # The projection weights match the online encoder's at init.
    torch.testing.assert_close(
        model.target_encoder.patch_embeddings.state_dict(),
        model.encoder.patch_embeddings.state_dict(),
    )
