"""Test LatentMIM with loss."""

import logging

import pytest
import torch

from helios.data.constants import Modality, ModalitySpec
from helios.data.transform import TransformConfig
from helios.nn.flexihelios import Encoder, Predictor
from helios.nn.latent_mim import LatentMIM
from helios.train.loss import PatchDiscriminationLoss
from helios.train.masking import MaskedHeliosSample, MaskValue

logger = logging.getLogger(__name__)


@pytest.fixture
def modality_band_set_len_and_total_bands(
    supported_modalities: list[ModalitySpec],
) -> dict[str, tuple[int, int]]:
    """Get the number of band sets and total bands for each modality.

    Returns:
        Dictionary mapping modality name to tuple of (num_band_sets, total_bands)
    """
    return {
        modality.name: (
            len(modality.band_sets),
            modality.num_bands,
        )
        for modality in supported_modalities
    }


def test_latentmim_with_loss(
    modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
) -> None:
    """Test the full end to end forward pass of the model with an exit configuration and loss."""
    supported_modalities = [
        Modality.SENTINEL2_L2A,
        Modality.LATLON,
        Modality.WORLDCOVER,
    ]
    sentinel2_l2a_num_band_sets, sentinel2_l2a_num_bands = (
        modality_band_set_len_and_total_bands["sentinel2_l2a"]
    )
    latlon_num_band_sets, latlon_num_bands = modality_band_set_len_and_total_bands[
        "latlon"
    ]
    B, H, W, T, C = (
        1,
        4,
        4,
        2,
        sentinel2_l2a_num_bands,
    )
    # Create dummy sentinel2_l2a data: shape (B, H, W, T, C)
    sentinel2_l2a = torch.randn(B, H, W, T, C)
    # Here we assume 0 (ONLINE_ENCODER) means the token is visible.
    sentinel2_l2a_mask = torch.zeros(B, H, W, T, C, dtype=torch.long)
    # Dummy latitude-longitude data.
    latlon = torch.randn(B, latlon_num_bands)
    latlon_mask = (
        torch.ones(B, latlon_num_bands, dtype=torch.float32) * MaskValue.DECODER.value
    )
    worldcover = torch.randn(B, H, W, 1, 1)
    worldcover_mask = (
        torch.ones(B, H, W, 1, 1, dtype=torch.float32) * MaskValue.DECODER.value
    )
    # Generate valid timestamps:
    # - days: range 1..31,
    # - months: range 1..13,
    # - years: e.g. 2018-2019.
    days = torch.randint(0, 25, (B, T, 1), dtype=torch.long)
    months = torch.randint(0, 12, (B, T, 1), dtype=torch.long)
    years = torch.randint(2018, 2020, (B, T, 1), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=-1)  # Shape: (B, T, 3)

    masked_sample_dict = {
        "sentinel2_l2a": sentinel2_l2a,
        "sentinel2_l2a_mask": sentinel2_l2a_mask,
        "latlon": latlon,
        "latlon_mask": latlon_mask,
        "worldcover": worldcover,
        "worldcover_mask": worldcover_mask,
        "timestamps": timestamps,
    }
    x = MaskedHeliosSample(**masked_sample_dict)

    patch_size = 4
    input_res = 1
    # Shared constants for encoder and predictor
    MAX_PATCH_SIZE = 8
    NUM_HEADS = 2
    MLP_RATIO = 4.0
    MAX_SEQ_LENGTH = 12
    DEPTH = 2
    DROP_PATH = 0.1
    ENCODER_EMBEDDING_SIZE = 16
    DECODER_EMBEDDING_SIZE = 16
    encoder = Encoder(
        supported_modalities=supported_modalities,
        embedding_size=ENCODER_EMBEDDING_SIZE,
        max_patch_size=MAX_PATCH_SIZE,
        min_patch_size=1,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        max_sequence_length=MAX_SEQ_LENGTH,
        use_channel_embs=True,
        depth=DEPTH,
        drop_path=DROP_PATH,
    )
    predictor = Predictor(
        supported_modalities=supported_modalities,
        encoder_embedding_size=ENCODER_EMBEDDING_SIZE,
        decoder_embedding_size=DECODER_EMBEDDING_SIZE,
        depth=DEPTH,
        mlp_ratio=MLP_RATIO,
        num_heads=NUM_HEADS,
        max_sequence_length=MAX_SEQ_LENGTH,
        drop_path=DROP_PATH,
        learnable_channel_embeddings=True,
    )
    transform = TransformConfig(transform_type="no_transform").build()
    latentmim = LatentMIM(encoder, predictor, transform)
    output = latentmim.forward(x, patch_size)
    output = predictor.forward(output, timestamps, patch_size, input_res)
    patched_H = H // patch_size
    patched_W = W // patch_size
    assert output.sentinel2_l2a is not None
    assert output.sentinel2_l2a_mask is not None
    assert output.latlon is not None
    assert output.latlon_mask is not None
    assert output.sentinel2_l2a.shape == (
        B,
        patched_H,
        patched_W,
        T,
        sentinel2_l2a_num_band_sets,
        predictor.output_embedding_size,
    )
    assert output.sentinel2_l2a_mask.shape == (
        B,
        patched_H,
        patched_W,
        T,
        sentinel2_l2a_num_band_sets,
    )
    assert output.latlon.shape == (
        B,
        latlon_num_band_sets,
        predictor.output_embedding_size,
    )
    assert output.latlon_mask.shape == (
        B,
        latlon_num_band_sets,
    )
    assert output.worldcover is not None
    assert output.worldcover_mask is not None
    assert output.worldcover.shape == (
        B,
        patched_H,
        patched_W,
        1,
        1,
        predictor.output_embedding_size,
    )
    assert output.worldcover_mask.shape == (
        B,
        patched_H,
        patched_W,
        1,
        1,
    )

    # this reflects the forward_model function in latentmim
    loss_fn = PatchDiscriminationLoss()
    with torch.no_grad():
        logger.info("target encoder running here")
        target_output = latentmim.target_encoder.forward(
            x.unmask(),
            patch_size=patch_size,
            token_exit_cfg={
                modality: 0 for modality in latentmim.encoder.supported_modality_names
            },
        )
    loss_fn.compute(output, target_output).backward()

    for name, param in latentmim.encoder.named_parameters():
        # worldcover and latlons are masked from the encoder
        if not any(
            ignore_param in name
            for ignore_param in [
                "pos_embed",
                "month_embed",
                "composite_encodings.per_modality_channel_embeddings.latlon",
                "composite_encodings.per_modality_channel_embeddings.worldcover",
                "patch_embeddings.per_modality_embeddings.latlon",
                "patch_embeddings.per_modality_embeddings.worldcover",
            ]
        ):
            assert param.grad is not None, name
    for name, param in latentmim.decoder.named_parameters():
        # sentinel2_l2a is "masked" from the decoder
        if not any(
            ignore_param in name
            for ignore_param in [
                "pos_embed",
                "month_embed",
                "composite_encodings.per_modality_channel_embeddings.latlon",
            ]
        ):
            assert param.grad is not None, name
    for name, param in latentmim.target_encoder.named_parameters():
        assert param.grad is None, name
