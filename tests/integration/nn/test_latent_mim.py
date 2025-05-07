"""Test LatentMIM with loss."""

import logging

import pytest
import torch

from helios.data.constants import Modality, ModalitySpec
from helios.nn.flexihelios import Encoder, Predictor
from helios.nn.latent_mim import LatentMIM
from helios.train.loss import PatchDiscriminationLoss
from helios.train.masking import MaskedHeliosSample

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
    masked_sample_dict: dict[str, torch.Tensor],
) -> None:
    """Test the full end to end forward pass of the model with an exit configuration and loss."""
    # Define supported modalities
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
    B, H, W, T, C = masked_sample_dict["sentinel2_l2a"].shape
    x = MaskedHeliosSample(**masked_sample_dict)

    patch_size = 4
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
    latentmim = LatentMIM(encoder, predictor)

    _, output = latentmim.forward(x, patch_size)
    output = predictor.forward(output, x.timestamps, patch_size, input_res=1)
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
    loss_fn = PatchDiscriminationLoss()
    with torch.no_grad():
        logger.info("target encoder running here")
        target_output, _ = latentmim.target_encoder.forward(
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
                "project_and_aggregate",
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
