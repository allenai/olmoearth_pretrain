"""Test LatentMIM with loss."""

import logging

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.nn.flexi_vit import (
    Encoder,
    EncoderConfig,
    Predictor,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIM, LatentMIMConfig
from olmoearth_pretrain.nn.utils import unpack_encoder_output
from olmoearth_pretrain.train.loss import PatchDiscriminationLoss
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

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
    x = MaskedOlmoEarthSample(**masked_sample_dict)

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
    )
    latentmim = LatentMIM(encoder, predictor)

    _, output, _, _, _ = latentmim.forward(x, patch_size)
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
        output_dict = latentmim.target_encoder.forward(
            x.unmask(),
            patch_size=patch_size,
            token_exit_cfg={
                modality: 0 for modality in latentmim.encoder.supported_modality_names
            },
        )
        target_output, _, _ = unpack_encoder_output(output_dict)
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


@pytest.mark.parametrize(
    "encoder_width, output_embedding_size, target_embedding_size, expect_expander",
    [
        # Approach 1: base bottleneck. Encoder emits a small deliverable
        # embedding; target width == encoder width, so it is the raw
        # patch-embedding with no expander.
        (16, 8, 16, False),
        # Approach 2: native small encoder projected up to the loss dim via a
        # frozen expander MLP.
        (16, None, 24, True),
    ],
)
def test_latentmim_smaller_embedding_loss_at_target_dim(
    masked_sample_dict: dict[str, torch.Tensor],
    encoder_width: int,
    output_embedding_size: int | None,
    target_embedding_size: int,
    expect_expander: bool,
) -> None:
    """Encoder emits a compact embedding while the loss/target run at 768-analog."""
    supported_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.LATLON.name,
        Modality.WORLDCOVER.name,
    ]
    x = MaskedOlmoEarthSample(**masked_sample_dict)
    patch_size = 4

    deliverable_size = output_embedding_size or encoder_width
    encoder_config = EncoderConfig(
        supported_modality_names=supported_modalities,
        embedding_size=encoder_width,
        max_patch_size=8,
        min_patch_size=1,
        num_heads=2,
        mlp_ratio=4.0,
        depth=2,
        drop_path=0.1,
        max_sequence_length=12,
        output_embedding_size=output_embedding_size,
    )
    decoder_config = PredictorConfig(
        supported_modality_names=supported_modalities,
        encoder_embedding_size=deliverable_size,
        decoder_embedding_size=encoder_width,
        depth=2,
        mlp_ratio=4.0,
        num_heads=2,
        max_sequence_length=12,
        output_embedding_size=target_embedding_size,
    )
    latentmim = LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        projection_only_target=True,
        target_embedding_size=target_embedding_size,
        target_expander_num_layers=2,
    ).build()

    assert (latentmim.target_encoder.expander is not None) == expect_expander
    # Target must skip the encoder's compressing projector.
    assert latentmim.target_encoder.embedding_projector is None

    latent, decoded, _, _, _ = latentmim.forward(x, patch_size)

    assert latent.sentinel2_l2a is not None
    assert decoded.sentinel2_l2a is not None
    # Encoder emits the compact deliverable embedding...
    assert latent.sentinel2_l2a.shape[-1] == deliverable_size
    # ...while the decoder predicts at the loss dimension.
    assert decoded.sentinel2_l2a.shape[-1] == target_embedding_size

    with torch.no_grad():
        output_dict = latentmim.target_encoder.forward(
            x.unmask(),
            patch_size=patch_size,
            token_exit_cfg={m: 0 for m in latentmim.encoder.supported_modality_names},
        )
        target_output, _, _ = unpack_encoder_output(output_dict)
    # Target operates at the loss dimension too.
    assert target_output.sentinel2_l2a is not None
    assert target_output.sentinel2_l2a.shape[-1] == target_embedding_size

    loss = PatchDiscriminationLoss().compute(decoded, target_output)
    loss.backward()

    # Frozen target (including any expander) never accumulates gradients.
    for name, param in latentmim.target_encoder.named_parameters():
        assert param.grad is None, name
