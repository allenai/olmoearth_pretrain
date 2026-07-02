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
from olmoearth_pretrain.nn.supervision_head import (
    SupervisionHeadConfig,
    SupervisionModalityConfig,
    SupervisionTaskType,
    compute_supervision_loss,
)
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


def test_latentmim_register_bottleneck(
    modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
    masked_sample_dict: dict[str, torch.Tensor],
) -> None:
    """End-to-end LatentMIM where the decoder reads ONLY the register bottleneck."""
    supported_modalities = [
        Modality.SENTINEL2_L2A,
        Modality.LATLON,
        Modality.WORLDCOVER,
    ]
    sentinel2_l2a_num_band_sets, _ = modality_band_set_len_and_total_bands[
        "sentinel2_l2a"
    ]
    B, H, W, T, _ = masked_sample_dict["sentinel2_l2a"].shape
    x = MaskedOlmoEarthSample(**masked_sample_dict)

    patch_size = 4
    grid_size, register_dim = 3, 8
    encoder_config = EncoderConfig(
        supported_modality_names=[m.name for m in supported_modalities],
        embedding_size=16,
        num_heads=2,
        depth=2,
        mlp_ratio=4.0,
        max_patch_size=8,
        min_patch_size=1,
        max_sequence_length=12,
        drop_path=0.1,
        spatial_pos_encoding="rope",
        use_register_bottleneck=True,
        register_grid_size=grid_size,
        register_dim=register_dim,
        register_read_depth=1,
        register_latent_depth=2,
    )
    decoder_config = PredictorConfig(
        supported_modality_names=[m.name for m in supported_modalities],
        encoder_embedding_size=16,
        decoder_embedding_size=16,
        num_heads=2,
        depth=2,
        mlp_ratio=4.0,
        max_sequence_length=12,
        drop_path=0.1,
        spatial_pos_encoding="rope",
        use_register_bottleneck=True,
        register_dim=register_dim,
    )
    # Low-weight supervision on the registers: a spatial-salience nudge, not a learning
    # signal. Regression (not classification) so the random worldcover target is valid.
    supervision_config = SupervisionHeadConfig(
        modality_configs={
            "worldcover": SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=1,
                weight=0.02,
                regression_loss_type="l1",
            )
        },
        register_supervision=True,
    )
    model = LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        supervision_head_config=supervision_config,
    ).build()

    _, decoded, _, _, _, supervision_preds = model.forward(x, patch_size)
    assert supervision_preds is not None
    assert model.supervision_head is not None
    patched_H, patched_W = H // patch_size, W // patch_size
    assert decoded.sentinel2_l2a is not None
    assert decoded.sentinel2_l2a.shape == (
        B,
        patched_H,
        patched_W,
        T,
        sentinel2_l2a_num_band_sets,
        model.decoder.output_embedding_size,
    )
    # Register supervision predicts worldcover at the target pixel resolution.
    worldcover_target = x.worldcover
    assert worldcover_target is not None
    assert supervision_preds["worldcover"].shape[1] == worldcover_target.shape[1]
    assert supervision_preds["worldcover"].shape[2] == worldcover_target.shape[2]

    loss_fn = PatchDiscriminationLoss()
    with torch.no_grad():
        output_dict = model.target_encoder.forward(
            x.unmask(),
            patch_size=patch_size,
            token_exit_cfg={
                modality: 0 for modality in model.encoder.supported_modality_names
            },
        )
        target_output, _, _ = unpack_encoder_output(output_dict)
    contrastive_loss = loss_fn.compute(decoded, target_output)
    supervision_loss, _ = compute_supervision_loss(
        supervision_preds, x, model.supervision_head
    )
    (contrastive_loss + supervision_loss).backward()

    # Gradients reach the register bottleneck (read + latent transformer), the decoder's
    # register->decoder projection (the only context it attends to), and the (register-fed)
    # supervision head.
    assert model.encoder.register_bottleneck.registers.grad is not None
    assert model.encoder.register_bottleneck.kv_proj.weight.grad is not None
    assert (
        model.encoder.register_bottleneck.read_blocks[0].attn.q.weight.grad is not None
    )
    assert model.decoder.register_to_decoder_embed.weight.grad is not None
    assert model.supervision_head.heads["worldcover"].weight.grad is not None


def test_latentmim_register_bottleneck_3d_encoder_2d_decoder(
    modality_band_set_len_and_total_bands: dict[str, tuple[int, int]],
    masked_sample_dict: dict[str, torch.Tensor],
) -> None:
    """3D-RoPE encoder + spatial (2D) register bottleneck + 2D decoder, end-to-end.

    The patch encoder self-attention keeps 3D RoPE over ``(t, row, col)``; the register
    grid is a spatial summary read with 2D RoPE; and the decoder cross-attends the mask
    tokens to that 2D register grid (so it runs in 2D and recovers the timestep from the
    additive slot-index + month encodings). Verifies gradients flow through the whole path.
    """
    supported_modalities = [
        Modality.SENTINEL2_L2A,
        Modality.LATLON,
        Modality.WORLDCOVER,
    ]
    B, H, W, T, _ = masked_sample_dict["sentinel2_l2a"].shape
    x = MaskedOlmoEarthSample(**masked_sample_dict)

    patch_size = 4
    register_dim = 8
    encoder_config = EncoderConfig(
        supported_modality_names=[m.name for m in supported_modalities],
        embedding_size=16,
        num_heads=2,
        depth=2,
        mlp_ratio=4.0,
        max_patch_size=8,
        min_patch_size=1,
        max_sequence_length=12,
        drop_path=0.1,
        position_encoding="rope_3d_mixed",  # 3D encoder self-attention
        use_register_bottleneck=True,
        register_grid_size=0,  # dynamic single-latent grid (gdyn)
        register_dim=register_dim,
        register_interleave=True,
        register_per_depth_read_proj=True,
    )
    decoder_config = PredictorConfig(
        supported_modality_names=[m.name for m in supported_modalities],
        encoder_embedding_size=16,
        decoder_embedding_size=16,
        num_heads=2,
        depth=2,
        mlp_ratio=4.0,
        max_sequence_length=12,
        drop_path=0.1,
        position_encoding="rope",  # 2D decoder: cross-attends the spatial register grid
        use_register_bottleneck=True,
        register_dim=register_dim,
    )
    model = LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
    ).build()

    _, decoded, _, _, _, _ = model.forward(x, patch_size)
    assert decoded.sentinel2_l2a is not None
    patched_H, patched_W = H // patch_size, W // patch_size
    assert decoded.sentinel2_l2a.shape == (
        B,
        patched_H,
        patched_W,
        T,
        modality_band_set_len_and_total_bands["sentinel2_l2a"][0],
        model.decoder.output_embedding_size,
    )

    loss_fn = PatchDiscriminationLoss()
    with torch.no_grad():
        output_dict = model.target_encoder.forward(
            x.unmask(),
            patch_size=patch_size,
            token_exit_cfg={
                modality: 0 for modality in model.encoder.supported_modality_names
            },
        )
        target_output, _, _ = unpack_encoder_output(output_dict)
    loss_fn.compute(decoded, target_output).backward()

    # Gradients reach the (2D) register read and the decoder's register projection, and the
    # 3D encoder self-attention still trains.
    assert model.encoder.register_bottleneck.register.grad is not None
    assert (
        model.encoder.register_bottleneck.read_blocks[0].attn.q.weight.grad is not None
    )
    assert model.decoder.register_to_decoder_embed.weight.grad is not None
    assert model.encoder.blocks[0].attn.q.weight.grad is not None


def test_eval_wrapper_probes_register_grid(
    masked_sample_dict: dict[str, torch.Tensor],
) -> None:
    """The frozen eval wrapper probes the register grid for bottleneck models."""
    from olmoearth_pretrain.evals.datasets.configs import TaskType
    from olmoearth_pretrain.evals.eval_wrapper import OlmoEarthEvalWrapper
    from olmoearth_pretrain.nn.pooling import PoolingType

    supported_modalities = [
        Modality.SENTINEL2_L2A,
        Modality.LATLON,
        Modality.WORLDCOVER,
    ]
    x = MaskedOlmoEarthSample(**masked_sample_dict).unmask()
    B = masked_sample_dict["sentinel2_l2a"].shape[0]
    grid_size, register_dim = 3, 8
    encoder = Encoder(
        supported_modalities=supported_modalities,
        embedding_size=16,
        max_patch_size=8,
        min_patch_size=1,
        num_heads=2,
        mlp_ratio=4.0,
        max_sequence_length=12,
        depth=2,
        drop_path=0.0,
        spatial_pos_encoding="rope",
        use_register_bottleneck=True,
        register_grid_size=grid_size,
        register_dim=register_dim,
        register_read_depth=1,
        register_latent_depth=2,
    )
    encoder.eval()
    labels = torch.zeros(B)

    # Classification: registers pooled across the grid -> [B, register_dim].
    clf_wrapper = OlmoEarthEvalWrapper(
        encoder, TaskType.CLASSIFICATION, patch_size=4, pooling_type=PoolingType.MEAN
    )
    clf_emb, _ = clf_wrapper(x, labels)
    assert clf_emb.shape == (B, register_dim)

    # Segmentation: registers kept as a coarse [B, n_h, n_w, register_dim] spatial map.
    seg_wrapper = OlmoEarthEvalWrapper(
        encoder, TaskType.SEGMENTATION, patch_size=4, pooling_type=PoolingType.MEAN
    )
    seg_emb, _ = seg_wrapper(x, labels)
    assert seg_emb.shape == (B, grid_size, grid_size, register_dim)


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

    _, output, _, _, _, _ = latentmim.forward(x, patch_size)
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
