"""Integration tests for the dual-encoder (main + NAIP) latent MIM model."""

import logging
from typing import Any

import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import (
    MaskedOlmoEarthSample,
    MaskValue,
    OlmoEarthSample,
)
from olmoearth_pretrain.nn.dual_encoder_latent_mim import DualEncoderLatentMIMConfig
from olmoearth_pretrain.nn.flexi_vit import (
    EncoderConfig,
    PredictorConfig,
    ReconstructorConfig,
)
from olmoearth_pretrain.train.loss import MAELoss
from olmoearth_pretrain.train.masking import MaskingConfig

logger = logging.getLogger(__name__)

EMB = 32
NAIP_EMB = 16
PATCH_SIZE = 4
MAIN_MODALITIES = [Modality.SENTINEL2_L2A.name]
NAIP = Modality.NAIP_10.name


def _build_config(detach: bool = False) -> DualEncoderLatentMIMConfig:
    common_kwargs: dict[str, Any] = dict(
        max_patch_size=8,
        max_sequence_length=12,
        num_heads=4,
        mlp_ratio=2.0,
        depth=2,
    )
    encoder_config = EncoderConfig(
        embedding_size=EMB,
        supported_modality_names=MAIN_MODALITIES,
        min_patch_size=1,
        drop_path=0.0,
        **common_kwargs,
    )
    naip_encoder_config = EncoderConfig(
        embedding_size=NAIP_EMB,
        output_embedding_size=EMB,
        supported_modality_names=[NAIP],
        min_patch_size=1,
        drop_path=0.0,
        **common_kwargs,
    )
    decoder_config = PredictorConfig(
        encoder_embedding_size=EMB,
        decoder_embedding_size=EMB,
        supported_modality_names=MAIN_MODALITIES,
        num_heads=4,
        mlp_ratio=2.0,
        depth=2,
        max_sequence_length=12,
    )
    reconstructor_predictor_config = PredictorConfig(
        encoder_embedding_size=EMB,
        decoder_embedding_size=EMB,
        supported_modality_names=MAIN_MODALITIES + [NAIP],
        num_heads=4,
        mlp_ratio=2.0,
        depth=2,
        max_sequence_length=12,
    )
    reconstructor_config = ReconstructorConfig(
        decoder_config=reconstructor_predictor_config,
        supported_modality_names=[NAIP],
        max_patch_size=8,
    )
    return DualEncoderLatentMIMConfig(
        encoder_config=encoder_config,
        naip_encoder_config=naip_encoder_config,
        decoder_config=decoder_config,
        reconstructor_config=reconstructor_config,
        naip_modality_name=NAIP,
        detach_main_context_for_reconstruction=detach,
    )


def _build_masked_sample() -> MaskedOlmoEarthSample:
    B, H, W, T = 2, 8, 8, 12
    s2_bands = Modality.SENTINEL2_L2A.num_bands
    naip_bands = Modality.NAIP_10.num_bands
    naip_hw = H * Modality.NAIP_10.image_tile_size_factor  # 8 * 4 = 32
    timestamps = torch.stack(
        [
            torch.randint(1, 28, (B, T)),
            torch.randint(1, 12, (B, T)),
            torch.full((B, T), 2023),
        ],
        dim=-1,
    )
    sample = OlmoEarthSample(
        sentinel2_l2a=torch.randn(B, H, W, T, s2_bands),
        naip_10=torch.randn(B, naip_hw, naip_hw, 1, naip_bands),
        timestamps=timestamps,
    )
    strategy = MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode_separate_encoder",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "random_ratio": 1.0,
            "separate_encoder_modalities": [NAIP],
            "separate_encode_ratio": 0.25,
            "separate_decode_ratio": 0.75,
        }
    ).build()
    return strategy.apply_mask(sample, patch_size=PATCH_SIZE)


def test_dual_encoder_builds_and_forward() -> None:
    """Model builds; reconstructs only NAIP; latent-MIM decoder has no NAIP."""
    model = _build_config().build()
    x = _build_masked_sample()

    latent, decoded, pooled, reconstructed, _ = model(x, patch_size=PATCH_SIZE)

    # Latent-MIM decoder path: main modality only, no NAIP context.
    assert decoded.sentinel2_l2a is not None
    assert decoded.naip_10 is None
    # Reconstruction path: only NAIP is reconstructed.
    assert reconstructed is not None
    assert reconstructed.naip_10 is not None
    assert reconstructed.sentinel2_l2a is None
    # Reconstructed NAIP matches input NAIP shape.
    assert x.naip_10 is not None
    assert reconstructed.naip_10.shape == x.naip_10.shape
    assert pooled is not None


def test_naip_mask_has_encode_and_decode() -> None:
    """The NAIP mask has both encoder-visible and to-be-decoded tokens (75% masked)."""
    x = _build_masked_sample()
    naip_mask = x.naip_10_mask
    assert naip_mask is not None
    assert bool((naip_mask == MaskValue.ONLINE_ENCODER.value).any())
    assert bool((naip_mask == MaskValue.DECODER.value).any())


def test_naip_encoder_params_disjoint_from_main() -> None:
    """The NAIP encoder and main encoder share no parameters."""
    model = _build_config().build()
    main_ids = {id(p) for p in model.encoder.parameters()}
    naip_ids = {id(p) for p in model.naip_encoder.parameters()}
    assert main_ids.isdisjoint(naip_ids)


def test_reconstruction_grads_reach_naip_encoder() -> None:
    """NAIP reconstruction loss trains the NAIP encoder and reconstructor."""
    model = _build_config(detach=False).build()
    x = _build_masked_sample()
    _, _, _, reconstructed, _ = model(x, patch_size=PATCH_SIZE)
    loss = MAELoss(loss_function="MSELoss").compute(reconstructed, x)
    loss.backward()
    assert model.reconstructor is not None
    assert any(p.grad is not None for p in model.naip_encoder.parameters())
    assert any(p.grad is not None for p in model.reconstructor.parameters())


def test_detach_isolates_main_encoder_from_reconstruction() -> None:
    """With detach=True, NAIP reconstruction loss does not touch the main encoder."""
    model = _build_config(detach=True).build()
    x = _build_masked_sample()
    _, _, _, reconstructed, _ = model(x, patch_size=PATCH_SIZE)
    loss = MAELoss(loss_function="MSELoss").compute(reconstructed, x)
    loss.backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert any(p.grad is not None for p in model.naip_encoder.parameters())


def test_mae_loss_finite_when_naip_all_missing() -> None:
    """MAE loss is finite (0) when NAIP is entirely missing (no decode tokens).

    NAIP is US-only, so on a global dataset whole batches can have NAIP missing. That
    yields zero decode tokens; the loss must not divide by zero and produce NaN (which
    would abort the step and desync distributed collectives).
    """
    from olmoearth_pretrain.data.constants import MISSING_VALUE

    model = _build_config().build()
    B, H, W, T = 2, 8, 8, 12
    s2_bands = Modality.SENTINEL2_L2A.num_bands
    naip_bands = Modality.NAIP_10.num_bands
    naip_hw = H * Modality.NAIP_10.image_tile_size_factor
    timestamps = torch.stack(
        [
            torch.randint(1, 28, (B, T)),
            torch.randint(1, 12, (B, T)),
            torch.full((B, T), 2023),
        ],
        dim=-1,
    )
    # NAIP entirely missing -> masking marks every NAIP token MISSING (0 decode).
    sample = OlmoEarthSample(
        sentinel2_l2a=torch.randn(B, H, W, T, s2_bands),
        naip_10=torch.full((B, naip_hw, naip_hw, 1, naip_bands), float(MISSING_VALUE)),
        timestamps=timestamps,
    )
    strategy = MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode_separate_encoder",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "random_ratio": 1.0,
            "separate_encoder_modalities": [NAIP],
            "separate_encode_ratio": 0.25,
            "separate_decode_ratio": 0.75,
        }
    ).build()
    x = strategy.apply_mask(sample, patch_size=PATCH_SIZE)

    naip_mask = x.naip_10_mask
    assert naip_mask is not None
    assert not bool((naip_mask == MaskValue.DECODER.value).any())  # no decode tokens

    _, _, _, reconstructed, _ = model(x, patch_size=PATCH_SIZE)
    loss = MAELoss(loss_function="MSELoss").compute(reconstructed, x)
    assert torch.isfinite(loss)
    assert float(loss) == 0.0


def test_forward_runs_when_naip_modality_absent() -> None:
    """When NAIP is absent from the batch, the model still runs the NAIP path.

    Under FSDP every rank must execute the same modules each step. NAIP is US-only, so
    some ranks' batches lack it entirely; the model must inject a synthetic NAIP so the
    NAIP encoder + reconstructor still run (avoiding a collective desync).
    """
    model = _build_config().build()
    B, H, W, T = 2, 8, 8, 12
    s2_bands = Modality.SENTINEL2_L2A.num_bands
    timestamps = torch.stack(
        [
            torch.randint(1, 28, (B, T)),
            torch.randint(1, 12, (B, T)),
            torch.full((B, T), 2023),
        ],
        dim=-1,
    )
    # No NAIP modality at all in the sample.
    sample = OlmoEarthSample(
        sentinel2_l2a=torch.randn(B, H, W, T, s2_bands),
        timestamps=timestamps,
    )
    strategy = MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode_separate_encoder",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "random_ratio": 1.0,
            "separate_encoder_modalities": [NAIP],
            "separate_encode_ratio": 0.25,
            "separate_decode_ratio": 0.75,
        }
    ).build()
    x = strategy.apply_mask(sample, patch_size=PATCH_SIZE)
    assert NAIP not in x.modalities  # NAIP genuinely absent from the batch

    # The NAIP path still runs (synthetic NAIP injected) and reconstructs.
    _, _, _, reconstructed, _ = model(x, patch_size=PATCH_SIZE)
    assert reconstructed is not None
    assert reconstructed.naip_10 is not None
    loss = MAELoss(loss_function="MSELoss").compute(
        reconstructed, model._ensure_naip_present(x)
    )
    assert torch.isfinite(loss)


def test_injected_naip_yields_encode_and_decode_tokens() -> None:
    """The synthetic NAIP injected for an absent batch has both encode + decode tokens.

    A purely MISSING injection would leave the NAIP encoder + reconstructor with zero
    surviving tokens (their block outputs get discarded), so their FSDP-sharded blocks
    would get no gradient and skip their backward all-gathers -- desyncing from ranks
    that do have NAIP. Injecting real encode (ONLINE_ENCODER) + decode (DECODER) tokens
    keeps every NAIP block connected.
    """
    model = _build_config().build()
    B, H, W, T = 2, 8, 8, 12
    s2_bands = Modality.SENTINEL2_L2A.num_bands
    timestamps = torch.stack(
        [
            torch.randint(1, 28, (B, T)),
            torch.randint(1, 12, (B, T)),
            torch.full((B, T), 2023),
        ],
        dim=-1,
    )
    sample = OlmoEarthSample(
        sentinel2_l2a=torch.randn(B, H, W, T, s2_bands),
        timestamps=timestamps,
    )
    strategy = MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode_separate_encoder",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "random_ratio": 1.0,
            "separate_encoder_modalities": [NAIP],
            "separate_encode_ratio": 0.25,
            "separate_decode_ratio": 0.75,
        }
    ).build()
    x = strategy.apply_mask(sample, patch_size=PATCH_SIZE)
    assert NAIP not in x.modalities

    injected = model._ensure_naip_present(x)
    naip_mask = injected.naip_10_mask
    assert naip_mask is not None
    # Injected data must be finite (benign zeros), never the MISSING_VALUE sentinel.
    assert injected.naip_10 is not None
    assert torch.isfinite(injected.naip_10).all()
    # Both encoder-visible and to-be-decoded tokens are present.
    assert bool((naip_mask == MaskValue.ONLINE_ENCODER.value).any())
    assert bool((naip_mask == MaskValue.DECODER.value).any())


def test_absent_naip_reconstruction_grads_reach_naip_modules() -> None:
    """When NAIP is absent, a keep-alive on the reconstruction still trains NAIP modules.

    This is the crux of the FSDP-desync fix: even on a rank whose batch has no NAIP, a
    zero-magnitude term over the reconstructed NAIP must propagate gradient to every
    NAIP encoder and reconstructor block, so their backward all-gathers fire and stay in
    sync with ranks that have real NAIP.
    """
    model = _build_config(detach=False).build()
    B, H, W, T = 2, 8, 8, 12
    s2_bands = Modality.SENTINEL2_L2A.num_bands
    timestamps = torch.stack(
        [
            torch.randint(1, 28, (B, T)),
            torch.randint(1, 12, (B, T)),
            torch.full((B, T), 2023),
        ],
        dim=-1,
    )
    sample = OlmoEarthSample(
        sentinel2_l2a=torch.randn(B, H, W, T, s2_bands),
        timestamps=timestamps,
    )
    strategy = MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode_separate_encoder",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "random_ratio": 1.0,
            "separate_encoder_modalities": [NAIP],
            "separate_encode_ratio": 0.25,
            "separate_decode_ratio": 0.75,
        }
    ).build()
    x = strategy.apply_mask(sample, patch_size=PATCH_SIZE)
    assert NAIP not in x.modalities

    _, _, _, reconstructed, _ = model(x, patch_size=PATCH_SIZE)
    assert reconstructed is not None and reconstructed.naip_10 is not None
    # Mimic the train module's zero keep-alive when the batch has no real NAIP target.
    keep_alive = 0.0 * reconstructed.naip_10.sum()
    keep_alive.backward()

    assert model.reconstructor is not None
    assert any(p.grad is not None for p in model.naip_encoder.parameters())
    assert any(p.grad is not None for p in model.reconstructor.parameters())
