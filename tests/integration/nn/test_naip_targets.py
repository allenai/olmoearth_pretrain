"""Pixel MIM targets with ``naip_10`` (a finer-grid modality) as an encoded input.

* ``gather_pixels`` keeps the ``f x f`` block of a finer-grid modality (naip_10, f = 4)
  under the drawn base pixel;
* the train module's pixel-target forward with naip_10 encoded (the NAIP "N2" arm);
* a v1.3-control-shaped model (single-band-set S2, S1, NISAR, naip_10, a decode-only
  map with a supervision head, location-token student) on a batch masked by the
  control's ``random_time_with_decode`` strategy, with NAIP missing in some samples and
  NISAR missing at some timesteps: finite loss and gradients at patch sizes 1 and 4.
"""

from unittest.mock import patch

import pytest
import torch
from olmo_core.optim.adamw import AdamWConfig

from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality
from olmoearth_pretrain.data.transform import TransformConfig
from olmoearth_pretrain.datatypes import OlmoEarthSample
from olmoearth_pretrain.nn.flexi_vit import (
    AttentionPoolerConfig,
    EncoderConfig,
    PredictorConfig,
)
from olmoearth_pretrain.nn.latent_mim import LatentMIM, LatentMIMConfig
from olmoearth_pretrain.nn.location_token_distillation_head import (
    LocationTokenDistillationHeadConfig,
)
from olmoearth_pretrain.nn.pixel_targets import (
    gather_pixels,
    sample_pixel_offsets,
    spatial_token_grid,
)
from olmoearth_pretrain.nn.supervision_head import (
    SupervisionHeadConfig,
    SupervisionModalityConfig,
    SupervisionTaskType,
)
from olmoearth_pretrain.nn.tokenization import ModalityTokenization, TokenizationConfig
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import (
    MaskedOlmoEarthSample,
    MaskingConfig,
    MaskValue,
)
from olmoearth_pretrain.train.train_module.latent_mim import (
    LatentMIMTrainModule,
    LatentMIMTrainModuleConfig,
)

B, H, W, T = 2, 8, 8, 2
F_NAIP = Modality.NAIP_10.image_tile_size_factor
S1 = Modality.SENTINEL1.name
S2 = Modality.SENTINEL2_L2A.name
NAIP = Modality.NAIP_10.name
NISAR = Modality.NISAR_ASC.name
SRTM = Modality.SRTM.name


def _make_sample() -> MaskedOlmoEarthSample:
    """S2 + naip_10; top-left 4x4 S2 block and the top-right naip_10 quarter decoded.

    Sample 1 has no NAIP (all MISSING), like most training samples.
    """
    torch.manual_seed(1234)
    s2_bands = Modality.SENTINEL2_L2A.num_bands
    s2_mask = torch.zeros(B, H, W, T, s2_bands, dtype=torch.long)
    s2_mask[:, 0:4, 0:4, 0, :] = MaskValue.DECODER.value
    naip = torch.randn(B, H * F_NAIP, W * F_NAIP, 1, Modality.NAIP_10.num_bands)
    naip[1] = MISSING_VALUE
    naip_mask = torch.zeros_like(naip, dtype=torch.long)
    naip_mask[:, 0 : 4 * F_NAIP, 4 * F_NAIP :] = MaskValue.DECODER.value
    naip_mask[1] = MaskValue.MISSING.value
    return MaskedOlmoEarthSample(
        sentinel2_l2a=torch.randn(B, H, W, T, s2_bands),
        sentinel2_l2a_mask=s2_mask,
        naip_10=naip,
        naip_10_mask=naip_mask,
        timestamps=torch.tensor(
            [[[1, 0, 2020], [2, 1, 2020]]], dtype=torch.long
        ).expand(B, -1, -1),
    )


def _model_config(
    encoded: list[str],
    *,
    tokenization_config: TokenizationConfig | None = None,
    supervision: dict[str, SupervisionModalityConfig] | None = None,
    student_dims: list[int] | None = None,
) -> LatentMIMConfig:
    """Small v1.3-shaped model: AttentionPooler + 2D-RoPE decoder, projection target."""
    encoder_config = EncoderConfig(
        supported_modality_names=encoded,
        embedding_size=32,
        num_heads=2,
        depth=2,
        mlp_ratio=2.0,
        max_patch_size=4,
        min_patch_size=1,
        max_sequence_length=12,
        drop_path=0.0,
        position_encoding="rope",
        tokenization_config=tokenization_config,
        attention_pooler_config=AttentionPoolerConfig(
            pooler_dim=16,
            pooler_depth=1,
            per_depth_read_proj=True,
            attn_dim=32,
            student_dims=student_dims,
            student_output_norm=student_dims is not None,
        ),
    )
    decoder_config = PredictorConfig(
        supported_modality_names=encoded,
        encoder_embedding_size=32,
        decoder_embedding_size=16,
        num_heads=2,
        depth=1,
        mlp_ratio=2.0,
        max_sequence_length=12,
        drop_path=0.0,
        position_encoding="rope",
        tokenization_config=tokenization_config,
        use_attention_pooler=True,
        pooler_dim=16,
    )
    return LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        projection_only_target=True,
        supervision_head_config=(
            SupervisionHeadConfig(modality_configs=supervision) if supervision else None
        ),
        location_token_distillation_head_config=(
            LocationTokenDistillationHeadConfig(back_projection_hidden=8)
            if student_dims
            else None
        ),
    )


def _train_module(
    model: LatentMIM, rank_microbatch_size: int = B
) -> LatentMIMTrainModule:
    config = LatentMIMTrainModuleConfig(
        optim_config=AdamWConfig(lr=1e-4),
        rank_microbatch_size=rank_microbatch_size,
        loss_config=LossConfig(
            loss_config={
                "type": "modality_patch_discrimination_masked_negatives_vec",
                "tau": 0.1,
                "same_target_threshold": 0.999,
            }
        ),
        masking_config=MaskingConfig(strategy_config={"type": "random"}),
        token_exit_cfg={modality: 0 for modality in Modality.names()},
        ema_decay=(1.0, 1.0),
        transform_config=TransformConfig(transform_type="no_transform"),
        pixel_targets=True,
    )
    with patch("olmoearth_pretrain.train.train_module.train_module.build_world_mesh"):
        return config.build(model, device=torch.device("cpu"))


@pytest.mark.parametrize("patch_size", [1, 2, 4])
def test_gather_keeps_the_naip_block_under_the_drawn_pixel(patch_size: int) -> None:
    """naip_10 cell (i, j) is the f x f block under S2 pixel (i*p + o_r, j*p + o_c)."""
    sample = _make_sample()
    grid = spatial_token_grid(sample, patch_size)
    assert grid == (H // patch_size, W // patch_size)
    offsets = sample_pixel_offsets(B, grid, patch_size, torch.device("cpu"))
    gathered = gather_pixels(sample, offsets, patch_size)
    assert gathered.naip_10 is not None and gathered.naip_10_mask is not None
    assert gathered.naip_10.shape == (B, grid[0] * F_NAIP, grid[1] * F_NAIP, 1, 4)
    naip, naip_mask = sample.naip_10, sample.naip_10_mask
    s2, s2_gathered = sample.sentinel2_l2a, gathered.sentinel2_l2a
    assert naip is not None and naip_mask is not None
    assert s2 is not None and s2_gathered is not None
    for b in range(B):
        for i in range(grid[0]):
            for j in range(grid[1]):
                r = (i * patch_size + int(offsets[b, i, j, 0])) * F_NAIP
                c = (j * patch_size + int(offsets[b, i, j, 1])) * F_NAIP
                got = gathered.naip_10[
                    b, i * F_NAIP : (i + 1) * F_NAIP, j * F_NAIP : (j + 1) * F_NAIP
                ]
                assert torch.equal(got, naip[b, r : r + F_NAIP, c : c + F_NAIP])
                got_mask = gathered.naip_10_mask[
                    b, i * F_NAIP : (i + 1) * F_NAIP, j * F_NAIP : (j + 1) * F_NAIP
                ]
                assert torch.equal(
                    got_mask, naip_mask[b, r : r + F_NAIP, c : c + F_NAIP]
                )
                # The S2 pixel is still the base-grid gather.
                assert torch.equal(
                    s2_gathered[b, i, j],
                    s2[b, r // F_NAIP, c // F_NAIP],
                )


def test_token_grid_rejects_a_misaligned_fine_grid() -> None:
    """A naip_10 field that is not f x the S2 grid is refused."""
    sample = _make_sample()
    assert sample.naip_10 is not None
    bad = sample._replace(naip_10=sample.naip_10[:, :-4], naip_10_mask=None)
    with pytest.raises(ValueError):
        spatial_token_grid(bad, 2)


@pytest.mark.parametrize("patch_size", [1, 2, 4])
def test_forward_with_naip_encoded(patch_size: int) -> None:
    """naip_10 encoded + pixel targets: finite loss, naip_10 decoded on the S2 grid."""
    torch.manual_seed(0)
    model: LatentMIM = _model_config([S2, NAIP]).build()
    train_module = _train_module(model)
    loss, _latent, decoded, target_output, _metrics = train_module.model_forward(
        _make_sample(),
        patch_size,
        train_module.token_exit_cfg,
        location_query_patch_size=1,
    )
    assert torch.isfinite(loss)
    assert decoded.naip_10 is not None and target_output.naip_10 is not None
    assert decoded.naip_10.shape[:3] == (B, H // patch_size, W // patch_size)
    assert target_output.naip_10.shape[:-1] == decoded.naip_10.shape[:-1]
    loss.backward()
    grads = [
        p.grad
        for name, p in model.encoder.named_parameters()
        if "naip_10" in name and p.grad is not None
    ]
    assert grads and all(torch.isfinite(g).all() for g in grads)


# --- v1.3-control-shaped model with S2 + S1 + NISAR + naip_10 ---------------------------

B_CTRL, H_CTRL, T_CTRL = 4, 8, 3
CONTROL_TOKENIZATION = TokenizationConfig(
    overrides={
        S2: ModalityTokenization(band_groups=[Modality.SENTINEL2_L2A.band_order]),
    }
)


def _control_batch() -> OlmoEarthSample:
    """S2 + S1 + nisar_asc + naip_10 + srtm, unmasked, already normalized.

    naip_10 is missing entirely in samples 1 and 3 (and is all-missing in sample 3's
    S1 too); nisar_asc is missing at t=0 everywhere and entirely in sample 2.
    """
    g = torch.Generator().manual_seed(7)
    n, h, t = B_CTRL, H_CTRL, T_CTRL

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=g)

    naip = rand(n, h * F_NAIP, h * F_NAIP, 1, Modality.NAIP_10.num_bands)
    naip[1] = MISSING_VALUE
    naip[3] = MISSING_VALUE
    nisar = rand(n, h, h, t, Modality.NISAR_ASC.num_bands)
    nisar[:, :, :, 0] = MISSING_VALUE
    nisar[2] = MISSING_VALUE
    s1 = rand(n, h, h, t, Modality.SENTINEL1.num_bands)
    s1[3] = MISSING_VALUE
    return OlmoEarthSample(
        sentinel2_l2a=rand(n, h, h, t, Modality.SENTINEL2_L2A.num_bands),
        sentinel1=s1,
        nisar_asc=nisar,
        naip_10=naip,
        srtm=rand(n, h, h, 1, Modality.SRTM.num_bands),
        timestamps=torch.tensor(
            [[[1, 0, 2025], [15, 4, 2025], [1, 9, 2025]]], dtype=torch.long
        ).expand(n, -1, -1),
    )


@pytest.mark.parametrize("patch_size", [1, 4])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_control_shaped_forward_with_naip_and_nisar(patch_size: int, seed: int) -> None:
    """random_time_with_decode masking + pixel targets with naip_10 partly missing."""
    torch.manual_seed(seed)
    encoded = [S2, S1, NISAR, NAIP, SRTM]
    masking = MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "random_ratio": 0.5,
            "only_decode_modalities": [SRTM],
        },
        tokenization_config=CONTROL_TOKENIZATION,
    ).build()
    masked = masking.apply_mask(_control_batch(), patch_size)
    assert masked.naip_10_mask is not None
    for b in (1, 3):
        assert (masked.naip_10_mask[b] == MaskValue.MISSING.value).all()
    for b in (0, 2):
        assert not (masked.naip_10_mask[b] == MaskValue.MISSING.value).any()

    model: LatentMIM = _model_config(
        encoded,
        tokenization_config=CONTROL_TOKENIZATION,
        supervision={
            SRTM: SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=Modality.SRTM.num_bands,
                regression_loss_type="l1",
            )
        },
        student_dims=[16, 8],
    ).build()
    train_module = _train_module(model, rank_microbatch_size=B_CTRL)
    for location_query_patch_size in sorted({1, patch_size}):
        model.zero_grad()
        loss, latent, decoded, target_output, _metrics = train_module.model_forward(
            masked,
            patch_size,
            train_module.token_exit_cfg,
            location_query_patch_size=location_query_patch_size,
        )
        assert torch.isfinite(loss), (patch_size, location_query_patch_size)
        for name in (S2, S1, NISAR, NAIP):
            for out in (decoded, target_output):
                tokens = getattr(out, name)
                assert tokens is not None
                assert torch.isfinite(tokens).all(), name
            assert getattr(decoded, name).shape[1:3] == (
                H_CTRL // patch_size,
                H_CTRL // patch_size,
            )
        loss.backward()
        grads = [
            (name, p.grad) for name, p in model.named_parameters() if p.grad is not None
        ]
        assert grads
        for name, grad in grads:
            assert torch.isfinite(grad).all(), name
