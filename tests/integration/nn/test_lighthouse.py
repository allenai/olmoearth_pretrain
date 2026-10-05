"""Lighthouse inference (``nn/lighthouse.py``) against the stock encoder forward."""

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
from olmoearth_pretrain.nn.flexi_vit import Encoder, EncoderConfig, PerceiverConfig
from olmoearth_pretrain.nn.lighthouse import (
    LighthouseSettings,
    _flex_mask_mod,
    _flex_na,
    _flex_tables,
    _reference_na,
    embed_domain,
    lighthouse_reach_px,
)

T = 3
MODALITIES = [Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name]


def _sample(size: int, missing_t: int | None = None) -> MaskedOlmoEarthSample:
    """One ``size`` px domain; ``missing_t`` marks a whole timestep MISSING."""
    torch.manual_seed(1234)
    fields: dict[str, torch.Tensor] = {}
    for name in MODALITIES:
        bands = Modality.get(name).num_bands
        mask = torch.full(
            (1, size, size, T, bands), MaskValue.ONLINE_ENCODER.value, dtype=torch.long
        )
        if missing_t is not None:
            mask[:, :, :, missing_t] = MaskValue.MISSING.value
        fields[name] = torch.randn(1, size, size, T, bands)
        fields[f"{name}_mask"] = mask
    fields["timestamps"] = torch.tensor([[[1, t, 2020] for t in range(T)]])
    return MaskedOlmoEarthSample(**fields)


def _encoder() -> Encoder:
    """A small v1.3-shaped encoder: 3D mixed RoPE ViT + per-depth-read Perceiver."""
    torch.manual_seed(0)
    return (
        EncoderConfig(
            supported_modality_names=MODALITIES,
            embedding_size=32,
            num_heads=2,
            depth=2,
            mlp_ratio=2.0,
            max_patch_size=8,
            min_patch_size=1,
            max_sequence_length=12,
            drop_path=0.0,
            position_encoding="rope_3d_mixed",
            perceiver_config=PerceiverConfig(
                register_dim=16,
                latent_depth=2,
                attn_dim=32,
                per_depth_read_proj=True,
                student_dims=[8],
                student_output_norm=True,
            ),
        )
        .build()
        .eval()
    )


@pytest.mark.parametrize("missing_t", [None, 1])
@pytest.mark.parametrize(
    ("patch_size", "latent_patch_size"), [(2, 1), (2, None), (4, 2), (1, None)]
)
def test_one_window_domain_matches_the_stock_forward(
    patch_size: int, latent_patch_size: int | None, missing_t: int | None
) -> None:
    """A domain exactly one FOV wide: every query's box is the whole window."""
    encoder = _encoder()
    sample = _sample(16, missing_t)
    kwargs = dict(
        patch_size=patch_size, input_res=10, latent_patch_size=latent_patch_size
    )
    with torch.no_grad():
        stock = encoder(sample, **kwargs)
        encoder.lighthouse = LighthouseSettings(fov_px=16)
        lighthouse = encoder(sample, **kwargs)
    for key in ("registers", "student_registers", "register_positions"):
        torch.testing.assert_close(lighthouse[key], stock[key], atol=1e-5, rtol=1e-5)
    for name in MODALITIES:
        torch.testing.assert_close(
            getattr(lighthouse["tokens_and_masks"], name),
            getattr(stock["tokens_and_masks"], name),
            atol=1e-5,
            rtol=1e-5,
        )


def test_reference_attention_is_the_sliding_box() -> None:
    """A query sees exactly the cells of its box, shifted inward at the edges."""
    h = w = 6
    fov = 4
    torch.manual_seed(0)
    q = torch.randn(h, w, 2, 1, 4)
    k = torch.randn(h, w, 3, 1, 4)
    # Values one-hot on the key's cell: the output's support is the attended cells.
    v = torch.eye(h * w).view(h, w, 1, 1, h * w).expand(h, w, 3, 1, h * w)
    seen = (_reference_na(q, k, v, fov)[..., 0, 0, :] > 0).view(h, w, h, w)
    starts = [0, 0, 0, 1, 2, 2]  # clip(r - 2, 0, 2)
    for r in range(h):
        for c in range(w):
            box = torch.zeros(h, w, dtype=torch.bool)
            box[starts[r] : starts[r] + fov, starts[c] : starts[c] + fov] = True
            assert torch.equal(seen[r, c], box)


@pytest.mark.parametrize(("kq", "kk"), [(4, 12), (12, 12), (1, 3)])
def test_flex_matches_the_reference(kq: int, kk: int) -> None:
    """FlexAttention's layout and mask (CPU eager applies ``mask_mod`` densely)."""
    torch.manual_seed(0)
    h, w, fov = 7, 9, 4
    q = torch.randn(h, w, kq, 2, 8)
    k, v = torch.randn(2, h, w, kk, 2, 8)
    out = _flex_na(q, k, v, fov, block=16, chunk=3)
    torch.testing.assert_close(out, _reference_na(q, k, v, fov))


@pytest.mark.parametrize(("kq", "kk"), [(4, 12), (12, 12), (1, 3)])
def test_flex_block_tables_cover_every_box(kq: int, kk: int) -> None:
    """Every (query, key) pair the box rule allows lies in a listed key block."""
    h, w, fov, block = 7, 9, 4, 16
    num, idx = _flex_tables(h, w, kq, kk, fov, block, torch.device("cpu"))
    lq = -(-w * kq // block) * block
    lk = -(-w * kk // block) * block
    listed = torch.zeros(num.numel(), h * lk // block, dtype=torch.bool)
    for b in range(num.numel()):
        blocks = idx[b, : num[b]]
        assert blocks.unique().numel() == blocks.numel()  # no block twice
        listed[b, blocks] = True
    listed = listed.repeat_interleave(block, 0).repeat_interleave(block, 1)
    qi = torch.arange(h * lq)[:, None]
    ki = torch.arange(h * lk)[None, :]
    allowed = _flex_mask_mod(h, w, kq, kk, fov, block, 0, "cpu")(0, 0, qi, ki)
    real_q = (qi % lq < w * kq).expand_as(allowed)
    assert not (allowed & real_q & ~listed).any()


def test_embed_domain_chunks_match_one_pass() -> None:
    """Cores + the exact halo reproduce the single-pass domain forward."""
    encoder = _encoder()
    sample = _sample(64)
    halo = lighthouse_reach_px(16, 4, vit_depth=2, perceiver_depth=2)
    one_pass = embed_domain(encoder, sample, 4, 2, core_px=64, halo_px=0)
    chunked = embed_domain(encoder, sample, 4, 2, core_px=16, halo_px=halo)
    assert one_pass.shape == (32, 32, 8)
    torch.testing.assert_close(chunked, one_pass, atol=1e-4, rtol=1e-4)


def test_per_pixel_missing_data_is_refused() -> None:
    """NATTEN needs the same tokens in every cell."""
    encoder = _encoder()
    sample = _sample(16)
    assert sample.sentinel1_mask is not None
    sample.sentinel1_mask[:, :4, :4, 0] = MaskValue.MISSING.value
    encoder.lighthouse = LighthouseSettings(fov_px=16)
    with pytest.raises(NotImplementedError, match="same number of tokens"):
        encoder(sample, patch_size=2, input_res=10)
