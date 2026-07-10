"""Integration tests for the dual-resolution (coarse + pixel branch) encoder."""

import logging

import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import (
    MaskedOlmoEarthSample,
    MaskValue,
    OlmoEarthSample,
)
from olmoearth_pretrain.nn.dual_res_encoder import (
    DualResEncoderConfig,
    PixelFiLM,
    PixelTemporalBlock,
    build_location_groupings,
    unit_grid_coords,
)
from olmoearth_pretrain.train.masking import MaskingConfig

logger = logging.getLogger(__name__)

EMB = 32
PIXEL_EMB = 16
PATCH_SIZE = 4
S2 = Modality.SENTINEL2_L2A.name
B, H, W, T = 2, 8, 8, 12  # base-res 8x8 -> coarse grid 2x2 at patch size 4


def _build_config(
    pixel_film_from_coarse: bool = True,
    coarse_cross_attn_to_pixel: bool = True,
) -> DualResEncoderConfig:
    return DualResEncoderConfig(
        supported_modality_names=[S2],
        embedding_size=EMB,
        max_patch_size=8,
        min_patch_size=1,
        num_heads=4,
        mlp_ratio=2.0,
        depth=2,
        drop_path=0.0,
        max_sequence_length=T,
        pixel_embedding_size=PIXEL_EMB,
        pixel_num_heads=4,
        pixel_mlp_ratio=2.0,
        pixel_film_from_coarse=pixel_film_from_coarse,
        coarse_cross_attn_to_pixel=coarse_cross_attn_to_pixel,
    )


def _timestamps() -> torch.Tensor:
    return torch.stack(
        [
            torch.randint(1, 28, (B, T)),
            torch.randint(1, 12, (B, T)),
            torch.full((B, T), 2023),
        ],
        dim=-1,
    )


def _masked_sample() -> MaskedOlmoEarthSample:
    s2_bands = Modality.SENTINEL2_L2A.num_bands
    sample = OlmoEarthSample(
        sentinel2_l2a=torch.randn(B, H, W, T, s2_bands),
        timestamps=_timestamps(),
    )
    strategy = MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "random_ratio": 1.0,
        }
    ).build()
    return strategy.apply_mask(sample, patch_size=PATCH_SIZE)


def test_builds_and_forward() -> None:
    """Model builds; forward returns pixel-enriched coarse tokens with correct shape."""
    model = _build_config().build()
    x = _masked_sample()

    out = model(x, patch_size=PATCH_SIZE)

    tokens = out["tokens_and_masks"]
    assert tokens.sentinel2_l2a is not None
    b, g1, g2, t = tokens.sentinel2_l2a.shape[:4]
    assert (b, g1, g2, t) == (B, H // PATCH_SIZE, W // PATCH_SIZE, T)
    assert tokens.sentinel2_l2a.shape[-1] == EMB
    assert torch.isfinite(tokens.sentinel2_l2a).all()
    assert out["project_aggregated"].shape == (B, EMB)


def test_unit_grid_coords_roundtrip() -> None:
    """unit_grid_coords decodes flat unit indices back to their grid coordinates."""
    grid = (2, 3, 2, 4, 2)  # (B, G1, G2, T, band_sets)
    b, g1, g2, t, bs = grid
    u = g1 * g2 * t * bs
    flat_idx = torch.arange(b * u)
    coords = unit_grid_coords(flat_idx, grid)

    # Rebuild the flat index from the decoded coordinates.
    rebuilt = coords.instance * u + (coords.patch * t + coords.t) * bs + coords.bandset
    assert bool((rebuilt == flat_idx).all())
    assert bool((coords.within == flat_idx % u).all())
    assert int(coords.patch.max()) == g1 * g2 - 1
    assert int(coords.t.max()) == t - 1
    assert int(coords.bandset.max()) == bs - 1


def test_build_location_groupings_by_pixel_location() -> None:
    """Units at the same location id land in the same padded group; no empty slots."""
    # 3 units at location 0, 1 unit at location 5.
    location_ids = torch.tensor([0, 5, 0, 0])
    loc = build_location_groupings(location_ids)
    assert loc.num_locations == 2
    assert loc.max_units == 3
    assert int(loc.valid.sum()) == location_ids.numel()
    # Location 0 has 3 valid slots, location 5 has 1.
    assert sorted(loc.valid.sum(dim=1).tolist()) == [1, 3]


def test_encoder_processes_online_units_only() -> None:
    """The pixel branch packs exactly ONLINE-unit pixels (none for masked patches)."""
    model = _build_config().build()
    x = _masked_sample()
    coarse = model.patch_embeddings.forward(x, PATCH_SIZE)
    pixels = model.pixel_embeddings.forward(x, PATCH_SIZE)
    _, masks, dims = model.split_tokens_masks_and_dims(coarse)

    ctx = model._build_pixel_context(pixels, masks, dims)
    assert ctx is not None
    state = ctx.states[S2]
    online = masks[f"{S2}_mask"] == MaskValue.ONLINE_ENCODER.value
    p2 = PATCH_SIZE**2

    assert state.num_online == int(online.sum())
    # Flat indices point at exactly the ONLINE entries of the row-major mask.
    assert bool((online.reshape(-1)[state.flat_idx]).all())
    assert state.pixels.shape[0] == state.num_online
    assert state.pixels.shape[1] == p2


def test_pixel_temporal_block_runs() -> None:
    """The temporal block maps [N, S, D] -> [N, S, D] and respects the key mask."""
    torch.manual_seed(0)
    block = PixelTemporalBlock(PIXEL_EMB, num_heads=4, mlp_ratio=2.0).eval()
    x = torch.randn(5, T, PIXEL_EMB)
    key_mask = torch.ones(5, T, dtype=torch.bool)
    out = block(x, key_mask)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_film_conditioning_is_pixel_dependent() -> None:
    """FiLM starts as identity and, once trained, affects each pixel differently.

    Cross-attending to a pixel's single coarse token is degenerate (softmax over one
    key broadcasts the same update to every pixel). FiLM's multiplicative term instead
    rescales each pixel's own feature channels by the coarse context, so the shared
    per-unit conditioning must produce DIFFERENT updates for different pixels of the
    same unit.
    """
    torch.manual_seed(0)
    film = PixelFiLM(coarse_dim=EMB, pixel_dim=PIXEL_EMB).eval()

    pixels = torch.randn(1, 4, PIXEL_EMB)  # one unit, 4 different pixels
    coarse = torch.randn(1, EMB)

    # Zero-initialized: the module is the identity (training starts unconditioned).
    film.zero_init()
    with torch.no_grad():
        assert torch.equal(film(pixels, coarse), pixels)

    # With (simulated) trained weights, the shared coarse token modulates each pixel
    # of the unit differently, and the modulation depends on the coarse token.
    torch.manual_seed(1)
    torch.nn.init.normal_(film.to_film.weight, std=0.2)
    torch.nn.init.normal_(film.to_film.bias, std=0.2)
    with torch.no_grad():
        out = film(pixels, coarse)
        update = out - pixels
        other = film(pixels, torch.randn(1, EMB)) - pixels
    assert not torch.allclose(update, torch.zeros_like(update))
    # Different pixels of the SAME unit get different updates (not a broadcast)...
    assert not torch.allclose(update[0, 0], update[0, 1], atol=1e-5)
    # ...and the update depends on the coarse context.
    assert not torch.allclose(update, other, atol=1e-5)


def test_grads_reach_every_pixel_module() -> None:
    """A coarse-only loss trains the pixel embed and every pixel/cross-attn block.

    Every pixel module must feed the coarse output each block (via the coarse<-pixel
    cross-attention) so the whole pixel branch trains even from coarse-only losses.
    """
    model = _build_config().build()
    x = _masked_sample()

    out = model(x, patch_size=PATCH_SIZE)
    out["tokens_and_masks"].sentinel2_l2a.sum().backward()

    assert any(p.grad is not None for p in model.pixel_embeddings.parameters())
    for i, block in enumerate(model.pixel_self_blocks):
        assert any(p.grad is not None for p in block.parameters()), (
            f"pixel_self_blocks[{i}] received no gradient"
        )
    for i, film in enumerate(model.pixel_film):
        assert any(p.grad is not None for p in film.parameters()), (
            f"pixel_film[{i}] received no gradient"
        )
    for i, block in enumerate(model.coarse_to_pixel):
        assert any(p.grad is not None for p in block.parameters()), (
            f"coarse_to_pixel[{i}] received no gradient"
        )


def test_cross_modality_pixel_self_attention() -> None:
    """Two pixel modalities (S2 + S1) are pooled into one per-location self-attention."""
    s1 = Modality.SENTINEL1.name
    cfg = DualResEncoderConfig(
        supported_modality_names=[S2, s1],
        embedding_size=EMB,
        max_patch_size=8,
        min_patch_size=1,
        num_heads=4,
        mlp_ratio=2.0,
        depth=2,
        drop_path=0.0,
        max_sequence_length=T,
        pixel_embedding_size=PIXEL_EMB,
        pixel_num_heads=4,
        pixel_mlp_ratio=2.0,
    )
    model = cfg.build()
    sample = OlmoEarthSample(
        sentinel2_l2a=torch.randn(B, H, W, T, Modality.SENTINEL2_L2A.num_bands),
        sentinel1=torch.randn(B, H, W, T, Modality.SENTINEL1.num_bands),
        timestamps=_timestamps(),
    )
    strategy = MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "random_ratio": 1.0,
        }
    ).build()
    x = strategy.apply_mask(sample, patch_size=PATCH_SIZE)

    out = model(x, patch_size=PATCH_SIZE)
    assert set(out["pixel_branch"]) == {S2, s1}
    (
        out["tokens_and_masks"].sentinel2_l2a.sum()
        + out["tokens_and_masks"].sentinel1.sum()
    ).backward()
    # The single shared pixel self-attention is trained by both modalities.
    assert any(p.grad is not None for p in model.pixel_self_blocks.parameters())


def test_pixels_do_not_affect_output_when_disconnected() -> None:
    """With no coarse<-pixel cross-attn, the pixel branch cannot change the output."""
    model = _build_config(coarse_cross_attn_to_pixel=False).build()
    x = _masked_sample()

    out = model(x, patch_size=PATCH_SIZE)
    out["tokens_and_masks"].sentinel2_l2a.sum().backward()

    # Pixel modules are disconnected from the coarse output -> no gradient.
    assert all(p.grad is None for p in model.pixel_embeddings.parameters())


def test_token_exit_cfg_target_encoder_path() -> None:
    """The target-encoder path (token_exit_cfg on the unmasked sample) runs and is finite."""
    model = _build_config().build()
    x = _masked_sample().unmask()

    out = model(x, patch_size=PATCH_SIZE, token_exit_cfg={S2: 1})
    assert torch.isfinite(out["tokens_and_masks"].sentinel2_l2a).all()


def test_fast_pass_eval() -> None:
    """The fast-pass (no mask) evaluation path produces finite tokens."""
    model = _build_config().build().eval()
    x = _masked_sample()

    out = model(x, patch_size=PATCH_SIZE, fast_pass=True)
    assert torch.isfinite(out["tokens_and_masks"].sentinel2_l2a).all()
    assert "project_aggregated" not in out
