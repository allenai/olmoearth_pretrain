"""Integration tests for the dual-resolution (coarse + pixel branch) encoder."""

import logging

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import (
    MaskedOlmoEarthSample,
    MaskValue,
    OlmoEarthSample,
)
from olmoearth_pretrain.nn.dual_res_encoder import (
    DualResEncoderConfig,
    PerceiverPixelStep,
    PixelAttentionBlock,
    PixelFiLM,
    WindowPixelStep,
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


def test_pixel_attention_block_runs() -> None:
    """The attention block maps [N, S, D] -> [N, S, D] and respects the key mask."""
    torch.manual_seed(0)
    block = PixelAttentionBlock(PIXEL_EMB, num_heads=4, mlp_ratio=2.0).eval()
    x = torch.randn(5, T, PIXEL_EMB)
    key_mask = torch.ones(5, T, dtype=torch.bool)
    out = block(x, key_mask)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_joint_patch_attention_mixes_space_and_time_within_patch_only() -> None:
    """Pixels mix across offsets AND units of their patch, never across patches.

    The joint per-patch attention exists so the pixel branch is non-degenerate with a
    single modality and timestep: spatial mixing within the patch must happen (it was
    structurally blocked when offsets were a pure batch dimension), while patch
    isolation is preserved.
    """
    torch.manual_seed(0)
    model = _build_config().build().eval()
    # Units 0 and 1 share location 0 (e.g. two timesteps); unit 2 is another patch.
    loc = build_location_groupings(torch.tensor([0, 0, 1]))
    pixels = torch.randn(3, PATCH_SIZE**2, PIXEL_EMB)
    perturbed = pixels.clone()
    # Perturb a single channel of one pixel of unit 0. (A constant added to ALL
    # channels would be removed by the pre-attention LayerNorm and prove nothing.)
    perturbed[0, 0, 0] += 10.0

    with torch.no_grad():
        out = model._pixel_patch_attn(0, pixels, loc)
        out_p = model._pixel_patch_attn(0, perturbed, loc)

    # Spatial mixing: another pixel of the SAME unit sees the perturbation.
    assert not torch.allclose(out[0, 1], out_p[0, 1], atol=1e-5)
    # Temporal/modal mixing: the other unit at the same location sees it too.
    assert not torch.allclose(out[1, 0], out_p[1, 0], atol=1e-5)
    # Patch isolation: the unit at the other location is untouched.
    assert torch.allclose(out[2], out_p[2], atol=1e-5)


def test_pixel_embed_encodes_within_patch_offset() -> None:
    """Identical inputs at different within-patch offsets get different embeddings.

    Without an offset signal the joint spatial attention could not tell pixels apart.
    The encoding is offset-relative: the same offset in different patches (and the
    same offset at any patch position) must embed identically.
    """
    model = _build_config().build()
    sample = OlmoEarthSample(
        sentinel2_l2a=torch.ones(B, H, W, T, Modality.SENTINEL2_L2A.num_bands),
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

    with torch.no_grad():
        tokens = model.pixel_embeddings.forward(x, PATCH_SIZE)[S2]
    tok = tokens[0, :, :, 0, 0]  # [H, W, Dp] at one (instance, timestep, band set)

    # Different offsets within a patch differ (constant input, so encoding-only).
    assert not torch.allclose(tok[0, 0], tok[0, 1], atol=1e-5)
    assert not torch.allclose(tok[0, 0], tok[1, 0], atol=1e-5)
    # The same offset in the neighboring patch (PATCH_SIZE away) is identical.
    assert torch.allclose(tok[0, 0], tok[PATCH_SIZE, PATCH_SIZE], atol=1e-6)
    assert torch.allclose(tok[1, 2], tok[1 + PATCH_SIZE, 2 + PATCH_SIZE], atol=1e-6)


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
    assert model.pixel_self_blocks is not None
    assert model.pixel_film is not None
    assert model.coarse_to_pixel is not None
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
    assert model.pixel_self_blocks is not None
    assert any(p.grad is not None for p in model.pixel_self_blocks.parameters())


def test_pixels_do_not_affect_output_when_disconnected() -> None:
    """With no coarse<-pixel cross-attn, the pixel branch cannot change the output."""
    model = _build_config(coarse_cross_attn_to_pixel=False).build()
    x = _masked_sample()

    out = model(x, patch_size=PATCH_SIZE)
    out["tokens_and_masks"].sentinel2_l2a.sum().backward()

    # Pixel modules are disconnected from the coarse output -> no gradient.
    assert all(p.grad is None for p in model.pixel_embeddings.parameters())


def _build_variant_config(
    pixel_branch_type: str, **overrides: object
) -> DualResEncoderConfig:
    kwargs: dict = dict(
        supported_modality_names=[S2, Modality.SENTINEL1.name],
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
        pixel_branch_type=pixel_branch_type,
    )
    kwargs.update(overrides)
    return DualResEncoderConfig(**kwargs)


def _masked_two_modality_sample() -> MaskedOlmoEarthSample:
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
    return strategy.apply_mask(sample, patch_size=PATCH_SIZE)


@pytest.mark.parametrize(
    "pixel_branch_type,overrides",
    [
        ("conv", {"pixel_conv_kernel": 3}),
        ("conv", {"pixel_every_k_blocks": 2}),
        ("window", {}),
        ("window", {"pixel_every_k_blocks": 2}),
        ("perceiver", {}),
        ("perceiver", {"pixel_coarse_read_interval": 2}),
    ],
)
def test_variant_forward_and_grads(pixel_branch_type: str, overrides: dict) -> None:
    """Every pixel-branch variant forwards finitely and trains every pixel module.

    A coarse-only loss must reach every pixel parameter (the pixel -> coarse fusion
    keeps the branch on the coarse loss path each step), so FSDP grad reduction stays
    rank-uniform even when a microbatch produces no pixel-head loss.
    """
    torch.manual_seed(0)
    model = _build_variant_config(pixel_branch_type, **overrides).build()
    x = _masked_two_modality_sample()

    out = model(x, patch_size=PATCH_SIZE)
    tokens = out["tokens_and_masks"]
    assert torch.isfinite(tokens.sentinel2_l2a).all()
    for state in out["pixel_branch"].values():
        assert state.pixels.shape[1] == PATCH_SIZE**2
        assert torch.isfinite(state.pixels).all()

    (tokens.sentinel2_l2a.sum() + tokens.sentinel1.sum()).backward()
    missing = [
        name
        for name, p in model.named_parameters()
        if name.startswith("pixel_") and p.grad is None
    ]
    assert not missing, f"pixel params without gradient: {missing}"


def test_conv_branch_no_masked_data_leakage() -> None:
    """Perturbing a non-ONLINE unit's raw pixels cannot change any ONLINE output.

    The conv branch runs on the dense pixel grid and its depthwise convolutions mix
    across patch boundaries, so masked-unit inputs MUST be zeroed before the first
    conv -- otherwise masked imagery would leak into the visible pixels used to
    reconstruct it.

    ``max_patch_size`` is pinned to the tested patch size: at any other patch size the
    COARSE branch's FlexiPatchEmbed bicubic-resizes the raw input (an intentional,
    pre-existing cross-patch mix that would fail the coarse-token assertion for
    reasons unrelated to the pixel branch).
    """
    torch.manual_seed(0)
    model = _build_variant_config("conv", max_patch_size=PATCH_SIZE).build().eval()
    x = _masked_two_modality_sample()

    assert x.sentinel2_l2a is not None and x.sentinel2_l2a_mask is not None
    mask = x.sentinel2_l2a_mask  # [B, H, W, T, bs] pixel-level mask values
    online_px = mask == MaskValue.ONLINE_ENCODER.value
    # Only perturb pixels where EVERY band set is masked: a channel can be shared by
    # several band sets, and perturbing a channel under an ONLINE band set would
    # legitimately change the outputs.
    masked_all = (~online_px).all(dim=-1)  # [B, H, W, T]
    assert bool(masked_all.any()) and bool(online_px.any())

    perturbed = torch.where(
        masked_all[..., None].expand_as(x.sentinel2_l2a),
        x.sentinel2_l2a + 100.0,
        x.sentinel2_l2a,
    )
    x_perturbed = x._replace(sentinel2_l2a=perturbed)

    with torch.no_grad():
        out = model(x, patch_size=PATCH_SIZE)
        out_p = model(x_perturbed, patch_size=PATCH_SIZE)

    for modality in (S2, Modality.SENTINEL1.name):
        st, st_p = out["pixel_branch"][modality], out_p["pixel_branch"][modality]
        assert torch.equal(st.flat_idx, st_p.flat_idx)
        torch.testing.assert_close(st.pixels, st_p.pixels, atol=1e-5, rtol=1e-4)
    # ONLINE coarse tokens are also unaffected (masked ones are zeroed anyway).
    online_units = mask[:, ::PATCH_SIZE, ::PATCH_SIZE] == (
        MaskValue.ONLINE_ENCODER.value
    )
    tok, tok_p = (
        out["tokens_and_masks"].sentinel2_l2a,
        out_p["tokens_and_masks"].sentinel2_l2a,
    )
    torch.testing.assert_close(
        tok[online_units], tok_p[online_units], atol=1e-5, rtol=1e-4
    )


def test_window_step_isolates_units() -> None:
    """Window attention mixes pixels within a unit but never across units."""
    torch.manual_seed(0)
    step = WindowPixelStep(
        coarse_dim=EMB, pixel_dim=PIXEL_EMB, num_heads=4, mlp_ratio=2.0
    ).eval()
    pixels = torch.randn(3, PATCH_SIZE**2, PIXEL_EMB)
    coarse = torch.randn(3, EMB)
    perturbed = pixels.clone()
    perturbed[0, 0, 0] += 10.0

    with torch.no_grad():
        out, delta = step(pixels, coarse)
        out_p, delta_p = step(perturbed, coarse)

    # Within-unit spatial mixing happens...
    assert not torch.allclose(out[0, 1], out_p[0, 1], atol=1e-5)
    # ...the unit's register (-> coarse update) sees it...
    assert not torch.allclose(delta[0], delta_p[0], atol=1e-6)
    # ...but other units are untouched.
    torch.testing.assert_close(out[1:], out_p[1:])
    torch.testing.assert_close(delta[1:], delta_p[1:])


def test_perceiver_step_never_mixes_pixels() -> None:
    """Perceiver pixels are pointwise: no pixel ever sees another pixel."""
    torch.manual_seed(0)
    step = PerceiverPixelStep(
        coarse_dim=EMB, pixel_dim=PIXEL_EMB, num_heads=4, mlp_ratio=2.0, with_read=True
    ).eval()
    pixels = torch.randn(2, PATCH_SIZE**2, PIXEL_EMB)
    coarse = torch.randn(2, EMB)
    perturbed = pixels.clone()
    # Single channel: a constant added to ALL channels of a pixel is a LayerNorm null
    # direction and would vanish before the read's key/value projection.
    perturbed[0, 0, 0] += 10.0

    with torch.no_grad():
        out, delta = step(pixels, coarse)
        out_p, delta_p = step(perturbed, coarse)

    # Only the perturbed pixel itself changes...
    torch.testing.assert_close(out[0, 1:], out_p[0, 1:])
    torch.testing.assert_close(out[1], out_p[1])
    assert not torch.allclose(out[0, 0], out_p[0, 0], atol=1e-5)
    # ...while its unit's coarse read sees the change; the other unit's does not.
    assert delta is not None and delta_p is not None
    assert not torch.allclose(delta[0], delta_p[0], atol=1e-6)
    torch.testing.assert_close(delta[1], delta_p[1])


def test_pixel_every_k_blocks_builds_reduced_steps() -> None:
    """pixel_every_k_blocks=k builds depth/k steps and still runs the last block."""
    model = _build_variant_config("window", pixel_every_k_blocks=2).build()
    assert model.pixel_steps is not None
    assert len(model.pixel_steps) == 1  # depth 2 // k 2
    assert model._is_pixel_block(1)
    assert not model._is_pixel_block(0)


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
