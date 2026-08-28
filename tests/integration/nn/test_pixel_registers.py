"""Tests for pixel-resolution registers and the conv pixel branch.

Covers the pixreg run group (see ``scripts/official/v1_2/regbtl_v1_2_pixreg_common``):

* pixel-grid register shapes/positions across patch sizes (run 1);
* init-equivalence of the branch runs to the branch-free run (zero-init fusion);
* the leakage guard: values at non-ONLINE units can never reach any output;
* ONLINE-only register-init pooling (fully masked pixels contribute exactly zero);
* the ``"thinconv"`` branch's independence from the coarse trunk (and, by contrast,
  the ``"conv"`` branch's dependence on it through FiLM);
* the ``embedread`` arm (``register_embed_read``): init-equivalence of the extra
  zero-init patch-embed read, its liveness once opened, and its leakage guard.
"""

import pytest
import torch
from torch import nn

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_vit import Encoder
from olmoearth_pretrain.nn.pixel_branch import (
    ConvPixelStep,
    PixelFrameContext,
    PixelRegisterBranch,
)
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, MaskValue

REGISTER_DIM = 16
PIXEL_DIM = 16
B, H, W, T = 2, 8, 8, 2


def _build_encoder(
    pixel_branch_type: str | None = None, embed_read: bool = False
) -> Encoder:
    """Small pixel-grid register encoder mirroring the pixreg run configs."""
    return Encoder(
        supported_modalities=[Modality.SENTINEL2_L2A, Modality.LATLON],
        embedding_size=16,
        max_patch_size=4,
        min_patch_size=1,
        num_heads=2,
        mlp_ratio=2.0,
        max_sequence_length=12,
        depth=4,
        drop_path=0.0,
        position_encoding="rope",
        use_register_bottleneck=True,
        register_grid_size=0,
        register_dim=REGISTER_DIM,
        register_read_depth=1,
        register_latent_depth=2,
        register_pixel_grid=True,
        register_latent_attn_dim=16,
        register_norm_affine=False,
        register_embed_read=embed_read,
        pixel_branch_type=pixel_branch_type,
        pixel_embedding_size=PIXEL_DIM,
        pixel_every_k_blocks=2,
        pixel_thin_depth=2,
    )


def _make_sample(mask_all_timesteps_block: bool = False) -> MaskedOlmoEarthSample:
    """Sample with unit-granularity mixed masking on 4x4 pixel blocks.

    The 4x4 blocks align with every patch size in {1, 2, 4}, so each
    ``(patch, timestep)`` unit carries a single mask value at any tested ps.

    Args:
        mask_all_timesteps_block: Mask the top-left 4x4 block at EVERY timestep
            (pixels with no ONLINE unit anywhere, for the pooling test) instead of
            the default one-block-per-timestep mix.
    """
    torch.manual_seed(1234)
    num_bands = Modality.SENTINEL2_L2A.num_bands
    latlon_bands = Modality.LATLON.num_bands
    mask = torch.zeros(B, H, W, T, num_bands, dtype=torch.long)
    if mask_all_timesteps_block:
        mask[:, 0:4, 0:4, :, :] = MaskValue.DECODER.value
    else:
        mask[:, 0:4, 0:4, 0, :] = MaskValue.DECODER.value
        mask[:, 4:8, 0:4, 1, :] = MaskValue.TARGET_ENCODER_ONLY.value
    return MaskedOlmoEarthSample(
        sentinel2_l2a=torch.randn(B, H, W, T, num_bands),
        sentinel2_l2a_mask=mask,
        latlon=torch.randn(B, latlon_bands),
        latlon_mask=torch.zeros(B, latlon_bands, dtype=torch.long),
        timestamps=torch.tensor(
            [[[1, 0, 2020], [2, 1, 2020]]], dtype=torch.long
        ).expand(B, -1, -1),
    )


def _forward(
    encoder: Encoder, sample: MaskedOlmoEarthSample, patch_size: int
) -> dict[str, torch.Tensor]:
    """Deterministic (eval, no-grad) forward pass."""
    encoder.eval()
    with torch.no_grad():
        return encoder.forward(sample, patch_size=patch_size, input_res=10)


def test_pixel_register_positions_math() -> None:
    """Pixel-center register coordinates follow ((p + 0.5) / ps - 0.5) * scale."""
    encoder = _build_encoder()
    bottleneck = encoder.register_bottleneck
    assert bottleneck is not None
    scale = 3.0
    positions = bottleneck.build_pixel_register_positions(
        batch_size=2,
        register_grid=(4, 6),
        patch_size=2,
        patch_coordinate_scale=scale,
        device=torch.device("cpu"),
    )
    assert positions.shape == (2, 24, 2)
    expected_rows = torch.tensor([((p + 0.5) / 2 - 0.5) * scale for p in range(4)])
    grid = positions[0].view(4, 6, 2)
    torch.testing.assert_close(grid[:, 0, 0], expected_rows)
    # All rows share the column coordinates, all columns share the row coordinates.
    assert torch.equal(grid[:, :1, 0].expand(-1, 6), grid[..., 0])
    assert torch.equal(grid[:1, :, 1].expand(4, -1), grid[..., 1])
    # At patch_size=1 the pixel positions reduce to the patch positions p * scale.
    ps1 = bottleneck.build_pixel_register_positions(
        1, (4, 4), 1, scale, torch.device("cpu")
    )
    torch.testing.assert_close(
        ps1[0].view(4, 4, 2)[:, 0, 0], torch.arange(4, dtype=torch.float32) * scale
    )


@pytest.mark.parametrize("patch_size", [1, 2, 4])
def test_pixel_grid_register_shapes(patch_size: int) -> None:
    """One register per pixel regardless of patch size; pixel-spacing positions."""
    torch.manual_seed(0)
    encoder = _build_encoder()
    sample = _make_sample()
    encoder.zero_grad()
    output = encoder.forward(sample, patch_size=patch_size, input_res=10)

    n_reg = H * W
    # The bottleneck returns the grid shaped; positions stay flat (row-major).
    assert output["registers"].shape == (B, H, W, REGISTER_DIM)
    assert output["register_positions"].shape == (B, n_reg, 2)
    # Consecutive pixels are evenly spaced along each axis.
    grid = output["register_positions"][0].view(H, W, 2)
    row_steps = grid[1:, 0, 0] - grid[:-1, 0, 0]
    col_steps = grid[0, 1:, 1] - grid[0, :-1, 1]
    torch.testing.assert_close(row_steps, row_steps[:1].expand(H - 1))
    torch.testing.assert_close(col_steps, col_steps[:1].expand(W - 1))

    output["registers"].sum().backward()
    bottleneck = encoder.register_bottleneck
    assert bottleneck is not None
    assert bottleneck.register.grad is not None
    assert torch.isfinite(bottleneck.register.grad).all()


def test_pixel_register_spacing_invariant_across_patch_sizes() -> None:
    """The pixel register spacing is a ground property: identical at every ps."""
    torch.manual_seed(0)
    encoder = _build_encoder()
    sample = _make_sample()
    spacings = []
    for patch_size in (1, 2, 4):
        output = _forward(encoder, sample, patch_size)
        grid = output["register_positions"][0].view(H, W, 2)
        spacings.append(grid[1, 0, 0] - grid[0, 0, 0])
    torch.testing.assert_close(spacings[0], spacings[1])
    torch.testing.assert_close(spacings[0], spacings[2])


def _randomize_fusion_paths(encoder: Encoder) -> None:
    """Open the zero-init fusion paths so the pixel branch actually contributes."""
    branch = encoder.pixel_branch
    assert branch is not None
    torch.manual_seed(7)
    for step in branch.steps:
        if isinstance(step, ConvPixelStep):
            nn.init.normal_(step.to_film.weight, std=0.05)
            nn.init.normal_(step.to_film.bias, std=0.05)
            nn.init.normal_(step.to_coarse.weight, std=0.05)
            nn.init.normal_(step.to_coarse.bias, std=0.05)
    nn.init.normal_(branch.to_register.weight, std=0.05)


@pytest.mark.parametrize("branch_type", ["conv", "thinconv"])
def test_pixel_branch_init_equivalence(branch_type: str) -> None:
    """At init the branch models equal the branch-free model bit-for-bit (run 1)."""
    torch.manual_seed(0)
    plain = _build_encoder()
    torch.manual_seed(0)
    branch = _build_encoder(branch_type)
    # Copy every shared parameter; the branch-only parameters keep their init, whose
    # fusion paths zero_init() closed.
    missing, unexpected = branch.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected
    assert all(key.startswith("pixel_branch.") for key in missing)

    sample = _make_sample()
    for patch_size in (1, 2, 4):
        out_plain = _forward(plain, sample, patch_size)
        out_branch = _forward(branch, sample, patch_size)
        assert out_plain.keys() == out_branch.keys()
        for key, value in out_plain.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, out_branch[key]), (key, patch_size)


def _perturb_masked_units(sample: MaskedOlmoEarthSample) -> MaskedOlmoEarthSample:
    """Copy of the sample with the values at every non-ONLINE unit perturbed."""
    mask: torch.Tensor = sample.sentinel2_l2a_mask
    masked_units = (mask != MaskValue.ONLINE_ENCODER.value).any(dim=-1, keepdim=True)
    assert masked_units.any()
    return MaskedOlmoEarthSample(
        sentinel2_l2a=sample.sentinel2_l2a
        + 100.0 * torch.randn_like(sample.sentinel2_l2a) * masked_units,
        sentinel2_l2a_mask=sample.sentinel2_l2a_mask,
        latlon=sample.latlon,
        latlon_mask=sample.latlon_mask,
        timestamps=sample.timestamps,
    )


@pytest.mark.parametrize("branch_type", ["conv", "thinconv"])
@pytest.mark.parametrize("patch_size", [1, 2, 4])
def test_pixel_branch_masked_leakage(branch_type: str, patch_size: int) -> None:
    """Values at non-ONLINE units never reach the branch's outputs (leakage guard).

    The branch zeroes non-ONLINE pixels BEFORE the first convolution, so even though
    the depthwise convs mix across unit boundaries, nothing derived from masked
    values can flow to the register init or (``"conv"``) the coarse fusion.
    Regression for the old dual-res branch's guard, exercised at the BRANCH level:
    the coarse trunk's own FlexiPatchEmbed bilinearly resizes the image whenever
    ``patch_size < max_patch_size``, mixing masked pixels into neighboring ONLINE
    patch tokens -- a pre-existing base-encoder property this test must not conflate
    with the branch (see test_encoder_masked_leakage_at_max_patch_size for the
    end-to-end check on the interpolation-free patch size).
    """
    torch.manual_seed(0)
    encoder = _build_encoder(branch_type)
    _randomize_fusion_paths(encoder)
    assert encoder.pixel_branch is not None
    # Rebound with a non-Optional annotation so the closure below type-checks.
    branch: PixelRegisterBranch = encoder.pixel_branch

    sample = _make_sample()

    def run(s: MaskedOlmoEarthSample) -> list[torch.Tensor]:
        frames, ctx = branch.build_frames(s, patch_size)
        assert frames is not None and ctx is not None
        outputs: list[torch.Tensor] = []
        with torch.no_grad():
            if branch_type == "thinconv":
                frames = branch.run_thin_steps(frames)
            else:
                state = next(iter(ctx.states.values()))
                _, g1, g2, t, bs = state.grid
                ctx.coarse_offsets = {"sentinel2_l2a": 0}
                torch.manual_seed(3)
                coarse = torch.randn(B, g1 * g2 * t * bs, 16)
                for step_idx in range(len(branch.steps)):
                    coarse, frames = branch.run_conv_step(step_idx, frames, coarse, ctx)
                outputs.append(coarse)
            outputs.append(frames)
            outputs.append(branch.register_init(frames, ctx))
        return outputs

    for out_a, out_b in zip(run(sample), run(_perturb_masked_units(sample))):
        assert torch.equal(out_a, out_b)


@pytest.mark.parametrize("branch_type", ["conv", "thinconv"])
def test_encoder_masked_leakage_at_max_patch_size(branch_type: str) -> None:
    """End-to-end: at ps = max_patch_size no output depends on masked values.

    At the maximum patch size the coarse FlexiPatchEmbed applies no resize, so the
    base encoder is exactly invariant to masked-unit values -- any end-to-end
    difference would be a leak introduced by the pixel branch (whose fusion paths
    are opened here so it genuinely contributes to every output).
    """
    torch.manual_seed(0)
    encoder = _build_encoder(branch_type)
    _randomize_fusion_paths(encoder)

    sample = _make_sample()
    out_a = _forward(encoder, sample, patch_size=4)
    out_b = _forward(encoder, _perturb_masked_units(sample), patch_size=4)
    assert out_a.keys() == out_b.keys()
    for key, value in out_a.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, out_b[key]), key


def test_register_init_pooling_is_online_only() -> None:
    """A pixel with no ONLINE unit anywhere contributes exactly zero to the init.

    The convolutions DO write into masked pixels' frames (local mixing is the
    point), but the register-init pooling weights by the ONLINE mask, so a
    fully-masked pixel's pooled feature is exactly zero -- its register cell starts
    from the bare learned latent.
    """
    torch.manual_seed(0)
    encoder = _build_encoder("thinconv")
    branch = encoder.pixel_branch
    assert branch is not None
    # Weight opened, bias kept at zero: a zero pooled feature maps to exactly zero.
    torch.manual_seed(7)
    nn.init.normal_(branch.to_register.weight, std=0.05)

    sample = _make_sample(mask_all_timesteps_block=True)
    frames, ctx = branch.build_frames(sample, patch_size=2)
    assert frames is not None and ctx is not None
    with torch.no_grad():
        frames = branch.run_thin_steps(frames)
        init = branch.register_init(frames, ctx).view(B, H, W, REGISTER_DIM)
    assert torch.all(init[:, 0:4, 0:4] == 0)
    assert init[:, 4:8, 4:8].abs().sum() > 0


@pytest.mark.parametrize("branch_type", ["conv", "thinconv"])
def test_pixel_branch_coarse_trunk_dependence(branch_type: str) -> None:
    """The thinconv register init ignores the coarse trunk; the conv init uses it.

    Perturbing the trunk block weights must leave the standalone thin stack's
    register initialization bit-identical (it never sees a coarse token), while the
    FiLM-conditioned conv branch's initialization must change.
    """
    torch.manual_seed(0)
    encoder = _build_encoder(branch_type)
    _randomize_fusion_paths(encoder)
    branch = encoder.pixel_branch
    assert branch is not None

    captured: dict[str, torch.Tensor] = {}
    original_register_init = branch.register_init

    def spy(frames: torch.Tensor, ctx: PixelFrameContext) -> torch.Tensor:
        init = original_register_init(frames, ctx)
        captured["init"] = init.detach().clone()
        return init

    branch.register_init = spy  # type: ignore[method-assign]

    sample = _make_sample()
    _forward(encoder, sample, patch_size=2)
    init_before = captured["init"]
    assert init_before.abs().sum() > 0

    torch.manual_seed(11)
    with torch.no_grad():
        for parameter in encoder.blocks.parameters():
            parameter.add_(0.1 * torch.randn_like(parameter))
    _forward(encoder, sample, patch_size=2)
    init_after = captured["init"]

    if branch_type == "thinconv":
        assert torch.equal(init_before, init_after)
    else:
        assert not torch.equal(init_before, init_after)


def _open_embed_read(encoder: Encoder) -> None:
    """Open the embed read's zero-init residual paths so it actually contributes."""
    bottleneck = encoder.register_bottleneck
    assert bottleneck is not None and bottleneck.embed_read
    torch.manual_seed(7)
    nn.init.normal_(bottleneck.embed_read_block.attn.proj.weight, std=0.05)
    nn.init.normal_(bottleneck.embed_read_block.mlp.fc2.weight, std=0.05)


def test_embed_read_init_equivalence() -> None:
    """At init the embed-read model equals the embed-read-free model bit-for-bit.

    The extra read block's residual paths (attention out-projection + MLP second
    linear) are zeroed after the blanket weight init, so the block is an exact
    identity on the registers -- the embedread arm IS run 1 at step 0.
    """
    torch.manual_seed(0)
    plain = _build_encoder()
    torch.manual_seed(0)
    embed = _build_encoder(embed_read=True)
    missing, unexpected = embed.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected
    assert missing
    assert all(key.startswith("register_bottleneck.embed_") for key in missing)

    sample = _make_sample()
    for patch_size in (1, 2, 4):
        out_plain = _forward(plain, sample, patch_size)
        out_embed = _forward(embed, sample, patch_size)
        assert out_plain.keys() == out_embed.keys()
        for key, value in out_plain.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, out_embed[key]), (key, patch_size)


@pytest.mark.parametrize("patch_size", [1, 2, 4])
def test_embed_read_contributes_once_opened(patch_size: int) -> None:
    """With the zero-init paths opened, the embed read changes the registers.

    Confirms the extra read is actually wired to a live source (the finalized
    block-0 tokens) rather than silently dropped.
    """
    torch.manual_seed(0)
    plain = _build_encoder()
    torch.manual_seed(0)
    embed = _build_encoder(embed_read=True)
    embed.load_state_dict(plain.state_dict(), strict=False)
    _open_embed_read(embed)

    sample = _make_sample()
    out_plain = _forward(plain, sample, patch_size)
    out_embed = _forward(embed, sample, patch_size)
    assert not torch.equal(out_plain["registers"], out_embed["registers"])


def test_embed_read_masked_leakage_at_max_patch_size() -> None:
    """End-to-end: the opened embed read introduces no masked-value leakage.

    At ps = max_patch_size the coarse FlexiPatchEmbed applies no resize, so the
    embed-read-free encoder is exactly invariant to masked-unit values; the embed
    source is captured AFTER masked-token removal (re-added rows are zeros and
    read-masked), so the embed-read model must be invariant too.
    """
    torch.manual_seed(0)
    encoder = _build_encoder(embed_read=True)
    _open_embed_read(encoder)

    sample = _make_sample()
    out_a = _forward(encoder, sample, patch_size=4)
    out_b = _forward(encoder, _perturb_masked_units(sample), patch_size=4)
    assert out_a.keys() == out_b.keys()
    for key, value in out_a.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, out_b[key]), key
