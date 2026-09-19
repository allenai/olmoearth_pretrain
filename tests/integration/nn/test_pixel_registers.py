"""Tests for pixel-resolution registers (``register_pixel_grid``).

Covers the pixreg run (see ``scripts/official/v1_3/experiments/pixreg_pixrecon.py``):

* pixel-center register coordinate math;
* one register per pixel regardless of patch size, evenly spaced positions, and a
  finite gradient into the cloned latent;
* the pixel spacing as a ground property, identical at every patch size;
* the decoupled latent self-attention width/heads and the affine-free block norms;
* the whole pixel-register LatentMIM forward (decoder over the larger grid, per-cell
  map supervision, time-conditioned raw-band reconstruction heads);
* the ``"thinconv"`` pixel branch (``pixreg_thinconv_pixrecon``): init equivalence
  with the branch-free encoder, the leakage guard at branch and encoder level, and
  the full LatentMIM forward with the branch attached.
"""

import pytest
import torch
from torch import nn

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_vit import Encoder, EncoderConfig, PredictorConfig
from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
from olmoearth_pretrain.nn.pixel_branch import PixelRegisterBranch
from olmoearth_pretrain.nn.supervision_head import (
    SupervisionHeadConfig,
    SupervisionModalityConfig,
    SupervisionTaskType,
    compute_supervision_loss,
)
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, MaskValue

REGISTER_DIM = 16
LATENT_ATTN_DIM = 16
LATENT_NUM_HEADS = 2
PIXEL_DIM = 16
PIXEL_THIN_DEPTH = 2
B, H, W, T = 2, 8, 8, 2


def _build_encoder(pixel_branch_type: str | None = None) -> Encoder:
    """Small pixel-grid register encoder mirroring the pixreg run config."""
    return Encoder(
        supported_modalities=[Modality.SENTINEL2_L2A, Modality.LATLON],
        embedding_size=32,
        max_patch_size=4,
        min_patch_size=1,
        num_heads=2,
        mlp_ratio=2.0,
        max_sequence_length=12,
        depth=2,
        drop_path=0.0,
        position_encoding="rope",
        use_register_bottleneck=True,
        register_dim=REGISTER_DIM,
        register_latent_depth=2,
        register_per_depth_read_proj=True,
        # Wideread: reads at encoder width; latent self-attention at register width.
        register_attn_dim=32,
        register_pixel_grid=True,
        register_latent_attn_dim=LATENT_ATTN_DIM,
        register_latent_num_heads=LATENT_NUM_HEADS,
        register_norm_affine=False,
        pixel_branch_type=pixel_branch_type,
        pixel_embedding_size=PIXEL_DIM,
        pixel_thin_depth=PIXEL_THIN_DEPTH,
    )


def _make_sample() -> MaskedOlmoEarthSample:
    """Sample with unit-granularity mixed masking on 4x4 pixel blocks.

    The 4x4 blocks align with every patch size in {1, 2, 4}, so each
    ``(patch, timestep)`` unit carries a single mask value at any tested ps.
    """
    torch.manual_seed(1234)
    num_bands = Modality.SENTINEL2_L2A.num_bands
    latlon_bands = Modality.LATLON.num_bands
    mask = torch.zeros(B, H, W, T, num_bands, dtype=torch.long)
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


def test_pixel_grid_matches_patch_grid_at_ps1() -> None:
    """At ps=1 the pixel-grid bottleneck lays down the same frame as the patch grid."""
    encoder = _build_encoder()
    bottleneck = encoder.register_bottleneck
    assert bottleneck is not None
    scale = 2.5
    patch_positions = torch.stack(
        torch.meshgrid(
            torch.arange(4, dtype=torch.float32) * scale,
            torch.arange(4, dtype=torch.float32) * scale,
            indexing="ij",
        ),
        dim=-1,
    ).reshape(1, 16, 2)
    patch_grid_positions = bottleneck.build_register_positions(patch_positions, (4, 4))
    pixel_positions = bottleneck.build_pixel_register_positions(
        1, (4, 4), 1, scale, torch.device("cpu")
    )
    torch.testing.assert_close(pixel_positions, patch_grid_positions)


def test_latent_attention_decoupled_and_norms_affine_free() -> None:
    """Reads at encoder width; latent blocks at the register width; no block affine."""
    encoder = _build_encoder()
    bottleneck = encoder.register_bottleneck
    assert bottleneck is not None
    for read_blk in bottleneck.read_blocks:
        assert read_blk.attn.num_heads == 2
        # Wideread: q/k/v run at the encoder width (32), not the register width.
        assert read_blk.attn.q.out_features == 32
    for latent_blk in bottleneck.latent_blocks:
        assert latent_blk.attn.num_heads == LATENT_NUM_HEADS
        # latent_attn_dim == register_dim -> classic tied-width attention.
        assert latent_blk.attn.q.out_features == REGISTER_DIM
    for blk in (*bottleneck.read_blocks, *bottleneck.latent_blocks):
        for module in blk.modules():
            if isinstance(module, nn.LayerNorm):
                assert not module.elementwise_affine
    # The K/V input norms and the output norm keep their affine.
    for norm in bottleneck.input_norms:
        assert norm.elementwise_affine
    assert bottleneck.norm.elementwise_affine


def test_pixel_grid_requires_bottleneck() -> None:
    """register_pixel_grid without the bottleneck is a config error."""
    config = EncoderConfig(
        supported_modality_names=[Modality.SENTINEL2_L2A.name],
        embedding_size=16,
        num_heads=2,
        depth=1,
        position_encoding="rope",
        register_pixel_grid=True,
    )
    with pytest.raises(ValueError, match="register_pixel_grid requires"):
        config.validate()


def test_latent_head_shape_validated() -> None:
    """A latent width that does not split into RoPE-sized heads is rejected."""
    config = EncoderConfig(
        supported_modality_names=[Modality.SENTINEL2_L2A.name],
        embedding_size=16,
        num_heads=2,
        depth=1,
        position_encoding="rope",
        use_register_bottleneck=True,
        register_dim=8,
        register_latent_attn_dim=8,
        register_latent_num_heads=3,
    )
    with pytest.raises(ValueError, match="register_latent_num_heads"):
        config.validate()


def _pixreg_pixrecon_model_config(
    pixel_branch_type: str | None = None,
) -> LatentMIMConfig:
    """Small version of the pixreg_pixrecon run: pixel registers + recon heads.

    ``pixel_branch_type="thinconv"`` gives the pixreg_thinconv_pixrecon shape.
    """
    modalities = [Modality.SENTINEL2_L2A.name, Modality.LATLON.name]
    encoder_config = EncoderConfig(
        supported_modality_names=modalities,
        embedding_size=32,
        num_heads=2,
        depth=2,
        mlp_ratio=2.0,
        max_patch_size=4,
        min_patch_size=1,
        max_sequence_length=12,
        drop_path=0.0,
        position_encoding="rope",
        use_register_bottleneck=True,
        register_dim=REGISTER_DIM,
        register_latent_depth=2,
        register_per_depth_read_proj=True,
        register_attn_dim=32,
        register_pixel_grid=True,
        register_latent_attn_dim=LATENT_ATTN_DIM,
        register_latent_num_heads=LATENT_NUM_HEADS,
        register_norm_affine=False,
        pixel_branch_type=pixel_branch_type,
        pixel_embedding_size=PIXEL_DIM,
        pixel_thin_depth=PIXEL_THIN_DEPTH,
    )
    decoder_config = PredictorConfig(
        supported_modality_names=modalities,
        encoder_embedding_size=32,
        decoder_embedding_size=16,
        num_heads=2,
        depth=1,
        mlp_ratio=2.0,
        max_sequence_length=12,
        drop_path=0.0,
        position_encoding="rope",
        use_register_bottleneck=True,
        register_dim=REGISTER_DIM,
    )
    supervision_config = SupervisionHeadConfig(
        modality_configs={
            # A static per-cell head (regression so the random target is valid).
            "latlon": SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=Modality.LATLON.num_bands,
                weight=0.1,
            ),
            # The pixrecon head: time-conditioned reconstruction of the S2 inputs.
            "sentinel2_l2a": SupervisionModalityConfig(
                task_type=SupervisionTaskType.REGRESSION,
                num_output_channels=Modality.SENTINEL2_L2A.num_bands,
                weight=0.05,
                regression_loss_type="mse",
                time_conditioned=True,
                time_harmonics=4,
            ),
        },
        spatial_unfold=1,
    )
    return LatentMIMConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        supervision_head_config=supervision_config,
    )


@pytest.mark.parametrize("patch_size", [1, 2, 4])
def test_pixreg_pixrecon_latent_mim_forward(patch_size: int) -> None:
    """The full model runs at every patch size and reconstructs per (pixel, timestep)."""
    torch.manual_seed(0)
    model = _pixreg_pixrecon_model_config().build()
    sample = _make_sample()
    model.train()
    (latent, decoded, _pooled, _recon, _metrics, supervision_preds, _proj) = (
        model.forward(sample, patch_size=patch_size)
    )
    assert supervision_preds is not None
    # Per-cell recon at every observed timestep, at the target's pixel resolution.
    assert supervision_preds["sentinel2_l2a"].shape == (
        B,
        H,
        W,
        T,
        Modality.SENTINEL2_L2A.num_bands,
    )
    assert supervision_preds["latlon"].shape == (B, Modality.LATLON.num_bands)
    # Decoder targets keep the input layout at this patch size.
    assert decoded.sentinel2_l2a is not None
    assert decoded.sentinel2_l2a.shape[:3] == (B, H // patch_size, W // patch_size)

    assert model.supervision_head is not None
    loss, per_modality = compute_supervision_loss(
        supervision_preds, sample, model.supervision_head
    )
    assert torch.isfinite(loss)
    assert set(per_modality) == {"sentinel2_l2a", "latlon"}
    loss.backward()
    # The reconstruction loss reaches the register latent through the pixel grid.
    assert model.encoder.register_bottleneck is not None
    grad = model.encoder.register_bottleneck.register.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


# --- thinconv pixel branch --------------------------------------------------------------


def _open_register_init(encoder: Encoder) -> None:
    """Open the zero-init handoff so the pixel branch actually contributes."""
    assert encoder.pixel_branch is not None
    torch.manual_seed(7)
    nn.init.normal_(encoder.pixel_branch.to_register.weight, std=0.05)


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


def test_pixel_branch_init_equivalence() -> None:
    """At init the branch model equals the branch-free model bit-for-bit.

    The register-init projection is zeroed by ``zero_init``, so the branch's
    parameters exist (and receive gradient) but contribute nothing at step 0 -- the
    thinconv arm starts exactly where ``pixreg_pixrecon`` does.
    """
    torch.manual_seed(0)
    plain = _build_encoder()
    torch.manual_seed(0)
    branch = _build_encoder("thinconv")
    assert isinstance(branch.pixel_branch, PixelRegisterBranch)
    assert len(branch.pixel_branch.steps) == PIXEL_THIN_DEPTH
    assert torch.equal(
        branch.pixel_branch.to_register.weight,
        torch.zeros_like(branch.pixel_branch.to_register.weight),
    )
    # Copy every shared parameter; the branch-only parameters keep their init.
    missing, unexpected = branch.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected
    assert missing and all(key.startswith("pixel_branch.") for key in missing)

    sample = _make_sample()
    for patch_size in (1, 2, 4):
        out_plain = _forward(plain, sample, patch_size)
        out_branch = _forward(branch, sample, patch_size)
        assert out_plain.keys() == out_branch.keys()
        for key, value in out_plain.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, out_branch[key]), (key, patch_size)


@pytest.mark.parametrize("patch_size", [1, 2, 4])
def test_pixel_branch_changes_registers_once_open(patch_size: int) -> None:
    """With the handoff open, the branch changes the register grid (it is wired in)."""
    torch.manual_seed(0)
    plain = _build_encoder()
    torch.manual_seed(0)
    branch = _build_encoder("thinconv")
    branch.load_state_dict(plain.state_dict(), strict=False)
    _open_register_init(branch)
    sample = _make_sample()
    regs_plain = _forward(plain, sample, patch_size)["registers"]
    regs_branch = _forward(branch, sample, patch_size)["registers"]
    assert regs_plain.shape == regs_branch.shape == (B, H, W, REGISTER_DIM)
    assert not torch.allclose(regs_plain, regs_branch)


@pytest.mark.parametrize("patch_size", [1, 2, 4])
def test_pixel_branch_masked_leakage(patch_size: int) -> None:
    """Values at non-ONLINE units never reach the branch's outputs (leakage guard).

    The branch zeroes non-ONLINE pixels BEFORE the first convolution, so even though
    the depthwise convs mix across unit boundaries, nothing derived from masked
    values can flow to the register init. Exercised at the BRANCH level because the
    coarse trunk's own FlexiPatchEmbed bilinearly resizes the image whenever
    ``patch_size < max_patch_size``, mixing masked pixels into neighbouring ONLINE
    patch tokens -- a pre-existing base-encoder property this test must not conflate
    with the branch (see the end-to-end check at the interpolation-free patch size).
    """
    torch.manual_seed(0)
    encoder = _build_encoder("thinconv")
    _open_register_init(encoder)
    assert encoder.pixel_branch is not None
    branch: PixelRegisterBranch = encoder.pixel_branch
    sample = _make_sample()

    def run(s: MaskedOlmoEarthSample) -> list[torch.Tensor]:
        with torch.no_grad():
            frames, ctx = branch.build_frames(s, patch_size)
            assert frames is not None and ctx is not None
            frames = branch.run_thin_steps(frames)
            init = branch.register_init(frames, ctx)
        assert init.shape == (B, H * W, REGISTER_DIM)
        return [frames, init]

    for out_a, out_b in zip(run(sample), run(_perturb_masked_units(sample))):
        assert torch.equal(out_a, out_b)


def test_encoder_masked_leakage_at_max_patch_size() -> None:
    """End-to-end: at ps = max_patch_size no output depends on masked values.

    At the maximum patch size the coarse FlexiPatchEmbed applies no resize, so the
    base encoder is exactly invariant to masked-unit values -- any end-to-end
    difference would be a leak introduced by the pixel branch (whose handoff is
    opened here so it genuinely contributes to every output).
    """
    torch.manual_seed(0)
    encoder = _build_encoder("thinconv")
    _open_register_init(encoder)
    sample = _make_sample()
    out_a = _forward(encoder, sample, patch_size=4)
    out_b = _forward(encoder, _perturb_masked_units(sample), patch_size=4)
    assert out_a.keys() == out_b.keys()
    for key, value in out_a.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, out_b[key]), key


def test_pixel_branch_register_init_is_online_only_pooling() -> None:
    """A pixel whose every unit is masked gets a zero init (bare latent).

    With the projection bias zero (as ``zero_init`` leaves it) and the weight open,
    the init at an all-masked pixel is exactly zero, while visible pixels are not.
    """
    torch.manual_seed(0)
    encoder = _build_encoder("thinconv")
    _open_register_init(encoder)
    assert encoder.pixel_branch is not None
    branch = encoder.pixel_branch
    num_bands = Modality.SENTINEL2_L2A.num_bands
    # The top-left 4x4 block is masked at EVERY timestep; everything else ONLINE.
    mask = torch.zeros(B, H, W, T, num_bands, dtype=torch.long)
    mask[:, 0:4, 0:4, :, :] = MaskValue.DECODER.value
    sample = MaskedOlmoEarthSample(
        sentinel2_l2a=torch.randn(B, H, W, T, num_bands),
        sentinel2_l2a_mask=mask,
        timestamps=torch.tensor(
            [[[1, 0, 2020], [2, 1, 2020]]], dtype=torch.long
        ).expand(B, -1, -1),
    )
    with torch.no_grad():
        frames, ctx = branch.build_frames(sample, patch_size=4)
        assert frames is not None and ctx is not None
        init = branch.register_init(branch.run_thin_steps(frames), ctx)
    init = init.view(B, H, W, REGISTER_DIM)
    assert torch.equal(init[:, 0:4, 0:4], torch.zeros_like(init[:, 0:4, 0:4]))
    assert init[:, 4:, 4:].abs().sum() > 0


def test_pixel_branch_requires_pixel_grid() -> None:
    """pixel_branch_type without register_pixel_grid is rejected at config time."""
    config = EncoderConfig(
        supported_modality_names=[Modality.SENTINEL2_L2A.name],
        embedding_size=16,
        num_heads=2,
        depth=1,
        position_encoding="rope",
        use_register_bottleneck=True,
        register_dim=8,
        pixel_branch_type="thinconv",
    )
    with pytest.raises(ValueError, match="register_pixel_grid"):
        config.validate()


def test_pixel_branch_type_validated() -> None:
    """Only the thinconv variant exists; the old 'conv' name is rejected."""
    config = EncoderConfig(
        supported_modality_names=[Modality.SENTINEL2_L2A.name],
        embedding_size=16,
        num_heads=2,
        depth=1,
        position_encoding="rope",
        use_register_bottleneck=True,
        register_dim=8,
        register_pixel_grid=True,
        pixel_branch_type="conv",
    )
    with pytest.raises(ValueError, match="thinconv"):
        config.validate()


@pytest.mark.parametrize("patch_size", [1, 2, 4])
def test_pixreg_thinconv_pixrecon_latent_mim_forward(patch_size: int) -> None:
    """The thinconv arm's full model trains end to end and the branch gets gradient.

    The handoff is opened so the conv stack lies on the loss path; the register
    latent and the branch's embedding both receive finite, non-zero gradient.
    """
    torch.manual_seed(0)
    model = _pixreg_pixrecon_model_config("thinconv").build()
    assert model.encoder.pixel_branch is not None
    _open_register_init(model.encoder)
    sample = _make_sample()
    model.train()
    (_latent, decoded, _pooled, _recon, _metrics, supervision_preds, _proj) = (
        model.forward(sample, patch_size=patch_size)
    )
    assert supervision_preds is not None
    assert supervision_preds["sentinel2_l2a"].shape == (
        B,
        H,
        W,
        T,
        Modality.SENTINEL2_L2A.num_bands,
    )
    assert decoded.sentinel2_l2a is not None
    assert decoded.sentinel2_l2a.shape[:3] == (B, H // patch_size, W // patch_size)

    assert model.supervision_head is not None
    loss, _ = compute_supervision_loss(
        supervision_preds, sample, model.supervision_head
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.encoder.register_bottleneck is not None
    grad = model.encoder.register_bottleneck.register.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    branch = model.encoder.pixel_branch
    embed_grads = [p.grad for p in branch.embed.parameters() if p.grad is not None]
    assert embed_grads and all(torch.isfinite(g).all() for g in embed_grads)
    assert sum(g.abs().sum() for g in embed_grads) > 0
