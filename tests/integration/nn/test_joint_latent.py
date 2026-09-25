"""Tests for the joint latent-token transformer (``nn/joint_latent.py``)."""

from typing import Any

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_vit import Encoder, EncoderConfig, PerceiverConfig
from olmoearth_pretrain.nn.joint_latent import (
    JointLatentConfig,
    JointLatentTransformer,
    joint_attention_allowed,
)
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, MaskValue


def test_joint_attention_mask_matches_the_written_out_rule() -> None:
    """The vectorised mask equals the two rules written as loops.

    Latent -> every latent + tokens of its own cell; token -> every latent + tokens of
    its own cell; padding is never a key; non-spatial tokens (cell -1) are read by no
    latent but see the latents and each other.
    """
    torch.manual_seed(0)
    n_tokens, n_latents = 11, 4
    cell_tok = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3, -1, 0, 1]])
    cell_id = torch.cat([cell_tok, torch.arange(n_latents)[None]], dim=1)
    is_latent = torch.cat(
        [
            torch.zeros(1, n_tokens, dtype=torch.bool),
            torch.ones(1, n_latents, dtype=torch.bool),
        ],
        1,
    )
    valid = torch.ones_like(is_latent)
    valid[0, 9:11] = False  # two padding tokens
    allowed = joint_attention_allowed(cell_id, is_latent, valid)[0]
    for q in range(n_tokens + n_latents):
        for kv in range(n_tokens + n_latents):
            expect = bool(valid[0, kv]) and (
                bool(is_latent[0, kv]) or int(cell_id[0, q]) == int(cell_id[0, kv])
            )
            assert bool(allowed[q, kv]) == expect, (q, kv)
    # A latent never reads a non-spatial (-1) token; that token does see the latents.
    assert not allowed[n_tokens:, 8].any()
    assert allowed[8, n_tokens:].all()


def test_joint_block_with_an_open_mask_is_a_plain_block() -> None:
    """With an open mask ``_joint_block`` equals ``Block.forward``.

    The only thing the joint path adds is the mask, so this pins the masked call
    against the stock one.
    """
    torch.manual_seed(0)
    module = JointLatentTransformer(
        embedding_size=32,
        num_heads=4,
        mlp_ratio=2.0,
        joint_depth=1,
        latent_only_depth=0,
        token_mlp=True,
        position_encoding="rope_3d_mixed",
        rope_base=10000.0,
        rope_mixed_base=10.0,
        temporal_rope_dim_frac=0.25,
        rope_temporal_base=None,
        qk_norm=False,
    ).eval()
    blk = module.joint_blocks[0]
    x = torch.randn(2, 13, 32)
    pos = torch.rand(2, 13, 3) * 10
    open_mask = torch.ones(2, 1, 13, 13, dtype=torch.bool)
    out_joint = module._joint_block(
        blk, x, n_tokens=9, rope_positions=pos, attn_kwargs={"attn_mask": open_mask}
    )
    out_block = blk(x=x, rope_positions=pos)
    torch.testing.assert_close(out_joint, out_block)


def _joint_encoder(**config_kwargs: object) -> Encoder:
    return Encoder(
        supported_modalities=[Modality.SENTINEL2_L2A, Modality.SENTINEL1],
        embedding_size=32,
        max_patch_size=4,
        min_patch_size=1,
        num_heads=4,
        mlp_ratio=2.0,
        max_sequence_length=12,
        depth=0,
        drop_path=0.0,
        position_encoding="rope_3d_mixed",
        perceiver_config=JointLatentConfig(register_dim=32, **config_kwargs),  # type: ignore[arg-type]
    )


def _sample(B: int = 2, H: int = 8, W: int = 8, T: int = 3) -> MaskedOlmoEarthSample:
    nb2 = Modality.SENTINEL2_L2A.num_bands
    nb1 = Modality.SENTINEL1.num_bands
    timestamps = torch.tensor([[[1, 0, 2020], [1, 3, 2020], [1, 6, 2020]]] * B).long()
    return MaskedOlmoEarthSample(
        sentinel2_l2a=torch.randn(B, H, W, T, nb2),
        sentinel2_l2a_mask=torch.full(
            (B, H, W, T, nb2), MaskValue.ONLINE_ENCODER.value
        ),
        sentinel1=torch.randn(B, H, W, T, nb1),
        sentinel1_mask=torch.full((B, H, W, T, nb1), MaskValue.ONLINE_ENCODER.value),
        timestamps=timestamps,
    )


@pytest.mark.parametrize("token_mlp", [True, False])
def test_encoder_with_joint_latent_produces_the_register_grid(token_mlp: bool) -> None:
    """A zero-depth encoder with a JointLatentConfig emits the register grid.

    Grid, positions, student output and gradients, at more than one patch size (the
    grid follows the input).
    """
    torch.manual_seed(0)
    encoder = _joint_encoder(
        joint_depth=2,
        latent_only_depth=1,
        token_mlp=token_mlp,
        student_dims=[8],
        student_output_norm=True,
    )
    assert len(encoder.blocks) == 0
    assert isinstance(encoder.perceiver, JointLatentTransformer)
    sample = _sample()
    for patch_size in (2, 4):
        encoder.zero_grad()
        out = encoder(sample, patch_size=patch_size, input_res=10)
        side = 8 // patch_size
        assert out["registers"].shape == (2, side, side, 32)
        assert out["register_positions"].shape == (2, side * side, 2)
        assert out["student_registers"].shape == (2, side, side, 8)
        assert torch.isfinite(out["registers"]).all()
        out["registers"].sum().backward()
        assert encoder.perceiver.register.grad is not None
        assert encoder.perceiver.joint_blocks[0].attn.q.weight.grad is not None
    # The inference fast path (no mask removal) gives the same shapes.
    out = encoder(sample, patch_size=2, input_res=10, fast_pass=True)
    assert out["registers"].shape == (2, 4, 4, 32)


def test_invalid_tokens_do_not_reach_the_registers() -> None:
    """Padding / invisible tokens are inert inside the joint blocks.

    They are excluded as keys and their query outputs are discarded, so perturbing them
    changes no register; perturbing a valid token does. Checked at the module boundary
    because the encoder's patch embedding itself mixes neighbouring pixels below the
    maximum patch size, which would confound a pixel-level version of this test.
    """
    torch.manual_seed(0)
    module = JointLatentTransformer(
        embedding_size=32,
        num_heads=4,
        mlp_ratio=2.0,
        joint_depth=2,
        latent_only_depth=1,
        token_mlp=True,
        position_encoding="rope_3d_mixed",
        rope_base=10000.0,
        rope_mixed_base=10.0,
        temporal_rope_dim_frac=0.25,
        rope_temporal_base=None,
        qk_norm=False,
    ).eval()
    B, n_h, n_w, T = 2, 3, 3, 4
    n_tokens = n_h * n_w * T
    tokens = torch.randn(B, n_tokens, 32)
    cells = torch.arange(n_h * n_w).repeat_interleave(T).expand(B, -1)
    positions = torch.stack(
        [
            torch.arange(T).repeat(n_h * n_w).float().expand(B, -1),
            (cells // n_w).float(),
            (cells % n_w).float(),
        ],
        dim=-1,
    )
    valid = torch.ones(B, n_tokens, dtype=torch.bool)
    valid[0, -5:] = False
    perturbed = tokens.clone()
    perturbed[0, -5:] += 100.0
    with torch.no_grad():
        ref, _ = module(tokens, positions, valid, cells, (n_h, n_w))
        same, _ = module(perturbed, positions, valid, cells, (n_h, n_w))
        touched = tokens.clone()
        touched[0, 0] += 100.0
        different, _ = module(touched, positions, valid, cells, (n_h, n_w))
    torch.testing.assert_close(ref, same)
    assert not torch.allclose(ref, different)


def test_encoder_config_dispatches_the_joint_config_class() -> None:
    """A serialised JointLatentConfig dict is rebuilt as that class.

    A dict carrying the JointLatentConfig ``_CLASS_`` must not become a PerceiverConfig;
    a plain Perceiver dict still does.
    """
    joint = EncoderConfig(
        supported_modality_names=["sentinel2_l2a"],
        embedding_size=32,
        num_heads=4,
        depth=0,
        position_encoding="rope_3d_mixed",
        perceiver_config={  # type: ignore[arg-type]
            "_CLASS_": "olmoearth_pretrain.nn.joint_latent.JointLatentConfig",
            "register_dim": 32,
            "joint_depth": 3,
        },
    )
    assert isinstance(joint.perceiver_config, JointLatentConfig)
    assert joint.perceiver_config.joint_depth == 3
    joint.validate()
    plain = EncoderConfig(
        supported_modality_names=["sentinel2_l2a"],
        embedding_size=32,
        num_heads=4,
        position_encoding="rope",
        perceiver_config={"register_dim": 32, "latent_depth": 2},  # type: ignore[arg-type]
    )
    assert isinstance(plain.perceiver_config, PerceiverConfig)
    with pytest.raises(ValueError, match="embedding_size"):
        EncoderConfig(
            supported_modality_names=["sentinel2_l2a"],
            embedding_size=32,
            num_heads=4,
            depth=0,
            position_encoding="rope_3d_mixed",
            perceiver_config=JointLatentConfig(register_dim=16),
        ).build()


def test_latent_reads_all_opens_only_the_latent_rows() -> None:
    """With ``latent_reads_all`` a latent query sees every valid key; tokens unchanged."""
    torch.manual_seed(0)
    n_tokens, n_latents = 11, 4
    cell_tok = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3, -1, 0, 1]])
    cell_id = torch.cat([cell_tok, torch.arange(n_latents)[None]], dim=1)
    is_latent = torch.cat(
        [
            torch.zeros(1, n_tokens, dtype=torch.bool),
            torch.ones(1, n_latents, dtype=torch.bool),
        ],
        1,
    )
    valid = torch.ones_like(is_latent)
    valid[0, 9:11] = False
    base = joint_attention_allowed(cell_id, is_latent, valid)[0]
    opened = joint_attention_allowed(cell_id, is_latent, valid, latent_reads_all=True)[
        0
    ]
    # Token rows are identical.
    torch.testing.assert_close(opened[:n_tokens], base[:n_tokens])
    # Latent rows: exactly the valid keys (including the non-spatial -1 token).
    for q in range(n_tokens, n_tokens + n_latents):
        assert torch.equal(opened[q], valid[0]), q
    assert opened[n_tokens:, 8].all()  # the -1 token is now read by every latent
    assert not base[n_tokens:, 8].any()


def _joint_module(**overrides: Any) -> JointLatentTransformer:
    kwargs: dict = dict(
        embedding_size=32,
        num_heads=4,
        mlp_ratio=2.0,
        joint_depth=2,
        latent_only_depth=1,
        token_mlp=True,
        position_encoding="rope_3d_mixed",
        rope_base=10000.0,
        rope_mixed_base=10.0,
        temporal_rope_dim_frac=0.25,
        rope_temporal_base=None,
        qk_norm=False,
    )
    kwargs.update(overrides)
    return JointLatentTransformer(**kwargs).eval()


def _encoder_order_inputs(
    B: int, n_h: int, n_w: int, T: int, n_mod: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokens in the encoder's collapsed order: per modality ``(h, w, t)``, then concat."""
    cells_one = torch.arange(n_h * n_w).repeat_interleave(T)
    cells = cells_one.repeat(n_mod).expand(B, -1)
    t = torch.arange(T).repeat(n_h * n_w).repeat(n_mod).float().expand(B, -1)
    positions = torch.stack([t, (cells // n_w).float(), (cells % n_w).float()], dim=-1)
    n_tokens = cells.shape[1]
    tokens = torch.randn(B, n_tokens, 32)
    valid = torch.ones(B, n_tokens, dtype=torch.bool)
    valid[0, 5:9] = False  # padding inside the first modality's run
    return tokens, positions, valid, cells


def test_cell_sorted_layout_is_numerically_a_no_op() -> None:
    """Sorting tokens by cell changes no register (attention is permutation-equivariant)."""
    torch.manual_seed(0)
    tokens, positions, valid, cells = _encoder_order_inputs(
        B=2, n_h=3, n_w=3, T=4, n_mod=2
    )
    unsorted = _joint_module(sort_by_cell=False)
    sorted_ = _joint_module(sort_by_cell=True)
    sorted_.load_state_dict(unsorted.state_dict())
    with torch.no_grad():
        ref, ref_pos = unsorted(tokens, positions, valid, cells, (3, 3))
        out, out_pos = sorted_(tokens, positions, valid, cells, (3, 3))
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_pos, ref_pos)


def test_sort_by_cell_groups_cells_and_parks_padding_last() -> None:
    """The permutation is stable within a cell, groups its runs, and sends padding last."""
    tokens, positions, valid, cells = _encoder_order_inputs(
        B=1, n_h=2, n_w=2, T=3, n_mod=2
    )
    n_latents = 4
    tok, pos, val, cid = JointLatentTransformer._sort_by_cell(
        tokens, positions, valid, cells, n_latents
    )
    n_valid = int(valid.sum())
    # Valid tokens first, sorted by cell; padding after.
    assert val[0, :n_valid].all() and not val[0, n_valid:].any()
    assert torch.equal(cid[0, :n_valid], cid[0, :n_valid].sort().values)
    # Positions travelled with their tokens.
    inv = torch.argsort(cells[0].masked_fill(~valid[0], n_latents), stable=True)
    torch.testing.assert_close(tok[0], tokens[0, inv])
    torch.testing.assert_close(pos[0], positions[0, inv])


def test_latent_reads_all_changes_registers_only_through_latent_rows() -> None:
    """A far-away token (other cell) reaches a register under latent_reads_all only."""
    torch.manual_seed(0)
    tokens, positions, valid, cells = _encoder_order_inputs(
        B=1, n_h=2, n_w=2, T=3, n_mod=1
    )
    local = _joint_module(latent_reads_all=False)
    opened = _joint_module(latent_reads_all=True)
    opened.load_state_dict(local.state_dict())
    with torch.no_grad():
        a, _ = local(tokens, positions, valid, cells, (2, 2))
        b, _ = opened(tokens, positions, valid, cells, (2, 2))
    # The two masks differ, so the registers differ.
    assert not torch.allclose(a, b)
    # Config round-trip carries both flags.
    cfg = JointLatentConfig(register_dim=32, latent_reads_all=True, sort_by_cell=False)
    built = cfg.build(
        encoder_embedding_size=32,
        encoder_num_heads=4,
        mlp_ratio=2.0,
        position_encoding="rope_3d_mixed",
        rope_base=10000.0,
        qk_norm=False,
    )
    assert built.latent_reads_all is True and built.sort_by_cell is False


def test_pixel_latents_at_patch_size_one_equal_patch_latents() -> None:
    """At patch size 1 a pixel is a patch, so pixel latents reproduce the patch grid."""
    torch.manual_seed(0)
    patch_model = _joint_encoder(joint_depth=2, latent_reads_all=True).eval()
    torch.manual_seed(0)
    pixel_model = _joint_encoder(
        joint_depth=2, latent_reads_all=True, pixel_latents=True
    ).eval()
    pixel_model.load_state_dict(patch_model.state_dict())
    sample = _sample()
    with torch.no_grad():
        a = patch_model(sample, patch_size=1, input_res=10)
        b = pixel_model(sample, patch_size=1, input_res=10)
    torch.testing.assert_close(a["registers"], b["registers"])
    torch.testing.assert_close(a["register_positions"], b["register_positions"])


@pytest.mark.parametrize("latent_reads_all", [False, True])
def test_pixel_latents_grid_positions_and_cells(latent_reads_all: bool) -> None:
    """Above patch size 1 the grid is at pixel resolution, centred inside its patch.

    Each pixel latent carries the cell id of its containing patch, so without
    ``latent_reads_all`` it reads only that patch's tokens.
    """
    torch.manual_seed(0)
    encoder = _joint_encoder(
        joint_depth=2, latent_reads_all=latent_reads_all, pixel_latents=True
    ).eval()
    sample = _sample()  # 8x8 pixels
    captured: dict[str, torch.Tensor] = {}
    original = encoder.perceiver._attention_masks

    def spy(
        cell_id: torch.Tensor, is_latent: torch.Tensor, valid: torch.Tensor
    ) -> dict[str, Any]:
        captured["cell_id"], captured["is_latent"] = cell_id, is_latent
        return original(cell_id, is_latent, valid)

    encoder.perceiver._attention_masks = spy  # type: ignore[method-assign]
    with torch.no_grad():
        out = encoder(sample, patch_size=2, input_res=10)
    assert out["registers"].shape == (2, 8, 8, 32)
    assert out["register_positions"].shape == (2, 64, 2)
    assert torch.isfinite(out["registers"]).all()
    # Pixel centres: patch i sits at i * spacing, its two pixels at (i -/+ 0.25) * s.
    rows = out["register_positions"][0, ::8, 0]
    spacing = rows[2] - rows[0]  # two pixels = one patch
    torch.testing.assert_close(rows[1] - rows[0], spacing / 2)
    torch.testing.assert_close(rows[0], -0.25 * spacing)
    # Latent cell ids: the 4x4 patch grid, each id repeated over its 2x2 pixels.
    latent_cells = captured["cell_id"][0][captured["is_latent"][0]].reshape(8, 8)
    expected = (torch.arange(8)[:, None] // 2) * 4 + torch.arange(8)[None, :] // 2
    assert torch.equal(latent_cells, expected)


def test_supervision_head_spatial_unfold_override() -> None:
    """spatial_unfold replaces max_patch_size as the per-cell unfold factor."""
    from olmoearth_pretrain.nn.supervision_head import SupervisionHeadConfig

    assert SupervisionHeadConfig(spatial_unfold=1).build(32, 4).max_patch_size == 1
    assert SupervisionHeadConfig().build(32, 4).max_patch_size == 4
    with pytest.raises(ValueError, match="spatial_unfold"):
        SupervisionHeadConfig(spatial_unfold=0)


def test_latent_stride_equal_to_patch_size_is_the_patch_grid() -> None:
    """Latents every ``patch_size`` pixels reproduce the patch-latent model exactly."""
    torch.manual_seed(0)
    patch_model = _joint_encoder(joint_depth=2, latent_reads_all=True).eval()
    torch.manual_seed(0)
    strided = _joint_encoder(
        joint_depth=2, latent_reads_all=True, pixel_latents=True, eval_latent_stride=2
    ).eval()
    strided.load_state_dict(patch_model.state_dict())
    sample = _sample()
    with torch.no_grad():
        a = patch_model(sample, patch_size=2, input_res=10)
        b = strided(sample, patch_size=2, input_res=10)
    torch.testing.assert_close(a["registers"], b["registers"])
    torch.testing.assert_close(a["register_positions"], b["register_positions"])


def test_random_latent_stride_respects_divisors_and_budget() -> None:
    """Training draws only divisors of the patch size within the latent budget.

    The patch stride is always allowed, and evaluation uses the fixed eval stride.
    """
    encoder = _joint_encoder(
        joint_depth=1,
        latent_reads_all=True,
        pixel_latents=True,
        random_latent_stride=True,
        max_latents=64,
    )
    module = encoder.perceiver
    assert isinstance(module, JointLatentTransformer)
    module.train()
    torch.manual_seed(0)
    # 4x4 patches at ps4: stride 1 -> 256 latents (over budget), 2 -> 64, 4 -> 16.
    assert {module.choose_latent_stride((4, 4), 4) for _ in range(200)} == {2, 4}
    # ps3: divisors 1 and 3; 2x2 patches at stride 1 = 36 latents, within budget.
    assert {module.choose_latent_stride((2, 2), 3) for _ in range(200)} == {1, 3}
    # A grid too large for any sub-patch stride still gets the patch stride.
    assert {module.choose_latent_stride((16, 16), 2) for _ in range(50)} == {2}
    module.eval()
    assert module.choose_latent_stride((4, 4), 4) == 1
    with torch.no_grad():
        out = encoder(_sample(), patch_size=2, input_res=10)
    assert out["registers"].shape == (2, 8, 8, 32)  # eval stride 1 = per pixel
    with pytest.raises(ValueError, match="max_latents"):
        _joint_encoder(pixel_latents=True, random_latent_stride=True)


def test_token_mlp_ratio_gives_tokens_their_own_light_mlp() -> None:
    """Tokens get a separate MLP of the given width.

    With weights copied from the shared MLP at the same width it reproduces the
    shared-MLP model exactly.
    """
    torch.manual_seed(0)
    shared = _joint_encoder(joint_depth=2, latent_reads_all=True).eval()
    torch.manual_seed(0)
    split = _joint_encoder(
        joint_depth=2, latent_reads_all=True, token_mlp_ratio=2.0
    ).eval()
    module = split.perceiver
    assert isinstance(module, JointLatentTransformer)
    assert module.token_mlps is not None and module.token_norms is not None
    # 32-d embedding, ratio 2 -> hidden 64 for tokens; latents keep the block's 2.0 * 32.
    assert module.token_mlps[0].fc1.out_features == 64
    split.load_state_dict(shared.state_dict(), strict=False)
    with torch.no_grad():
        for blk, norm, mlp in zip(
            module.joint_blocks, module.token_norms, module.token_mlps
        ):
            norm.load_state_dict(blk.norm2.state_dict())
            mlp.load_state_dict(blk.mlp.state_dict())
    sample = _sample()
    with torch.no_grad():
        a = shared(sample, patch_size=2, input_res=10)["registers"]
        b = split(sample, patch_size=2, input_res=10)["registers"]
    torch.testing.assert_close(a, b)
    light = _joint_encoder(joint_depth=1, token_mlp_ratio=0.5).perceiver
    assert isinstance(light, JointLatentTransformer) and light.token_mlps is not None
    assert light.token_mlps[0].fc1.out_features == 16
    with pytest.raises(ValueError, match="token_mlp_ratio"):
        _joint_encoder(joint_depth=1, token_mlp=False, token_mlp_ratio=1.0)
