"""Tests for the Perceiver's neighbourhood token mixing (``token_mix_layout``)."""

import pytest
import torch
import torch.nn.functional as F

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_vit import (
    Encoder,
    Perceiver,
    PerceiverConfig,
)
from olmoearth_pretrain.nn.joint_latent import (
    neighbourhood_attention_allowed,
    sort_tokens_by_cell,
    token_mix_attention_kwargs,
)
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, MaskValue


def test_neighbourhood_mask_matches_the_written_out_rule() -> None:
    """The vectorised mask equals the rule written as loops.

    Spatial tokens see valid keys whose cell is within ``radius`` in row and column;
    non-spatial tokens (cell -1) see each other; padding is never a key; every token
    sees itself.
    """
    n_w = 3
    cells = torch.tensor([[0, 0, 1, 2, 4, 4, 5, 8, 6, -1, -1, 3, 7]])
    valid = torch.ones_like(cells, dtype=torch.bool)
    valid[0, 11:] = False
    for radius in (0, 1):
        allowed = neighbourhood_attention_allowed(cells, valid, n_w, radius)[0]
        for q in range(cells.shape[1]):
            for kv in range(cells.shape[1]):
                cq, ck = int(cells[0, q]), int(cells[0, kv])
                if cq >= 0 and ck >= 0:
                    kind = (
                        abs(cq // n_w - ck // n_w) <= radius
                        and abs(cq % n_w - ck % n_w) <= radius
                    )
                else:
                    kind = cq < 0 and ck < 0
                expect = (bool(valid[0, kv]) and kind) or q == kv
                assert bool(allowed[q, kv]) == expect, (radius, q, kv)
    # 3x3 around the centre cell (4) covers the whole 3x3 grid; radius 0 only cell 4.
    centre = 4
    assert neighbourhood_attention_allowed(cells, valid, n_w, 1)[0, centre, :9].all()
    assert neighbourhood_attention_allowed(cells, valid, n_w, 0)[
        0, centre, :9
    ].tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
    ]


def test_sort_tokens_by_cell_makes_cells_contiguous_and_is_stable() -> None:
    """Cells come out contiguous in row-major order, padding last, ties in order."""
    cells = torch.tensor([[3, 0, 3, -1, 0, 2, 1]])
    valid = torch.tensor([[True, True, True, True, True, False, True]])
    tokens = torch.arange(7.0).view(1, 7, 1)
    positions = torch.arange(7.0).view(1, 7, 1).expand(1, 7, 3)
    t, p, v, c = sort_tokens_by_cell(tokens, positions, valid, cells)
    assert c[0].tolist() == [-1, 0, 0, 1, 3, 3, 2]
    assert t[0, :, 0].tolist() == [3.0, 1.0, 4.0, 6.0, 0.0, 2.0, 5.0]
    assert v[0].tolist() == [True] * 6 + [False]
    torch.testing.assert_close(p[..., 0], t[..., 0])


def _mixing_perceiver(layout: str = "MR", **kwargs: object) -> Perceiver:
    return Perceiver(
        encoder_embedding_size=32,
        register_dim=32,
        num_heads=4,
        mlp_ratio=2.0,
        latent_transformer_depth=layout.count("R"),
        use_2d_rope=True,
        per_depth_read_proj=True,
        attn_dim=32,
        time_rope_encoding="rope_3d_mixed",
        token_mix_layout=layout,
        **kwargs,  # type: ignore[arg-type]
    ).eval()


def _grid_inputs(
    n_h: int = 4, n_w: int = 4, T: int = 3, B: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokens ordered (h, w, t) with (t, row, col) positions and their cell ids."""
    hh, ww, tt = torch.meshgrid(
        torch.arange(n_h), torch.arange(n_w), torch.arange(T), indexing="ij"
    )
    positions = torch.stack([tt * 30.0, hh.float(), ww.float()], -1).reshape(1, -1, 3)
    cells = (hh * n_w + ww).reshape(1, -1)
    tokens = torch.randn(B, positions.shape[1], 32)
    return tokens, positions.expand(B, -1, -1), cells.expand(B, -1)


def test_one_mixing_block_only_reaches_the_neighbourhood() -> None:
    """Perturbing cell (0, 0) changes cell (1, 1) after one 3x3 block, not (2, 2)."""
    torch.manual_seed(0)
    perceiver = _mixing_perceiver()
    blk = perceiver.token_mix_blocks[0]
    tokens, positions, cells = _grid_inputs()
    kwargs = token_mix_attention_kwargs(cells, None, n_w=4, radius=1)
    base = blk(x=tokens, rope_positions=positions, **kwargs)
    bumped = tokens.clone()
    # A random bump: a constant shift would be removed by the block's pre-norm.
    bumped[0, cells[0] == 0] += torch.randn_like(bumped[0, cells[0] == 0])
    out = blk(x=bumped, rope_positions=positions, **kwargs)
    changed = (out - base).abs().amax(-1)[0]
    assert changed[cells[0] == 5].min() > 1e-4  # cell (1, 1): a neighbour
    torch.testing.assert_close(  # cell (2, 2): two cells away
        out[0, cells[0] == 10], base[0, cells[0] == 10]
    )


def test_mixing_perceiver_output_is_invariant_to_token_order() -> None:
    """Shuffling the input tokens (with their positions, mask and cells) is a no-op.

    Pins the cell sort: mixing and reads are permutation-equivariant over tokens, so
    the register grid must not depend on the order the encoder hands them in.
    """
    torch.manual_seed(0)
    perceiver = _mixing_perceiver("MRMRL", token_mix_radius=1)
    tokens, positions, cells = _grid_inputs(B=2)
    valid = torch.ones(cells.shape, dtype=torch.bool)
    valid[1, -5:] = False
    out, _ = perceiver(
        tokens, positions, valid, spatial_grid=(4, 4), cell_ids=cells.clone()
    )
    perm = torch.randperm(cells.shape[1])
    out_perm, _ = perceiver(
        tokens[:, perm],
        positions[:, perm],
        valid[:, perm],
        spatial_grid=(4, 4),
        grid_extent_positions=positions,
        cell_ids=cells[:, perm].clone(),
    )
    torch.testing.assert_close(out_perm, out, atol=1e-5, rtol=1e-5)


def test_default_perceiver_has_no_mixing_parameters() -> None:
    """Without a layout the module is the plain Perceiver (checkpoint-compatible)."""
    perceiver = Perceiver(
        encoder_embedding_size=32,
        register_dim=32,
        num_heads=4,
        mlp_ratio=2.0,
        latent_transformer_depth=2,
        use_2d_rope=True,
    )
    names = [n for n, _ in perceiver.named_parameters()]
    assert not any("token_mix" in n or "extra_latent" in n for n in names)


def _mixing_encoder(**config_kwargs: object) -> Encoder:
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
        perceiver_config=PerceiverConfig(
            register_dim=32,
            per_depth_read_proj=True,
            attn_dim=32,
            read_time_rope=True,
            student_dims=[16, 8],
            **config_kwargs,  # type: ignore[arg-type]
        ),
    )


def _sample(B: int = 2, H: int = 8, W: int = 8, T: int = 3) -> MaskedOlmoEarthSample:
    nb2 = Modality.SENTINEL2_L2A.num_bands
    nb1 = Modality.SENTINEL1.num_bands
    timestamps = torch.tensor([[[1, 0, 2020], [1, 3, 2020], [1, 6, 2020]]] * B).long()
    mask2 = torch.full((B, H, W, T, nb2), MaskValue.ONLINE_ENCODER.value)
    mask2[:, :2, :, 1] = MaskValue.DECODER.value  # some tokens removed in training
    return MaskedOlmoEarthSample(
        sentinel2_l2a=torch.randn(B, H, W, T, nb2),
        sentinel2_l2a_mask=mask2,
        sentinel1=torch.randn(B, H, W, T, nb1),
        sentinel1_mask=torch.full((B, H, W, T, nb1), MaskValue.ONLINE_ENCODER.value),
        timestamps=timestamps,
    )


@pytest.mark.parametrize(
    "config",
    [
        dict(latent_depth=2, token_mix_layout="MRMRL"),
        dict(latent_depth=2, token_mix_layout="MRMRL", token_mix_radius=0),
        dict(
            latent_depth=2,
            token_mix_layout="MMRMRL",
            token_mix_dim=16,
            token_mix_num_heads=2,
        ),
    ],
)
@pytest.mark.parametrize("patch_size", [1, 2])
def test_encoder_with_token_mixing_trains_and_evaluates(
    config: dict, patch_size: int
) -> None:
    """Grid shape, student output, gradients into every mixing block, both passes."""
    torch.manual_seed(0)
    encoder = _mixing_encoder(**config)
    sample = _sample()
    encoder.train()
    out = encoder(sample, patch_size=patch_size)
    registers = out["registers"]
    n = 8 // patch_size
    assert registers.shape == (2, n, n, 32)
    assert out["student_registers"].shape[-1] == 16
    registers.sum().backward()
    perceiver = encoder.perceiver
    assert isinstance(perceiver, Perceiver)
    for blk in perceiver.token_mix_blocks:
        grads = [p.grad for p in blk.parameters() if p.grad is not None]
        assert grads and any(g.abs().sum() > 0 for g in grads)
    if config.get("token_mix_dim"):
        assert perceiver.token_mix_in is not None
    encoder.eval()
    with torch.no_grad():
        eval_out = encoder(sample, patch_size=patch_size, fast_pass=True)
    assert eval_out["registers"].shape == (2, n, n, 32)


@pytest.mark.parametrize(
    "bad, message",
    [
        (dict(latent_depth=2, token_mix_layout="MRL"), "reads"),
        (
            dict(latent_depth=1, token_mix_layout="MR", read_time_rope=False),
            "read_time_rope",
        ),
        (dict(latent_depth=1, token_mix_layout="MRX"), "M, R and L"),
        (dict(latent_depth=1, token_mix_radius=1), "need token_mix_layout"),
    ],
)
def test_bad_token_mix_configs_are_rejected(bad: dict, message: str) -> None:
    """Layout/read-count mismatches and orphan settings fail at build time."""
    base = dict(
        register_dim=32, per_depth_read_proj=True, attn_dim=32, read_time_rope=True
    )
    base.update(bad)
    config = PerceiverConfig(**base)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=message):
        config.validate(encoder_num_heads=4, position_encoding="rope_3d_mixed")
        config.build(
            encoder_embedding_size=32,
            encoder_num_heads=4,
            mlp_ratio=2.0,
            position_encoding="rope_3d_mixed",
            rope_base=10000.0,
            qk_norm=False,
        )


def test_mixing_block_runs_through_the_block_mask_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CUDA call path (``block_mask`` via ``Block``) matches the dense path.

    FlexAttention needs CUDA, so the kernel is swapped for dense SDPA over a boolean
    mask passed in the ``block_mask`` slot; what this pins is the kwarg plumbing that
    the GPU path uses and the CPU path never touches (it once raised
    ``Block.forward() got an unexpected keyword argument 'block_mask'``).
    """
    import olmoearth_pretrain.nn.joint_latent as jl

    def dense_flex(q, k, v, block_mask):  # type: ignore[no-untyped-def]
        return F.scaled_dot_product_attention(q, k, v, attn_mask=block_mask)

    monkeypatch.setattr(jl, "flex_attention_cuda", dense_flex)
    torch.manual_seed(0)
    blk = _mixing_perceiver().token_mix_blocks[0]
    tokens, positions, cells = _grid_inputs()
    dense = token_mix_attention_kwargs(cells, None, n_w=4, radius=1)
    via_dense = blk(x=tokens, rope_positions=positions, **dense)
    via_block_mask = blk(
        x=tokens, rope_positions=positions, block_mask=dense["attn_mask"]
    )
    torch.testing.assert_close(via_block_mask, via_dense)


# --- Pixel (sub-patch) latents on the Perceiver ----------------------------------------


def test_perceiver_latent_stride_equal_to_patch_size_is_the_patch_grid() -> None:
    """Latents every ``patch_size`` pixels reproduce the patch-latent Perceiver exactly."""
    torch.manual_seed(0)
    patch_model = _mixing_encoder(latent_depth=2, token_mix_layout="MRMR").eval()
    strided = _mixing_encoder(
        latent_depth=2,
        token_mix_layout="MRMR",
        pixel_latents=True,
        eval_latent_stride=2,
    ).eval()
    strided.load_state_dict(patch_model.state_dict())
    sample = _sample()
    with torch.no_grad():
        a = patch_model(sample, patch_size=2, input_res=10)
        b = strided(sample, patch_size=2, input_res=10)
    torch.testing.assert_close(a["registers"], b["registers"])
    torch.testing.assert_close(a["register_positions"], b["register_positions"])


def test_perceiver_pixel_latents_train_and_eval_grids() -> None:
    """Stride-1 eval gives one latent per pixel; training strides stay in budget.

    Gradients reach the mixing blocks and the reads at a sub-patch stride.
    """
    torch.manual_seed(0)
    encoder = _mixing_encoder(
        latent_depth=2,
        token_mix_layout="MMRR",
        pixel_latents=True,
        random_latent_stride=True,
        max_latents=64,
    )
    sample = _sample()
    encoder.eval()
    with torch.no_grad():
        out = encoder(sample, patch_size=2, input_res=10, fast_pass=True)
    assert out["registers"].shape == (2, 8, 8, 32)  # 4x4 patches x 2x2 pixels
    assert out["register_positions"].shape == (2, 64, 2)
    encoder.train()
    shapes = set()
    for _ in range(20):
        out = encoder(sample, patch_size=2, input_res=10)
        shapes.add(tuple(out["registers"].shape[1:3]))
    # 4x4 patches at ps2: stride 1 = 64 latents (== budget), stride 2 = 16.
    assert shapes <= {(8, 8), (4, 4)} and (8, 8) in shapes
    out["registers"].sum().backward()
    perceiver = encoder.perceiver
    assert isinstance(perceiver, Perceiver)
    for blk in list(perceiver.token_mix_blocks) + list(perceiver.read_blocks):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in blk.parameters()
        )


def test_latent_stride_bias_prefers_fine_strides_and_keeps_uniform_default() -> None:
    """Bias 0 is the old uniform draw (same RNG path); a large bias picks the finest."""
    from olmoearth_pretrain.nn.joint_latent import choose_latent_stride

    kwargs = dict(
        training=True,
        spatial_grid=(2, 2),
        patch_size=4,
        random_latent_stride=True,
        max_latents=1024,
        eval_latent_stride=1,
    )
    torch.manual_seed(0)
    uniform = [choose_latent_stride(**kwargs) for _ in range(300)]  # type: ignore[arg-type]
    torch.manual_seed(0)
    allowed = [1, 2, 4]
    old = [allowed[int(torch.randint(3, (1,)).item())] for _ in range(300)]
    assert uniform == old
    torch.manual_seed(0)
    biased = [
        choose_latent_stride(**kwargs, stride_bias=8.0)  # type: ignore[arg-type]
        for _ in range(300)
    ]
    assert biased.count(1) > 250 and set(biased) <= {1, 2, 4}
    torch.manual_seed(0)
    mild = [
        choose_latent_stride(**kwargs, stride_bias=2.0)  # type: ignore[arg-type]
        for _ in range(3000)
    ]
    # Weights 1 : 1/4 : 1/16 -> about 76% / 19% / 5%.
    assert 0.70 < mild.count(1) / 3000 < 0.82


def test_pixel_latent_settings_need_pixel_latents() -> None:
    """Stride settings without pixel latents fail validation."""
    config = PerceiverConfig(register_dim=32, latent_stride_bias=2.0)
    with pytest.raises(ValueError, match="pixel_latents"):
        config.validate(encoder_num_heads=4, position_encoding="rope_3d_mixed")
    config = PerceiverConfig(
        register_dim=32, pixel_latents=True, random_latent_stride=True
    )
    with pytest.raises(ValueError, match="max_latents"):
        config.validate(encoder_num_heads=4, position_encoding="rope_3d_mixed")
