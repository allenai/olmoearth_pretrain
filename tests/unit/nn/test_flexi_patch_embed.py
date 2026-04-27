"""Unit tests for the flexi_patch_embed module."""

import pytest
import torch
from einops import rearrange

from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.nn.flexi_patch_embed import (
    CrossAttentionPatchEmbed,
    FlexiPatchEmbed,
)
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig


def _make_seeded_conv_and_linear_pair(
    modality: ModalitySpec,
    in_chans: int,
    embedding_size: int,
    patch_size_at_16: int,
    seed: int,
) -> tuple[FlexiPatchEmbed, FlexiPatchEmbed]:
    """Build conv/linear pairs from the same RNG seed (no weight copying)."""
    torch.manual_seed(seed)
    embed_conv = FlexiPatchEmbed(
        modality_spec=modality,
        base_patch_size_at_16=patch_size_at_16,
        in_chans=in_chans,
        embedding_size=embedding_size,
        use_linear_patch_embed=False,
    )
    torch.manual_seed(seed)
    embed_linear = FlexiPatchEmbed(
        modality_spec=modality,
        base_patch_size_at_16=patch_size_at_16,
        in_chans=in_chans,
        embedding_size=embedding_size,
        use_linear_patch_embed=True,
    )
    return embed_conv, embed_linear


def _make_weight_matched_conv_and_linear_pair(
    modality: ModalitySpec,
    in_chans: int,
    embedding_size: int,
    patch_size_at_16: int,
) -> tuple[FlexiPatchEmbed, FlexiPatchEmbed]:
    """Build conv/linear pair and explicitly match weights for forward equivalence checks."""
    torch.manual_seed(0)
    embed_conv = FlexiPatchEmbed(
        modality_spec=modality,
        base_patch_size_at_16=patch_size_at_16,
        in_chans=in_chans,
        embedding_size=embedding_size,
        use_linear_patch_embed=False,
    )
    embed_linear = FlexiPatchEmbed(
        modality_spec=modality,
        base_patch_size_at_16=patch_size_at_16,
        in_chans=in_chans,
        embedding_size=embedding_size,
        use_linear_patch_embed=True,
    )
    conv_weight = embed_conv.proj.weight.data
    linear_weight = rearrange(conv_weight, "o c p1 p2 -> o (p1 p2 c)")
    embed_linear.proj.weight.data.copy_(linear_weight)
    embed_linear.proj.bias.data.copy_(embed_conv.proj.bias.data)
    return embed_conv, embed_linear


@pytest.mark.parametrize("patch_size_at_16", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "modality_name",
    ["sentinel2_l2a", "sentinel1", "worldcover"],
)
def test_linear_vs_conv_patch_embed_init_equivalence(
    patch_size_at_16: int, modality_name: str
) -> None:
    """Linear and Conv2d patch projection params initialize identically with same seed."""
    modality = getattr(Modality, modality_name.upper())
    in_chans = modality.num_bands
    embedding_size = 32

    embed_conv, embed_linear = _make_seeded_conv_and_linear_pair(
        modality=modality,
        in_chans=in_chans,
        embedding_size=embedding_size,
        patch_size_at_16=patch_size_at_16,
        seed=123,
    )

    conv_weight_flat = embed_conv.proj.weight.detach().reshape(embedding_size, -1)
    linear_weight = embed_linear.proj.weight.detach()
    assert torch.allclose(conv_weight_flat, linear_weight), (
        f"Weight init mismatch ({modality_name}, ps={patch_size_at_16})"
    )
    assert torch.allclose(embed_conv.proj.bias, embed_linear.proj.bias), (
        f"Bias init mismatch ({modality_name}, ps={patch_size_at_16})"
    )


@pytest.mark.parametrize("patch_size_at_16", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "modality_name",
    ["sentinel2_l2a", "sentinel1", "worldcover"],
)
def test_linear_vs_conv_patch_embed_equivalence(
    patch_size_at_16: int, modality_name: str
) -> None:
    """Linear and Conv2d patch embeddings produce identical outputs with equivalent weights."""
    modality = getattr(Modality, modality_name.upper())
    in_chans = modality.num_bands
    embedding_size = 32

    embed_conv, embed_linear = _make_weight_matched_conv_and_linear_pair(
        modality, in_chans, embedding_size, patch_size_at_16
    )

    p_h, _ = embed_conv.base_patch_size
    H = W = max(p_h * 2, 16)

    x_4d = torch.randn(2, H, W, in_chans)
    out_conv = embed_conv(x_4d)
    out_linear = embed_linear(x_4d)
    assert out_conv.shape == out_linear.shape
    assert torch.allclose(out_conv, out_linear, atol=1e-5), (
        f"4D mismatch ({modality_name}, ps={patch_size_at_16}): "
        f"max diff={(out_conv - out_linear).abs().max().item()}"
    )

    x_5d = torch.randn(2, H, W, 3, in_chans)
    out_conv_5d = embed_conv(x_5d)
    out_linear_5d = embed_linear(x_5d)
    assert out_conv_5d.shape == out_linear_5d.shape
    assert torch.allclose(out_conv_5d, out_linear_5d, atol=1e-5), (
        f"5D mismatch ({modality_name}, ps={patch_size_at_16}): "
        f"max diff={(out_conv_5d - out_linear_5d).abs().max().item()}"
    )


@pytest.mark.parametrize("patch_size_at_16", [1, 2, 4, 8])
def test_linear_vs_conv_spatial_ordering(patch_size_at_16: int) -> None:
    """Verify linear patch embed preserves spatial ordering — no pixel swaps or flips."""
    modality = Modality.SENTINEL2_L2A
    in_chans = modality.num_bands
    embedding_size = 32

    torch.manual_seed(0)
    embed_conv = FlexiPatchEmbed(
        modality_spec=modality,
        base_patch_size_at_16=patch_size_at_16,
        in_chans=in_chans,
        embedding_size=embedding_size,
        use_linear_patch_embed=False,
    )

    p_h, _ = embed_conv.base_patch_size
    with torch.no_grad():
        embed_conv.proj.weight.zero_()
        embed_conv.proj.bias.zero_()
        for i in range(min(embedding_size, in_chans)):
            embed_conv.proj.weight[i, i, :, :] = 1.0

    embed_linear = FlexiPatchEmbed(
        modality_spec=modality,
        base_patch_size_at_16=patch_size_at_16,
        in_chans=in_chans,
        embedding_size=embedding_size,
        use_linear_patch_embed=True,
    )
    conv_weight = embed_conv.proj.weight.data
    linear_weight = rearrange(conv_weight, "o c p1 p2 -> o (p1 p2 c)")
    embed_linear.proj.weight.data.copy_(linear_weight)
    embed_linear.proj.bias.data.copy_(embed_conv.proj.bias.data)

    H = W = max(p_h * 4, 32)
    x = torch.arange(H * W * in_chans, dtype=torch.float32).reshape(1, H, W, in_chans)

    out_conv = embed_conv(x)
    out_linear = embed_linear(x)

    assert torch.allclose(out_conv, out_linear, atol=1e-5), (
        f"Spatial ordering mismatch (ps={patch_size_at_16}): "
        f"max diff={(out_conv - out_linear).abs().max().item()}"
    )
    assert not torch.allclose(out_conv[0, 0, 0], out_conv[0, 0, 1]), (
        "Adjacent patches should differ with spatially-varying input"
    )


@pytest.mark.parametrize("patch_size_at_16", [2, 4, 8])
@pytest.mark.parametrize("runtime_patch_size", [1, 2, 4])
def test_linear_vs_conv_with_flexi_patch_resize(
    patch_size_at_16: int, runtime_patch_size: int
) -> None:
    """Linear and conv match when runtime patch_size differs from base (triggers interpolation)."""
    if runtime_patch_size >= patch_size_at_16:
        pytest.skip("runtime_patch_size must be smaller than base to trigger resize")

    modality = Modality.SENTINEL2_L2A
    in_chans = modality.num_bands
    embedding_size = 32

    embed_conv, embed_linear = _make_weight_matched_conv_and_linear_pair(
        modality, in_chans, embedding_size, patch_size_at_16
    )

    p_h, _ = embed_conv.base_patch_size
    H = W = max(p_h * 4, 32)
    x = torch.randn(2, H, W, 2, in_chans)

    out_conv = embed_conv(x, patch_size=runtime_patch_size)
    out_linear = embed_linear(x, patch_size=runtime_patch_size)

    assert out_conv.shape == out_linear.shape
    assert torch.allclose(out_conv, out_linear, atol=1e-5), (
        f"Flexi resize mismatch (base={patch_size_at_16}, runtime={runtime_patch_size}): "
        f"max diff={(out_conv - out_linear).abs().max().item()}"
    )


def test_encoder_config_all_modalities_patch_embed_init_equivalence() -> None:
    """EncoderConfig with all modalities initializes conv/linear patch embeds equivalently."""
    all_modalities = Modality.names()

    base_kwargs = dict(
        supported_modality_names=all_modalities,
        embedding_size=32,
        num_heads=4,
        depth=2,
        mlp_ratio=2.0,
        max_patch_size=8,
        max_sequence_length=12,
    )

    torch.manual_seed(321)
    conv_encoder = EncoderConfig(
        **base_kwargs,  # type: ignore
        use_linear_patch_embed=False,
    ).build()

    torch.manual_seed(321)
    linear_encoder = EncoderConfig(
        **base_kwargs,  # type: ignore
        use_linear_patch_embed=True,
    ).build()

    compared_modules = 0
    for (
        modality_name,
        conv_modality_modules,
    ) in conv_encoder.patch_embeddings.per_modality_embeddings.items():
        linear_modality_modules = (
            linear_encoder.patch_embeddings.per_modality_embeddings[modality_name]
        )
        for module_name, conv_module in conv_modality_modules.items():
            linear_module = linear_modality_modules[module_name]
            if not isinstance(conv_module, FlexiPatchEmbed):
                continue
            assert isinstance(linear_module, FlexiPatchEmbed)
            compared_modules += 1

            conv_weight_flat = conv_module.proj.weight.detach().reshape(
                conv_module.proj.weight.shape[0], -1
            )
            linear_weight = linear_module.proj.weight.detach()
            assert torch.allclose(conv_weight_flat, linear_weight), (
                f"Init mismatch for {modality_name}/{module_name} weights"
            )
            assert torch.allclose(conv_module.proj.bias, linear_module.proj.bias), (
                f"Init mismatch for {modality_name}/{module_name} bias"
            )

    assert compared_modules > 0, "Expected at least one spatial FlexiPatchEmbed module"


# ---------------------------------------------------------------------------
# CrossAttentionPatchEmbed tests
# ---------------------------------------------------------------------------


def _make_cross_attn_embed(
    in_chans: int = 12,
    embedding_size: int = 32,
    cross_attn_embedding_size: int = 16,
    patch_size_at_16: int = 8,
    band_dropout_rate: float = 0.0,
    random_band_dropout: bool = False,
    per_patch_band_dropout: bool = False,
) -> CrossAttentionPatchEmbed:
    """Helper to build a CrossAttentionPatchEmbed for testing."""
    modality = Modality.SENTINEL2_L2A
    return CrossAttentionPatchEmbed(
        modality_spec=modality,
        base_patch_size_at_16=patch_size_at_16,
        in_chans=in_chans,
        embedding_size=embedding_size,
        cross_attn_embedding_size=cross_attn_embedding_size,
        band_dropout_rate=band_dropout_rate,
        random_band_dropout=random_band_dropout,
        per_patch_band_dropout=per_patch_band_dropout,
    )


@pytest.mark.parametrize("patch_size_at_16", [4, 8])
def test_cross_attn_patch_embed_output_shape_4d(patch_size_at_16: int) -> None:
    """CrossAttentionPatchEmbed produces correct output shape for 4D input (no time)."""
    in_chans = 12
    embedding_size = 32
    embed = _make_cross_attn_embed(
        in_chans=in_chans,
        embedding_size=embedding_size,
        patch_size_at_16=patch_size_at_16,
    )
    p_h = patch_size_at_16  # image_tile_size_factor=1 for S2
    H = W = p_h * 4
    B = 2
    x = torch.randn(B, H, W, in_chans)
    out = embed(x)
    expected_h = H // p_h
    expected_w = W // p_h
    assert out.shape == (B, expected_h, expected_w, embedding_size)


@pytest.mark.parametrize("patch_size_at_16", [4, 8])
def test_cross_attn_patch_embed_output_shape_5d(patch_size_at_16: int) -> None:
    """CrossAttentionPatchEmbed produces correct output shape for 5D input (with time)."""
    in_chans = 12
    embedding_size = 32
    T = 3
    embed = _make_cross_attn_embed(
        in_chans=in_chans,
        embedding_size=embedding_size,
        patch_size_at_16=patch_size_at_16,
    )
    p_h = patch_size_at_16
    H = W = p_h * 4
    B = 2
    x = torch.randn(B, H, W, T, in_chans)
    out = embed(x)
    expected_h = H // p_h
    expected_w = W // p_h
    assert out.shape == (B, expected_h, expected_w, T, embedding_size)


def test_cross_attn_patch_embed_eval_no_dropout() -> None:
    """In eval mode, band dropout mask is None (all bands participate)."""
    embed = _make_cross_attn_embed(band_dropout_rate=0.5)
    embed.eval()
    mask = embed._generate_band_dropout_mask(
        batch_size=4, num_bands=12, device=torch.device("cpu")
    )
    assert mask is None


def test_cross_attn_patch_embed_train_no_dropout_when_rate_zero() -> None:
    """With rate=0, no dropout mask is generated even in training."""
    embed = _make_cross_attn_embed(band_dropout_rate=0.0)
    embed.train()
    mask = embed._generate_band_dropout_mask(
        batch_size=4, num_bands=12, device=torch.device("cpu")
    )
    assert mask is None


def test_cross_attn_patch_embed_train_no_dropout_single_band() -> None:
    """With a single band, no dropout mask is generated."""
    embed = _make_cross_attn_embed(in_chans=1, band_dropout_rate=0.5)
    embed.train()
    mask = embed._generate_band_dropout_mask(
        batch_size=4, num_bands=1, device=torch.device("cpu")
    )
    assert mask is None


def test_cross_attn_patch_embed_at_least_one_band_kept() -> None:
    """Even with very high dropout rate, at least one band is always kept."""
    embed = _make_cross_attn_embed(band_dropout_rate=0.99)
    embed.train()
    torch.manual_seed(42)
    for _ in range(50):
        mask = embed._generate_band_dropout_mask(
            batch_size=16, num_bands=12, device=torch.device("cpu")
        )
        assert mask is not None
        assert mask.any(dim=1).all(), "Every sample must keep at least one band"


def test_cross_attn_patch_embed_dropout_masks_some_bands() -> None:
    """With moderate dropout rate, some bands should be masked out."""
    embed = _make_cross_attn_embed(band_dropout_rate=0.5)
    embed.train()
    torch.manual_seed(0)
    mask = embed._generate_band_dropout_mask(
        batch_size=64, num_bands=12, device=torch.device("cpu")
    )
    assert mask is not None
    # With rate=0.5 and 64*12 = 768 entries, it's extremely unlikely all are True
    assert not mask.all(), "Expected some bands to be dropped"


def test_cross_attn_patch_embed_per_image_dropout_consistency() -> None:
    """With per_patch_band_dropout=False, all spatial patches in the same image share the mask."""
    in_chans = 6
    embedding_size = 16
    patch_size = 4
    embed = _make_cross_attn_embed(
        in_chans=in_chans,
        embedding_size=embedding_size,
        cross_attn_embedding_size=8,
        patch_size_at_16=patch_size,
        band_dropout_rate=0.5,
        per_patch_band_dropout=False,
    )
    embed.train()
    H = W = patch_size * 4  # 4x4 = 16 spatial patches
    B = 3

    # Run forward twice with same seed to capture the mask behavior.
    # We verify indirectly: with the same input at all spatial positions,
    # per-image dropout should produce identical outputs across patches.
    # (With per-patch dropout they would differ due to different masks.)
    x_uniform = torch.ones(B, H, W, in_chans)
    torch.manual_seed(123)
    out = embed(x_uniform)
    # All spatial patches within each image should be identical
    for b in range(B):
        patch_00 = out[b, 0, 0]
        for h in range(out.shape[1]):
            for w in range(out.shape[2]):
                assert torch.allclose(out[b, h, w], patch_00, atol=1e-5), (
                    f"Patches differ at image {b}, position ({h},{w})"
                )


def test_cross_attn_patch_embed_per_patch_dropout_varies() -> None:
    """With per_patch_band_dropout=True, different patches can get different masks."""
    in_chans = 6
    embedding_size = 16
    patch_size = 4
    embed = _make_cross_attn_embed(
        in_chans=in_chans,
        embedding_size=embedding_size,
        cross_attn_embedding_size=8,
        patch_size_at_16=patch_size,
        band_dropout_rate=0.5,
        per_patch_band_dropout=True,
    )
    embed.train()
    H = W = patch_size * 4
    B = 2
    x_uniform = torch.ones(B, H, W, in_chans)

    # Run many times; at least once, patches should differ
    found_difference = False
    for seed in range(50):
        torch.manual_seed(seed)
        out = embed(x_uniform)
        patch_00 = out[0, 0, 0]
        for h in range(out.shape[1]):
            for w in range(out.shape[2]):
                if not torch.allclose(out[0, h, w], patch_00, atol=1e-5):
                    found_difference = True
                    break
            if found_difference:
                break
        if found_difference:
            break
    assert found_difference, (
        "Expected per-patch dropout to produce varying outputs across spatial positions"
    )
