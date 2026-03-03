"""Unit tests for the flexi_patch_embed module."""

import pytest
import torch
from einops import rearrange

from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.nn.flexi_patch_embed import FlexiPatchEmbed
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
        patch_size_at_16=patch_size_at_16,
        in_chans=in_chans,
        embedding_size=embedding_size,
        use_linear_patch_embed=False,
    )
    torch.manual_seed(seed)
    embed_linear = FlexiPatchEmbed(
        modality_spec=modality,
        patch_size_at_16=patch_size_at_16,
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
        patch_size_at_16=patch_size_at_16,
        in_chans=in_chans,
        embedding_size=embedding_size,
        use_linear_patch_embed=False,
    )
    embed_linear = FlexiPatchEmbed(
        modality_spec=modality,
        patch_size_at_16=patch_size_at_16,
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

    p_h, _ = embed_conv.patch_size
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
        patch_size_at_16=patch_size_at_16,
        in_chans=in_chans,
        embedding_size=embedding_size,
        use_linear_patch_embed=False,
    )

    p_h, _ = embed_conv.patch_size
    with torch.no_grad():
        embed_conv.proj.weight.zero_()
        embed_conv.proj.bias.zero_()
        for i in range(min(embedding_size, in_chans)):
            embed_conv.proj.weight[i, i, :, :] = 1.0

    embed_linear = FlexiPatchEmbed(
        modality_spec=modality,
        patch_size_at_16=patch_size_at_16,
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

    p_h, _ = embed_conv.patch_size
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
