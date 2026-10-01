"""Tests for grouped self-attention blocks and the factorized encoder."""

import numpy as np
import pytest
import torch

from olmoearth_pretrain.data.collate import collate_single_masked_batched
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.datatypes import (
    MaskedOlmoEarthSample,
    MaskValue,
    OlmoEarthSample,
)
from olmoearth_pretrain.nn.attention import Block, DropPath, build_token_groups
from olmoearth_pretrain.nn.flexi_vit import EncoderConfig
from olmoearth_pretrain.train.masking import MaskingConfig

torch.set_default_device("cpu")

DIM, HEADS = 16, 2


def _block(drop_path: float = 0.0) -> Block:
    return Block(
        DIM,
        HEADS,
        mlp_ratio=2.0,
        qkv_bias=True,
        drop_path=drop_path,
        position_encoding="rope_3d_mixed",
    )


def test_build_token_groups_layout() -> None:
    """Groups gather each key's tokens; token_position restores the packed order."""
    key = torch.tensor([2, 0, 2, 1, 0, 2])
    sample = torch.zeros(6, dtype=torch.long)
    groups = build_token_groups(key, sample, batch_size=1)
    assert groups.index.shape == (3, 3)
    assert groups.valid.sum(dim=1).tolist() == [2, 1, 3]
    # Every packed token appears exactly once.
    assert sorted(groups.index[groups.valid].tolist()) == list(range(6))
    for g, k in enumerate([0, 1, 2]):
        members = groups.index[g][groups.valid[g]]
        assert (key[members] == k).all()
    packed = torch.arange(6.0)
    regrouped = torch.cat([packed, packed.new_zeros(1)])[groups.index]
    torch.testing.assert_close(regrouped.reshape(-1)[groups.token_position], packed)


def test_grouped_block_one_group_per_sample_matches_full_attention(
    set_random_seeds: None,
) -> None:
    """With one group per sample, grouped attention is ordinary attention."""
    block = _block().eval()
    x = torch.randn(2, 5, DIM)
    positions = torch.randn(2, 5, 3)
    expected = block(x=x, rope_positions=positions)

    sample = torch.arange(2).repeat_interleave(5)
    groups = build_token_groups(sample, sample, batch_size=2)
    actual = block(
        x=x.reshape(10, DIM),
        rope_positions=positions.reshape(10, 3),
        token_groups=groups,
    )
    torch.testing.assert_close(actual.reshape(2, 5, DIM), expected)


def test_grouped_block_isolates_groups(set_random_seeds: None) -> None:
    """Changing one group's tokens leaves every other group's output unchanged."""
    block = _block().eval()
    key = torch.tensor([0, 1, 0, 1, 2, 2])
    groups = build_token_groups(key, torch.zeros(6, dtype=torch.long), batch_size=1)
    x = torch.randn(6, DIM)
    positions = torch.randn(6, 3)
    base = block(x=x, rope_positions=positions, token_groups=groups)
    perturbed = x.clone()
    perturbed[key == 0] += 1.0
    out = block(x=perturbed, rope_positions=positions, token_groups=groups)
    torch.testing.assert_close(out[key != 0], base[key != 0])
    assert not torch.allclose(out[key == 0], base[key == 0])


def test_drop_path_is_per_sample_on_packed_tokens(set_random_seeds: None) -> None:
    """Packed tokens of one sample are all kept or all dropped together."""
    drop = DropPath(0.5).train()
    sample = torch.arange(64).repeat_interleave(3)
    groups = build_token_groups(sample, sample, batch_size=64)
    out = drop(torch.ones(sample.shape[0], DIM), groups)
    per_sample = out.reshape(64, 3 * DIM)
    assert ((per_sample == 0).all(dim=1) | (per_sample == 2).all(dim=1)).all()
    assert (per_sample == 0).all(dim=1).any() and (per_sample == 2).all(dim=1).any()


def _encoder_config(attention_mode: str, depth: int) -> EncoderConfig:
    return EncoderConfig(
        supported_modality_names=[
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.WORLDCOVER.name,
        ],
        embedding_size=DIM,
        max_patch_size=8,
        num_heads=HEADS,
        mlp_ratio=2.0,
        depth=depth,
        drop_path=0.0,
        position_encoding="rope_3d_mixed",
        rope_temporal_coordinate_scale=1.0 / 30.0,
        attention_mode=attention_mode,
    )


def _masked_batch(
    patch_size: int, num_timesteps: int = 5
) -> tuple[int, MaskedOlmoEarthSample]:
    rng = np.random.default_rng(0)
    timestamps = np.array(
        [[1 + 3 * t, t % 12, 2022] for t in range(num_timesteps)], dtype=np.int32
    )
    samples = [
        (
            patch_size,
            OlmoEarthSample(
                sentinel2_l2a=rng.standard_normal((8, 8, num_timesteps, 12)).astype(
                    np.float32
                ),
                sentinel1=rng.standard_normal((8, 8, num_timesteps, 2)).astype(
                    np.float32
                ),
                worldcover=rng.standard_normal((8, 8, 1, 1)).astype(np.float32),
                timestamps=timestamps,
            ),
        )
        for _ in range(3)
    ]
    masking = MaskingConfig(strategy_config={"type": "random"}).build()
    return collate_single_masked_batched(
        samples, transform=None, masking_strategy=masking
    )


def test_factorized_matches_full_attention_on_a_single_location(
    set_random_seeds: None,
) -> None:
    """One location: the local (block 0) group is the whole sample."""
    full = _encoder_config("full", depth=1).build()
    factorized = _encoder_config("factorized", depth=1).build()
    factorized.load_state_dict(full.state_dict())
    # Training mode: the full path only applies its padding mask when training;
    # drop path is 0 so both are deterministic.
    full.train()
    factorized.train()
    patch_size, batch = _masked_batch(patch_size=8)  # 8x8 pixels -> 1x1 patch grid
    expected = full(batch, patch_size=patch_size)["tokens_and_masks"]
    actual = factorized(batch, patch_size=patch_size)["tokens_and_masks"]
    for modality in expected.modalities:
        mask = getattr(expected, f"{modality}_mask") == MaskValue.ONLINE_ENCODER.value
        torch.testing.assert_close(
            getattr(actual, modality)[mask], getattr(expected, modality)[mask]
        )


def test_factorized_encoder_forward_backward(set_random_seeds: None) -> None:
    """Multi-location factorized encoder runs, is finite and backpropagates."""
    encoder = _encoder_config("factorized", depth=4).build().train()
    patch_size, batch = _masked_batch(patch_size=2)  # 4x4 patch grid
    output = encoder(batch, patch_size=patch_size)
    tokens = output["tokens_and_masks"]
    loss = output["project_aggregated"].sum()
    for modality in tokens.modalities:
        assert torch.isfinite(getattr(tokens, modality)).all()
        loss = loss + getattr(tokens, modality).sum()
    loss.backward()
    grads = [p.grad for p in encoder.blocks.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_factorized_encoder_config_rejects_unsupported_options() -> None:
    """Flash attention / registers are not supported with factorized attention."""
    config = _encoder_config("factorized", depth=2)
    config.use_flash_attn = True
    with pytest.raises(ValueError, match="factorized"):
        config.validate()
    config = _encoder_config("bogus", depth=2)
    with pytest.raises(ValueError, match="attention_mode"):
        config.validate()
