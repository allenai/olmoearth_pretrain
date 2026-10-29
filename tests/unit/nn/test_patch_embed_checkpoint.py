"""checkpoint_activations in FlexiPatchEmbed: same outputs and gradients."""

import copy

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.nn.flexi_patch_embed import FlexiPatchEmbed


@pytest.mark.parametrize("patch_size", [1, 2, 8])
def test_checkpointed_patch_embed_matches(patch_size: int) -> None:
    """naip_10 (tile factor 4) with a per-pixel MLP: identical forward and grads."""
    torch.manual_seed(0)
    plain = FlexiPatchEmbed(
        modality_spec=Modality.NAIP_10,
        base_patch_size_at_16=8,
        in_chans=4,
        embedding_size=32,
        patch_embed_hidden_sizes=[16],
    )
    ckpt = copy.deepcopy(plain)
    ckpt.checkpoint_activations = True
    hw = 3
    x = torch.randn(2, hw * patch_size * 4, hw * patch_size * 4, 4)
    out_plain = plain(x, patch_size)
    out_ckpt = ckpt(x, patch_size)
    torch.testing.assert_close(out_ckpt, out_plain)
    assert out_plain.shape == (2, hw, hw, 32)
    weight = torch.randn_like(out_plain)
    (out_plain * weight).sum().backward()
    (out_ckpt * weight).sum().backward()
    for (name, p_plain), (_, p_ckpt) in zip(
        plain.named_parameters(), ckpt.named_parameters()
    ):
        torch.testing.assert_close(p_ckpt.grad, p_plain.grad, msg=name)
