"""The encoder keeps its attention mask outside training (padding of ragged batches)."""

from types import SimpleNamespace

import torch

from olmoearth_pretrain.nn.flexi_vit import Encoder


def test_attn_mask_kept_in_eval() -> None:
    """Eval mode keeps the mask; only fast_pass drops it."""
    mask = torch.tensor([[True, True, False]])
    for training in (True, False):
        module = SimpleNamespace(training=training)
        assert Encoder._maybe_get_attn_mask(module, mask, fast_pass=False) is mask  # type: ignore[arg-type]
        assert Encoder._maybe_get_attn_mask(module, mask, fast_pass=True) is None  # type: ignore[arg-type]
