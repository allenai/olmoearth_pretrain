"""Test the train utils."""

import pytest
import torch

from helios.data.dataset import HeliosSample
from helios.train.utils import split_batch


@pytest.mark.parametrize("microbatch_size", [1, 2, 5])
def test_split_batch(microbatch_size: int) -> None:
    """Test the split_batch function."""
    B, H, W, T, D = 10, 2, 2, 2, 4
    sentinel2_tokens = torch.zeros(B, H, W, T, D)
    latlon_tokens = torch.randn(B, 1, D)
    x = {"sentinel2_l2a": sentinel2_tokens, "latlon": latlon_tokens}
    batch = HeliosSample(**x)
    micro_batches = split_batch(batch, microbatch_size)
    assert len(micro_batches) == (B + microbatch_size - 1) // microbatch_size
    for i, micro_batch in enumerate(micro_batches):
        if i == len(micro_batches) - 1:
            microbatch_size = B - i * microbatch_size
        assert micro_batch.batch_size == microbatch_size
        assert micro_batch.sentinel2_l2a is not None
        assert micro_batch.sentinel2_l2a.shape == (microbatch_size, H, W, T, D)
        assert micro_batch.latlon is not None
        assert micro_batch.latlon.shape == (microbatch_size, 1, D)
