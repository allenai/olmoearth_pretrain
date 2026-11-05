"""Tests for the PANGAEA rslearn export's window tiling."""

import numpy as np
import pytest

from olmoearth_pretrain.evals.datasets.pangaea_rslearn_export import window_offsets


@pytest.mark.parametrize("size", [128, 256, 300, 497, 807])
def test_window_offsets_cover_every_pixel_once(size: int) -> None:
    """Scored pixels across windows tile [0, size) exactly once."""
    hits = np.zeros(size, dtype=int)
    for start, first_new in window_offsets(size, 128):
        assert 0 <= start <= size - 128
        hits[start + first_new : start + 128] += 1
    assert (hits == 1).all()


def test_window_offsets_rejects_small_tile() -> None:
    """A tile smaller than one window cannot be cut."""
    with pytest.raises(ValueError):
        window_offsets(100, 128)
