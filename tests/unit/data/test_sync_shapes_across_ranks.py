"""sync_shapes_across_ranks: every dp rank draws the same per-batch shapes."""

from pathlib import Path

import numpy as np

from olmoearth_pretrain.data.dataloader import (
    OlmoEarthDataLoader,
    _IterableDatasetWrapper,
)

from .test_encoder_budget_half_t import _build


def _rank_shapes(dl: OlmoEarthDataLoader, n: int) -> list[tuple]:
    """(patch_size, hw, t, location-query patch size, e, k) per instance, without the index."""
    dl.reshuffle()
    return [
        item[1:]
        for item in _IterableDatasetWrapper(dl)._get_batch_item_params_iterator(
            np.arange(n), dl.patch_sizes, dl.sampled_hw_p_list, rank_batch_size=1
        )
    ]


def _loader(tmp_path: Path, h5_dir: Path, rank: int, sync: bool) -> OlmoEarthDataLoader:
    return _build(
        tmp_path / f"r{rank}_{sync}",
        h5_dir,
        encoder_token_budget=128,
        min_encoder_tokens=10,
        max_num_visible=4,
        encoder_budget_half_t=True,
        dp_world_size=2,
        dp_rank=rank,
        sync_shapes_across_ranks=sync,
    )


def test_sync_shapes_across_ranks(tmp_path: Path, setup_h5py_dir: Path) -> None:
    """Ranks match with the option on, differ without it; rank 0 is unchanged."""
    synced = [_rank_shapes(_loader(tmp_path, setup_h5py_dir, r, True), 200) for r in (0, 1)]
    unsynced = [
        _rank_shapes(_loader(tmp_path, setup_h5py_dir, r, False), 200) for r in (0, 1)
    ]
    assert synced[0] == synced[1]
    assert unsynced[0] != unsynced[1]
    assert synced[0] == unsynced[0]
