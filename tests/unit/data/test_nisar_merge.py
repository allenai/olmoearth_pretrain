"""Tests for merging the stored NISAR passes into the one-slot ``nisar`` modality."""

import numpy as np

from olmoearth_pretrain.data.dataset import _dense_months, merge_nisar_passes


def _ragged(mask: np.ndarray, value: float) -> np.ndarray:
    """A ragged (2, 2, T_present, 2) array whose month-t entries are value + t."""
    months = np.flatnonzero(mask)
    out = np.zeros((2, 2, len(months), 2), dtype=np.float32)
    for i, t in enumerate(months):
        out[:, :, i] = value + t
    return out


def test_merge_takes_one_pass_per_month_and_unions_masks() -> None:
    """Each present month comes from a pass present that month; the mask is the union."""
    asc_mask = np.array([0, 1, 1, 0, 1, 0], dtype=bool)
    desc_mask = np.array([0, 0, 1, 1, 1, 0], dtype=bool)
    arrays = {
        "nisar_asc": _ragged(asc_mask, 100.0),
        "nisar_desc": _ragged(desc_mask, 200.0),
    }
    masks = {"nisar_asc": asc_mask, "nisar_desc": desc_mask}
    picks: dict[int, set[float]] = {2: set(), 4: set()}
    for seed in range(50):
        result = merge_nisar_passes(arrays, masks, rng=np.random.default_rng(seed))
        assert result is not None
        merged, union = result
        np.testing.assert_array_equal(union, asc_mask | desc_mask)
        dense = _dense_months(merged, union)
        assert dense[0, 0, 1, 0] == 101.0  # asc only
        assert dense[0, 0, 3, 0] == 203.0  # desc only
        for t in picks:  # both passes present
            assert dense[0, 0, t, 0] in (100.0 + t, 200.0 + t)
            picks[t].add(float(dense[0, 0, t, 0]))
    assert all(len(v) == 2 for v in picks.values()), picks


def test_merge_single_pass_and_none() -> None:
    """One stored pass is passed through; no stored pass gives None."""
    mask = np.array([1, 0, 1], dtype=bool)
    asc = _ragged(mask, 10.0)
    result = merge_nisar_passes({"nisar_asc": asc}, {"nisar_asc": mask})
    assert result is not None
    merged, union = result
    np.testing.assert_array_equal(merged, asc)
    np.testing.assert_array_equal(union, mask)
    assert merge_nisar_passes({}, {}) is None
