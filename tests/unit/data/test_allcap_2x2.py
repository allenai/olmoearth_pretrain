"""Tests for the E5 follow-up 2x2 selection rules (A_real, B_slot)."""

import numpy as np
import pytest
from upath import UPath

from olmoearth_pretrain.data.allcap import (
    ALLCAP_LEAST_CLOUDY_SLOT_REAL_DATES,
    ALLCAP_RANDOM_CLEAR_SLOT,
    ALLCAP_SELECTIONS,
    NUM_SLOTS,
    SLOT_DAYS,
    select_least_cloudy_per_slot,
    select_least_cloudy_per_slot_real_dates,
    select_random_clear_per_slot,
    select_slots,
    epoch_days_to_timestamps,
)
from olmoearth_pretrain.data.dataset import GetItemArgs, OlmoEarthDataset

from .test_allcap import MODS, START, allcap_dir  # noqa: F401  (fixture reuse)


def _series(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    s2_days = START + np.sort(rng.choice(360, size=70, replace=False))
    cloud = rng.random(70)
    s1_days = START + np.arange(2, 330, 12)  # no S1 in the last slot
    return s2_days, cloud, s1_days


def test_selections_registered() -> None:
    """Both new rules are valid ``allcap_selection`` values."""
    assert ALLCAP_LEAST_CLOUDY_SLOT_REAL_DATES in ALLCAP_SELECTIONS
    assert ALLCAP_RANDOM_CLEAR_SLOT in ALLCAP_SELECTIONS


class TestArmAReal:
    """Arm A's captures exactly; each slot stamped with its capture's real date."""

    def test_same_captures_as_arm_a_real_dates(self) -> None:
        s2_days, cloud, s1_days = _series()
        a = select_least_cloudy_per_slot(s2_days, cloud, s1_days, START)
        r = select_least_cloudy_per_slot_real_dates(s2_days, cloud, s1_days, START)
        np.testing.assert_array_equal(r.s2_idx, a.s2_idx)
        np.testing.assert_array_equal(r.s1_idx, a.s1_idx)
        for k in range(NUM_SLOTS):
            if r.s2_idx[k] >= 0:
                assert r.days[k] == s2_days[r.s2_idx[k]]
            elif r.s1_idx[k] >= 0:
                assert r.days[k] == s1_days[r.s1_idx[k]]
            else:
                assert r.days[k] == a.days[k]
        # every date stays inside its slot, so the timeline is sorted
        assert (np.diff(r.days) > 0).all()
        assert ((r.days - START) // SLOT_DAYS == np.arange(NUM_SLOTS)).all()

    def test_s1_only_slot_takes_s1_date(self) -> None:
        s2_days = START + np.array([3])
        s1_days = START + np.array([40])
        r = select_least_cloudy_per_slot_real_dates(s2_days, np.array([0.1]), s1_days, START)
        assert r.days[0] == s2_days[0] and r.s2_idx[0] == 0
        assert r.days[1] == s1_days[0] and r.s2_idx[1] == -1 and r.s1_idx[1] == 0
        assert r.days[2] == START + 2 * SLOT_DAYS  # empty slot keeps the slot start

    def test_deterministic(self) -> None:
        s2_days, cloud, s1_days = _series()
        x = select_slots(ALLCAP_LEAST_CLOUDY_SLOT_REAL_DATES, epoch_days_to_timestamps(s2_days), cloud,
                         epoch_days_to_timestamps(s1_days), START, 1.0, np.random.default_rng(1))
        y = select_slots(ALLCAP_LEAST_CLOUDY_SLOT_REAL_DATES, epoch_days_to_timestamps(s2_days), cloud,
                         epoch_days_to_timestamps(s1_days), START, 1.0, np.random.default_rng(2))
        np.testing.assert_array_equal(x.days, y.days)
        np.testing.assert_array_equal(x.s2_idx, y.s2_idx)


class TestArmBSlot:
    """Arm A's slot grid and stamps; per-slot random clear S2 and random S1, redrawn per read."""

    def test_grid_stamps_and_slot_membership(self) -> None:
        s2_days, cloud, s1_days = _series()
        tau = 0.5
        b = select_random_clear_per_slot(s2_days, cloud, s1_days, START, tau, np.random.default_rng(0))
        a = select_least_cloudy_per_slot(s2_days, cloud, s1_days, START)
        np.testing.assert_array_equal(b.days, START + SLOT_DAYS * np.arange(NUM_SLOTS))
        for k in range(NUM_SLOTS):
            in_slot = np.flatnonzero((s2_days - START) // SLOT_DAYS == k)
            clear = in_slot[cloud[in_slot] <= tau]
            if len(clear):
                assert b.s2_idx[k] in clear
            else:
                assert b.s2_idx[k] == a.s2_idx[k]  # least-cloudy fallback (or -1)
            s1_in = np.flatnonzero((s1_days - START) // SLOT_DAYS == k)
            if len(s1_in):
                assert b.s1_idx[k] in s1_in
            else:
                assert b.s1_idx[k] == -1

    def test_redrawn_across_reads(self) -> None:
        s2_days, cloud, s1_days = _series()
        rng = np.random.default_rng(0)
        picks = {tuple(select_random_clear_per_slot(s2_days, cloud, s1_days, START, 0.6, rng).s2_idx)
                 for _ in range(20)}
        assert len(picks) > 1

    def test_no_clear_in_slot_falls_back_to_least_cloudy(self) -> None:
        s2_days = START + np.array([1, 2, 31])
        cloud = np.array([0.9, 0.8, 0.95])
        b = select_random_clear_per_slot(s2_days, cloud, np.zeros(0, dtype=np.int64), START, 0.5,
                                         np.random.default_rng(0))
        assert b.s2_idx[0] == 1 and b.s2_idx[1] == 2 and (b.s2_idx[2:] == -1).all()


@pytest.mark.parametrize("selection", [ALLCAP_LEAST_CLOUDY_SLOT_REAL_DATES, ALLCAP_RANDOM_CLEAR_SLOT])
def test_dataset_getitem_2x2(allcap_dir: tuple[UPath, dict], selection: str) -> None:  # noqa: F811
    """End to end through OlmoEarthDataset: a standard 12-slot sample."""
    d, _ = allcap_dir
    ds = OlmoEarthDataset(h5py_dir=d, training_modalities=MODS, dtype=np.float32, normalize=False,
                          allcap_selection=selection, allcap_cloud_tau=0.3)
    ds.prepare()
    np.random.seed(0)
    _, sample = ds[GetItemArgs(idx=0, patch_size=1, sampled_hw_p=4, token_budget=10000)]
    assert sample.timestamps.shape == (NUM_SLOTS, 3)
    assert sample.sentinel2_l2a.shape[2] == NUM_SLOTS
