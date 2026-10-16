"""Tests for the every-capture (allcap) -> 12-slot sampler used by experiment E5."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from upath import UPath

from olmoearth_pretrain.data.allcap import (
    ALLCAP_LEAST_CLOUDY_SLOT,
    align_cloud_fraction,
    ALLCAP_RANDOM_CLEAR,
    NUM_SLOTS,
    S1_MATCH_TOLERANCE_DAYS,
    SLOT_DAYS,
    epoch_days_to_timestamps,
    read_allcap_sample,
    select_least_cloudy_per_slot,
    select_random_clear,
    timestamps_to_epoch_days,
)
from olmoearth_pretrain.data.constants import MISSING_VALUE
from olmoearth_pretrain.data.dataset import GetItemArgs, OlmoEarthDataset

START = int(np.datetime64("2021-01-03", "D").astype(np.int64))


def test_timestamp_roundtrip() -> None:
    """[day, month-1, year] <-> epoch days is exact, incl. leap days and year ends."""
    ts = np.array([[1, 0, 2020], [29, 1, 2020], [31, 11, 2021], [1, 2, 2023]])
    days = timestamps_to_epoch_days(ts)
    assert days[1] - days[0] == 59  # Jan 1 -> Feb 29 2020
    np.testing.assert_array_equal(epoch_days_to_timestamps(days), ts)
    assert epoch_days_to_timestamps(days).dtype == np.int32


def test_align_cloud_fraction() -> None:
    """SCL with an extra capture is matched by date; a missing SCL gives all-NaN."""
    s2 = epoch_days_to_timestamps(START + np.array([0, 5, 10]))
    scl = epoch_days_to_timestamps(START + np.array([0, 3, 5, 10]))
    cf = np.array([0.1, 0.9, 0.5, 0.2])
    np.testing.assert_allclose(align_cloud_fraction(s2, scl, cf), [0.1, 0.5, 0.2])
    np.testing.assert_allclose(align_cloud_fraction(s2, s2, cf[:3]), cf[:3])
    assert np.isnan(align_cloud_fraction(s2, None, None)).all()
    out = align_cloud_fraction(s2, scl[:2], cf[:2])
    assert out[0] == 0.1 and np.isnan(out[1:]).all()


class TestArmA:
    """Least-cloudy S2 per 30-day slot, midpoint-nearest S1, slot-start stamps."""

    def _series(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        s2_days = START + np.sort(rng.choice(360, size=70, replace=False))
        cloud = rng.random(70)
        s1_days = START + np.arange(2, 330, 12)  # no S1 in the last slot
        return s2_days, cloud, s1_days

    def test_least_cloudy_and_slot_start(self) -> None:
        """Each slot holds the least-cloudy S2 capture of that slot; stamps are slot starts."""
        s2_days, cloud, s1_days = self._series()
        sel = select_least_cloudy_per_slot(s2_days, cloud, s1_days, START)
        assert len(sel.days) == NUM_SLOTS
        np.testing.assert_array_equal(
            sel.days, START + SLOT_DAYS * np.arange(NUM_SLOTS)
        )
        for k in range(NUM_SLOTS):
            in_slot = np.flatnonzero((s2_days - START) // SLOT_DAYS == k)
            if len(in_slot) == 0:
                assert sel.s2_idx[k] == -1
            else:
                assert sel.s2_idx[k] == in_slot[np.argmin(cloud[in_slot])]
            s1_in = np.flatnonzero((s1_days - START) // SLOT_DAYS == k)
            if len(s1_in) == 0:
                assert sel.s1_idx[k] == -1
            else:
                mid = START + SLOT_DAYS * k + SLOT_DAYS // 2
                assert sel.s1_idx[k] == s1_in[np.argmin(np.abs(s1_days[s1_in] - mid))]
        assert sel.s1_idx[-1] == -1

    def test_deterministic(self) -> None:
        """Arm A takes no RNG and returns the same selection every time."""
        s2_days, cloud, s1_days = self._series()
        a = select_least_cloudy_per_slot(s2_days, cloud, s1_days, START)
        b = select_least_cloudy_per_slot(s2_days, cloud, s1_days, START)
        np.testing.assert_array_equal(a.s2_idx, b.s2_idx)
        np.testing.assert_array_equal(a.s1_idx, b.s1_idx)

    def test_empty_slot_and_nan_cloud(self) -> None:
        """A slot with no S2 is missing; NaN cloud fractions count as fully cloudy."""
        s2_days = START + np.array([0, 5, 70])
        cloud = np.array([np.nan, 0.5, 0.2])
        sel = select_least_cloudy_per_slot(s2_days, cloud, np.zeros(0), START)
        assert sel.s2_idx[0] == 1
        assert sel.s2_idx[1] == -1
        assert sel.s2_idx[2] == 2
        assert (sel.s1_idx == -1).all()


class TestArmB:
    """Random clear S2 captures with real dates; S1 paired or S1-only slots."""

    def test_clear_only_real_dates_sorted(self) -> None:
        """Chosen S2 captures are all <= tau, distinct, stamped at their own dates."""
        rng_data = np.random.default_rng(1)
        s2_days = START + np.sort(rng_data.choice(360, size=80, replace=False))
        cloud = rng_data.random(80)
        s1_days = START + np.arange(0, 360, 12)
        tau = 0.4
        for seed in range(20):
            sel = select_random_clear(
                s2_days, cloud, s1_days, tau, np.random.default_rng(seed)
            )
            assert len(sel.days) == NUM_SLOTS
            assert (sel.s2_idx >= 0).all()
            assert len(set(sel.s2_idx.tolist())) == NUM_SLOTS
            assert (cloud[sel.s2_idx] <= tau).all()
            np.testing.assert_array_equal(sel.days, s2_days[sel.s2_idx])
            assert (np.diff(sel.days) >= 0).all()
            paired = sel.s1_idx >= 0
            assert paired.any()
            assert (
                np.abs(s1_days[sel.s1_idx[paired]] - sel.days[paired])
                <= S1_MATCH_TOLERANCE_DAYS
            ).all()
            assert len(set(sel.s1_idx[paired].tolist())) == paired.sum()

    def test_random_across_reads(self) -> None:
        """Different draws give different captures (per sample per epoch)."""
        s2_days = START + np.arange(0, 360, 5)
        cloud = np.zeros(len(s2_days))
        picks = {
            tuple(
                select_random_clear(
                    s2_days, cloud, np.zeros(0), 0.1, np.random.default_rng(s)
                ).s2_idx
            )
            for s in range(10)
        }
        assert len(picks) > 1

    def test_fewer_than_12_clear(self) -> None:
        """5 clear S2 -> 5 S2 slots + up to 7 S1-only slots with real S1 dates."""
        s2_days = START + np.arange(0, 360, 10)  # 36 captures
        cloud = np.ones(len(s2_days))
        clear = np.array([1, 7, 13, 20, 30])
        cloud[clear] = 0.0
        s1_days = START + np.arange(3, 360, 24)  # 15 captures
        sel = select_random_clear(
            s2_days, cloud, s1_days, 0.1, np.random.default_rng(0)
        )
        assert len(sel.days) == NUM_SLOTS
        assert sorted(sel.s2_idx[sel.s2_idx >= 0].tolist()) == clear.tolist()
        s1_only = sel.s2_idx == -1
        assert s1_only.sum() == NUM_SLOTS - len(clear)
        assert (sel.s1_idx[s1_only] >= 0).all()
        np.testing.assert_array_equal(
            sel.days[s1_only], s1_days[sel.s1_idx[s1_only]]
        )
        used = sel.s1_idx[sel.s1_idx >= 0]
        assert len(set(used.tolist())) == len(used)
        assert (np.diff(sel.days) >= 0).all()

    def test_fewer_than_12_total_is_missing(self) -> None:
        """3 clear S2 and 2 S1 captures -> fewer than 12 slots (rest padded as missing)."""
        s2_days = START + np.array([10, 100, 200, 300])
        cloud = np.array([0.0, 0.0, 0.9, 0.0])
        s1_days = START + np.array([12, 250])
        sel = select_random_clear(
            s2_days, cloud, s1_days, 0.5, np.random.default_rng(0)
        )
        assert len(sel.days) == 4
        assert sel.s2_idx.tolist() == [0, 1, -1, 3]
        assert sel.s1_idx.tolist() == [0, -1, 1, -1]
        assert sel.days.tolist() == (START + np.array([10, 100, 250, 300])).tolist()

    def test_no_clear_no_s1_falls_back_to_least_cloudy(self) -> None:
        """Without clear S2 and without S1, keep the least-cloudy S2 capture."""
        s2_days = START + np.array([10, 100, 200])
        cloud = np.array([0.9, 0.6, 0.95])
        sel = select_random_clear(
            s2_days, cloud, np.zeros(0), 0.1, np.random.default_rng(0)
        )
        assert sel.s2_idx.tolist() == [1]
        assert sel.days.tolist() == [START + 100]


def _write_allcap_h5(path: Path, rng: np.random.Generator) -> dict:
    """Write a small allcap-layout H5 (16x16 px) and return its raw arrays."""
    t2, t1, tl = 40, 25, 10
    start = np.datetime64("2021-01-03", "D").astype(np.int64)
    s2_days = start + np.sort(rng.choice(360, size=t2, replace=False))
    s1_days = start + 1 + np.sort(rng.choice(355, size=t1, replace=False))
    l_days = start + 4 + np.sort(rng.choice(350, size=tl, replace=False))
    arrays = {
        "sentinel2_l2a": rng.integers(1, 10000, (16, 16, t2, 12)).astype(np.uint16),
        "sentinel2_scl": rng.integers(0, 11, (16, 16, t2, 1)).astype(np.uint8),
        "sentinel2_scl_cloud_fraction": rng.random(t2).astype(np.float32),
        "sentinel1": rng.normal(-12, 3, (16, 16, t1, 2)).astype(np.float32),
        "landsat_l2": rng.integers(1, 30000, (16, 16, tl, 8)).astype(np.uint16),
        "timestamps_sentinel2_l2a": epoch_days_to_timestamps(s2_days),
        "timestamps_sentinel2_scl": epoch_days_to_timestamps(s2_days),
        "timestamps_sentinel1": epoch_days_to_timestamps(s1_days),
        "timestamps_landsat_l2": epoch_days_to_timestamps(l_days),
        "worldcover": np.full((16, 16, 1, 1), 10, dtype=np.uint8),
        "srtm": rng.integers(0, 500, (16, 16, 1, 1)).astype(np.int32),
        "latlon": np.array([10.0, 20.0], dtype=np.float32),
    }
    with h5py.File(path, "w") as f:
        for k, v in arrays.items():
            f.create_dataset(k, data=v)
    return arrays


@pytest.fixture
def allcap_dir(tmp_path: Path) -> tuple[UPath, dict]:
    """An allcap-style h5py dir with one sample."""
    d = tmp_path / "allcap" / "1"
    d.mkdir(parents=True)
    arrays = _write_allcap_h5(d / "sample_0.h5", np.random.default_rng(3))
    pd.DataFrame(
        [
            {
                "sample_index": 0,
                "sentinel2_l2a": 1,
                "sentinel1": 1,
                "landsat_l2": 1,
                "worldcover": 1,
                "srtm": 1,
            }
        ]
    ).to_csv(d / "sample_metadata.csv", index=False)
    np.save(d / "latlon_distribution.npy", np.array([[10.0, 20.0]]))
    return UPath(d), arrays


MODS = ["sentinel2_l2a", "sentinel1", "worldcover", "srtm"]


def test_read_allcap_sample_arm_a(allcap_dir: tuple[UPath, dict]) -> None:
    """The reader emits compact present-only stacks + masks matching the selection."""
    d, arrays = allcap_dir
    with h5py.File(d / "sample_0.h5", "r") as f:
        sample, masks = read_allcap_sample(f, MODS, ALLCAP_LEAST_CLOUDY_SLOT, 1.0)
    assert set(sample) == {"timestamps", *MODS}
    assert sample["timestamps"].shape == (NUM_SLOTS, 3)
    assert "landsat_l2" not in sample and "sentinel2_scl" not in sample
    # span start = earliest capture of any sensor
    starts = [
        timestamps_to_epoch_days(arrays[f"timestamps_{m}"]).min()
        for m in ("sentinel2_l2a", "sentinel1", "landsat_l2")
    ]
    assert timestamps_to_epoch_days(sample["timestamps"])[0] == min(starts)
    for m in ("sentinel2_l2a", "sentinel1"):
        assert sample[m].shape[2] == masks[m].sum()
    s2_days = timestamps_to_epoch_days(arrays["timestamps_sentinel2_l2a"])
    sel = select_least_cloudy_per_slot(
        s2_days,
        arrays["sentinel2_scl_cloud_fraction"],
        timestamps_to_epoch_days(arrays["timestamps_sentinel1"]),
        min(starts),
    )
    np.testing.assert_array_equal(
        sample["sentinel2_l2a"],
        arrays["sentinel2_l2a"][:, :, sel.s2_idx[sel.s2_idx >= 0]],
    )


@pytest.mark.parametrize("selection", [ALLCAP_LEAST_CLOUDY_SLOT, ALLCAP_RANDOM_CLEAR])
def test_dataset_getitem_allcap(
    allcap_dir: tuple[UPath, dict], selection: str
) -> None:
    """End to end through OlmoEarthDataset: a standard 12-slot sample with missing slots."""
    d, arrays = allcap_dir
    ds = OlmoEarthDataset(
        h5py_dir=d,
        training_modalities=MODS,
        dtype=np.float32,
        normalize=False,
        allcap_selection=selection,
        allcap_cloud_tau=0.3,
    )
    ds.prepare()
    np.random.seed(0)
    _, sample = ds[GetItemArgs(idx=0, patch_size=1, sampled_hw_p=4, token_budget=10000)]
    assert sample.timestamps.shape == (NUM_SLOTS, 3)
    s2 = sample.sentinel2_l2a
    assert s2.shape[2] == NUM_SLOTS and s2.shape[-1] == 12
    present = ~(s2 == MISSING_VALUE).all(axis=(0, 1, 3))
    cloud = arrays["sentinel2_scl_cloud_fraction"]
    s2_days = timestamps_to_epoch_days(arrays["timestamps_sentinel2_l2a"])
    slot_days = timestamps_to_epoch_days(sample.timestamps)
    if selection == ALLCAP_RANDOM_CLEAR:
        # every present S2 slot is a clear capture stamped at its real date
        n_clear = int((cloud <= 0.3).sum())
        assert present.sum() == min(NUM_SLOTS, n_clear)
        for k in np.flatnonzero(present):
            j = np.flatnonzero(s2_days == slot_days[k])
            assert len(j) == 1 and cloud[j[0]] <= 0.3
    else:
        np.testing.assert_array_equal(
            np.diff(slot_days), np.full(NUM_SLOTS - 1, SLOT_DAYS)
        )
    assert sample.worldcover.shape[:2] == s2.shape[:2]


def test_dataset_rejects_unknown_selection(allcap_dir: tuple[UPath, dict]) -> None:
    """A typo in the selection name fails at construction."""
    d, _ = allcap_dir
    with pytest.raises(ValueError):
        OlmoEarthDataset(
            h5py_dir=d,
            training_modalities=MODS,
            dtype=np.float32,
            allcap_selection="least_cloudy",
        )
