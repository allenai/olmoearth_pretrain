"""Per-modality capture dates packed into ``timestamps`` (v1.3 + per-modality DOY)."""

import h5py
import numpy as np
import pytest
from upath import UPath

from olmoearth_pretrain.data.allcap import (
    ALLCAP_LEAST_CLOUDY_SLOT,
    ALLCAP_LEAST_CLOUDY_SLOT_REAL_DATES,
    ALLCAP_RANDOM_CLEAR,
    ALLCAP_SELECTIONS,
    NUM_SLOTS,
    SlotSelection,
    read_allcap_sample,
    read_allcap_sample_two_draws,
    select_least_cloudy_per_slot_real_dates,
    selection_to_sample,
    timestamps_to_epoch_days,
)
from olmoearth_pretrain.data.constants import MISSING_VALUE
from olmoearth_pretrain.data.dataset import GetItemArgs, OlmoEarthDataset
from olmoearth_pretrain.data.per_modality_timestamps import (
    PACKED_TIMESTAMP_WIDTH,
    PER_MODALITY_TIMESTAMP_MODALITIES,
    has_per_modality_timestamps,
    modality_timestamps,
    pack_per_modality_timestamps,
    shared_timestamps,
)

from .test_allcap import MODS, START, allcap_dir  # noqa: F401

S2, S1, LANDSAT = PER_MODALITY_TIMESTAMP_MODALITIES


def test_pack_and_unpack_roundtrip() -> None:
    shared = np.array([[1, 0, 2022], [1, 1, 2022]], dtype=np.int32)
    s1 = np.array([[9, 0, 2022], [20, 1, 2022]], dtype=np.int32)
    packed = pack_per_modality_timestamps(shared, {S1: s1})
    assert packed.shape == (2, PACKED_TIMESTAMP_WIDTH) and packed.dtype == shared.dtype
    assert has_per_modality_timestamps(packed) and not has_per_modality_timestamps(shared)
    np.testing.assert_array_equal(shared_timestamps(packed), shared)
    np.testing.assert_array_equal(modality_timestamps(packed, S1), s1)
    # modalities without a block, and modalities outside the layout, read the shared stamp
    np.testing.assert_array_equal(modality_timestamps(packed, S2), shared)
    np.testing.assert_array_equal(modality_timestamps(packed, LANDSAT), shared)
    np.testing.assert_array_equal(modality_timestamps(packed, "worldcover"), shared)
    # an unpacked (T, 3) array is its own stamp for every modality
    np.testing.assert_array_equal(modality_timestamps(shared, S1), shared)
    with pytest.raises(ValueError, match="not in PER_MODALITY"):
        pack_per_modality_timestamps(shared, {"worldcover": s1})


def test_selection_to_sample_packs_each_modalitys_capture_dates() -> None:
    sel = SlotSelection(
        days=START + np.array([0, 30, 60]),
        s2_idx=np.array([0, -1, 1]),
        s1_idx=np.array([1, 0, -1]),
    )
    s2_days = START + np.array([2, 65])
    s1_days = START + np.array([31, 7])
    series = {
        S2: np.zeros((2, 2, 2, 12), dtype=np.uint16),
        S1: np.zeros((2, 2, 2, 2), dtype=np.float32),
    }
    plain, masks_plain = selection_to_sample(sel, series)
    packed, masks = selection_to_sample(sel, series, {S2: s2_days, S1: s1_days})
    assert plain["timestamps"].shape == (3, 3)
    assert packed["timestamps"].shape == (3, PACKED_TIMESTAMP_WIDTH)
    np.testing.assert_array_equal(shared_timestamps(packed["timestamps"]), plain["timestamps"])
    for k in masks_plain:
        np.testing.assert_array_equal(masks[k], masks_plain[k])
    s2 = timestamps_to_epoch_days(modality_timestamps(packed["timestamps"], S2))
    s1 = timestamps_to_epoch_days(modality_timestamps(packed["timestamps"], S1))
    # present: the capture's own date; missing: the slot's date
    assert s2.tolist() == [START + 2, START + 30, START + 65]
    assert s1.tolist() == [START + 7, START + 31, START + 60]


@pytest.mark.parametrize("selection", ALLCAP_SELECTIONS)
def test_reader_default_unchanged_and_packed_shared_matches(
    allcap_dir: tuple[UPath, dict],  # noqa: F811
    selection: str,
) -> None:
    """Flag off = the old sample exactly; flag on = same sample, wider timestamps."""
    d, _ = allcap_dir
    with h5py.File(d / "sample_0.h5", "r") as f:
        old = read_allcap_sample(f, MODS, selection, 0.5, np.random.default_rng(1))
        new = read_allcap_sample(
            f, MODS, selection, 0.5, np.random.default_rng(1), per_modality_timestamps=True
        )
    assert old[0]["timestamps"].shape == (len(old[0]["timestamps"]), 3)
    assert new[0]["timestamps"].shape[1] == PACKED_TIMESTAMP_WIDTH
    np.testing.assert_array_equal(shared_timestamps(new[0]["timestamps"]), old[0]["timestamps"])
    for k in set(old[0]) - {"timestamps"}:
        np.testing.assert_array_equal(old[0][k], new[0][k])
    for k in old[1]:
        np.testing.assert_array_equal(old[1][k], new[1][k])


def test_areal_s1_carries_its_own_dates(allcap_dir: tuple[UPath, dict]) -> None:  # noqa: F811
    """A_real: S2 blocks equal the shared stamp; S1 blocks are the real S1 capture dates."""
    d, arrays = allcap_dir
    with h5py.File(d / "sample_0.h5", "r") as f:
        sample, masks = read_allcap_sample(
            f, MODS, ALLCAP_LEAST_CLOUDY_SLOT_REAL_DATES, 1.0, per_modality_timestamps=True
        )
    ts = sample["timestamps"]
    s2_days_all = timestamps_to_epoch_days(arrays["timestamps_sentinel2_l2a"])
    s1_days_all = timestamps_to_epoch_days(arrays["timestamps_sentinel1"])
    shared = timestamps_to_epoch_days(shared_timestamps(ts))
    s2 = timestamps_to_epoch_days(modality_timestamps(ts, S2))
    s1 = timestamps_to_epoch_days(modality_timestamps(ts, S1))
    np.testing.assert_array_equal(s2, shared)  # A_real stamps a slot with its S2 date
    starts = [
        timestamps_to_epoch_days(arrays[f"timestamps_{m}"]).min()
        for m in ("sentinel2_l2a", "sentinel1", "landsat_l2")
    ]
    sel = select_least_cloudy_per_slot_real_dates(
        s2_days_all, arrays["sentinel2_scl_cloud_fraction"], s1_days_all, min(starts)
    )
    present = sel.s1_idx >= 0
    np.testing.assert_array_equal(present, masks[S1])
    np.testing.assert_array_equal(s1[present], s1_days_all[sel.s1_idx[present]])
    np.testing.assert_array_equal(s1[~present], shared[~present])
    both = present & (sel.s2_idx >= 0)
    # the change this experiment is about: S1 and S2 dates differ within shared slots
    assert (s1[both] != s2[both]).sum() >= both.sum() // 2
    assert np.abs(s1[both] - s2[both]).max() < 30
    # S1 dates in the S1 block are real S1 captures, not the slot's S2 date
    assert set(s1[present].tolist()) <= set(s1_days_all.tolist())


def test_two_draws_pack_per_view(allcap_dir: tuple[UPath, dict]) -> None:  # noqa: F811
    d, _ = allcap_dir
    with h5py.File(d / "sample_0.h5", "r") as f:
        rngs = tuple(np.random.default_rng(5).spawn(2))
        views = read_allcap_sample_two_draws(
            f, MODS, ALLCAP_RANDOM_CLEAR, 0.6, rngs, per_modality_timestamps=True
        )
    for sample, masks in views:
        ts = sample["timestamps"]
        assert ts.shape[1] == PACKED_TIMESTAMP_WIDTH
        s1 = modality_timestamps(ts, S1)
        # arm B pairs S1 within +/-15 d of the S2 date, now visible in the S1 block
        both = masks[S1] & masks[S2]
        d1 = timestamps_to_epoch_days(s1)[both]
        d2 = timestamps_to_epoch_days(modality_timestamps(ts, S2))[both]
        assert (np.abs(d1 - d2) <= 15).all()


def _dataset(d: UPath, per_mod: bool, selection: str) -> OlmoEarthDataset:
    ds = OlmoEarthDataset(
        h5py_dir=d,
        training_modalities=MODS,
        dtype=np.float32,
        normalize=False,
        allcap_selection=selection,
        allcap_cloud_tau=1.0,
        allcap_per_modality_timestamps=per_mod,
    )
    ds.prepare()
    return ds


@pytest.mark.parametrize("selection", [ALLCAP_LEAST_CLOUDY_SLOT_REAL_DATES, ALLCAP_LEAST_CLOUDY_SLOT])
def test_dataset_getitem_packed(allcap_dir: tuple[UPath, dict], selection: str) -> None:  # noqa: F811
    """End to end through OlmoEarthDataset: crop/pad/subset keep the packed columns aligned."""
    d, _ = allcap_dir
    args = GetItemArgs(idx=0, patch_size=1, sampled_hw_p=4, token_budget=10000)
    np.random.seed(0)
    _, old = _dataset(d, False, selection)[args]
    np.random.seed(0)
    _, new = _dataset(d, True, selection)[args]
    assert old.timestamps.shape == (NUM_SLOTS, 3)
    assert new.timestamps.shape == (NUM_SLOTS, PACKED_TIMESTAMP_WIDTH)
    np.testing.assert_array_equal(shared_timestamps(new.timestamps), old.timestamps)
    for name in MODS:
        np.testing.assert_array_equal(getattr(old, name), getattr(new, name))
    s1_present = ~(new.sentinel1 == MISSING_VALUE).all(axis=(0, 1, 3))
    s1 = timestamps_to_epoch_days(modality_timestamps(new.timestamps, S1))
    shared = timestamps_to_epoch_days(shared_timestamps(new.timestamps))
    assert (s1[s1_present] != shared[s1_present]).any()
    np.testing.assert_array_equal(s1[~s1_present], shared[~s1_present])


def test_dataset_flag_needs_allcap(allcap_dir: tuple[UPath, dict]) -> None:  # noqa: F811
    d, _ = allcap_dir
    with pytest.raises(ValueError, match="requires allcap_selection"):
        OlmoEarthDataset(
            h5py_dir=d,
            training_modalities=MODS,
            dtype=np.float32,
            allcap_per_modality_timestamps=True,
        )
