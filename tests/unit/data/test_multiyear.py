"""Tests for 16-step (multi-year quarterly) samples and their mix with 12-step ones."""

import functools
from datetime import date, timedelta
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from upath import UPath

from olmoearth_pretrain.data.collate import (
    collate_olmoearth_pretrain,
    collate_single_masked_batched,
)
from olmoearth_pretrain.data.concat import (
    OlmoEarthConcatDataset,
    OlmoEarthConcatDatasetConfig,
)
from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoader
from olmoearth_pretrain.data.dataset import (
    GetItemArgs,
    OlmoEarthDataset,
    OlmoEarthDatasetConfig,
    choose_year_spread_timesteps,
    is_multiyear_sample,
    timestamps_to_ordinal_days,
)
from olmoearth_pretrain.dataset.convert_to_h5py import ConvertToH5py
from olmoearth_pretrain.train.masking import MaskingConfig, MaskValue

TRAINING_MODALITIES = [Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name]
HW = 8
PERIOD_DAYS = 91


def _ts(dates: list[date]) -> np.ndarray:
    return np.array([[d.day, d.month - 1, d.year] for d in dates], dtype=np.int32)


def monthly_dates(start: date = date(2020, 3, 5)) -> list[date]:
    """12 monthly period starts (30 d apart), like osm_sampling."""
    return [start + timedelta(days=30 * i) for i in range(12)]


def quarterly_dates(start: date = date(2018, 6, 18)) -> list[date]:
    """16 quarterly period starts (91 d apart), like osm_quarterly4y."""
    return [start + timedelta(days=PERIOD_DAYS * i) for i in range(16)]


def write_h5_dir(
    root: Path,
    samples: list[tuple[list[date], dict[str, np.ndarray]]],
) -> UPath:
    """Write a tiny H5 corpus in the ConvertToH5py layout.

    Each sample is (timestamps, {modality: bool presence mask}). Modality data has one
    frame per present timestep (as the H5 builder writes it); the S2 value of a frame
    encodes its timestep index (value 1000 + index) so tests can check placement.
    """
    h5_dir = (
        root
        / "h5py_data_w_missing_timesteps_zstd_3_128_x_4"
        / "sentinel1_sentinel2_l2a"
        / str(len(samples))
    )
    h5_dir.mkdir(parents=True)
    rows = []
    for i, (dates, masks) in enumerate(samples):
        with h5py.File(
            h5_dir / ConvertToH5py.sample_file_pattern.format(index=i), "w"
        ) as f:
            f.create_dataset("timestamps", data=_ts(dates))
            f.create_dataset("latlon", data=np.array([10.0, 20.0], dtype=np.float32))
            grp = f.create_group(ConvertToH5py.missing_timesteps_mask_group_name)
            for name, mask in masks.items():
                present_idx = np.flatnonzero(mask)
                num_bands = Modality.get(name).num_bands
                data = np.zeros((HW, HW, len(present_idx), num_bands), dtype=np.float32)
                for k, t in enumerate(present_idx):
                    data[:, :, k, :] = 1000 + t
                f.create_dataset(name, data=data)
                grp.create_dataset(name, data=np.asarray(mask, dtype=bool))
        rows.append(
            {"sample_index": i, **{m: int(m in masks) for m in TRAINING_MODALITIES}}
        )
    pd.DataFrame(rows).to_csv(h5_dir / ConvertToH5py.sample_metadata_fname, index=False)
    np.save(
        h5_dir / ConvertToH5py.latlon_distribution_fname, np.zeros((len(samples), 2))
    )
    return UPath(h5_dir)


@pytest.fixture
def monthly_h5_dir(tmp_path: Path) -> UPath:
    """Two 12-step monthly samples (S1 missing month 3 in the second)."""
    s1_mask = np.ones(12, dtype=bool)
    s1_mask[3] = False
    return write_h5_dir(
        tmp_path / "monthly",
        [
            (
                monthly_dates(),
                {"sentinel2_l2a": np.ones(12, bool), "sentinel1": np.ones(12, bool)},
            ),
            (
                monthly_dates(),
                {"sentinel2_l2a": np.ones(12, bool), "sentinel1": s1_mask},
            ),
        ],
    )


@pytest.fixture
def quarterly_h5_dir(tmp_path: Path) -> UPath:
    """Three 16-step quarterly samples on the full grid with gaps.

    0: complete. 1: S1 missing periods 0 and 9, S2 missing 5. 2: period 0 missing in
    every modality (empty grid point kept in the sequence), S1 entirely absent.
    """
    full = np.ones(16, dtype=bool)
    s1_gap = full.copy()
    s1_gap[[0, 9]] = False
    s2_gap = full.copy()
    s2_gap[5] = False
    first_empty = full.copy()
    first_empty[0] = False
    return write_h5_dir(
        tmp_path / "quarterly",
        [
            (quarterly_dates(), {"sentinel2_l2a": full, "sentinel1": full}),
            (quarterly_dates(), {"sentinel2_l2a": s2_gap, "sentinel1": s1_gap}),
            (quarterly_dates(), {"sentinel2_l2a": first_empty}),
        ],
    )


def make_dataset(
    h5_dir: UPath, max_sequence_length: int = 16, **kwargs
) -> OlmoEarthDataset:  # type: ignore[no-untyped-def]
    """Build and prepare an unnormalized dataset."""
    dataset = OlmoEarthDatasetConfig(
        h5py_dir=str(h5_dir),
        training_modalities=TRAINING_MODALITIES,
        normalize=False,
        max_sequence_length=max_sequence_length,
        **kwargs,
    ).build()
    dataset.prepare()
    return dataset


def get(
    dataset: OlmoEarthDataset | OlmoEarthConcatDataset,
    idx: int,
    target_t: int,
    hw_p: int = 4,
):  # type: ignore[no-untyped-def]
    """Fetch one sample at patch size 1 with a large token budget."""
    return dataset[
        GetItemArgs(
            idx=idx,
            patch_size=1,
            sampled_hw_p=hw_p,
            token_budget=100000,
            target_t=target_t,
        )
    ][1]


def present_from_sample(sample) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Timesteps where any spacetime modality has data."""
    present = np.zeros(sample.time, dtype=bool)
    for name in TRAINING_MODALITIES:
        present |= getattr(sample, name)[0, 0, :, 0] != MISSING_VALUE
    return present


# --- crop / pad ---------------------------------------------------------------------


def test_crop_timestamps_and_masks_without_masks_keeps_16() -> None:
    """Without masks, a 16-step sample is not truncated (was cropped to 13)."""
    timestamps = _ts(quarterly_dates())
    cropped, masks = OlmoEarthDataset._crop_timestamps_and_masks(timestamps, {})
    assert len(cropped) == 16 and masks == {}


def test_crop_timestamps_and_masks_16_steps() -> None:
    """Masks beyond index 12 are handled with 16 timesteps."""
    timestamps = _ts(quarterly_dates())
    first_late = np.zeros(16, dtype=bool)
    first_late[[13, 14]] = True
    cropped, masks = OlmoEarthDataset._crop_timestamps_and_masks(
        timestamps, {"sentinel2_l2a": first_late}
    )
    np.testing.assert_array_equal(cropped, timestamps[13:15])
    np.testing.assert_array_equal(masks["sentinel2_l2a"], [True, True])

    full = np.ones(16, dtype=bool)
    full[0] = False
    cropped, masks = OlmoEarthDataset._crop_timestamps_and_masks(
        timestamps, {"sentinel2_l2a": full}
    )
    np.testing.assert_array_equal(cropped, timestamps[1:])
    assert len(masks["sentinel2_l2a"]) == 15


def test_12_step_sample_padded_to_16(monthly_h5_dir: UPath) -> None:
    """A monthly sample read with max_sequence_length=16 is 12 real + 4 missing steps."""
    dataset = make_dataset(monthly_h5_dir, max_sequence_length=16)
    sample = get(dataset, 1, target_t=16)
    assert sample.time == 16
    assert sample.sentinel2_l2a.shape == (4, 4, 16, 12)
    s2 = sample.sentinel2_l2a[0, 0, :, 0]
    np.testing.assert_array_equal(s2[:12], 1000 + np.arange(12))
    assert np.all(sample.sentinel2_l2a[:, :, 12:] == MISSING_VALUE)
    s1 = sample.sentinel1[0, 0, :, 0]
    assert s1[3] == MISSING_VALUE and s1[4] == 1004 and np.all(s1[12:] == MISSING_VALUE)
    # Padded timestamps repeat the last real one.
    np.testing.assert_array_equal(
        sample.timestamps[12:], np.repeat(sample.timestamps[11:12], 4, 0)
    )
    # Contiguous behaviour is unchanged for T <= 12.
    sample = get(dataset, 0, target_t=5)
    assert sample.time == 5
    s2 = sample.sentinel2_l2a[0, 0, :, 0]
    assert np.all(np.diff(s2) == 1) and s2[0] >= 1000


def test_12_step_default_unchanged(monthly_h5_dir: UPath) -> None:
    """The default (max_sequence_length=12) still yields 12-step samples."""
    dataset = make_dataset(monthly_h5_dir, max_sequence_length=12)
    assert get(dataset, 0, target_t=12).time == 12


def test_16_step_h5_requires_max_sequence_length_16(quarterly_h5_dir: UPath) -> None:
    """A 16-step H5 read at the default 12 fails loudly rather than truncating."""
    dataset = make_dataset(quarterly_h5_dir, max_sequence_length=12)
    with pytest.raises(AssertionError):
        get(dataset, 0, target_t=12)


# --- year-spread subsetting ---------------------------------------------------------


def _check_year_spread(
    slots: np.ndarray, present: np.ndarray, num_t: int
) -> tuple[int, int]:
    """Assert the year-spread invariants; return (n_years, n_quarters)."""
    assert len(slots) == num_t
    real = slots[slots >= 0]
    assert len(real) >= 1
    # Padding only at the end, real slots unique and chronological.
    assert np.all(slots[: len(real)] == real) and np.all(slots[len(real) :] == -1)
    assert np.all(np.diff(real) > 0)
    years = sorted({int(s) // 4 for s in real})
    quarters = sorted({int(s) % 4 for s in real})
    # Full grid: every chosen year has the same quarter positions.
    assert sorted(real.tolist()) == [y * 4 + q for y in years for q in quarters]
    assert len(years) * len(quarters) <= num_t
    # Year recovery from dates (the loss's rule): with d0 = the earliest present
    # timestep, or d0 = the first returned timestep, round((d - d0) / P) // 4 equals
    # the year up to a constant shift, for every returned slot.
    days = real * PERIOD_DAYS
    is_present = present[real]
    refs = [days[0]] + ([days[is_present][0]] if is_present.any() else [])
    for ref in refs:
        chunk = np.round((days - ref) / PERIOD_DAYS).astype(int) // 4
        shift = chunk - real // 4
        assert np.all(shift == shift[0])
    return len(years), len(quarters)


def test_choose_year_spread_full_data_factorizations() -> None:
    """With complete data, n*q is maximal, n >= 2 preferred."""
    present = np.ones(16, dtype=bool)
    rng = np.random.default_rng(0)
    best = {
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 4,
        6: 6,
        7: 6,
        8: 8,
        9: 9,
        10: 9,
        11: 9,
        12: 12,
        13: 12,
        14: 12,
        15: 12,
        16: 16,
    }
    for num_t in range(1, 17):
        seen = set()
        for _ in range(50):
            slots = choose_year_spread_timesteps(present, num_t, 4, rng)
            n, q = _check_year_spread(slots, present, num_t)
            assert n * q == best[num_t]
            if num_t >= 2:
                assert n >= 2
            seen.add((n, q))
        if num_t == 4:
            assert seen == {(4, 1), (2, 2)}
        if num_t == 12:
            assert seen == {(4, 3), (3, 4)}


def test_choose_year_spread_invariants_random_gaps() -> None:
    """Invariants hold for random missing patterns, including empty first slots."""
    rng = np.random.default_rng(1)
    for _ in range(2000):
        present = rng.random(16) < rng.uniform(0.2, 1.0)
        if not present.any():
            present[rng.integers(16)] = True
        num_t = int(rng.integers(1, 17))
        slots = choose_year_spread_timesteps(present, num_t, 4, rng)
        _check_year_spread(slots, present, num_t)
        # At least one chosen timestep has data.
        assert any(present[s] for s in slots if s >= 0)


def test_choose_year_spread_prefers_years_with_data() -> None:
    """An empty year is never chosen while enough years have data."""
    present = np.ones(16, dtype=bool)
    present[4:8] = False  # year 1 empty
    rng = np.random.default_rng(2)
    for _ in range(200):
        slots = choose_year_spread_timesteps(present, 6, 4, rng)
        assert not any(4 <= s < 8 for s in slots)


def test_is_multiyear_sample() -> None:
    """Monthly samples are never multi-year; quarterly spans are."""
    assert not is_multiyear_sample(_ts(monthly_dates()), {})
    assert is_multiyear_sample(_ts(quarterly_dates()), {})
    only_first_year = np.zeros(16, dtype=bool)
    only_first_year[:4] = True
    assert not is_multiyear_sample(
        _ts(quarterly_dates()), {"sentinel2_l2a": only_first_year}
    )


def test_year_spread_getitem(quarterly_h5_dir: UPath) -> None:
    """Dataset-level year-spread subsetting of 16-step samples."""
    dataset = make_dataset(
        quarterly_h5_dir, max_sequence_length=16, year_spread_steps_per_year=4
    )
    grid_days = timestamps_to_ordinal_days(_ts(quarterly_dates()))
    for idx in range(3):
        for num_t in range(1, 17):
            for _ in range(10):
                sample = get(dataset, idx, target_t=num_t)
                assert sample.time == num_t
                days = timestamps_to_ordinal_days(sample.timestamps)
                present = present_from_sample(sample)
                assert present.any()
                # S2 values encode the source slot; check alignment with timestamps.
                s2 = sample.sentinel2_l2a[0, 0, :, 0]
                for t in np.flatnonzero(s2 != MISSING_VALUE):
                    assert grid_days[int(s2[t]) - 1000] == days[t]
                s1 = sample.sentinel1[0, 0, :, 0]
                for t in np.flatnonzero(s1 != MISSING_VALUE):
                    assert grid_days[int(s1[t]) - 1000] == days[t]
                # Loss-side year recovery from present timesteps.
                d = days[present]
                assert np.all(np.diff(d) > 0)
                slot = (d - grid_days[0]) // PERIOD_DAYS
                chunk = np.round((d - d[0]) / PERIOD_DAYS).astype(int) // 4
                year = slot // 4
                assert np.all(chunk == year - year[0])
                # Same quarter positions in every chosen year: the distinct
                # timestamps (pads repeat the last one) form a full years x quarters grid.
                real_slots = np.unique((days - grid_days[0]) // PERIOD_DAYS)
                qs = {int(s) % 4 for s in real_slots}
                ys = {int(s) // 4 for s in real_slots}
                assert len(real_slots) == len(qs) * len(ys)
                assert len(real_slots) <= num_t


def test_year_spread_disabled_is_contiguous(quarterly_h5_dir: UPath) -> None:
    """Without year_spread_steps_per_year a 16-step sample is a contiguous slice."""
    dataset = make_dataset(quarterly_h5_dir, max_sequence_length=16)
    sample = get(dataset, 0, target_t=6)
    s2 = sample.sentinel2_l2a[0, 0, :, 0]
    assert np.all(np.diff(s2) == 1)


# --- concat / batches ---------------------------------------------------------------


def test_concat_mixed_12_and_16_steps_stack(
    monthly_h5_dir: UPath, quarterly_h5_dir: UPath
) -> None:
    """12-step and 16-step corpora mix into stackable, maskable batches."""
    config = OlmoEarthConcatDatasetConfig(
        dataset_configs=[
            OlmoEarthDatasetConfig(
                h5py_dir=str(monthly_h5_dir),
                training_modalities=TRAINING_MODALITIES,
                normalize=False,
            ),
            OlmoEarthDatasetConfig(
                h5py_dir=str(quarterly_h5_dir),
                training_modalities=TRAINING_MODALITIES,
                normalize=False,
                year_spread_steps_per_year=4,
            ),
        ],
        max_sequence_length=16,
    )
    concat = config.build()
    concat.prepare()
    assert concat.max_sequence_length == 16
    masking = MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode",
            "encode_ratio": 0.5,
            "decode_ratio": 0.5,
            "random_ratio": 0.5,
            "only_decode_modalities": [],
        }
    ).build()
    for num_t in (1, 3, 5, 8, 12, 14, 16):
        batch = [(1, get(concat, i, target_t=num_t)) for i in range(len(concat))]
        _, stacked = collate_olmoearth_pretrain(batch)
        assert stacked.timestamps.shape == (len(concat), num_t, 3)
        assert stacked.sentinel2_l2a.shape == (len(concat), 4, 4, num_t, 12)
        _, masked = collate_single_masked_batched(
            batch, transform=None, masking_strategy=masking
        )
        # Missing (padded) timesteps stay MISSING in the masks.
        s2_missing = stacked.sentinel2_l2a[..., 0] == MISSING_VALUE
        s2_mask = masked.sentinel2_l2a_mask[..., 0]
        assert np.all(s2_mask[s2_missing].numpy() == MaskValue.MISSING.value)


def test_concat_rejects_mismatched_max_sequence_length(
    monthly_h5_dir: UPath, quarterly_h5_dir: UPath
) -> None:
    """Sub datasets with different max_sequence_length cannot be concatenated."""
    concat = OlmoEarthConcatDataset(
        [make_dataset(monthly_h5_dir, 12), make_dataset(quarterly_h5_dir, 16)]
    )
    with pytest.raises(ValueError, match="max_sequence_length"):
        concat.prepare()


def test_dataloader_max_timesteps_16(
    tmp_path: Path, monthly_h5_dir: UPath, quarterly_h5_dir: UPath
) -> None:
    """The dataloader yields 16-step batches from a mixed concat and checks the cap."""
    concat = OlmoEarthConcatDataset(
        [
            make_dataset(monthly_h5_dir, 16),
            make_dataset(quarterly_h5_dir, 16, year_spread_steps_per_year=4),
        ]
    )
    concat.prepare()
    masking = MaskingConfig(strategy_config={"type": "random"}).build()
    kwargs = dict(
        dataset=concat,
        work_dir=tmp_path,
        global_batch_size=5,
        min_patch_size=1,
        max_patch_size=1,
        sampled_hw_p_list=[2, 4],
        token_budget=2 * 16 * 16 + 10,
        temporal_bias=2.0,
        collator=functools.partial(
            collate_single_masked_batched, transform=None, masking_strategy=masking
        ),
        masking_strategy=masking,
        target_device_type="cpu",
        tile_size=HW,
    )
    with pytest.raises(ValueError, match="max_timesteps"):
        OlmoEarthDataLoader(max_timesteps=17, **kwargs)  # type: ignore[arg-type]
    loader = OlmoEarthDataLoader(max_timesteps=16, **kwargs)  # type: ignore[arg-type]
    ts = []
    for epoch in range(1, 9):
        loader.reshuffle(epoch=epoch, in_memory=True)
        for _, masked, _ in loader:
            ts.append(masked.timestamps.shape[1])
            assert masked.sentinel2_l2a.shape[3] == ts[-1]
        loader.reset()
    assert max(ts) <= 16 and max(ts) > 12
