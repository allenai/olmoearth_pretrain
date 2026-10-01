"""Tests for per-modality-timestamp (every-capture) samples."""

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest
from upath import UPath

from olmoearth_pretrain.data.collate import collate_olmoearth_pretrain
from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality
from olmoearth_pretrain.data.dataset import (
    GetItemArgs,
    OlmoEarthDataset,
    build_union_timeline,
    timestamps_to_ordinals,
)
from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, OlmoEarthSample
from olmoearth_pretrain.nn.tokenization import ModalityTokenization, TokenizationConfig

MODALITIES = [
    Modality.SENTINEL2_L2A.name,
    Modality.SENTINEL1.name,
    Modality.LANDSAT_L2.name,
    Modality.SRTM.name,
]


def _ts(*dates: tuple[int, int, int]) -> np.ndarray:
    """(day, month 1-12, year) tuples -> stored [day, month-1, year] timestamps."""
    return np.array([[d, m - 1, y] for d, m, y in dates], dtype=np.int32)


def test_build_union_timeline_merges_same_day_captures() -> None:
    """Same-day captures share a step; a modality's repeat captures get their own."""
    s2 = _ts((1, 1, 2022), (6, 1, 2022), (11, 1, 2022))
    s1 = _ts((6, 1, 2022), (6, 1, 2022), (20, 1, 2022))
    union, steps = build_union_timeline({"s2": s2, "s1": s1})
    # Steps: Jan 1, Jan 6 (s2 + s1's 1st), Jan 6 (s1's 2nd), Jan 11, Jan 20.
    assert union[:, 0].tolist() == [1, 6, 6, 11, 20]
    assert steps["s2"].tolist() == [0, 1, 3]
    assert steps["s1"].tolist() == [1, 2, 4]
    assert (np.diff(timestamps_to_ordinals(union)) >= 0).all()


def test_build_union_timeline_handles_empty_modalities() -> None:
    """A modality with no captures contributes no steps."""
    union, steps = build_union_timeline(
        {"s2": _ts((3, 2, 2021)), "s1": np.zeros((0, 3), dtype=np.int32)}
    )
    assert union.shape == (1, 3)
    assert steps["s1"].shape == (0,)


@pytest.fixture
def allcap_h5py_dir(tmp_path: Path) -> UPath:
    """Two every-capture samples in the allcap h5 layout."""
    h5py_dir = (
        tmp_path
        / "h5py_data_w_missing_timesteps_zstd_3_128_x_4"
        / "landsat_l2_sentinel1_sentinel2_l2a_srtm"
        / "2"
    )
    h5py_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    # Sample 0: S2 every 5 days for a year, S1 every 12 days (one same-day
    # double capture), Landsat every 16 days.
    s2_ts = [(1 + d % 28, 1 + d // 30 % 12, 2022) for d in range(0, 360, 5)]
    s1_ts = [(1 + d % 28, 1 + d // 30 % 12, 2022) for d in range(0, 360, 12)]
    s1_ts.insert(3, s1_ts[3])
    ls_ts = [(1 + d % 28, 1 + d // 30 % 12, 2022) for d in range(0, 360, 16)]
    for idx in range(2):
        with h5py.File(h5py_dir / f"sample_{idx}.h5", "w") as f:
            for name, dates, bands in (
                ("sentinel2_l2a", s2_ts, 12),
                ("sentinel1", s1_ts, 2),
                ("landsat_l2", ls_ts, 8),
            ):
                if idx == 1 and name == "landsat_l2":
                    continue  # sample 1 has no Landsat
                order = np.argsort(timestamps_to_ordinals(_ts(*dates)), kind="stable")
                ts = _ts(*dates)[order]
                f.create_dataset(f"timestamps_{name}", data=ts)
                data = rng.integers(1, 3000, (16, 16, len(dates), bands))
                f.create_dataset(name, data=data.astype(np.uint16))
            f.create_dataset("srtm", data=rng.integers(0, 500, (16, 16, 1, 1)))
            f.create_dataset("latlon", data=np.array([1.0, 2.0], dtype=np.float32))
    pd.DataFrame(
        {
            "sample_index": [0, 1],
            "sentinel2_l2a": [1, 1],
            "sentinel1": [1, 1],
            "landsat_l2": [1, 0],
            "srtm": [1, 1],
        }
    ).to_csv(h5py_dir / "sample_metadata.csv", index=False)
    np.save(h5py_dir / "latlon_distribution.npy", np.zeros((2, 2), dtype=np.float32))
    return UPath(h5py_dir)


def _dataset(h5py_dir: UPath, normalize: bool = False) -> OlmoEarthDataset:
    dataset = OlmoEarthDataset(
        h5py_dir=h5py_dir,
        training_modalities=MODALITIES,
        dtype=np.float32,
        normalize=normalize,
        per_modality_timestamps=True,
    )
    dataset.prepare()
    return dataset


S2_SINGLE = TokenizationConfig(
    overrides={
        "sentinel2_l2a": ModalityTokenization(
            band_groups=[Modality.SENTINEL2_L2A.band_order]
        )
    }
)


def _field(sample: OlmoEarthSample | MaskedOlmoEarthSample, name: str) -> Any:
    """A present field of a sample (asserting it is not None)."""
    value = getattr(sample, name)
    assert value is not None
    return value


def _real_tokens(sample: OlmoEarthSample, hw_p: int, exclude: set[str]) -> int:
    """Tokens of present (modality, step) pairs with single-bandset tokenization."""
    total = 0
    for name in sample.modalities:
        if name in exclude:
            continue
        data = getattr(sample, name)
        present = (data != MISSING_VALUE).any(axis=(0, 1, 3))
        total += int(present.sum()) * hw_p * hw_p
    return total


def test_whole_span_keeps_every_capture(allcap_h5py_dir: UPath) -> None:
    """No budget, no range: every capture lands on the union timeline."""
    dataset = _dataset(allcap_h5py_dir)
    _, sample = dataset[GetItemArgs(idx=0, patch_size=4, sampled_hw_p=4)]
    assert _field(sample, "sentinel2_l2a").shape[:3] == (
        16,
        16,
        _field(sample, "timestamps").shape[0],
    )
    with h5py.File(allcap_h5py_dir / "sample_0.h5") as f:
        for name in ("sentinel2_l2a", "sentinel1", "landsat_l2"):
            present = (getattr(sample, name) != MISSING_VALUE).all(axis=(0, 1, 3))
            assert present.sum() == f[name].shape[2]
            # Data land on the right steps, in order.
            np.testing.assert_array_equal(
                getattr(sample, name)[:, :, present], f[name][()].astype(np.float32)
            )
    assert _field(sample, "srtm").shape == (16, 16, 1, 1)


def test_missing_modality_is_filled(allcap_h5py_dir: UPath) -> None:
    """A modality absent from the file is all MISSING."""
    dataset = _dataset(allcap_h5py_dir, normalize=True)
    _, sample = dataset[GetItemArgs(idx=1, patch_size=2, sampled_hw_p=4)]
    assert (_field(sample, "landsat_l2") == MISSING_VALUE).all()
    assert (_field(sample, "sentinel2_l2a") != MISSING_VALUE).any()


@pytest.mark.parametrize("hw_p,budget", [(4, 300), (2, 100), (8, 2000)])
def test_budget_counts_real_tokens_in_a_contiguous_run(
    allcap_h5py_dir: UPath, hw_p: int, budget: int
) -> None:
    """The kept steps fit the real-token budget and are consecutive captures."""
    dataset = _dataset(allcap_h5py_dir)
    np.random.seed(0)
    for _ in range(20):
        _, sample = dataset[
            GetItemArgs(
                idx=0,
                patch_size=2,
                sampled_hw_p=hw_p,
                token_budget=budget,
                tokenization_config=S2_SINGLE,
                budget_exclude_modalities=frozenset({"srtm"}),
            )
        ]
        assert _field(sample, "sentinel2_l2a").shape[:2] == (2 * hw_p, 2 * hw_p)
        assert _real_tokens(sample, hw_p, {"srtm"}) <= max(budget, 3 * hw_p * hw_p)
        # Every step of the run holds at least one capture.
        any_present = np.zeros(_field(sample, "timestamps").shape[0], dtype=bool)
        for name in ("sentinel2_l2a", "sentinel1", "landsat_l2"):
            any_present |= (getattr(sample, name) != MISSING_VALUE).any(axis=(0, 1, 3))
        assert any_present.all()


def test_time_range_limits_the_span(allcap_h5py_dir: UPath) -> None:
    """A 30-day range keeps only captures within 30 days of each other."""
    dataset = _dataset(allcap_h5py_dir)
    np.random.seed(0)
    for _ in range(20):
        _, sample = dataset[
            GetItemArgs(idx=0, patch_size=2, sampled_hw_p=2, time_range_days=30)
        ]
        ordinals = timestamps_to_ordinals(_field(sample, "timestamps"))
        assert 1 <= len(ordinals) and ordinals[-1] - ordinals[0] <= 30


def test_collate_pads_to_the_longest_sample(allcap_h5py_dir: UPath) -> None:
    """Different-length samples stack with MISSING-padded tails."""
    dataset = _dataset(allcap_h5py_dir)
    np.random.seed(1)
    batch = [
        dataset[GetItemArgs(idx=0, patch_size=2, sampled_hw_p=2, time_range_days=r)]
        for r in (7, 365)
    ]
    lengths = [_field(s, "timestamps").shape[0] for _, s in batch]
    assert lengths[0] < lengths[1]
    _, collated = collate_olmoearth_pretrain(batch)
    assert _field(collated, "timestamps").shape == (2, lengths[1], 3)
    short = _field(collated, "sentinel2_l2a")[0, :, :, lengths[0] :]
    assert (short == MISSING_VALUE).all()
    # Padded timestamps repeat the last real one.
    assert (
        _field(collated, "timestamps")[0, lengths[0] :]
        == _field(collated, "timestamps")[0, lengths[0] - 1]
    ).all()


def test_uint8_masks_round_trip(allcap_h5py_dir: UPath) -> None:
    """uint8 transport masks come back from to_device identical to int64 masks."""
    import torch

    from olmoearth_pretrain.data.collate import collate_double_masked_batched
    from olmoearth_pretrain.train.masking import MaskingConfig

    dataset = _dataset(allcap_h5py_dir, normalize=True)
    np.random.seed(2)
    batch = [
        dataset[GetItemArgs(idx=i, patch_size=2, sampled_hw_p=4, time_range_days=90)]
        for i in (0, 1)
    ]
    masking = MaskingConfig(strategy_config={"type": "random_time_with_decode"}).build()
    outputs = []
    for uint8_masks in (False, True):
        np.random.seed(3)
        torch.manual_seed(3)
        outputs.append(
            collate_double_masked_batched(
                batch, None, masking, None, uint8_masks=uint8_masks
            )
        )
    (_, ref_a, ref_b), (_, small_a, small_b) = outputs
    assert _field(small_a, "sentinel2_l2a_mask").dtype == torch.uint8
    for ref, small in ((ref_a, small_a), (ref_b, small_b)):
        moved = small.to_device(torch.device("cpu"))
        for name, value in ref.as_dict().items():
            restored = getattr(moved, name)
            assert restored.dtype == value.dtype, name
            assert torch.equal(restored, value), name
