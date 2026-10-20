"""Tests for per-modality-timestamp (every-capture) samples."""

import warnings
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
    nanmedian_axis2,
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
                if name == "sentinel2_l2a":
                    # Scene classification per S2 capture; sample 1's has one
                    # extra capture S2 lacks (as in ~0.3% of the corpus).
                    # (Day 29 never occurs in the S2 dates above.)
                    scl_ts = (
                        ts
                        if idx == 0
                        else np.concatenate([ts[:4], [[29, 0, 2022]], ts[4:]])
                    )
                    f.create_dataset("timestamps_sentinel2_scl", data=scl_ts)
                    scl = rng.integers(0, 12, (16, 16, len(scl_ts), 1))
                    f.create_dataset("sentinel2_scl", data=scl.astype(np.uint8))
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


def _dataset(
    h5py_dir: UPath, normalize: bool = False, load_s2_cloud_mask: bool = False
) -> OlmoEarthDataset:
    dataset = OlmoEarthDataset(
        h5py_dir=h5py_dir,
        training_modalities=MODALITIES,
        dtype=np.float32,
        normalize=normalize,
        per_modality_timestamps=True,
        load_s2_cloud_mask=load_s2_cloud_mask,
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


TEMPORAL = ("sentinel2_l2a", "sentinel1", "landsat_l2")


def _dense(sample: OlmoEarthSample) -> dict[str, np.ndarray]:
    """Unbatched compact sample -> each temporal modality on the union timeline."""
    num_steps = _field(sample, "timestamps").shape[0]
    dense = {}
    for name in TEMPORAL:
        data = _field(sample, name)
        index = _field(sample, f"{name}_time_index")
        out = np.full((*data.shape[:2], num_steps, data.shape[3]), MISSING_VALUE)
        out[:, :, index[index >= 0]] = data[:, :, index >= 0]
        dense[name] = out.astype(np.float32)
    return dense


def test_whole_span_keeps_every_capture(allcap_h5py_dir: UPath) -> None:
    """No budget, no range: each modality keeps all its captures, in order."""
    dataset = _dataset(allcap_h5py_dir)
    _, sample = dataset[GetItemArgs(idx=0, patch_size=4, sampled_hw_p=4)]
    num_steps = _field(sample, "timestamps").shape[0]
    with h5py.File(allcap_h5py_dir / "sample_0.h5") as f:
        for name in TEMPORAL:
            # Compact: exactly the file's captures, chronological.
            np.testing.assert_array_equal(
                _field(sample, name), f[name][()].astype(np.float32)
            )
            index = _field(sample, f"{name}_time_index")
            assert index.shape == (f[name].shape[2],)
            assert (np.diff(index) > 0).all() and 0 <= index.min()
            assert index.max() < num_steps
            # The indexed timeline rows are the modality's own dates.
            np.testing.assert_array_equal(
                _field(sample, "timestamps")[index], f[f"timestamps_{name}"][()]
            )
    assert _field(sample, "srtm").shape == (16, 16, 1, 1)


def test_missing_modality_is_filled(allcap_h5py_dir: UPath) -> None:
    """A modality absent from the file is one MISSING padding slot."""
    dataset = _dataset(allcap_h5py_dir, normalize=True)
    _, sample = dataset[GetItemArgs(idx=1, patch_size=2, sampled_hw_p=4)]
    assert _field(sample, "landsat_l2").shape == (8, 8, 1, 8)
    assert (_field(sample, "landsat_l2") == MISSING_VALUE).all()
    assert _field(sample, "landsat_l2_time_index").tolist() == [-1]
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
        dense = _dense(sample)
        assert dense["sentinel2_l2a"].shape[:2] == (2 * hw_p, 2 * hw_p)
        real = sum(
            int((d != MISSING_VALUE).any(axis=(0, 1, 3)).sum()) for d in dense.values()
        )
        assert real * hw_p * hw_p <= max(budget, 3 * hw_p * hw_p)
        # Every step of the run holds at least one capture.
        any_present = np.zeros(_field(sample, "timestamps").shape[0], dtype=bool)
        for d in dense.values():
            any_present |= (d != MISSING_VALUE).any(axis=(0, 1, 3))
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


def test_collate_pads_each_modality_to_its_longest(allcap_h5py_dir: UPath) -> None:
    """Each modality pads to its own batch max (MISSING data, -1 time index)."""
    dataset = _dataset(allcap_h5py_dir)
    np.random.seed(1)
    batch = [
        dataset[GetItemArgs(idx=0, patch_size=2, sampled_hw_p=2, time_range_days=r)]
        for r in (7, 365)
    ]
    _, collated = collate_olmoearth_pretrain(batch)
    for name in TEMPORAL:
        lengths = [_field(s, name).shape[2] for _, s in batch]
        assert _field(collated, name).shape[3] == max(lengths)
        assert _field(collated, f"{name}_time_index").shape == (2, max(lengths))
        short = int(np.argmin(lengths))
        if lengths[short] < max(lengths):
            tail = slice(lengths[short], None)
            assert (_field(collated, name)[short, :, :, tail] == MISSING_VALUE).all()
            assert (_field(collated, f"{name}_time_index")[short, tail] == -1).all()
    t_lengths = [_field(s, "timestamps").shape[0] for _, s in batch]
    assert _field(collated, "timestamps").shape == (2, max(t_lengths), 3)


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


def test_microbatches_trim_shared_time_padding(allcap_h5py_dir: UPath) -> None:
    """Each microbatch keeps only its own longest per-modality time axis."""
    import torch

    from olmoearth_pretrain.data.collate import collate_double_masked_batched
    from olmoearth_pretrain.train.masking import MaskingConfig
    from olmoearth_pretrain.train.utils import split_masked_batch

    dataset = _dataset(allcap_h5py_dir, normalize=True)
    np.random.seed(4)
    batch = [
        dataset[GetItemArgs(idx=0, patch_size=2, sampled_hw_p=2, time_range_days=r)]
        for r in (7, 365)
    ]
    masking = MaskingConfig(strategy_config={"type": "random_time_with_decode"}).build()
    _, masked, _ = collate_double_masked_batched(batch, None, masking, None)
    for i, micro in enumerate(split_masked_batch(masked, microbatch_size=1)):
        for name in TEMPORAL:
            length = max(_field(batch[i][1], name).shape[2], 1)
            mask_name = f"{name}_mask"
            assert _field(micro, f"{name}_time_index").shape == (1, length)
            assert _field(micro, name).shape[3] == length
            assert _field(micro, mask_name).shape[3] == length
            for field in (name, mask_name, f"{name}_time_index"):
                assert torch.equal(
                    getattr(micro, field),
                    _field(masked, field)[i : i + 1, ..., :length]
                    if field.endswith("_time_index")
                    else _field(masked, field)[i : i + 1, :, :, :length],
                ), field


def test_s2_cloud_flags_follow_s2_captures(allcap_h5py_dir: UPath) -> None:
    """Cloud flags = cloudy SCL classes on S2's own captures (matched by date)."""
    import h5py

    from olmoearth_pretrain.data.dataset import CLOUD_SCL_CLASSES

    dataset = _dataset(allcap_h5py_dir, load_s2_cloud_mask=True)
    for idx in (0, 1):
        _, sample = dataset[GetItemArgs(idx=idx, patch_size=1, sampled_hw_p=16)]
        with h5py.File(allcap_h5py_dir / f"sample_{idx}.h5", "r") as f:
            scl = f["sentinel2_scl"][()][..., 0]
            scl_ts = f["timestamps_sentinel2_scl"][()]
            s2_ts = f["timestamps_sentinel2_l2a"][()]
        rows = [next(i for i, t in enumerate(scl_ts) if (t == s).all()) for s in s2_ts]
        expected = np.isin(scl[:, :, rows], CLOUD_SCL_CLASSES).astype(np.uint8)
        np.testing.assert_array_equal(_field(sample, "sentinel2_l2a_cloud"), expected)
        assert "sentinel2_l2a_cloud" not in sample.modalities


def test_sliced_read_matches_the_file(allcap_h5py_dir: UPath) -> None:
    """A sliced read returns the file's values at the crop and kept captures.

    Crop + time range + budget read only part of the file; values, dates and cloud
    flags must match the file at that crop and those captures.
    """
    from olmoearth_pretrain.data.dataset import CLOUD_SCL_CLASSES

    dataset = _dataset(allcap_h5py_dir, load_s2_cloud_mask=True)
    np.random.seed(1)
    for trial in range(12):
        idx = trial % 2
        _, sample = dataset[
            GetItemArgs(
                idx=idx,
                patch_size=2,
                sampled_hw_p=3,
                token_budget=200,
                time_range_days=90.0,
                tokenization_config=S2_SINGLE,
                budget_exclude_modalities=frozenset({"srtm"}),
            )
        ]
        union = _field(sample, "timestamps")
        with h5py.File(allcap_h5py_dir / f"sample_{idx}.h5", "r") as f:
            s2 = _field(sample, "sentinel2_l2a")
            # Locate the crop from the first S2 capture.
            file_s2 = f["sentinel2_l2a"][()].astype(np.float32)
            offsets = [
                (h, w)
                for h in range(16 - 6 + 1)
                for w in range(16 - 6 + 1)
                if any(
                    (file_s2[h : h + 6, w : w + 6, t] == s2[:, :, 0]).all()
                    for t in range(file_s2.shape[2])
                )
            ]
            assert len(offsets) == 1
            h, w = offsets[0]
            for name in TEMPORAL:
                if name not in f:
                    continue
                data = _field(sample, name)
                index = _field(sample, f"{name}_time_index")
                file_data = f[name][()].astype(np.float32)[h : h + 6, w : w + 6]
                file_ts = f[f"timestamps_{name}"][()]
                for slot, step in enumerate(index):
                    if step < 0:  # no capture in the run: one MISSING slot
                        assert (data[:, :, slot] == MISSING_VALUE).all()
                        continue
                    rows = np.flatnonzero((file_ts == union[step]).all(axis=1))
                    assert any(
                        (file_data[:, :, r] == data[:, :, slot]).all() for r in rows
                    )
            scl = f["sentinel2_scl"][()][h : h + 6, w : w + 6, :, 0]
            scl_ts = f["timestamps_sentinel2_scl"][()]
            s2_index = _field(sample, "sentinel2_l2a_time_index")
            if (s2_index < 0).any():
                continue
            s2_dates = union[s2_index]
            rows = [np.flatnonzero((scl_ts == d).all(axis=1))[0] for d in s2_dates]
            expected = np.isin(scl[:, :, rows], CLOUD_SCL_CLASSES).astype(np.uint8)
            np.testing.assert_array_equal(
                _field(sample, "sentinel2_l2a_cloud"), expected
            )


def test_s2_cloud_flags_leave_sampling_unchanged(allcap_h5py_dir: UPath) -> None:
    """Loading cloud flags draws the same crops, ranges and captures (same batches)."""
    plain = _dataset(allcap_h5py_dir)
    cloudy = _dataset(allcap_h5py_dir, load_s2_cloud_mask=True)
    for seed in range(3):
        samples = []
        for dataset in (plain, cloudy):
            np.random.seed(seed)
            args = GetItemArgs(
                idx=seed % 2,
                patch_size=2,
                sampled_hw_p=3,
                time_range_days=90,
                token_budget=400,
                tokenization_config=S2_SINGLE,
            )
            samples.append(dataset[args][1])
        for name, value in samples[0].as_dict().items():
            np.testing.assert_array_equal(getattr(samples[1], name), value)
        cloud = _field(samples[1], "sentinel2_l2a_cloud")
        assert cloud.shape == _field(samples[1], "sentinel2_l2a").shape[:3]


def test_cloudy_s2_targets_become_missing(allcap_h5py_dir: UPath) -> None:
    """Only S2 decoder targets whose patch is mostly cloud change (to MISSING)."""
    import torch

    from olmoearth_pretrain.data.collate import (
        CLOUDY_TOKEN_FRACTION,
        collate_double_masked_batched,
    )
    from olmoearth_pretrain.datatypes import MaskValue
    from olmoearth_pretrain.train.masking import MaskingConfig

    dataset = _dataset(allcap_h5py_dir, normalize=True, load_s2_cloud_mask=True)
    np.random.seed(5)
    items = [
        dataset[
            GetItemArgs(idx=i % 2, patch_size=2, sampled_hw_p=4, time_range_days=180)
        ]
        for i in range(4)
    ]
    stripped = [(p, s._replace(sentinel2_l2a_cloud=None)) for p, s in items]
    masking = MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode",
            "only_decode_modalities": ["srtm"],
        }
    ).build()
    outputs = []
    for batch in (stripped, items):
        np.random.seed(6)
        torch.manual_seed(6)
        outputs.append(collate_double_masked_batched(batch, None, masking, None))
    (_, ref_a, ref_b), (_, out_a, out_b) = outputs
    cloud = _field(collate_olmoearth_pretrain(items)[1], "sentinel2_l2a_cloud")
    fraction = cloud.float().reshape(4, 4, 2, 4, 2, -1).mean(dim=(2, 4))
    cloudy = (
        (fraction > CLOUDY_TOKEN_FRACTION)
        .repeat_interleave(2, 1)
        .repeat_interleave(2, 2)
    )
    dropped = 0
    for ref, out in ((ref_a, out_a), (ref_b, out_b)):
        for name, value in ref.as_dict().items():
            if name != "sentinel2_l2a_mask":
                assert torch.equal(getattr(out, name), value), name
        before, after = (
            _field(ref, "sentinel2_l2a_mask"),
            _field(out, "sentinel2_l2a_mask"),
        )
        expect_drop = cloudy.unsqueeze(-1) & (before == MaskValue.DECODER.value)
        assert torch.equal(
            after[expect_drop],
            torch.full_like(after[expect_drop], MaskValue.MISSING.value),
        )
        assert torch.equal(after[~expect_drop], before[~expect_drop])
        dropped += int(expect_drop.sum())
    assert dropped > 0


def _densify(batch: OlmoEarthSample) -> OlmoEarthSample:
    """Batched compact sample -> the previous union-timeline layout (no time index)."""
    import torch

    timestamps = _field(batch, "timestamps")
    num_steps = timestamps.shape[1]
    fields = {k: v for k, v in batch.as_dict().items() if not k.endswith("_time_index")}
    for name in TEMPORAL:
        data = _field(batch, name)
        index = _field(batch, f"{name}_time_index")
        out = torch.full(
            (*data.shape[:3], num_steps, data.shape[4]), MISSING_VALUE, dtype=data.dtype
        )
        for b in range(data.shape[0]):
            valid = index[b] >= 0
            out[b][:, :, index[b][valid]] = data[b][:, :, valid]
        fields[name] = out
    return OlmoEarthSample(**fields)


def _densify_mask(mask: Any, index: Any, num_steps: int) -> Any:
    """[B, H, W, T_m, bs] compact mask -> [B, H, W, T_u, bs] (MISSING elsewhere)."""
    import torch

    from olmoearth_pretrain.datatypes import MaskValue

    out = torch.full(
        (*mask.shape[:3], num_steps, mask.shape[4]),
        MaskValue.MISSING.value,
        dtype=mask.dtype,
    )
    for b in range(mask.shape[0]):
        valid = index[b] >= 0
        out[b][:, :, index[b][valid]] = mask[b][:, :, valid]
    return out


def _densify_tokens(tokens: Any, index: Any, num_steps: int) -> Any:
    """[B, H, W, T_m, bs, D] compact tokens -> [B, H, W, T_u, bs, D] (zeros)."""
    import torch

    out = torch.zeros((*tokens.shape[:3], num_steps, *tokens.shape[4:]))
    for b in range(tokens.shape[0]):
        valid = index[b] >= 0
        out[b][:, :, index[b][valid]] = tokens[b][:, :, valid].float()
    return out


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("random_ratio", [0.0, 1.0])  # time / random masking
def test_compact_layout_is_equivalent_to_union_layout(
    allcap_h5py_dir: UPath, seed: int, random_ratio: float
) -> None:
    """Own-time-axis tensors give the same masks, model outputs and loss.

    The previous layout put every modality on the union timeline with MISSING
    elsewhere; the compact layout drops those slots and keeps a time index. Real
    tokens, their order, dates and masking RNG draws are unchanged, so masking
    (random_time_with_decode, both its random and time branches), the
    encoder, the decoder, pooling and the patch-discrimination loss must agree.
    """
    import torch

    from olmoearth_pretrain.datatypes import (
        MaskedOlmoEarthSample,
        MaskValue,
        time_index_field,
    )
    from olmoearth_pretrain.nn.flexi_vit import EncoderConfig, PredictorConfig
    from olmoearth_pretrain.nn.latent_mim import LatentMIMConfig
    from olmoearth_pretrain.train.loss import LossConfig
    from olmoearth_pretrain.train.masking import MaskingConfig

    dataset = _dataset(allcap_h5py_dir, normalize=True)
    np.random.seed(seed)
    items = [
        dataset[
            GetItemArgs(idx=i % 2, patch_size=2, sampled_hw_p=4, time_range_days=90)
        ]
        for i in range(3)
    ]
    _, compact = collate_olmoearth_pretrain(items)
    dense = _densify(compact)
    num_steps = _field(compact, "timestamps").shape[1]

    masking = MaskingConfig(
        strategy_config={
            "type": "random_time_with_decode",
            "only_decode_modalities": ["srtm"],
            "random_ratio": random_ratio,
        }
    ).build()
    masked: dict[str, MaskedOlmoEarthSample] = {}
    for name, batch in (("dense", dense), ("compact", compact)):
        np.random.seed(100 + seed)
        torch.manual_seed(100 + seed)
        out = masking.apply_mask(batch, patch_size=2)
        if name == "compact":
            out = out._replace(
                **{
                    time_index_field(m): getattr(compact, time_index_field(m))
                    for m in TEMPORAL
                }
            )
        masked[name] = out
    for m in TEMPORAL:
        index = _field(compact, f"{m}_time_index")
        torch.testing.assert_close(
            _densify_mask(_field(masked["compact"], f"{m}_mask"), index, num_steps),
            _field(masked["dense"], f"{m}_mask"),
        )
    # Non-trivial masks: some tokens are encoded and some decoded.
    all_masks = torch.cat(
        [_field(masked["dense"], f"{m}_mask").flatten() for m in TEMPORAL]
    )
    assert (all_masks == MaskValue.ONLINE_ENCODER.value).any()
    assert (all_masks == MaskValue.DECODER.value).any()

    modalities = [*TEMPORAL, "srtm"]
    model = LatentMIMConfig(
        encoder_config=EncoderConfig(
            supported_modality_names=modalities,
            embedding_size=16,
            num_heads=2,
            depth=2,
            mlp_ratio=2.0,
            drop_path=0.0,
            position_encoding="rope_3d_mixed",
            rope_temporal_coordinate_scale=1.0 / 30.0,
        ),
        decoder_config=PredictorConfig(
            supported_modality_names=modalities,
            encoder_embedding_size=16,
            decoder_embedding_size=16,
            depth=2,
            mlp_ratio=2.0,
            num_heads=2,
            position_encoding="rope_3d_mixed",
            rope_temporal_coordinate_scale=1.0 / 30.0,
        ),
        projection_only_target=True,
    ).build()
    model.train()
    loss_fn = LossConfig(
        loss_config={
            "type": "modality_patch_discrimination_masked_negatives_vec",
            "mask_negatives_for_modalities": ["srtm"],
        }
    ).build()
    outputs: dict[str, tuple[Any, Any, Any, Any]] = {}
    for name in ("dense", "compact"):
        masked_batch = masked[name]
        latent, decoded, pooled, *_ = model(masked_batch, patch_size=2)
        target = model.target_encoder(masked_batch.unmask(), patch_size=2)[
            "tokens_and_masks"
        ]
        outputs[name] = (latent, decoded, pooled, loss_fn.compute(decoded, target))

    (lat_d, dec_d, pool_d, loss_d), (lat_c, dec_c, pool_c, loss_c) = (
        outputs["dense"],
        outputs["compact"],
    )
    torch.testing.assert_close(pool_c, pool_d, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(loss_c, loss_d, rtol=1e-4, atol=1e-5)
    for m in TEMPORAL:
        index = _field(compact, f"{m}_time_index")
        mask = _field(masked["dense"], f"{m}_mask")[:, ::2, ::2]
        for got, want, value in (
            (lat_c, lat_d, MaskValue.ONLINE_ENCODER),
            (dec_c, dec_d, MaskValue.DECODER),
        ):
            sel = mask == value.value
            torch.testing.assert_close(
                _densify_tokens(getattr(got, m), index, num_steps)[sel],
                getattr(want, m)[sel].float(),
                rtol=1e-4,
                atol=1e-5,
            )


def _raw(h5py_dir: UPath, idx: int) -> dict[str, np.ndarray]:
    import h5py

    with h5py.File(h5py_dir / f"sample_{idx}.h5", "r") as f:
        return {k: f[k][()] for k in f.keys()}


def _expected_composites(
    values: np.ndarray,
    ordinals: np.ndarray,
    origin: int,
    days: float,
    clear: np.ndarray | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Per-pixel median per occupied bin, written out with plain loops."""
    bins = sorted({int((o - origin) // days) for o in ordinals})
    out = np.zeros((*values.shape[:2], len(bins), values.shape[3]), dtype=np.float32)
    for j, b in enumerate(bins):
        members = [i for i, o in enumerate(ordinals) if int((o - origin) // days) == b]
        for r in range(values.shape[0]):
            for c in range(values.shape[1]):
                use = members
                if clear is not None:
                    ok = [i for i in members if clear[r, c, i]]
                    use = ok or members
                out[r, c, j] = np.median(values[r, c, use].astype(np.float32), axis=0)
    return out, bins


def test_composites_are_per_bin_medians(allcap_h5py_dir: UPath) -> None:
    """Composites are per-bin medians, one slot per occupied bin.

    S2 = median over SCL-clear captures (falling back to all captures), S1 and
    Landsat = median of all captures; slots are dated at the bin centre and bins
    are shared across modalities.
    """
    from olmoearth_pretrain.data.dataset import CLOUD_SCL_CLASSES

    dataset = _dataset(allcap_h5py_dir, load_s2_cloud_mask=True)
    raw = _raw(allcap_h5py_dir, 0)
    ords = {
        m: timestamps_to_ordinals(raw[f"timestamps_{m}"])
        for m in ("sentinel2_l2a", "sentinel1", "landsat_l2")
    }
    origin = min(int(o.min()) for o in ords.values())
    for days in (30.0, 14.0):
        _, sample = dataset[
            GetItemArgs(idx=0, patch_size=1, sampled_hw_p=16, composite_days=days)
        ]
        union = _field(sample, "timestamps")
        clear = ~np.isin(raw["sentinel2_scl"][..., 0], CLOUD_SCL_CLASSES)
        for m in ("sentinel2_l2a", "sentinel1", "landsat_l2"):
            expected, bins = _expected_composites(
                raw[m],
                ords[m],
                origin,
                days,
                clear if m == "sentinel2_l2a" else None,
            )
            np.testing.assert_allclose(_field(sample, m), expected, rtol=1e-6)
            index = _field(sample, f"{m}_time_index")
            centres = [origin + int(b * days + days / 2) for b in bins]
            np.testing.assert_array_equal(
                timestamps_to_ordinals(union[index]), np.array(centres)
            )
        # A composite pixel is flagged cloudy only when no capture in its bin is clear.
        expected_flag = np.zeros(_field(sample, "sentinel2_l2a").shape[:3], np.uint8)
        _, s2_bins = _expected_composites(
            raw["sentinel2_l2a"][:1, :1], ords["sentinel2_l2a"], origin, days
        )
        for j, b in enumerate(s2_bins):
            members = [
                i
                for i, o in enumerate(ords["sentinel2_l2a"])
                if int((o - origin) // days) == b
            ]
            expected_flag[:, :, j] = ~clear[:, :, members].any(axis=2)
        np.testing.assert_array_equal(
            _field(sample, "sentinel2_l2a_cloud"), expected_flag
        )


def test_composites_shrink_the_time_axis(allcap_h5py_dir: UPath) -> None:
    """Every capture > biweekly > monthly in S2 slots; 0 days = every capture."""
    dataset = _dataset(allcap_h5py_dir)
    lengths = {}
    for days in (None, 0.0, 14.0, 30.0):
        np.random.seed(0)
        _, sample = dataset[
            GetItemArgs(idx=0, patch_size=1, sampled_hw_p=16, composite_days=days)
        ]
        lengths[days] = _field(sample, "sentinel2_l2a").shape[2]
        if days == 0.0:
            np.random.seed(0)
            _, plain = dataset[GetItemArgs(idx=0, patch_size=1, sampled_hw_p=16)]
            for name, value in plain.as_dict().items():
                np.testing.assert_array_equal(getattr(sample, name), value)
    assert lengths[None] == lengths[0.0] > lengths[14.0] > lengths[30.0] >= 12


def test_composites_respect_the_token_budget(allcap_h5py_dir: UPath) -> None:
    """The real-token budget is applied to the composites, not the raw captures."""
    dataset = _dataset(allcap_h5py_dir)
    for seed in range(3):
        np.random.seed(seed)
        args = GetItemArgs(
            idx=0,
            patch_size=2,
            sampled_hw_p=3,
            token_budget=300,
            tokenization_config=S2_SINGLE,
            composite_days=30.0,
            budget_exclude_modalities=frozenset({"srtm"}),
        )
        _, sample = dataset[args]
        assert _real_tokens(sample, 3, {"srtm"}) <= 300


def test_decoder_token_cap() -> None:
    """At most max_decode_tokens DECODER tokens per sample survive the cap.

    The rest become TARGET_ENCODER_ONLY; encoder tokens and patch-uniformity are
    untouched.
    """
    import torch

    from olmoearth_pretrain.data.collate import _cap_decoder_tokens
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

    torch.manual_seed(0)
    p, b = 2, 3
    masks = {}
    for name, t in (("sentinel2_l2a", 5), ("sentinel1", 4)):
        tok = torch.randint(0, 2, (b, 4, 4, t, 1))  # 0 encoder, 1 -> decoder
        tok = torch.where(
            tok == 1, MaskValue.DECODER.value, MaskValue.ONLINE_ENCODER.value
        )
        masks[f"{name}_mask"] = tok.repeat_interleave(p, 1).repeat_interleave(p, 2)
    sample = MaskedOlmoEarthSample(
        timestamps=torch.zeros(b, 5, 3, dtype=torch.long),
        sentinel2_l2a=torch.zeros(b, 8, 8, 5, 12),
        sentinel1=torch.zeros(b, 8, 8, 4, 2),
        **masks,
    )
    capped = _cap_decoder_tokens(sample, p, 20)
    for name in ("sentinel2_l2a_mask", "sentinel1_mask"):
        before, after = getattr(sample, name), getattr(capped, name)
        enc = before == MaskValue.ONLINE_ENCODER.value
        assert torch.equal(after[enc], before[enc])
        changed = before != after
        assert (after[changed] == MaskValue.TARGET_ENCODER_ONLY.value).all()
        # still uniform within each p x p patch
        assert torch.equal(
            after, after[:, ::p, ::p].repeat_interleave(p, 1).repeat_interleave(p, 2)
        )
    counts = sum(
        (getattr(capped, n)[:, ::p, ::p] == MaskValue.DECODER.value)
        .reshape(b, -1)
        .sum(1)
        for n in ("sentinel2_l2a_mask", "sentinel1_mask")
    )
    before_counts = sum(
        (getattr(sample, n)[:, ::p, ::p] == MaskValue.DECODER.value)
        .reshape(b, -1)
        .sum(1)
        for n in ("sentinel2_l2a_mask", "sentinel1_mask")
    )
    assert torch.equal(counts, torch.clamp(before_counts, max=20))
    assert _cap_decoder_tokens(sample, p, None) is sample


@pytest.mark.parametrize("seed", range(5))
def test_nanmedian_axis2_matches_numpy(seed: int) -> None:
    """The sort-based median equals np.nanmedian exactly (ties, all-NaN pixels)."""
    rng = np.random.default_rng(seed)
    for n in range(1, 9):
        x = np.round(rng.random((6, 5, n, 3)) * 50).astype(np.float32)
        x[rng.random((6, 5, n)) < 0.5] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels
            expected = np.nanmedian(x, axis=2)
        np.testing.assert_array_equal(nanmedian_axis2(x), expected)


def _assert_same_item(
    a: tuple[int, OlmoEarthSample], b: tuple[int, OlmoEarthSample]
) -> None:
    assert a[0] == b[0]
    da, db = a[1].as_dict(), b[1].as_dict()
    assert set(da) == set(db)
    for name, value in da.items():
        np.testing.assert_array_equal(value, db[name], err_msg=name)


@pytest.mark.parametrize("composite_days", [None, 30.0])
def test_read_then_finish_equals_getitem(
    allcap_h5py_dir: UPath, composite_days: float | None
) -> None:
    """read_item + finish_item is self[args]; a seeded item is reproducible."""
    dataset = _dataset(allcap_h5py_dir, normalize=True, load_s2_cloud_mask=True)
    for seed in range(6):
        args = GetItemArgs(
            idx=seed % 2,
            patch_size=2,
            sampled_hw_p=4,
            token_budget=200,
            tokenization_config=S2_SINGLE,
            time_range_days=120.0,
            composite_days=composite_days,
            seed=seed,
        )
        whole = dataset[args]
        _assert_same_item(whole, dataset.finish_item(args, dataset.read_item(args)))
        _assert_same_item(whole, dataset[args])


def test_read_ahead_matches_sequential_items(allcap_h5py_dir: UPath) -> None:
    """The read-ahead iterator yields the seeded items in order, at any depth."""
    from types import SimpleNamespace

    from olmoearth_pretrain.data.dataloader import (
        OlmoEarthDataLoader,
        _IterableDatasetWrapper,
    )

    dataset = _dataset(allcap_h5py_dir)
    params = [
        (i % 2, 2, 4, None, 90.0, 30.0 if i % 3 == 0 else None) for i in range(10)
    ]

    loader = SimpleNamespace(
        token_budget=200,
        tokenization_config=S2_SINGLE,
        budget_exclude_modalities=frozenset(),
    )

    def make_args(*p: Any, seed: int | None = None) -> GetItemArgs:
        get_args: Any = OlmoEarthDataLoader._get_item_args
        return get_args(loader, *p, seed=seed)

    np.random.seed(0)
    expected = [
        dataset[make_args(*p, seed=int(np.random.randint(2**31 - 1)))] for p in params
    ]
    for depth in (1, 3):
        wrapper = SimpleNamespace(
            data_loader=SimpleNamespace(read_ahead=depth, _get_item_args=make_args),
            dataset=dataset,
        )
        np.random.seed(0)
        got = list(
            _IterableDatasetWrapper._read_ahead_items(wrapper, iter(params))  # type: ignore[arg-type]
        )
        assert len(got) == len(expected)
        for a, b in zip(got, expected):
            _assert_same_item(a, b)
