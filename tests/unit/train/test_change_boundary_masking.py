"""Tests for the pre/post guarantee of change samples under time masking."""

import numpy as np
import torch

from olmoearth_pretrain.data.change_boundary import (
    has_change_boundary,
    restrict_start_ts_to_change_boundary,
    timestep_is_post,
)
from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.train.masking import (
    MaskValue,
    RandomTimeWithDecodeMaskingStrategy,
)

# Monthly timestamps Jan..Dec 2021 in the [day, month0, year] convention; the
# boundary is July 1st so months 0-5 are pre and 6-11 are post.
_TIMESTAMPS = torch.tensor([[1, m, 2021] for m in range(12)], dtype=torch.int64)
_BOUNDARY = torch.tensor([1, 6, 2021], dtype=torch.int64)


def test_timestep_is_post_is_lexicographic_and_accepts_numpy() -> None:
    """The comparison is lexicographic on (y, m, d); numpy inputs are accepted."""
    timestamps = torch.tensor(
        [[[30, 5, 2021], [1, 6, 2021], [2, 6, 2021], [1, 6, 2020]]]
    )
    boundary = torch.tensor([[1, 6, 2021]])
    expected = torch.tensor([[False, True, True, False]])
    assert torch.equal(timestep_is_post(timestamps, boundary), expected)
    # Unbatched numpy inputs (the dataset path).
    assert torch.equal(
        timestep_is_post(timestamps[0].numpy(), boundary[0].numpy()), expected[0]
    )


def test_has_change_boundary_detects_missing_fill() -> None:
    """Missing-filled boundaries mark non-change samples."""
    boundary = torch.tensor([[1, 6, 2021], [MISSING_VALUE] * 3])
    assert torch.equal(has_change_boundary(boundary), torch.tensor([True, False]))
    assert not bool(has_change_boundary(boundary[1]))
    assert not bool(has_change_boundary(None))


def test_restrict_start_ts_keeps_only_straddling_windows() -> None:
    """With max_t=2 the only window covering both sides starts at 5."""
    starts = list(range(11))
    kept = restrict_start_ts_to_change_boundary(starts, 2, _TIMESTAMPS, _BOUNDARY)
    assert kept == [5]
    # All-post timestamps: the constraint cannot be met, so it is dropped.
    all_post = _TIMESTAMPS.clone()
    all_post[:, 2] = 2022
    assert (
        restrict_start_ts_to_change_boundary(starts, 2, all_post, _BOUNDARY) == starts
    )
    # No boundary -> unchanged; max_t == 1 -> unchanged.
    assert (
        restrict_start_ts_to_change_boundary(
            starts, 2, _TIMESTAMPS, np.full(3, MISSING_VALUE)
        )
        == starts
    )
    assert (
        restrict_start_ts_to_change_boundary(starts, 1, _TIMESTAMPS, _BOUNDARY)
        == starts
    )


def test_ensure_pre_and_post_encoded_swaps_in_missing_side() -> None:
    """An encode set with no post timestep gets one from the decode set."""
    batch = OlmoEarthSample(
        sentinel2_l2a=torch.ones((1, 8, 8, 12, 12)),
        open_set_change_boundary=_BOUNDARY.unsqueeze(0),
        timestamps=_TIMESTAMPS.unsqueeze(0),
    )
    encode = torch.tensor([0, 1, 2])
    decode = torch.tensor([3, 6, 7])
    new_encode, new_decode = (
        RandomTimeWithDecodeMaskingStrategy._ensure_pre_and_post_encoded(
            batch, 0, encode, decode
        )
    )
    # One post step moves from decode to encode; nothing else changes.
    assert len(new_encode) == 4 and len(new_decode) == 2
    assert (new_encode >= 6).sum() == 1 and (new_encode < 6).sum() == 3
    assert set(new_encode.tolist()) | set(new_decode.tolist()) == {0, 1, 2, 3, 6, 7}

    # Already balanced encode sets are untouched.
    encode = torch.tensor([0, 6])
    decode = torch.tensor([1, 7])
    same_encode, same_decode = (
        RandomTimeWithDecodeMaskingStrategy._ensure_pre_and_post_encoded(
            batch, 0, encode, decode
        )
    )
    assert torch.equal(same_encode, encode) and torch.equal(same_decode, decode)

    # Non-change samples (missing-filled boundary) are untouched.
    plain = OlmoEarthSample(
        sentinel2_l2a=torch.ones((1, 8, 8, 12, 12)),
        open_set_change_boundary=torch.full((1, 3), MISSING_VALUE),
        timestamps=_TIMESTAMPS.unsqueeze(0),
    )
    encode = torch.tensor([0, 1, 2])
    decode = torch.tensor([3, 6, 7])
    same_encode, same_decode = (
        RandomTimeWithDecodeMaskingStrategy._ensure_pre_and_post_encoded(
            plain, 0, encode, decode
        )
    )
    assert torch.equal(same_encode, encode) and torch.equal(same_decode, decode)


def test_time_masking_encodes_both_sides_of_change() -> None:
    """Under pure time masking every change sample encodes a pre AND a post step."""
    b, h, w, t = 8, 8, 8, 12
    batch = OlmoEarthSample(
        sentinel2_l2a=torch.ones((b, h, w, t, Modality.SENTINEL2_L2A.num_bands)),
        sentinel1=torch.ones((b, h, w, t, Modality.SENTINEL1.num_bands)),
        worldcover=torch.ones((b, h, w, 1, Modality.WORLDCOVER.num_bands)),
        open_set_change_boundary=_BOUNDARY.unsqueeze(0).expand(b, 3).clone(),
        timestamps=_TIMESTAMPS.unsqueeze(0).expand(b, t, 3).clone(),
    )
    strategy = RandomTimeWithDecodeMaskingStrategy(
        encode_ratio=0.25,
        decode_ratio=0.75,
        random_ratio=0.0,  # always time masking
        only_decode_modalities=[
            Modality.WORLDCOVER.name,
            Modality.OPEN_SET_CHANGE_BOUNDARY.name,
        ],
    )
    for _ in range(10):
        masked = strategy.apply_mask(batch, patch_size=4)
        # Per-timestep visibility to the online encoder over all spatial modalities.
        visible_t = torch.zeros(b, t, dtype=torch.bool)
        for name in ("sentinel2_l2a", "sentinel1"):
            mask = getattr(masked, f"{name}_mask")  # (B, H, W, T, bandsets)
            visible_t |= (mask == MaskValue.ONLINE_ENCODER.value).any(dim=(1, 2, 4))
        assert visible_t[:, :6].any(dim=1).all(), "some sample has no pre timestep"
        assert visible_t[:, 6:].any(dim=1).all(), "some sample has no post timestep"
        # The boundary itself is a decode-only label, never encoded.
        boundary_mask = masked.open_set_change_boundary_mask
        assert boundary_mask is not None
        assert (boundary_mask != MaskValue.ONLINE_ENCODER.value).all()
