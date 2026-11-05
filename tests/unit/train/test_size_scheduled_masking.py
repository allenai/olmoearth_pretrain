"""Size-scheduled random_time_with_decode masking (neighborhood-attention embedding run)."""

import numpy as np
import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.nn.tokenization import ModalityTokenization, TokenizationConfig
from olmoearth_pretrain.train.masking import (
    MaskValue,
    RandomTimeWithDecodeMaskingStrategy,
    SizeScheduledRandomTimeWithDecodeMaskingStrategy,
    propagate_tokenization_config,
)

SCHEDULE = [[8, 0.5, 0.5], [12, 0.35, 0.4], [16, 0.25, 0.3], [24, 0.15, 0.25]]
PS = 2
T = 12


def _batch(hw: int, b: int = 8) -> OlmoEarthSample:
    px = hw * PS
    timestamps = torch.stack(
        [
            torch.ones(b, T, dtype=torch.long),
            torch.arange(1, T + 1).expand(b, T),
            torch.full((b, T), 2022),
        ],
        dim=-1,
    )
    return OlmoEarthSample(
        sentinel2_l2a=torch.ones(b, px, px, T, Modality.SENTINEL2_L2A.num_bands),
        sentinel1=torch.ones(b, px, px, T, Modality.SENTINEL1.num_bands),
        landsat=torch.ones(b, px, px, T, Modality.LANDSAT.num_bands),
        timestamps=timestamps,
    )


def _single_bandset(strategy) -> None:
    """v1.3 tokenization: one band set each for S2 and Landsat (S1 has one)."""
    tok = TokenizationConfig(
        overrides={
            "sentinel2_l2a": ModalityTokenization(
                band_groups=[list(Modality.SENTINEL2_L2A.band_order)]
            ),
            "landsat": ModalityTokenization(
                band_groups=[list(Modality.LANDSAT.band_order)]
            ),
        }
    )
    propagate_tokenization_config(strategy, tok)


def _shares(masked, hw: int) -> tuple[float, float]:
    """Visible and decoded share of all S2+S1+Landsat tokens."""
    counts = {v: 0 for v in MaskValue}
    total = 0
    for name in ("sentinel2_l2a", "sentinel1", "landsat"):
        m = getattr(masked, f"{name}_mask")[:, ::PS, ::PS]
        total += m.numel()
        for v in MaskValue:
            counts[v] += int((m == v.value).sum())
    return counts[MaskValue.ONLINE_ENCODER] / total, counts[MaskValue.DECODER] / total


@pytest.mark.parametrize("hw", [8, 12, 16, 24])
def test_shares_follow_the_schedule(hw: int) -> None:
    """2 of 3 band sets encode: visible ~ 2/3 * e, decoded ~ 1/3 * d (random + time)."""
    np.random.seed(0)
    torch.manual_seed(0)
    strategy = SizeScheduledRandomTimeWithDecodeMaskingStrategy(schedule=SCHEDULE)
    _single_bandset(strategy)
    e, d = strategy.ratios_for(hw)
    vis, dec = zip(
        *(_shares(strategy.apply_mask(_batch(hw), PS), hw) for _ in range(6))
    )
    assert abs(np.mean(vis) - 2 / 3 * e) < 0.03, (np.mean(vis), e)
    assert abs(np.mean(dec) - 1 / 3 * d) < 0.03, (np.mean(dec), d)


def test_ratios_for_picks_the_largest_entry_below() -> None:
    """An unscheduled size uses the largest scheduled size below it."""
    strategy = SizeScheduledRandomTimeWithDecodeMaskingStrategy(schedule=SCHEDULE)
    assert strategy.ratios_for(4) == (0.5, 0.5)
    assert strategy.ratios_for(13) == (0.35, 0.4)
    assert strategy.ratios_for(40) == (0.15, 0.25)


def test_release_semantics_unchanged() -> None:
    """With the new options at their defaults the v1.3 strategy masks as before."""
    batch = _batch(8)
    out = []
    for kwargs in ({}, {"bandset_encode_ratio": 0.5}):
        np.random.seed(1)
        torch.manual_seed(1)
        strategy = RandomTimeWithDecodeMaskingStrategy(
            encode_ratio=0.5, decode_ratio=0.5, **kwargs
        )
        _single_bandset(strategy)
        out.append(strategy.apply_mask(batch, PS))
    for name in ("sentinel2_l2a", "sentinel1", "landsat"):
        assert torch.equal(
            getattr(out[0], f"{name}_mask"), getattr(out[1], f"{name}_mask")
        )
