"""Tests for RandomTimeWithDecodeMaskingStrategy's uniform-visible options."""

import math
from collections import Counter

import numpy as np
import pytest
import torch

from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality
from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.nn.tokenization import ModalityTokenization, TokenizationConfig
from olmoearth_pretrain.train.masking import (
    MaskedOlmoEarthSample,
    MaskValue,
    RandomTimeWithDecodeMaskingStrategy,
    propagate_tokenization_config,
)

ENCODE_DECODE = [
    Modality.SENTINEL2_L2A,
    Modality.SENTINEL1,
    Modality.LANDSAT,
    Modality.NISAR_ASC,
    Modality.NISAR_DESC,
]
H = W = 8
T = 6


def _single_bandset_tokenization() -> TokenizationConfig:
    """One band set per modality, as in v1.3."""
    return TokenizationConfig(
        overrides={
            m.name: ModalityTokenization(band_groups=[list(m.band_order)])
            for m in ENCODE_DECODE
        }
    )


def _batch(present: list[list[bool]]) -> OlmoEarthSample:
    """A batch where instance i has ENCODE_DECODE[j] iff present[i][j], plus worldcover."""
    b = len(present)
    timestamps = torch.stack(
        [
            torch.randint(1, 28, (b, T)),
            torch.randint(0, 12, (b, T)),
            torch.full((b, T), 2025),
        ],
        dim=-1,
    )
    fields: dict = {"timestamps": timestamps}
    for j, m in enumerate(ENCODE_DECODE):
        x = torch.ones((b, H, W, T, m.num_bands))
        for i in range(b):
            if not present[i][j]:
                x[i] = MISSING_VALUE
        fields[m.name] = x
    fields["worldcover"] = torch.ones((b, H, W, Modality.WORLDCOVER.num_bands))
    return OlmoEarthSample(**fields)


def _strategy(**kwargs) -> RandomTimeWithDecodeMaskingStrategy:
    strategy = RandomTimeWithDecodeMaskingStrategy(
        only_decode_modalities=["worldcover"], num_visible="uniform", **kwargs
    )
    propagate_tokenization_config(strategy, _single_bandset_tokenization())
    return strategy


def _roles(masked: MaskedOlmoEarthSample, i: int, patch_size: int) -> dict[str, set]:
    """Per present encode-decode modality of instance i: the set of mask values it uses."""
    out = {}
    for m in ENCODE_DECODE:
        mask = getattr(masked, MaskedOlmoEarthSample.get_masked_modality_name(m.name))
        tokens = mask[i, ::patch_size, ::patch_size]
        values = set(torch.unique(tokens).tolist())
        if values != {MaskValue.MISSING.value}:
            out[m.name] = values
    return out


def _num_visible(roles: dict[str, set]) -> int:
    return sum(MaskValue.ONLINE_ENCODER.value in v for v in roles.values())


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_uniform_visible_count(n: int) -> None:
    """e ~ U{1..n-1}: at least one visible and one hidden modality, uniform counts."""
    np.random.seed(0)
    torch.manual_seed(0)
    present = [[j < n for j in range(len(ENCODE_DECODE))]] * 400
    masked = _strategy().apply_mask(_batch(present), patch_size=2)
    counts = Counter(_num_visible(_roles(masked, i, 2)) for i in range(len(present)))
    assert set(counts) == set(range(1, n)), counts
    expected = len(present) / (n - 1)
    for e in range(1, n):
        assert abs(counts[e] - expected) < 5 * math.sqrt(expected), counts


def test_hidden_modalities_have_no_input_and_visible_remainder_is_decoded() -> None:
    """Visible: input + predicted (decode_visible_remainder); hidden: predicted + unused."""
    np.random.seed(1)
    present = [[True] * len(ENCODE_DECODE)] * 50
    for random_ratio in (1.0, 0.0):  # random mode, then time mode
        strategy = _strategy(decode_visible_remainder=True, random_ratio=random_ratio)
        masked = strategy.apply_mask(_batch(present), patch_size=1)
        for i in range(len(present)):
            for name, values in _roles(masked, i, 1).items():
                if MaskValue.ONLINE_ENCODER.value in values:
                    assert MaskValue.DECODER.value in values, (random_ratio, name)
                else:
                    assert values <= {
                        MaskValue.DECODER.value,
                        MaskValue.TARGET_ENCODER_ONLY.value,
                    }
                    assert MaskValue.DECODER.value in values
        assert (masked.worldcover_mask == MaskValue.DECODER.value).all()


def test_time_mode_visible_split_by_month() -> None:
    """Time mode: a visible modality's encode months are input, all other months predicted."""
    np.random.seed(2)
    present = [[True] * len(ENCODE_DECODE)] * 20
    strategy = _strategy(decode_visible_remainder=True, random_ratio=0.0)
    masked = strategy.apply_mask(_batch(present), patch_size=1)
    for i in range(len(present)):
        for m in ENCODE_DECODE:
            mask = getattr(masked, f"{m.name}_mask")[i, :, :, :, 0]  # H, W, T
            per_t = [set(torch.unique(mask[:, :, t]).tolist()) for t in range(T)]
            assert all(len(v) == 1 for v in per_t)
            values = {v.pop() for v in per_t}
            if MaskValue.ONLINE_ENCODER.value in values:
                assert values == {MaskValue.ONLINE_ENCODER.value, MaskValue.DECODER.value}


def test_batch_num_visible() -> None:
    """A per-batch e is used where n > e; instances with n <= e draw U{1..n-1}."""
    np.random.seed(3)
    present = [[True] * 5] * 50 + [[True, True, False, False, False]] * 50
    masked = _strategy().apply_mask(_batch(present), patch_size=1, num_visible=3)
    for i in range(50):
        assert _num_visible(_roles(masked, i, 1)) == 3
    for i in range(50, 100):
        assert _num_visible(_roles(masked, i, 1)) == 1  # n = 2 -> U{1..1}
    masked = _strategy().apply_mask(_batch(present[50:]), patch_size=1, num_visible=1)
    for i in range(50):
        assert _num_visible(_roles(masked, i, 1)) == 1


def test_ratio_mode_rejects_batch_num_visible() -> None:
    """The default ("ratio") mode keeps ceil(n/2) and refuses a per-batch count."""
    strategy = RandomTimeWithDecodeMaskingStrategy(only_decode_modalities=["worldcover"])
    propagate_tokenization_config(strategy, _single_bandset_tokenization())
    batch = _batch([[True] * 5] * 4)
    masked = strategy.apply_mask(batch, patch_size=1)
    for i in range(4):
        assert _num_visible(_roles(masked, i, 1)) == 3
    with pytest.raises(ValueError):
        strategy.apply_mask(batch, patch_size=1, num_visible=2)


@pytest.mark.parametrize("patch_size", [1, 2])
def test_max_decode_tokens(patch_size: int) -> None:
    """At most max_decode_tokens DECODER tokens per instance; masks stay patch-constant."""
    np.random.seed(4)
    cap = 50
    present = [[True] * 5] * 20
    strategy = _strategy(decode_visible_remainder=True, max_decode_tokens=cap)
    masked = strategy.apply_mask(_batch(present), patch_size=patch_size)
    for i in range(len(present)):
        total = 0
        for name in [m.name for m in ENCODE_DECODE] + ["worldcover"]:
            mask = getattr(masked, f"{name}_mask")[i]
            tokens = mask[::patch_size, ::patch_size]
            total += int((tokens == MaskValue.DECODER.value).sum())
            # every pixel equals its token's value
            expanded = tokens.repeat_interleave(patch_size, 0).repeat_interleave(
                patch_size, 1
            )
            assert torch.equal(expanded, mask)
        assert total == cap  # uncapped this batch decodes far more than 50


def test_decode_cap_skips_decode_only_modalities() -> None:
    """With decode_cap_skips_decode_only the cap leaves every map token DECODER."""
    from olmoearth_pretrain.datatypes import MaskValue
    from olmoearth_pretrain.train.masking import RandomTimeWithDecodeMaskingStrategy

    n, h, w, t = 2, 16, 16, 4
    masks = {
        "sentinel2_l2a_mask": torch.full((n, h, w, t, 1), MaskValue.DECODER.value),
        "worldcover_mask": torch.full((n, h, w, 1, 1), MaskValue.DECODER.value),
    }
    for skip, expect_maps in [(frozenset(), None), (frozenset({"worldcover"}), h * w)]:
        out = {k: v.clone() for k, v in masks.items()}
        for i in range(n):
            RandomTimeWithDecodeMaskingStrategy._cap_decode_tokens(
                out, i, 1, 100, skip
            )
        for i in range(n):
            s2 = int((out["sentinel2_l2a_mask"][i] == MaskValue.DECODER.value).sum())
            wc = int((out["worldcover_mask"][i] == MaskValue.DECODER.value).sum())
            if expect_maps is None:
                assert s2 + wc == 100
            else:
                assert wc == expect_maps and s2 == 100
