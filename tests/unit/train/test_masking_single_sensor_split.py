"""random_time_with_decode with a lone sensor bandset (E5 S2-only runs)."""

import numpy as np
import pytest
import torch

from olmoearth_pretrain.datatypes import MaskValue, OlmoEarthSample
from olmoearth_pretrain.train.masking import RandomTimeWithDecodeMaskingStrategy

MAPS = ["worldcover", "srtm"]
B, H, W, T = 64, 4, 4, 12


def _batch(with_s1: bool = False) -> OlmoEarthSample:
    g = torch.Generator().manual_seed(0)
    kw = dict(
        # S2 with one 12-band bandset is the v1.3 tokenization; the strategy only
        # counts bandsets through the mask's last dim, so 1 bandset is set via
        # tokenization below.
        sentinel2_l2a=torch.rand(B, H, W, T, 12, generator=g),
        worldcover=torch.rand(B, H, W, 1, 1, generator=g),
        srtm=torch.rand(B, H, W, 1, 1, generator=g),
        timestamps=torch.tensor([[1, m, 2021] for m in range(T)]).repeat(B, 1, 1),
    )
    if with_s1:
        kw["sentinel1"] = torch.rand(B, H, W, T, 2, generator=g)
    return OlmoEarthSample(**kw)


def _strategy(split: bool) -> RandomTimeWithDecodeMaskingStrategy:
    from olmoearth_pretrain.nn.tokenization import (
        ModalityTokenization,
        TokenizationConfig,
    )

    s = RandomTimeWithDecodeMaskingStrategy(
        encode_ratio=0.5,
        decode_ratio=0.5,
        random_ratio=0.5,
        only_decode_modalities=MAPS,
        single_sensor_split=split,
    )
    s.tokenization_config = TokenizationConfig(
        overrides={
            "sentinel2_l2a": ModalityTokenization(
                band_groups=[
                    [
                        "B02",
                        "B03",
                        "B04",
                        "B08",
                        "B05",
                        "B06",
                        "B07",
                        "B8A",
                        "B11",
                        "B12",
                        "B01",
                        "B09",
                    ]
                ]
            )
        }
    )
    return s


def _apply(split: bool, with_s1: bool = False, seed: int = 0):  # noqa: ANN202
    np.random.seed(seed)
    torch.manual_seed(seed)
    return _strategy(split).apply_mask(_batch(with_s1), patch_size=1)


def test_default_lone_sensor_is_encode_only() -> None:
    """Original v1.3 rule: with maps present, a lone S2 bandset is never decoded."""
    out = _apply(split=False)
    m = out.sentinel2_l2a_mask
    assert m.shape[-1] == 1
    assert not (m == MaskValue.DECODER.value).any()
    assert (m == MaskValue.ONLINE_ENCODER.value).any()
    for name in MAPS:
        assert (getattr(out, f"{name}_mask") == MaskValue.DECODER.value).all()


def test_split_lone_sensor_encodes_and_decodes_every_sample() -> None:
    """single_sensor_split: every sample has S2 encoder inputs AND S2 decode targets."""
    out = _apply(split=True)
    m = out.sentinel2_l2a_mask.reshape(B, H, W, T)
    enc = m == MaskValue.ONLINE_ENCODER.value
    dec = m == MaskValue.DECODER.value
    assert enc.flatten(1).any(1).all() and dec.flatten(1).any(1).all()
    # Either random (both roles at some timestep) or time (each timestep one role).
    n_time = 0
    for i in range(B):
        e_t = enc[i].flatten(0, 1).any(0)
        d_t = dec[i].flatten(0, 1).any(0)
        if not (e_t & d_t).any():
            n_time += 1
            # time masking: whole timesteps go to one role
            assert (enc[i].flatten(0, 1).all(0) == e_t).all()
            assert (dec[i].flatten(0, 1).all(0) == d_t).all()
            assert e_t.sum() == 6 and d_t.sum() == 6
        else:
            assert enc[i].sum() == dec[i].sum() == H * W * T // 2
    assert 0 < n_time < B
    for name in MAPS:
        assert (getattr(out, f"{name}_mask") == MaskValue.DECODER.value).all()


def test_split_respects_missing_timesteps() -> None:
    """Missing tokens stay MISSING under the split."""
    np.random.seed(1)
    torch.manual_seed(1)
    batch = _batch()
    s2 = batch.sentinel2_l2a.clone()
    s2[:, :, :, 7:] = -99999.0
    batch = batch._replace(sentinel2_l2a=s2)
    out = _strategy(True).apply_mask(batch, patch_size=1)
    m = out.sentinel2_l2a_mask
    assert (m[:, :, :, 7:] == MaskValue.MISSING.value).all()
    assert (m[:, :, :, :7] != MaskValue.MISSING.value).all()


@pytest.mark.parametrize("seed", [0, 1])
def test_split_flag_is_noop_with_two_sensors(seed: int) -> None:
    """With S1 + S2 the flag changes nothing (cross-sensor split as in v1.3)."""
    a = _apply(split=False, with_s1=True, seed=seed)
    b = _apply(split=True, with_s1=True, seed=seed)
    for name in ("sentinel2_l2a", "sentinel1", *MAPS):
        assert torch.equal(getattr(a, f"{name}_mask"), getattr(b, f"{name}_mask"))
