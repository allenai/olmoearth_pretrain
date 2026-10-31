"""Tests for FlexModalityMaskingStrategy (``flex_modality_with_decode``)."""

import numpy as np
import pytest
import torch

from olmoearth_pretrain.data.constants import MISSING_VALUE, Modality
from olmoearth_pretrain.datatypes import MaskValue, OlmoEarthSample
from olmoearth_pretrain.nn.tokenization import ModalityTokenization, TokenizationConfig
from olmoearth_pretrain.train.masking import MaskingConfig

ENC = MaskValue.ONLINE_ENCODER.value
TGT = MaskValue.TARGET_ENCODER_ONLY.value
DEC = MaskValue.DECODER.value
MIS = MaskValue.MISSING.value
ENCODABLE = ["sentinel2_l2a", "sentinel1", "landsat"]

# v1.2 / v1.3 tokenization: S2 and Landsat as one band set each (S1 has one already).
TOKENIZATION = TokenizationConfig(
    overrides={
        "sentinel2_l2a": ModalityTokenization(
            band_groups=[Modality.SENTINEL2_L2A.band_order]
        ),
        "landsat": ModalityTokenization(band_groups=[Modality.LANDSAT.band_order]),
    }
)


def _strategy(**kwargs):
    config = {
        "type": "flex_modality_with_decode",
        "only_decode_modalities": ["worldcover"],
        **kwargs,
    }
    return MaskingConfig(
        strategy_config=config, tokenization_config=TOKENIZATION
    ).build()


def _batch(b=8, h=16, w=16, t=6, missing_landsat=(), missing_s1_t=()):
    timestamps = torch.cat(
        [
            torch.randint(1, 28, (b, 1, t)),
            torch.randint(1, 13, (b, 1, t)),
            torch.full((b, 1, t), 2020),
        ],
        dim=1,
    )
    landsat = torch.ones((b, h, w, t, Modality.LANDSAT.num_bands))
    for i in missing_landsat:
        landsat[i] = MISSING_VALUE
    s1 = torch.ones((b, h, w, t, Modality.SENTINEL1.num_bands))
    for ti in missing_s1_t:
        s1[:, :, :, ti] = MISSING_VALUE
    return OlmoEarthSample(
        sentinel2_l2a=torch.ones((b, h, w, t, Modality.SENTINEL2_L2A.num_bands)),
        sentinel1=s1,
        landsat=landsat,
        worldcover=torch.ones((b, h, w, 1, Modality.WORLDCOVER.num_bands)),
        timestamps=timestamps,
    )


def _tokens(masked, name, i, patch_size):
    """Per-token (h_p, w_p, T) view of instance i's single band set."""
    m = getattr(masked, f"{name}_mask")[i, ..., 0]
    return m[::patch_size, ::patch_size]


def _visible(masked, name, i, ps):
    return bool((_tokens(masked, name, i, ps) == ENC).any())


def test_roles_maps_and_patch_consistency() -> None:
    """Maps only decode; visible sets never TGT; hidden never ENC; masks constant per patch."""
    torch.manual_seed(0)
    np.random.seed(0)
    ps = 4
    strategy = _strategy()
    for _ in range(20):
        masked = strategy.apply_mask(_batch(), patch_size=ps)
        wc = masked.worldcover_mask
        assert set(wc.unique().tolist()) <= {DEC, MIS}
        for name in ENCODABLE:
            full = getattr(masked, f"{name}_mask")
            # each token's value repeated over its ps x ps pixels
            rep = full[:, ::ps, ::ps].repeat_interleave(ps, 1).repeat_interleave(ps, 2)
            assert torch.equal(rep, full)
            for i in range(full.shape[0]):
                tok = _tokens(masked, name, i, ps)
                vals = set(tok.unique().tolist())
                if ENC in vals:
                    assert TGT not in vals and DEC in vals
                else:
                    assert vals <= {DEC, TGT} and DEC in vals


def test_visible_count_per_call_and_uniform() -> None:
    """E is shared by the n=3 instances of a call, and uniform over 1..n per instance."""
    np.random.seed(1)
    torch.manual_seed(1)
    ps = 4
    strategy = _strategy()
    counts3 = {1: 0, 2: 0, 3: 0}
    counts2 = {1: 0, 2: 0}
    for _ in range(300):
        masked = strategy.apply_mask(_batch(b=4, missing_landsat=(3,)), patch_size=ps)
        e3 = [sum(_visible(masked, n, i, ps) for n in ENCODABLE) for i in range(3)]
        assert len(set(e3)) == 1
        counts3[e3[0]] += 1
        e2 = sum(_visible(masked, n, 3, ps) for n in ENCODABLE[:2])
        counts2[e2] += 1
        assert not (getattr(masked, "landsat_mask")[3] != MIS).any()
    for k, v in counts3.items():
        assert abs(v / 300 - 1 / 3) < 0.08, counts3
    for k, v in counts2.items():
        assert abs(v / 300 - 1 / 2) < 0.09, counts2


def test_mask_ratio_range_random_mode() -> None:
    """Visible band sets decode r ~ U(0.3, 0.7) of their tokens."""
    np.random.seed(2)
    torch.manual_seed(2)
    ps = 2
    strategy = _strategy(mode_weights=[1.0, 0.0, 0.0])
    fracs = []
    for _ in range(30):
        masked = strategy.apply_mask(_batch(), patch_size=ps)
        for name in ENCODABLE:
            for i in range(8):
                tok = _tokens(masked, name, i, ps)
                if (tok == ENC).any():
                    fracs.append(float((tok == DEC).float().mean()))
    fracs = np.array(fracs)
    assert fracs.min() >= 0.29 and fracs.max() <= 0.71
    assert abs(fracs.mean() - 0.5) < 0.03


def test_temporal_mode_shared_timesteps() -> None:
    """Temporal: every band set decodes the same whole timesteps; visible encode the rest."""
    np.random.seed(3)
    torch.manual_seed(3)
    ps = 4
    strategy = _strategy(mode_weights=[0.0, 1.0, 0.0])
    for _ in range(10):
        masked = strategy.apply_mask(_batch(), patch_size=ps)
        for i in range(8):
            dec_t = None
            for name in ENCODABLE:
                tok = _tokens(masked, name, i, ps)
                per_t = (tok == DEC).flatten(0, 1)  # (hw, T)
                assert (per_t.all(0) | ~per_t.any(0)).all()  # whole timesteps
                t_dec = per_t.all(0)
                if dec_t is None:
                    dec_t = t_dec
                assert torch.equal(dec_t, t_dec)
                if (tok == ENC).any():
                    assert (tok[..., ~t_dec] == ENC).all()


def test_spatial_mode_through_time_and_per_modality() -> None:
    """Spatial: masked cells span all timesteps, drawn independently per band set."""
    np.random.seed(4)
    torch.manual_seed(4)
    ps = 2
    strategy = _strategy(mode_weights=[0.0, 0.0, 1.0])
    differs = 0
    for _ in range(10):
        masked = strategy.apply_mask(_batch(), patch_size=ps)
        for i in range(8):
            cell_sets = []
            for name in ENCODABLE:
                tok = _tokens(masked, name, i, ps)
                dec = tok == DEC
                assert (dec.all(-1) | ~dec.any(-1)).all()  # through all time
                frac = float(dec.all(-1).float().mean())
                assert 0.29 <= frac <= 0.71
                cell_sets.append(dec.all(-1))
            differs += int(not torch.equal(cell_sets[0], cell_sets[1]))
    assert differs > 70


def test_missing_preserved_and_temporal_fallback() -> None:
    """Missing S1 timesteps stay MISSING; tiny inputs fall back without errors."""
    np.random.seed(5)
    torch.manual_seed(5)
    strategy = _strategy()
    masked = strategy.apply_mask(_batch(missing_s1_t=(0, 2)), patch_size=4)
    assert (masked.sentinel1_mask[:, :, :, [0, 2]] == MIS).all()
    assert not (masked.sentinel1_mask[:, :, :, [1, 3, 4, 5]] == MIS).any()
    # 1x1 grid and a single timestep: spatial/temporal fall back to random.
    masked = strategy.apply_mask(_batch(h=4, w=4, t=1), patch_size=4)
    for name in ENCODABLE:
        m = getattr(masked, f"{name}_mask")
        assert not (m == MIS).any()
    for i in range(8):
        assert any(_visible(masked, n, i, 4) for n in ENCODABLE)


@pytest.mark.parametrize("num_bands", [2, 11, 12])
def test_band_count_dropout_distribution(num_bands: int) -> None:
    """Count band dropout: P(all kept) = 0.25 + 0.75 / N, else k ~ U{1..N}; same over positions."""
    from olmoearth_pretrain.nn.flexi_vit import MultiModalPatchEmbeddings

    torch.manual_seed(6)
    b = 20000
    data = torch.ones(b, 3, 3, 2, num_bands)
    out = MultiModalPatchEmbeddings._apply_band_count_dropout(data, 0.75)
    kept_map = out != 0
    assert (kept_map == kept_map[:, :1, :1, :1]).all()
    kept = kept_map[:, 0, 0, 0].sum(-1)
    assert kept.min() >= 1
    p_all = float((kept == num_bands).float().mean())
    assert abs(p_all - (0.25 + 0.75 / num_bands)) < 0.02
    if num_bands > 2:
        p_one = float((kept == 1).float().mean())
        assert abs(p_one - 0.75 / num_bands) < 0.015
