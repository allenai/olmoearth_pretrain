"""Unit tests for the Tessera v2 PASTIS inference pipeline (pure logic)."""

import numpy as np
import torch

from olmoearth_pretrain.evals.datasets.pastis_tessera_v2 import (
    TESSERA_S2_INDICES,
    s1_raw_to_tessera_units,
    scl_to_valid_mask,
)
from olmoearth_pretrain.evals.models.tessera.tessera_v2_infer import (
    encode_tile,
    get_bin_size,
)
from olmoearth_pretrain.evals.models.tessera.tessera_v2_model import (
    S2_BAND_ORDER,
    PixelStudent,
    count_params,
)


def test_scl_to_valid_mask() -> None:
    """SCL nodata/saturated/dark/shadow/cloud are invalid; cirrus is valid."""
    scl = np.arange(12)
    mask = scl_to_valid_mask(scl)
    assert mask.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1]


def test_s1_raw_to_tessera_units() -> None:
    """Replicates their (20*log10(raw)+50)*200 int16 encoding with 0 sentinel."""
    raw = np.array([0.06, 1.0, 0.0, -1.0, np.nan, np.inf, 1e6], dtype=np.float32)
    out = s1_raw_to_tessera_units(raw)
    assert out.dtype == np.float32
    # 20*log10(0.06) = -24.437 -> (50 - 24.437) * 200 = 5112 (int16-rounded).
    assert out[0] == 5112.0
    assert out[1] == 10000.0
    # Non-positive / non-finite raw values become the 0 missing sentinel.
    assert out[2] == out[3] == out[4] == 0.0
    # Clipped to int16 range.
    assert out[6] == 32767.0


def test_tessera_s2_band_order() -> None:
    """The reorder indices map config band order onto the model's band order."""
    concat_order = [
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
    ]
    assert [concat_order[i] for i in TESSERA_S2_INDICES] == S2_BAND_ORDER


def test_student_param_counts_match_model_cards() -> None:
    """Vendored architecture reproduces the published student sizes exactly."""
    sizes = {
        (36, 2, 384): 1_066_402,  # nano
        (64, 4, 1024): 7_112_322,  # small
        (110, 4, 1792): 21_031_506,  # medium
        (160, 4, 2560): 43_831_170,  # large
    }
    for (latent, layers, ffn), expected in sizes.items():
        model = PixelStudent(latent_dim=latent, num_layers=layers, dim_feedforward=ffn)
        assert count_params(model) == expected


def test_bin_sizes() -> None:
    """Valid-observation counts bucketize to the {8,...,256} ladder."""
    assert get_bin_size(0) == 0
    assert get_bin_size(1) == 8
    assert get_bin_size(43) == 48
    assert get_bin_size(135) == 136
    assert get_bin_size(500) == 256


def test_encode_tile_shapes_and_mask_handling() -> None:
    """encode_tile runs the full bucketize/pad/forward path on a tiny tile."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = PixelStudent(latent_dim=8, num_layers=1, nhead=2, dim_feedforward=16)
    model.eval()

    t_s2, t_s1, h, w = 5, 4, 3, 3
    s2 = rng.uniform(0, 4000, (t_s2, h, w, 10)).astype(np.float32)
    masks = np.ones((t_s2, h, w), dtype=np.uint8)
    masks[:, 0, 0] = 0  # one fully-cloudy pixel: s2_bin drops to 0
    s1 = rng.uniform(2000, 8000, (t_s1, h, w, 2)).astype(np.float32)

    out = encode_tile(
        model,
        s2_bands=s2,
        s2_doys=np.array([10, 60, 150, 220, 300]),
        s2_masks=masks,
        s1_asc_bands=s1,
        s1_asc_doys=np.array([15, 105, 195, 285]),
        s1_desc_bands=None,
        s1_desc_doys=None,
        batch_pixels=4,
        device=torch.device("cpu"),
    )
    assert out.shape == (h, w, 128)
    assert np.isfinite(out).all()
    # The final non-affine LayerNorm makes every pixel mean-0/std-1.
    np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-4)
