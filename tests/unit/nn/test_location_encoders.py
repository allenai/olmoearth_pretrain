"""Tests for the frozen location encoders and the additive location conditioning."""

import math

import pytest
import torch

from olmoearth_pretrain.nn.location_encoders import (
    LocationConditioning,
    SatCLIPLocationEncoder,
    TTELocationEncoder,
    denormalize_latlon,
    real_spherical_harmonics_satclip,
)


def _tte_weights(path: str) -> str:
    g = torch.Generator().manual_seed(0)
    p, r = "location.", "location.semantic_registers."
    sd = {
        p + "_directions": torch.randn(32, 3, generator=g),
        p + "_log_tau": torch.full((32,), math.log(45.0)),
        p + "_embeddings": torch.randn(32, 8, generator=g),
        r + "loc_to_logits.weight": torch.randn(4, 8, generator=g),
        r + "loc_to_logits.bias": torch.zeros(4),
        r + "log_temperature": torch.tensor(math.log(0.5)),
        r + "registers": torch.randn(4, 512, generator=g),
        r + "site_to_out.weight": torch.randn(512, 8, generator=g),
        r + "site_to_out.bias": torch.zeros(512),
        r + "norm.weight": torch.ones(512),
        r + "norm.bias": torch.zeros(512),
        p + "mlp.4.weight": torch.randn(512, 512, generator=g) * 0.01,
    }
    for k in ["0", "2.w1", "2.w2", "3.w1", "3.w2"]:
        sd[f"{p}mlp.{k}.weight"] = torch.randn(512, 512, generator=g) * 0.01
        sd[f"{p}mlp.{k}.bias"] = torch.zeros(512)
    torch.save(sd, path)
    return path


def _satclip_weights(path: str) -> str:
    g = torch.Generator().manual_seed(0)
    sd = {
        "layers.0.weight": torch.randn(512, 1600, generator=g, dtype=torch.float64)
        * 0.01,
        "layers.0.bias": torch.zeros(512, dtype=torch.float64),
        "layers.1.weight": torch.randn(512, 512, generator=g, dtype=torch.float64)
        * 0.01,
        "layers.1.bias": torch.zeros(512, dtype=torch.float64),
        "last_layer.weight": torch.randn(256, 512, generator=g, dtype=torch.float64),
        "last_layer.bias": torch.zeros(256, dtype=torch.float64),
    }
    torch.save(sd, path)
    return path


def test_spherical_harmonics_match_satclip_low_degrees() -> None:
    """Spot-check against SatCLIP's generated expressions (Y00, Y1-1, Y10, Y22)."""
    lat = torch.tensor([10.0, -45.0, 70.0], dtype=torch.float64)
    lon = torch.tensor([20.0, 150.0, -100.0], dtype=torch.float64)
    y = real_spherical_harmonics_satclip(lat, lon, 3)
    theta = torch.deg2rad(lat + 90)
    phi = torch.deg2rad(lon + 180)
    c = torch.cos(theta)
    torch.testing.assert_close(y[:, 0], torch.full_like(c, 0.886226925452758))
    torch.testing.assert_close(
        y[:, 1], 0.48860251190292 * (1 - c**2) ** 0.5 * torch.sin(phi)
    )
    torch.testing.assert_close(y[:, 2], 1.53499006191973 * c)
    torch.testing.assert_close(
        y[:, 8], 0.18209140509868 * (3 - 3 * c**2) * torch.cos(2 * phi)
    )


def test_encoders_shapes(tmp_path) -> None:  # noqa: ANN001
    """Both encoders map (N, 2) degrees to their output widths."""
    pts = torch.tensor([[47.6, -122.3], [-3.1, -60.0]])
    tte = TTELocationEncoder(_tte_weights(str(tmp_path / "tte.pt")))
    out = tte(pts)
    assert out.shape == (2, 512)
    torch.testing.assert_close(out.norm(dim=-1), torch.ones(2))
    sat = SatCLIPLocationEncoder(_satclip_weights(str(tmp_path / "sc.pt")))
    assert sat(pts.double()).shape == (2, 256)


@pytest.mark.parametrize("kind", ["tte", "satclip_l40"])
def test_conditioning_zero_init_and_null_path(tmp_path, kind: str) -> None:  # noqa: ANN001
    """Zero at init; missing latlon (None / MISSING_VALUE) takes the null vector."""
    path = (
        _tte_weights(str(tmp_path / "w.pt"))
        if kind == "tte"
        else _satclip_weights(str(tmp_path / "w.pt"))
    )
    cond = LocationConditioning(kind, path, embedding_size=16, dropout=0.5).eval()
    latlon = torch.tensor([[0.7, 0.3], [-99999.0, -99999.0]])
    out = cond(latlon, batch_size=2, device=torch.device("cpu"))
    assert out.shape == (2, 16) and float(out.abs().max()) == 0.0
    with torch.no_grad():
        cond.proj.weight.normal_()
        cond.null_embedding.fill_(1.0)
    out = cond(latlon, batch_size=2, device=torch.device("cpu"))
    torch.testing.assert_close(out[1], torch.ones(16))
    assert not torch.allclose(out[0], torch.ones(16))
    torch.testing.assert_close(
        cond(None, batch_size=2, device=torch.device("cpu")), torch.ones(2, 16)
    )
    # Only the projection, norm and null vector are trainable.
    assert {n for n, _ in cond.named_parameters()} == {
        "norm.weight",
        "norm.bias",
        "proj.weight",
        "proj.bias",
        "null_embedding",
    }


def test_denormalize_latlon() -> None:
    """[0, 1] predefined normalization round-trips to degrees."""
    torch.testing.assert_close(
        denormalize_latlon(torch.tensor([[0.5, 0.5], [1.0, 0.0]])),
        torch.tensor([[0.0, 0.0], [90.0, -180.0]]),
    )
