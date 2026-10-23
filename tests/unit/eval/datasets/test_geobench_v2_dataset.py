"""CPU unit tests for GeoBench v2 band mapping.

All tests use fake samples (no GeoBench v2 data needed).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

import olmoearth_pretrain.evals.datasets.geobench_v2_dataset as gb2
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.datasets.geobench_v2_loaders import SLUG_TO_DATASET

S2 = list(Modality.SENTINEL2_L2A.band_order)
H = 32


def _const_bands(values: list[float], h: int = H, dtype: Any = torch.float32) -> Any:
    """(C, H, W) tensor whose channel i is the constant values[i]."""
    return torch.stack([torch.full((h, h), float(v)) for v in values]).to(dtype)


def _channel_values(hwtc: torch.Tensor, names: list[str]) -> dict[str, float]:
    """Map band name -> the (constant) value of that channel."""
    return {n: float(hwtc[..., i].mean()) for i, n in enumerate(names)}


# Distinct raw values per source band so a mis-mapping is detectable.
FLAIR2_RAW = {"red": 10, "green": 20, "blue": 30, "nir": 40, "elevation": 50}
SN7_RAW = {"red": 11, "green": 22, "blue": 33, "nir": 255}  # "nir" is alpha
SN2_RAW = {
    "coastal": 100,
    "blue": 200,
    "green": 300,
    "yellow": 400,
    "red": 500,
    "red_edge": 600,
    "nir1": 700,
    "nir2": 800,
}
SN2_PAN = 999


def _fake_raw(slug: str, h: int = H) -> dict[str, torch.Tensor]:
    band_order = SLUG_TO_DATASET[slug].band_order
    if slug == "flair2":
        assert list(band_order) == list(FLAIR2_RAW)
        image = _const_bands(list(FLAIR2_RAW.values()), h, torch.uint8)
        return {"image": image, "mask": torch.zeros(h, h, dtype=torch.long)}
    if slug == "spacenet7":
        assert band_order["s2"] == list(SN7_RAW)
        image = _const_bands(list(SN7_RAW.values()), h, torch.uint8)
        return {"image": image, "mask": torch.zeros(h, h, dtype=torch.long)}
    if slug == "spacenet2":
        assert band_order["worldview"] == list(SN2_RAW)
        return {
            "image_worldview": _const_bands(list(SN2_RAW.values()), h, torch.int32),
            "image_pan": _const_bands([SN2_PAN], h, torch.int32),
            "mask": torch.zeros(h, h, dtype=torch.long),
        }
    raise ValueError(slug)


# ---------------------------------------------------------------------------
# sentinel2_l2a path: band mapping fixes
# ---------------------------------------------------------------------------


def test_flair2_s2_maps_rgbn_by_name_and_drops_elevation() -> None:
    """FLAIR2 red lands in B04 (not B02) and elevation is not fed to any band."""
    olmo = gb2._sample_to_olmoearth(
        _fake_raw("flair2"), SLUG_TO_DATASET["flair2"].band_order, "flair2"
    )
    assert olmo.sentinel2_l2a is not None
    assert olmo.sentinel2_l2a.shape == (H, H, 1, len(S2))
    vals = _channel_values(olmo.sentinel2_l2a, S2)
    k = 10000.0 / 255.0
    assert vals["B04"] == pytest.approx(FLAIR2_RAW["red"] * k, rel=1e-5)
    assert vals["B03"] == pytest.approx(FLAIR2_RAW["green"] * k, rel=1e-5)
    assert vals["B02"] == pytest.approx(FLAIR2_RAW["blue"] * k, rel=1e-5)
    assert vals["B08"] == pytest.approx(FLAIR2_RAW["nir"] * k, rel=1e-5)
    for band in set(S2) - {"B02", "B03", "B04", "B08"}:
        assert vals[band] == 0.0, band  # incl. B05, where elevation used to go


def test_spacenet2_s2_maps_worldview_by_name_and_rescales() -> None:
    """SpaceNet2 WV-3 bands go to their S2 counterparts in S2 DN scale; pan dropped."""
    olmo = gb2._sample_to_olmoearth(
        _fake_raw("spacenet2"), SLUG_TO_DATASET["spacenet2"].band_order, "spacenet2"
    )
    assert olmo.sentinel2_l2a is not None
    assert olmo.sentinel2_l2a.shape == (H, H, 1, len(S2))
    vals = _channel_values(olmo.sentinel2_l2a, S2)
    k = 10000.0 / 2047.0
    expected = {
        "B01": SN2_RAW["coastal"],
        "B02": SN2_RAW["blue"],
        "B03": SN2_RAW["green"],
        "B04": SN2_RAW["red"],
        "B05": SN2_RAW["red_edge"],
        "B08": SN2_RAW["nir1"],
    }
    for band in S2:
        assert vals[band] == pytest.approx(expected.get(band, 0) * k, rel=1e-5), band
    assert not torch.isclose(olmo.sentinel2_l2a, torch.tensor(SN2_PAN * k)).any()


def test_spacenet7_s2_drops_alpha_channel() -> None:
    """SpaceNet7's 4th ("nir") channel is alpha: B08 stays zero."""
    olmo = gb2._sample_to_olmoearth(
        _fake_raw("spacenet7"), SLUG_TO_DATASET["spacenet7"].band_order, "spacenet7"
    )
    assert olmo.sentinel2_l2a is not None
    vals = _channel_values(olmo.sentinel2_l2a, S2)
    k = 10000.0 / 255.0
    assert vals["B04"] == pytest.approx(SN7_RAW["red"] * k, rel=1e-5)
    assert vals["B03"] == pytest.approx(SN7_RAW["green"] * k, rel=1e-5)
    assert vals["B02"] == pytest.approx(SN7_RAW["blue"] * k, rel=1e-5)
    assert vals["B08"] == 0.0


def test_treesatai_uses_only_s2() -> None:
    """TreeSatAI feeds only image_s2 (not aerial or S1) through sentinel2_l2a."""
    band_order = SLUG_TO_DATASET["treesatai"].band_order
    s2_names = band_order["s2"]
    raw = {
        "image_s2": _const_bands(list(range(1, len(s2_names) + 1)), 6),
        "image_aerial": _const_bands([777] * 4, 6),
        "image_s1": _const_bands([888] * 2, 6),
        "label": torch.zeros(15),
    }
    olmo = gb2._sample_to_olmoearth(raw, band_order, "treesatai")
    assert olmo.modalities == ["sentinel2_l2a"]
    assert olmo.sentinel2_l2a is not None
    vals = _channel_values(olmo.sentinel2_l2a, S2)
    for i, name in enumerate(s2_names):
        if name in S2:
            assert vals[name] == i + 1


def test_dataset_stats_are_keyed_to_new_layout() -> None:
    """geobench2_norm_stats.json no longer has elevation/alpha/pan in S2 slots."""
    stats = gb2._load_geobench_norm_stats()
    zero = {"mean": 0.0, "std": 1.0}
    assert stats["flair2"]["sentinel2_l2a"]["B05"] == zero  # was elevation
    assert stats["spacenet7"]["sentinel2_l2a"]["B08"] == zero  # was alpha
    assert stats["spacenet2"]["sentinel2_l2a"]["B11"] == zero  # was nir2
    for slug in ("flair2", "spacenet2", "spacenet7"):
        assert set(stats[slug]["sentinel2_l2a"]) == set(S2)
