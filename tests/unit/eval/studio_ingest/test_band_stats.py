"""Tests for band stats over datasets that store a subset of a modality's bands."""

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.evals.studio_ingest.band_stats import (
    _fill_absent_bands,
    _get_bands_by_modality,
)


def _config(bands: list[str] | None) -> dict:
    s2 = {"layers": ["sentinel2_l2a"], "data_type": "raster"}
    if bands is not None:
        s2["bands"] = bands
    return {
        "data": {
            "init_args": {
                "inputs": {
                    "sentinel2_l2a": s2,
                    "targets": {"layers": ["label"], "is_target": True},
                }
            }
        }
    }


def test_declared_subset_bands_are_used_in_order() -> None:
    """Stats follow the bands model.yaml declares, in its order."""
    bands = ["B04", "B02", "B03", "B08"]
    assert _get_bands_by_modality(_config(bands)) == {
        Modality.SENTINEL2_L2A.name: bands
    }


def test_undeclared_bands_fall_back_to_canonical_order() -> None:
    """Without a declared band list, the canonical band order is used."""
    assert _get_bands_by_modality(_config(None)) == {
        Modality.SENTINEL2_L2A.name: list(Modality.SENTINEL2_L2A.band_order)
    }


def test_fill_absent_bands_covers_canonical_order() -> None:
    """Absent canonical bands get marked placeholders; stored bands are kept."""
    stored = {"mean": 5.0, "std": 2.0, "min": 0.0, "max": 9.0}
    stats = _fill_absent_bands({Modality.SENTINEL2_L2A.name: {"B02": dict(stored)}})
    per_band = stats[Modality.SENTINEL2_L2A.name]
    assert set(per_band) == set(Modality.SENTINEL2_L2A.band_order)
    assert per_band["B02"] == stored
    assert per_band["B05"]["absent"] is True
