"""Tests for eval constants and resolvers."""

from olmoearth_pretrain.evals.constants import (
    resolve_rslearn_layer_name,
    resolve_rslearn_modality,
)
from olmoearth_pretrain.modalities import Modality


def test_resolve_rslearn_layer_name_handles_direct_and_prefixed_layers() -> None:
    """Known rslearn layers should resolve after optional pre/post prefixes."""
    assert resolve_rslearn_layer_name("sentinel2") == "sentinel2"
    assert resolve_rslearn_layer_name("pre_sentinel2") == "sentinel2"
    assert resolve_rslearn_layer_name("post_landsat") == "landsat"


def test_resolve_rslearn_layer_name_returns_none_for_unknown_layers() -> None:
    """Unknown layers should remain unresolved for caller-specific fallback."""
    assert resolve_rslearn_layer_name("unknown_layer") is None
    assert resolve_rslearn_modality("unknown_layer") is None


def test_resolve_rslearn_modality_returns_modality_spec() -> None:
    """Known rslearn layers should resolve to OlmoEarth modality specs."""
    assert resolve_rslearn_modality("pre_sentinel1") is Modality.SENTINEL1
    assert resolve_rslearn_modality("sentinel2_l2a") is Modality.SENTINEL2_L2A
