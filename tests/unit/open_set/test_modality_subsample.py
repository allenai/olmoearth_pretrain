"""Tests for the modality subsampling transform."""

from __future__ import annotations

import random

import pytest
import torch

from olmoearth_pretrain.datatypes import OlmoEarthSample
from olmoearth_pretrain.open_set.data.modality_subsample import (
    ModalitySubsampleConfig,
    subsample_modalities,
)


def _sample_with(*modalities: str) -> OlmoEarthSample:
    """Build a sample with each named modality set to a tiny tensor."""
    fields = {}
    for m in modalities:
        # The encoder code does not look inside these tensors here, so any
        # nonzero placeholder is fine. The exact shape would matter if we
        # were running through the encoder.
        fields[m] = torch.zeros(1, 4, 4, 1, 1)
    return OlmoEarthSample(**fields)


class TestSubsampleModalities:
    """``subsample_modalities`` drops only configured input modalities."""

    def test_drops_at_least_one_input(self) -> None:
        """With three present inputs and max_kept=2 we drop at least one."""
        sample = _sample_with("sentinel2_l2a", "sentinel1", "landsat")
        cfg = ModalitySubsampleConfig(min_kept=1, max_kept=2, p_subsample=1.0)
        out = subsample_modalities(sample, cfg, rng=random.Random(0))
        kept = [
            m
            for m in ("sentinel2_l2a", "sentinel1", "landsat")
            if getattr(out, m) is not None
        ]
        assert 1 <= len(kept) <= 2

    def test_does_not_drop_label_sources(self) -> None:
        """OSM (a label source, not in input_modalities) is preserved."""
        sample = _sample_with("sentinel2_l2a", "sentinel1", "openstreetmap_raster")
        cfg = ModalitySubsampleConfig(
            min_kept=1, max_kept=1, p_subsample=1.0
        )  # drop one of the two inputs
        out = subsample_modalities(sample, cfg, rng=random.Random(0))
        assert out.openstreetmap_raster is not None

    def test_p_zero_is_passthrough(self) -> None:
        """``p_subsample=0`` returns the sample unchanged."""
        sample = _sample_with("sentinel2_l2a", "sentinel1")
        cfg = ModalitySubsampleConfig(p_subsample=0.0)
        out = subsample_modalities(sample, cfg, rng=random.Random(0))
        assert out is sample

    def test_min_kept_bound(self) -> None:
        """If we already have ``min_kept`` inputs we do not drop further."""
        sample = _sample_with("sentinel2_l2a")
        cfg = ModalitySubsampleConfig(min_kept=1, max_kept=1, p_subsample=1.0)
        out = subsample_modalities(sample, cfg, rng=random.Random(0))
        assert out.sentinel2_l2a is not None

    def test_invalid_config(self) -> None:
        """``min_kept < 1`` is rejected."""
        with pytest.raises(ValueError):
            ModalitySubsampleConfig(min_kept=0)
