"""Tests for the per-image mask extractor."""

from __future__ import annotations

import pytest
import torch

from olmoearth_pretrain.datatypes import OlmoEarthSample
from olmoearth_pretrain.open_set.catalog import (
    BandIndexExtractor,
    ClassEntry,
)
from olmoearth_pretrain.open_set.data.mask_extractor import (
    extract_per_image_class_assignments,
)


def _osm_sample(
    batch_size: int = 2,
    h: int = 4,
    w: int = 4,
    num_bands: int = 3,
) -> OlmoEarthSample:
    """Build a minimal sample with only ``openstreetmap_raster`` populated."""
    osm = torch.zeros(batch_size, h, w, 1, num_bands)
    return OlmoEarthSample(openstreetmap_raster=osm)


class TestExtractPerImage:
    """Selections drive which (image, class) records get returned."""

    def test_returns_one_record_per_selection(self) -> None:
        """Two selections produce two records."""
        sample = _osm_sample()
        e0 = ClassEntry(
            text="a", source="openstreetmap_raster", extractor=BandIndexExtractor(0)
        )
        e1 = ClassEntry(
            text="b", source="openstreetmap_raster", extractor=BandIndexExtractor(1)
        )
        out = extract_per_image_class_assignments(sample, [(0, e0), (1, e1)])
        assert len(out) == 2
        assert out.assignments[0].image_index == 0
        assert out.assignments[1].class_entry.text == "b"
        for a in out.assignments:
            assert a.mask.shape == (4, 4)

    def test_extractor_caches_per_class(self) -> None:
        """Selecting the same class for two images runs the extractor once.

        We verify by counting ``__call__`` invocations on a wrapper extractor.
        """
        sample = _osm_sample()

        class CountingExtractor:
            """Wrapper that satisfies the ClassExtractor protocol."""

            def __init__(self, base: BandIndexExtractor) -> None:
                self._base = base
                self.calls = 0

            def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
                self.calls += 1
                return self._base(tensor)

        ext = CountingExtractor(BandIndexExtractor(0))
        e = ClassEntry(
            text="a",
            source="openstreetmap_raster",
            extractor=ext,
        )
        extract_per_image_class_assignments(sample, [(0, e), (1, e)])
        assert ext.calls == 1

    def test_missing_source_raises(self) -> None:
        """Selecting an entry whose source is absent should raise a clear error."""
        sample = OlmoEarthSample()  # no modalities set
        e = ClassEntry(
            text="a", source="openstreetmap_raster", extractor=BandIndexExtractor(0)
        )
        with pytest.raises(KeyError):
            extract_per_image_class_assignments(sample, [(0, e)])
