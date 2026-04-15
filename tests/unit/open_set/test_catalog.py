"""Tests for the open-set class catalog."""

from __future__ import annotations

import pytest
import torch

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.open_set.catalog import (
    BandIndexExtractor,
    ClassEntry,
    ClassRegistry,
    ValueEqExtractor,
    build_default_registry,
    build_osm_entries,
)


class TestBandIndexExtractor:
    """BandIndexExtractor must select one channel and binarize it."""

    def test_selects_correct_band(self) -> None:
        """Picks band 2 and binarizes anything > threshold."""
        ex = BandIndexExtractor(band_index=2, threshold=0.0)
        # [B=2, H=3, W=3, 1, C=4]
        tensor = torch.zeros(2, 3, 3, 1, 4)
        tensor[0, 1, 1, 0, 2] = 1.0  # only this pixel positive in batch 0
        tensor[1, 2, 2, 0, 2] = 0.5  # batch 1 has a small positive

        mask = ex(tensor)
        assert mask.shape == (2, 3, 3)
        assert mask[0, 1, 1].item() == 1.0
        assert mask[0, 0, 0].item() == 0.0
        assert mask[1, 2, 2].item() == 1.0
        # other-band pixels do not leak in
        tensor[0, 0, 0, 0, 1] = 9.0
        assert ex(tensor)[0, 0, 0].item() == 0.0

    def test_threshold(self) -> None:
        """Anything <= threshold is zero."""
        ex = BandIndexExtractor(band_index=0, threshold=0.5)
        tensor = torch.tensor([[[[[0.4]], [[0.6]]]]])  # [1, 1, 2, 1, 1]
        mask = ex(tensor)
        assert mask[0, 0, 0].item() == 0.0
        assert mask[0, 0, 1].item() == 1.0

    def test_rejects_wrong_rank(self) -> None:
        """A tensor of the wrong rank is rejected explicitly."""
        ex = BandIndexExtractor(band_index=0)
        with pytest.raises(ValueError):
            ex(torch.zeros(2, 3, 3))


class TestValueEqExtractor:
    """ValueEqExtractor masks pixels equal to a class id."""

    def test_equality(self) -> None:
        """Pixels with the right value get a 1; everything else a 0."""
        ex = ValueEqExtractor(class_id=5)
        tensor = torch.tensor([[1, 5, 5], [0, 5, 9]]).reshape(1, 2, 3, 1, 1)
        mask = ex(tensor)
        assert mask.shape == (1, 2, 3)
        assert mask[0, 0, 1].item() == 1.0
        assert mask[0, 1, 1].item() == 1.0
        assert mask[0, 1, 2].item() == 0.0


class TestClassRegistry:
    """Registry rejects duplicates and supports lookup."""

    def test_dedup(self) -> None:
        """Two entries with the same (source, text) should error out."""
        e = ClassEntry(text="a", source="x", extractor=BandIndexExtractor(0))
        with pytest.raises(ValueError):
            ClassRegistry([e, e])

    def test_by_source(self) -> None:
        """``by_source`` filters by the modality field name."""
        reg = ClassRegistry(
            [
                ClassEntry(text="a", source="x", extractor=BandIndexExtractor(0)),
                ClassEntry(text="b", source="x", extractor=BandIndexExtractor(1)),
                ClassEntry(text="c", source="y", extractor=BandIndexExtractor(0)),
            ]
        )
        assert [e.text for e in reg.by_source("x")] == ["a", "b"]
        assert [e.text for e in reg.by_source("y")] == ["c"]
        assert reg.sources() == ["x", "y"]

    def test_by_text(self) -> None:
        """``by_text`` returns the first matching entry, optionally scoped by source."""
        reg = ClassRegistry(
            [
                ClassEntry(text="a", source="x", extractor=BandIndexExtractor(0)),
                ClassEntry(text="a", source="y", extractor=BandIndexExtractor(1)),
            ]
        )
        assert reg.by_text("a", source="y").source == "y"
        with pytest.raises(KeyError):
            reg.by_text("missing")


class TestOSMEntries:
    """OSM entries should be one per band, in band-order."""

    def test_one_entry_per_band(self) -> None:
        """Number of entries matches number of OSM bands."""
        entries = build_osm_entries()
        assert len(entries) == len(Modality.OPENSTREETMAP_RASTER.band_order)

    def test_extractor_indices_match_bands(self) -> None:
        """Each entry's extractor reads the band at the entry's index."""
        entries = build_osm_entries()
        bands = Modality.OPENSTREETMAP_RASTER.band_order
        for idx, entry in enumerate(entries):
            assert entry.source == "openstreetmap_raster"
            assert isinstance(entry.extractor, BandIndexExtractor)
            assert entry.extractor.band_index == idx
            # Spot-check one well-known band has a sensible prompt.
            if bands[idx] == "aerodrome":
                assert entry.text == "aerodrome"

    def test_default_registry_includes_osm(self) -> None:
        """The default registry should at minimum contain OSM."""
        reg = build_default_registry()
        assert any(e.source == "openstreetmap_raster" for e in reg)
