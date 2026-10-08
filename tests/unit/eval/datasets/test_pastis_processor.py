"""Tests for the PASTIS processor's embedding support."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform

from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.evals.datasets.pastis_processor import (
    PASTISRProcessor,
    patch_grid_from_geometry,
)
from olmoearth_pretrain.evals.embedding_materializer.fetchers import EmbeddingFetcher

# A PASTIS-like 1280m x 1280m footprint on the 10m UTM 31N grid.
UTM_CRS = CRS.from_epsg(32631)
X0, Y1 = 500_000.0, 6_700_000.0  # min x, max y
X1, Y0 = X0 + 1280.0, Y1 - 1280.0


def _square_geometry(xs: list[float], ys: list[float]) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [list(zip(xs, ys, strict=True))],
    }


def test_patch_grid_native_utm() -> None:
    """A UTM-native footprint snaps exactly onto its own 10m grid."""
    geometry = _square_geometry([X0, X1, X1, X0, X0], [Y1, Y1, Y0, Y0, Y1])
    bounds, projection = patch_grid_from_geometry(geometry, UTM_CRS)

    assert projection.crs == UTM_CRS
    assert projection.x_resolution == 10.0
    assert projection.y_resolution == -10.0
    assert bounds == (50_000, -670_000, 50_128, -669_872)


def test_patch_grid_from_lonlat_footprint() -> None:
    """A footprint given in EPSG:4326 recovers the same native UTM grid."""
    lons, lats = warp_transform(
        UTM_CRS,
        CRS.from_epsg(4326),
        [X0, X1, X1, X0, X0],
        [Y1, Y1, Y0, Y0, Y1],
    )
    geometry = _square_geometry(lons, lats)
    bounds, projection = patch_grid_from_geometry(geometry, CRS.from_epsg(4326))

    assert projection.crs == UTM_CRS
    assert bounds == (50_000, -670_000, 50_128, -669_872)


class FakeFetcher(EmbeddingFetcher):
    """Deterministic in-memory fetcher for embeddings-only tests."""

    def __init__(self) -> None:
        """Initialize with an empty call log."""
        self.calls: list[tuple] = []

    @property
    def modality(self) -> ModalitySpec:
        """Pretend to produce GSE."""
        return Modality.GSE

    @property
    def product_version(self) -> str:
        """Fake version string."""
        return "fake-v1"

    def fetch(self, bounds, projection, year):  # type: ignore[no-untyped-def]
        """Return a (C, 128, 128) array marked with the request's min-x."""
        self.calls.append((bounds, year))
        num_bands = len(self.modality.band_order)
        return np.full((num_bands, 128, 128), float(bounds[0]), dtype=np.float32)


# 12 acquisition dates covering Sep 2018 - Aug 2019 (one per month), so the
# replayed month sequence is exactly 12 months long.
DATES_S2 = {
    str(i): date
    for i, date in enumerate(
        [
            20180915,
            20181015,
            20181115,
            20181215,
            20190115,
            20190215,
            20190315,
            20190415,
            20190515,
            20190615,
            20190715,
            20190815,
        ]
    )
}
EXPECTED_MONTHS = [
    201809,
    201810,
    201811,
    201812,
    201901,
    201902,
    201903,
    201904,
    201905,
    201906,
    201907,
    201908,
]


def _make_metadata(data_dir: Path) -> None:
    """Write a metadata.geojson with one patch per fold (folds 1-5)."""
    features = []
    for fold in range(1, 6):
        lon0 = 5.0 + fold * 0.1
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "ID_PATCH": 10_000 + fold,
                    "Fold": fold,
                    "dates-S2": DATES_S2,
                },
                "geometry": _square_geometry(
                    [lon0, lon0 + 0.017, lon0 + 0.017, lon0, lon0],
                    [47.2, 47.2, 47.2115, 47.2115, 47.2],
                ),
            }
        )
    with open(data_dir / "metadata.geojson", "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)


def _make_existing_splits(output_dir: Path) -> None:
    """Write months.pt/targets.pt matching the metadata replay (64x64 mode)."""
    month_row = torch.tensor(EXPECTED_MONTHS, dtype=torch.long)
    for split, num_patches in [("train", 3), ("valid", 1), ("test", 1)]:
        split_dir = output_dir / f"pastis_r_{split}"
        split_dir.mkdir(parents=True, exist_ok=True)
        torch.save(month_row.repeat(num_patches * 4, 1), split_dir / "months.pt")
        torch.save(
            torch.zeros(num_patches * 4, 64, 64, dtype=torch.long),
            split_dir / "targets.pt",
        )


@pytest.fixture
def embeddings_only_processor(tmp_path: Path) -> tuple[PASTISRProcessor, FakeFetcher]:
    """A processor wired to a fake fetcher over synthetic metadata + splits."""
    data_dir = tmp_path / "raw"
    output_dir = tmp_path / "splits"
    data_dir.mkdir()
    _make_metadata(data_dir)
    _make_existing_splits(output_dir)
    processor = PASTISRProcessor(data_dir=str(data_dir), output_dir=str(output_dir))
    fetcher = FakeFetcher()
    processor.embedding_fetchers = [fetcher]
    return processor, fetcher


def test_embeddings_only_writes_aligned_files(
    embeddings_only_processor: tuple[PASTISRProcessor, FakeFetcher],
) -> None:
    """Embeddings are quadrant-split and written per verified sample index."""
    processor, fetcher = embeddings_only_processor
    processor.process_embeddings_only(workers=2)

    num_bands = len(Modality.GSE.band_order)
    for split, num_patches in [("train", 3), ("valid", 1), ("test", 1)]:
        gse_dir = processor.output_dir / f"pastis_r_{split}" / "gse_images"
        files = sorted(gse_dir.iterdir(), key=lambda p: int(p.stem))
        assert len(files) == num_patches * 4
        sample = torch.load(files[0])
        assert sample.shape == (num_bands, 64, 64)
    # One fetch per patch (5 patches), not per quadrant sample.
    assert len(fetcher.calls) == 5
    # Nothing else was touched.
    train_dir = processor.output_dir / "pastis_r_train"
    assert not (train_dir / "s2_images").exists()


def test_embeddings_only_rerun_skips(
    embeddings_only_processor: tuple[PASTISRProcessor, FakeFetcher],
) -> None:
    """A re-run skips patches whose files already exist."""
    processor, fetcher = embeddings_only_processor
    processor.process_embeddings_only()
    assert len(fetcher.calls) == 5
    processor.process_embeddings_only()
    assert len(fetcher.calls) == 5  # no new fetches
    processor.process_embeddings_only(overwrite=True)
    assert len(fetcher.calls) == 10


def test_embeddings_only_aborts_on_month_mismatch(
    embeddings_only_processor: tuple[PASTISRProcessor, FakeFetcher],
) -> None:
    """A months.pt row that disagrees with the replay aborts before writing."""
    processor, fetcher = embeddings_only_processor
    months_path = processor.output_dir / "pastis_r_valid" / "months.pt"
    months = torch.load(months_path)
    months[2, 5] = 201907  # corrupt one row
    torch.save(months, months_path)

    with pytest.raises(ValueError, match="disagrees with the metadata replay"):
        processor.process_embeddings_only()
    assert len(fetcher.calls) == 0
    assert not (processor.output_dir / "pastis_r_train" / "gse_images").exists()


def test_embeddings_only_aborts_on_count_mismatch(
    embeddings_only_processor: tuple[PASTISRProcessor, FakeFetcher],
) -> None:
    """A sample-count drift (e.g. skipped patches) aborts before writing."""
    processor, fetcher = embeddings_only_processor
    months_path = processor.output_dir / "pastis_r_train" / "months.pt"
    months = torch.load(months_path)
    torch.save(months[:-4], months_path)  # drop one patch's samples

    with pytest.raises(ValueError, match="cannot be aligned"):
        processor.process_embeddings_only()
    assert len(fetcher.calls) == 0
