"""Fetchers for precomputed embedding products (AlphaEarth/GSE, Tessera).

A fetcher retrieves a precomputed embedding raster for an arbitrary
(bounds, projection, year) request at the product's native 10 m resolution.

Array layout convention: all fetchers return float32 arrays with shape
(C, H, W), where C equals ``len(fetcher.modality.band_order)``,
H = bounds[3] - bounds[1] and W = bounds[2] - bounds[0]. This matches
rslearn's ``RasterArray(chw_array=...)`` and rasterio conventions, so the
result can be written to a GeoTIFF without any transposition.
"""

import logging
import os
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import rasterio.warp
import shapely
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rslearn.config import QueryConfig, SpaceMode
from rslearn.data_sources.aws_google_satellite_embedding_v1 import (
    BANDS as AEF_BANDS,
)
from rslearn.data_sources.aws_google_satellite_embedding_v1 import (
    GoogleSatelliteEmbeddingV1,
)
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry

from olmoearth_pretrain.data.constants import Modality, ModalitySpec

logger = logging.getLogger(__name__)

# Dequantized AlphaEarth (GSE) rasters use -1.0 as the nodata marker.
AEF_NODATA = -1.0

# Default location for caching the AEF spatial index CSV.
DEFAULT_AEF_CACHE_DIR = "~/.cache/olmoearth_pretrain/aef_index"


def year_time_range(year: int) -> tuple[datetime, datetime]:
    """Return the UTC (start, end) datetimes covering the given calendar year.

    Args:
        year: the calendar year.

    Returns:
        tuple of (start, end) datetimes spanning the year.
    """
    return (
        datetime(year, 1, 1, tzinfo=UTC),
        datetime(year, 12, 31, 23, 59, 59, tzinfo=UTC),
    )


class EmbeddingFetcher(ABC):
    """Fetches precomputed embedding rasters for a bounds/projection/year request."""

    @property
    @abstractmethod
    def modality(self) -> ModalitySpec:
        """The OlmoEarth modality this fetcher produces."""

    @property
    @abstractmethod
    def product_version(self) -> str:
        """Version identifier of the embedding product."""

    @property
    def nodata_value(self) -> float:
        """The nodata value used for pixels with no product coverage."""
        return float("nan")

    @abstractmethod
    def fetch(
        self, bounds: PixelBounds, projection: Projection, year: int
    ) -> np.ndarray | None:
        """Fetch the embedding raster covering bounds/projection for a year.

        Args:
            bounds: pixel bounds (x0, y0, x1, y1) in the given projection.
            projection: the target rslearn Projection (CRS + resolution).
            year: the calendar year of the annual embedding product.

        Returns:
            float32 array with shape (C, H, W) aligned to the requested
            bounds/projection, or None if the product has no coverage for
            (bounds, year). Pixels covered by the request but missing from the
            product are filled with ``self.nodata_value``.
        """


@dataclass
class SourceTile:
    """A source raster tile in the product's native grid.

    Attributes:
        array: float32 array with shape (C, H, W). Pixels with no data must be
            NaN.
        crs: the rasterio CRS of the tile.
        transform: the affine geotransform of the tile.
    """

    array: np.ndarray
    crs: CRS
    transform: Affine


def mosaic_tiles_to_bounds(
    tiles: Iterable[SourceTile],
    projection: Projection,
    bounds: PixelBounds,
    num_bands: int,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray | None:
    """Reproject and mosaic source tiles onto the requested pixel grid.

    Each tile is warped onto the (projection, bounds) grid with rasterio, and
    tiles are composited first-valid: the first tile providing data for a pixel
    wins. NaN marks pixels no tile covers.

    Args:
        tiles: source tiles in their native grids, with NaN as nodata.
        projection: the target rslearn Projection.
        bounds: pixel bounds (x0, y0, x1, y1) in the target projection.
        num_bands: number of bands in the output (and in every tile).
        resampling: rasterio resampling method used when warping.

    Returns:
        float32 (num_bands, H, W) array with NaN where no tile has data, or
        None if no tile contributed any valid pixel.
    """
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    dst_transform = Affine(
        projection.x_resolution,
        0,
        bounds[0] * projection.x_resolution,
        0,
        projection.y_resolution,
        bounds[1] * projection.y_resolution,
    )

    mosaic = np.full((num_bands, height, width), np.nan, dtype=np.float32)
    filled = np.zeros((height, width), dtype=bool)
    for tile in tiles:
        if tile.array.ndim != 3 or tile.array.shape[0] != num_bands:
            raise ValueError(
                f"expected tile array of shape ({num_bands}, H, W), "
                f"got {tile.array.shape}"
            )
        warped = np.full((num_bands, height, width), np.nan, dtype=np.float32)
        rasterio.warp.reproject(
            source=tile.array,
            destination=warped,
            src_transform=tile.transform,
            src_crs=tile.crs,
            src_nodata=np.nan,
            dst_transform=dst_transform,
            dst_crs=projection.crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )
        valid = ~np.all(np.isnan(warped), axis=0)
        update = valid & ~filled
        if update.any():
            mosaic[:, update] = warped[:, update]
            filled |= update

    if not filled.any():
        return None
    return mosaic


class AEFFetcher(EmbeddingFetcher):
    """Fetches AlphaEarth Foundations (GSE) embeddings from AWS Open Data.

    Wraps rslearn's GoogleSatelliteEmbeddingV1 data source, which reads
    Cloud-Optimized GeoTIFFs directly from S3 and dequantizes int8 values to
    float32 in [-1, 1] (nodata -1.0). Available years: 2018-2024.
    """

    def __init__(self, metadata_cache_dir: str | None = None) -> None:
        """Initialize an AEFFetcher.

        Args:
            metadata_cache_dir: directory used to cache the AEF spatial index
                CSV. Defaults to ~/.cache/olmoearth_pretrain/aef_index.
        """
        if metadata_cache_dir is None:
            metadata_cache_dir = os.path.expanduser(DEFAULT_AEF_CACHE_DIR)
        self._metadata_cache_dir = metadata_cache_dir
        self._source: GoogleSatelliteEmbeddingV1 | None = None
        self._source_lock = threading.Lock()

    @property
    def modality(self) -> ModalitySpec:
        """The GSE modality (64 bands A00..A63)."""
        return Modality.GSE

    @property
    def product_version(self) -> str:
        """AEF product version (the AWS Open Data bucket serves v1)."""
        return "v1"

    @property
    def nodata_value(self) -> float:
        """Dequantized AEF rasters mark nodata as -1.0."""
        return AEF_NODATA

    def _get_source(self) -> GoogleSatelliteEmbeddingV1:
        """Lazily construct the underlying rslearn data source.

        Guarded by a lock so concurrent fetch threads share one source (and
        therefore one cached spatial index) instead of each building their own.
        """
        with self._source_lock:
            if self._source is None:
                self._source = GoogleSatelliteEmbeddingV1(
                    metadata_cache_dir=self._metadata_cache_dir
                )
            return self._source

    def fetch(
        self, bounds: PixelBounds, projection: Projection, year: int
    ) -> np.ndarray | None:
        """Fetch dequantized AEF embeddings for the given bounds and year.

        Args:
            bounds: pixel bounds (x0, y0, x1, y1) in the given projection.
            projection: the target rslearn Projection.
            year: the calendar year to fetch.

        Returns:
            float32 (64, H, W) array in [-1, 1] with -1.0 where the product
            has no data, or None if no AEF item covers (bounds, year).
        """
        source = self._get_source()
        geometry = STGeometry(projection, shapely.box(*bounds), year_time_range(year))
        query_config = QueryConfig(space_mode=SpaceMode.MOSAIC, max_matches=1)
        groups = source.get_items([geometry], query_config)[0]
        items = [item for group in groups for item in group.items]
        if not items:
            return None

        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        num_bands = len(AEF_BANDS)
        mosaic = np.full((num_bands, height, width), AEF_NODATA, dtype=np.float32)
        filled = np.zeros((height, width), dtype=bool)
        for item in items:
            raster = source.read_raster(
                self.modality.name, item, AEF_BANDS, projection, bounds
            )
            chw = raster.get_chw_array().astype(np.float32)
            valid = ~np.all(chw == AEF_NODATA, axis=0)
            update = valid & ~filled
            if update.any():
                mosaic[:, update] = chw[:, update]
                filled |= update

        if not filled.any():
            return None
        return mosaic
