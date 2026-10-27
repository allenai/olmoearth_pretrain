"""Earth Engine data sources for OlmoEarth Pretrain dataset creation.

These subclass rslearn's generic GEE data source and are referenced by class path
from data/rslearn_dataset_configs/.

Masked pixels are returned as NODATA_VALUE (the layer configs set the same
nodata_value) so that rslearn's mosaicking fills them from other items instead of
treating Earth Engine's default 0 as valid data.
"""

import ee
import numpy as np
from rasterio.enums import Resampling
from rslearn.config import DType
from rslearn.data_sources import Item
from rslearn.data_sources.data_source import DataSourceContext
from rslearn.data_sources.google_earth_engine import GEE
from rslearn.utils.geometry import PixelBounds, Projection
from rslearn.utils.raster_array import RasterArray, RasterMetadata
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds

NODATA_VALUE = -99999

# Earth Engine endpoint intended for many small concurrent requests.
HIGH_VOLUME_URL = "https://earthengine-highvolume.googleapis.com"


class OpenBuildingsTemporal(GEE):
    """Google Open Buildings 2.5D Temporal, aggregated to the window resolution.

    The rasters have 0.5 m pixels (~4 m effective resolution). building_presence and
    building_height are averaged over each output pixel, so they become the
    building-confidence fraction and the mean height including non-building area
    (built volume per unit area). building_fractional_count is summed, so it
    becomes the building count per output pixel.

    Windows are small (256x256 at 10 m/pixel), so each one is fetched with a
    synchronous computePixels request instead of the base class's batch export to
    GCS, which takes tens of seconds per window.

    See https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_Research_open-buildings-temporal_v1
    """

    COLLECTION_NAME = "GOOGLE/Research/open-buildings-temporal/v1"
    MEAN_BANDS = ["building_presence", "building_height"]
    SUM_BANDS = ["building_fractional_count"]
    # reduceResolution reads at most this many input pixels per output pixel; 10 m
    # from 0.5 m is 400 inputs.
    MAX_INPUT_PIXELS = 1024

    def __init__(
        self,
        gcs_bucket_name: str,
        index_cache_dir: str,
        service_account_name: str,
        service_account_credentials: str,
        context: DataSourceContext = DataSourceContext(),
    ):
        """Create a new OpenBuildingsTemporal. See GEE for the arguments."""
        super().__init__(
            collection_name=self.COLLECTION_NAME,
            gcs_bucket_name=gcs_bucket_name,
            index_cache_dir=index_cache_dir,
            service_account_name=service_account_name,
            service_account_credentials=service_account_credentials,
            dtype=DType.FLOAT32,
            context=context,
        )
        ee.Initialize(
            ee.ServiceAccountCredentials(
                service_account_name, service_account_credentials
            ),
            opt_url=HIGH_VOLUME_URL,
        )

    def item_to_image(self, item: Item) -> ee.image.Image:
        """Get the Image for one tile, reduced to the requested resolution."""
        image = super().item_to_image(item)
        means = (
            image.select(self.MEAN_BANDS)
            .toFloat()
            .reduceResolution(ee.Reducer.mean(), maxPixels=self.MAX_INPUT_PIXELS)
        )
        sums = (
            image.select(self.SUM_BANDS)
            .toFloat()
            .reduceResolution(ee.Reducer.sum(), maxPixels=self.MAX_INPUT_PIXELS)
        )
        # Reducer.mean outputs doubles, and Earth Engine exports need one data type.
        return (
            ee.Image.cat(means, sums)
            .select(self.bands)
            .unmask(NODATA_VALUE, False)
            .toFloat()
        )

    def get_raster_metadata(
        self, layer_name: str, item: Item, bands: list[str]
    ) -> RasterMetadata:
        """Return the nodata value without exporting the item."""
        return RasterMetadata(nodata_value=NODATA_VALUE)

    def read_raster(
        self,
        layer_name: str,
        item: Item,
        bands: list[str],
        projection: Projection,
        bounds: PixelBounds,
        resampling: Resampling = Resampling.bilinear,
    ) -> RasterArray:
        """Fetch the item on the requested grid with one computePixels request."""
        transform = get_transform_from_projection_and_bounds(projection, bounds)
        array = ee.data.computePixels(
            {
                "expression": self.item_to_image(item).select(bands),
                "fileFormat": "NUMPY_NDARRAY",
                "grid": {
                    "dimensions": {
                        "width": bounds[2] - bounds[0],
                        "height": bounds[3] - bounds[1],
                    },
                    "affineTransform": {
                        "scaleX": transform.a,
                        "shearX": transform.b,
                        "translateX": transform.c,
                        "shearY": transform.d,
                        "scaleY": transform.e,
                        "translateY": transform.f,
                    },
                    "crsCode": projection.crs.to_string(),
                },
            }
        )
        return RasterArray(
            chw_array=np.stack([array[band] for band in bands]).astype(np.float32),
            time_range=item.geometry.time_range,
            metadata=RasterMetadata(nodata_value=NODATA_VALUE),
        )
