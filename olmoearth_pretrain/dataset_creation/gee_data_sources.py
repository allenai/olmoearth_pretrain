"""Earth Engine data sources for OlmoEarth Pretrain dataset creation.

These subclass rslearn's generic GEE data source, which exports each window's extent
from Earth Engine to GCS at the window's own projection (10 m/pixel UTM). They are
referenced by class path from data/rslearn_dataset_configs/.

Masked pixels are exported as NODATA_VALUE (the layer configs set the same
nodata_value) so that rslearn's mosaicking fills them from other items instead of
treating Earth Engine's default 0 as valid data.
"""

import ee
from rslearn.config import DType
from rslearn.data_sources import Item
from rslearn.data_sources.data_source import DataSourceContext
from rslearn.data_sources.google_earth_engine import GEE

NODATA_VALUE = -99999


class OpenBuildingsTemporal(GEE):
    """Google Open Buildings 2.5D Temporal, aggregated to the window resolution.

    The rasters have ~4 m effective resolution. building_presence and
    building_height are averaged over each output pixel, so they become the
    building-confidence fraction and the mean height including non-building area
    (built volume per unit area). building_fractional_count is summed, so it
    becomes the building count per output pixel.

    See https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_Research_open-buildings-temporal_v1
    """

    COLLECTION_NAME = "GOOGLE/Research/open-buildings-temporal/v1"
    MEAN_BANDS = ["building_presence", "building_height"]
    SUM_BANDS = ["building_fractional_count"]
    # reduceResolution reads at most this many input pixels per output pixel. The
    # catalog lists both 4 m and 0.5 m pixels; 10 m from 0.5 m is 400 inputs.
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

    def item_to_image(self, item: Item) -> ee.image.Image:
        """Get the Image for one tile, reduced to the export resolution."""
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
        return ee.Image.cat(means, sums).select(self.bands).unmask(NODATA_VALUE, False)
