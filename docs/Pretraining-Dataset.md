## OlmoEarth Pre-training Dataset

The OlmoEarth pre-training dataset consists of 285,288 samples of multi-modal satellite
image time series and auxiliary raster data. Each sample is specified by a 2560 x 2560 m
grid cell in a UTM projection, along with a 360-day time range. The data for each sample
consists of up to three satellite image time series, with one mosaic for each 30-day
period during the 360-day time range, along with up to seven auxiliary rasters.

- Download Link: https://huggingface.co/datasets/allenai/olmoearth_pretrain_dataset/

The satellite image modalities are:

- [Sentinel-2 L2A](https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-data/sentinel-2). We include all 12 bands.
- [Sentinel-1 IW GRD vv+vh](https://dataspace.copernicus.eu/data-collections/sentinel-data/sentinel-1). We include both bands.
- [Landsat 8/9 OLI-TIRS images](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-landsat-archives-landsat-8-oli-operational-land-imager-and).
  We use [Landsat Collection 2 Level-1 Data](https://www.usgs.gov/landsat-missions/landsat-collection-2-level-1-data), and include all 11 bands.

The auxiliary rasters are:

- The [USDA Cropland Data Layer](https://www.nass.usda.gov/Research_and_Science/Cropland/SARS1a.php) from USDA, public domain.
- [ERA5 monthly reanalysis data from ECMWF](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means), available under [CC-BY](https://spdx.org/licenses/CC-BY-4.0).
- [OpenStreetMap vector data](https://www.openstreetmap.org/), available under [ODbL](https://www.openstreetmap.org/copyright).
- [Shuttle Radar Topography Mission (SRTM)](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-srtm-1) Digital Elevation from USGS, public domain.
- [WorldCereal 2021](https://esa-worldcereal.org/en/products/global-maps) cropland map from ESA, available under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
- [WorldCover 2021](https://esa-worldcover.org/) land cover map from ESA, available under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
- [WRI Canopy Height Maps](https://dataforgood.facebook.com/dfg/tools/canopy-height-maps) from World Resources Institute and Meta, available under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

Note that ERA5 is a time series, while the others consist of up to one raster per
sample. USDA CDL is available annually in the continental US, so for samples in the
continental US, we include CDL data from the closest matching year. The other rasters
are uni-temporal, and we include them regardless of the sample's time range, as long as
there is data intersecting the sample spatially.

We re-sample data from their native resolution to resolutions that are powers of 2 off
from 10 m/pixel. For example, the 10 m/pixel bands in Sentinel-2 L2A are stored at
10 m/pixel, while the 60 m/pixel bands are resampled to 40 m/pixel. The canopy height
maps are an exception, where we resample from ~1 m/pixel down to 10 m/pixel, as are
OpenStreetMap features, which are rasterized from vector features to 2.5 m/pixel.

Some samples may be missing some modalities, or individual timesteps of multi-temporal
modalities, if no data is available at the location and time range of the sample.

## Spatial and Temporal Distribution

The dataset is sampled based on OpenStreetMap features. We select about 120 categories
of map features in OpenStreetMap, and enumerate the 2560 x 2560 m UTM grid cells that
contain at least one feature in each category. We then randomly sample up to 10K tiles
per category to get the 285,288 samples.

The categories can be found in the `olmoearth_pretrain/dataset_creation/openstreetmap/osm_sampling.go` script.

The time range for each sample is selected as follows:

- For samples in the continental US, with 75% probability, we look for National
  Agriculture Imagery Program images from 2016-2024 containing the sample's grid cell,
  and use the 360-day period centered at a random matching image.
- For other samples, or with 25% probability for samples in the continental US, we
  uniformly sample the time range between January 2016 and December 2024.

For each UTM projection, we define a grid with 2560 x 2560 m cells. The corner
coordinates of cells are multiples of 2560 m (projection units). We name a cell with
topleft coordinates `(x_off, y_off)` by `(x_off / 2560, y_off / -2560)`. Thus, the tile
"EPSG:32610_248_-2176" corresponds to the bounding box from (634880 m, 5570560 m) to
(637440 m, 5568000 m) in projection units in EPSG:32610.

## Dataset Format

The dataset is available on
[Hugging Face](https://huggingface.co/datasets/allenai/olmoearth_pretrain_dataset/tree/main).
GeoTIFFs corresponding to each modality have been archived into 50 GB tar files, which
must be extracted first, e.g.:

```
wget https://huggingface.co/datasets/allenai/olmoearth_pretrain_dataset/resolve/main/10_cdl/0000.tar -O 10_cdl/0000.tar
tar xvf 10_cdl/0000.tar
```

Once extracted, the dataset layout is like this:

```
10_cdl/
  EPSG:32610_150_-2089_10.tif
  ...
10_era5_10_monthly/
  EPSG:32610_150_-2089_2560.tif
  ...
10_landsat_monthly/
  EPSG:32610_150_-2089_10.tif
  EPSG:32610_150_-2089_20.tif
  ...
10_openstreetmap_raster/
  EPSG:32610_150_-2089_2.5.tif
  ...
10_sentinel1_monthly/
  EPSG:32610_150_-2089_10.tif
  ...
10_sentinel2_l2a_monthly/
  EPSG:32610_150_-2089_10.tif
  EPSG:32610_150_-2089_20.tif
  EPSG:32610_150_-2089_40.tif
  ...
10_srtm/
  EPSG:32610_150_-2089_10.tif
  ...
10_worldcereal/
  EPSG:32610_150_-2089_10.tif
  ...
10_worldcover/
  EPSG:32610_150_-2089_10.tif
  ...
10_wri_canopy_height_map/
  EPSG:32610_150_-2089_10.tif
  ...
```

So, each subfolder contains GeoTIFFs for one modality, and the data for one sample can
be found in files with the same prefix across the modality folders. Above, we show the
filenames containing data for the "EPSG:32610_150_-2089" sample.

The filename suffix, like "_2.5.tif" for OpenStreetMap, indicates the resolution that
the data is stored at. For example, the ERA5 GeoTIFFs are stored at 2560 m/pixel,
meaning it is a 1x1 image.

For multitemporal modalities (ERA5, Sentinel-1, Sentinel-2, and Landsat), the timesteps
are stacked on the channel axis in the GeoTIFF.

ERA5 consists of 6 bands:
- 2m-temperature
- 2m-dewpoint-temperature
- surface-pressure
- 10m-u-component-of-wind
- 10m-v-component-of-wind
- total-precipitation

Thus, samples with complete ERA5 data will contain 12*6 bands.

Sentinel-1 consists of 2 bands: vv, vh.

Sentinel-2 is split into three files since bands are captured at different resolutions.

- 10 m/pixel: B02, B03, B04, B08
- 20 m/pixel: B05, B06, B07, B8A, B11, B12
- 40 m/pixel (resampled from 60 m/pixel): B01, B09

Landsat is split into two files for the same reason:

- 10 m/pixel (resampled from 15 m/pixel): B8
- 20 m/pixel (resampled from 30 m/pixel): B1, B2, B3, B4, B5, B6, B7, B9, B10, B11

SRTM and canopy height maps are single-band. The value is the elevation and canopy
height, respectively, in meters.

CDL and WorldCover are also single-band, and the value is a modality-specific class ID.
Please refer to the original data sources for more details about the classes.

OpenStreetMap has 30 bands. See `olmoearth_pretrain/data/constants.py` for the band
meanings. See that file as well for WorldCereal, which has 8 bands.

You can use qgis or similar software to quickly visualize all of the data pertaining to
one sample:

```
qgis */EPSG:32610_150_-2089_10.tif
```

## CSV Files

For each modality, there is a CSV file (like "10_cdl.csv") that specifies the time
range of each image/timestep contained in the dataset for a sample. For multi-temporal
modalities, there will be multiple rows in the CSV, one for each per-timestep mosaic
that is stacked on the band axis in the corresponding GeoTIFF(s).

Here is an example for Sentinel-2 L2A:

```
crs,col,row,tile_time,image_idx,start_time,end_time
EPSG:32610,150,-2089,2018-12-12T00:00:00+00:00,0,2018-06-08T00:00:00+00:00,2018-07-08T00:00:00+00:00
EPSG:32610,150,-2089,2018-12-12T00:00:00+00:00,1,2018-07-08T00:00:00+00:00,2018-08-07T00:00:00+00:00
EPSG:32610,150,-2089,2018-12-12T00:00:00+00:00,2,2018-08-07T00:00:00+00:00,2018-09-06T00:00:00+00:00
EPSG:32610,150,-2089,2018-12-12T00:00:00+00:00,3,2018-09-06T00:00:00+00:00,2018-10-06T00:00:00+00:00
EPSG:32610,150,-2089,2018-12-12T00:00:00+00:00,4,2018-10-06T00:00:00+00:00,2018-11-05T00:00:00+00:00
EPSG:32610,150,-2089,2018-12-12T00:00:00+00:00,5,2018-11-05T00:00:00+00:00,2018-12-05T00:00:00+00:00
EPSG:32610,150,-2089,2018-12-12T00:00:00+00:00,6,2018-12-05T00:00:00+00:00,2019-01-04T00:00:00+00:00
EPSG:32610,150,-2089,2018-12-12T00:00:00+00:00,7,2019-01-04T00:00:00+00:00,2019-02-03T00:00:00+00:00
EPSG:32610,150,-2089,2018-12-12T00:00:00+00:00,8,2019-02-03T00:00:00+00:00,2019-03-05T00:00:00+00:00
EPSG:32610,150,-2089,2018-12-12T00:00:00+00:00,9,2019-03-05T00:00:00+00:00,2019-04-04T00:00:00+00:00
EPSG:32610,150,-2089,2018-12-12T00:00:00+00:00,10,2019-04-04T00:00:00+00:00,2019-05-04T00:00:00+00:00
EPSG:32610,150,-2089,2018-12-12T00:00:00+00:00,11,2019-05-04T00:00:00+00:00,2019-06-03T00:00:00+00:00
```

The `tile_time` column is the center of the 360-day time range for the sample, while
the `start_time` and `end_time` columns indicate the start/end time used to create the
mosaic. The `image_idx` is the index in the GeoTIFF; for Sentinel-2 L2A 10 m/pixel
data, the second timestep would be at 0-indexed bands 4, 5, 6, and 7.

## H5 Files

For training, the samples are split up from 2560 m x 2560 m to 1280 m x 1280 m, and all
of the data for each 1280 m x 1280 m sub-sample is packaged together in an H5 file.
This enables fast data loading.

The tar files containing these per-sample H5s are in the `h5py_data` folder.

These tar files can be used for pre-training (see [Pretraining](Pretraining.md)), and
can be generated from the GeoTIFFs (see [DatasetCreation](Dataset-Creation.md)).
