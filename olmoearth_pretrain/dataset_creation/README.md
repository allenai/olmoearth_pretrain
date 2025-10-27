## Creating the Pre-training Dataset

This guide details how to reproduce the pre-training dataset from data sources like
Microsoft Planetary Computer and the Copernicus Climate Data Store. If just want to use
the pre-training dataset, you can download it from
[Hugging Face](https://huggingface.co/datasets/allenai/olmoearth_pretrain_dataset).

There are three steps:
1. Initialize an rslearn dataset with windows corresponding to desired UTM tiles.
2. Use rslearn to materialize data from various data sources.
3. Convert the data to OlmoEarth format.

## Initialize the rslearn Dataset

The first step is to create windows in an rslearn dataset, where each window
corresponds to a 256x256 tile in a UTM zone, at 10 m/pixel. We divide each UTM zone
into a grid where the grid size is 256x256 pixels or 2560x2560 meters, so for example
"EPSG:32610_0_1" corresponds to (0, 256) to (256, 512) pixels ((0, 2560) to
(2560, 5120) meters) in EPSG:32610, which is one of the UTM zones.

The dataset is sampled based on OpenStreetMap features. Specifically, we select about
120 categories of map features in OpenStreetMap. For each category, we enumerate the
UTM tiles that contain at least one feature in that category. We then randomly sample
up to 10K tiles per category (some categories appear in fewer than 10K tiles). All of
the tiles across the categories get included in the dataset.

We start by downloading and processing the OpenStreetMap data:

```
cd olmoearth_pretrain/dataset_creation/openstreetmap
# Download the latest OpenStreetMap complete data.
wget https://planet.openstreetmap.org/pbf/planet-latest.osm.pbf
# Enumerate the tiles containing at least one feature in each category.
# This produces a file named tiles_by_category.json.
go mod init olmoearth_pretrain
go mod tidy
go run osm_sampling.go
# Randomly sample 10K tiles per category, and output the longitude and latitude at the
# center of each tile to a JSON file.
cd ../../..
python -m olmoearth_pretrain.dataset_creation.openstreetmap.osm_tiles_by_category_to_lonlats --in_fname olmoearth_pretrain/dataset_creation/openstreetmap/tiles_by_category.json --out_fname openstreetmap_lonlats.json --tiles_per_category 10000
```

We can then use these longitudes and latitudes to initialize an rslearn dataset with
one window corresponding to each tile.

Make a new folder for the dataset and copy the dataset configuration files that is used
for initializing the dataset (it has NAIP and Sentinel-2 data sources configured, which
are used to pick the window timestamp and filter out windows that don't have Sentinel-2
coverage).

    mkdir dataset/
    cp data/rslearn_dataset_configs/config_init.json dataset/config.json

The NAIP data source derives data from an AWS bucket, and so AWS credentials must be
set (i.e. the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables).

Then run:

```
python -m olmoearth_pretrain.dataset_creation.create_windows.from_lon_lat_list --ds_path dataset/ --fname openstreetmap_lonlats.json
```

To use the same tiles as our dataset, download the JSON file of longitudes/latitudes
from https://huggingface.co/datasets/allenai/olmoearth_pretrain_dataset/blob/main/lon_lats.json.

### Alternative Sampling Strategies

We tested two alternative sampling strategies: sampling based on the land cover
distribution using WorldCover, and random sampling. However, the final dataset is based
on OpenStreetMap features as described above.

For creating windows with random sampling:

```
python -m olmoearth_pretrain.dataset_creation.create_windows.random --ds_path dataset/ --count 50000
```

## Materialize the Data

Now we use [rslearn](https://github.com/allenai/rslearn) to materialize the data. It
can be installed via pip:

```
pip install rslearn
```

For simplicity, we provide a different dataset configuration file for each modality.
These need to be copied to the dataset path before running rslearn commands to
materialize the data.

Sentinel-1, Sentinel-2 L2A, and NAIP are ingested from Microsoft Planetary Computer, which
supports random access. Here are the steps:

```
# Set DATASET_PATH to the location of your rslearn dataset from the previous step.
export DATASET_PATH=./dataset/
# Choose one configuration file to copy and run the rslearn commands with at a time.
cp data/rslearn_dataset_configs/config_{sentinel1,sentinel2_l2a,naip_10}.json $DATASET_PATH/config.json
rslearn dataset prepare --root $DATASET_PATH --group res_10 --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60 --jobs-per-process 16
rslearn dataset materialize --root $DATASET_PATH --group res_10 --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60
```

OpenStreetMap and WorldCereal can be processed on one machine. We use 16 workers for
preparing and ingesting since processing the PBF (for OpenStreetMap) or GeoTIFF (for
WorldCereal) can use a lot of memory:

```
cp data/rslearn_dataset_configs/config_{openstreetmap,worldcereal}.json $DATASET_PATH/config.json
rslearn dataset prepare --root $DATASET_PATH --group res_10 --workers 16 --load-workers 64
rslearn dataset ingest --root $DATASET_PATH --group res_10 --workers 16 --load-workers 64 --no-use-initial-job
rslearn dataset materialize --root $DATASET_PATH --group res_10 --workers 64 --no-use-initial-job
```

WorldCover, SRTM, CDL, and WRI Canopy Height Map can also be processed on one machine:

```
cp data/rslearn_dataset_configs/config_{worldcover,srtm,cdl,wri_canopy_height_map}.json $DATASET_PATH/config.json
rslearn dataset prepare --root $DATASET_PATH --group res_10 --workers 64
rslearn dataset ingest --root $DATASET_PATH --group res_10 --workers 64 --no-use-initial-job
rslearn dataset materialize --root $DATASET_PATH --group res_10 --workers 64 --no-use-initial-job
```

For ERA5 data:

```
cp data/rslearn_dataset_configs/config_era5_10.json $DATASET_PATH/config.json
rslearn dataset prepare --root $DATASET_PATH --group res_10 --workers 64
rslearn dataset ingest --root $DATASET_PATH --group res_10 --workers 64 --no-use-initial-job
rslearn dataset materialize --root $DATASET_PATH --group res_10 --workers 64 --no-use-initial-job
```

### Landsat

Landsat 8/9 data is from an AWS bucket and should be materialized on an AWS machine.
Then the data can be transferred back after converting to OlmoEarth Pretrain format. This minimizes
the egress fee.

First copy the res_10 windows to the AWS machine:

    rsync -av --exclude layers --exclude items.json $DATASET_PATH/windows/res_10/ ubuntu@X:/mnt/rslearn_dataset/windows/res_10/

Then materialize the data on the AWS machine:

    export DATASET_PATH=/mnt/rslearn_dataset
    cp data/rslearn_dataset_configs/config_landsat.json $DATASET_PATH/config.json
    rslearn dataset prepare --root $DATASET_PATH --group res_10 --workers 64 --no-use-initial-job
    rslearn dataset materialize --root $DATASET_PATH --group res_10 --workers 64 --no-use-initial-job --ignore-errors --retry-max-attempts 4 --retry-backoff-seconds 1

### Other Modalities

These modalities are supported in the dataset creation code, but were not used to train
the final OlmoEarth models.

ERA5 data at 160 m/pixel:

```
cp data/rslearn_dataset_configs/config_era5.json $DATASET_PATH/config.json
rslearn dataset prepare --root $DATASET_PATH --group res_160 --workers 64
rslearn dataset ingest --root $DATASET_PATH --group res_160 --workers 64 --no-use-initial-job
rslearn dataset materialize --root $DATASET_PATH --group res_160 --workers 64 --no-use-initial-job
```

NAIP at 0.625 m/pixel windows:

```
cp data/rslearn_dataset_configs/config_naip.json $DATASET_PATH/config.json
rslearn dataset prepare --root $DATASET_PATH --group res_0.625 --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60 --jobs-per-process 16
rslearn dataset materialize --root $DATASET_PATH --group res_0.625 --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60
```

WorldPop world population data. We found the data was not fine-grained enough for it to
help understand the ~10 m/pixel images we use in OlmoEarth:

```
cp data/rslearn_dataset_configs/config_worldpop.json $DATASET_PATH/config.json
rslearn dataset prepare --root $DATASET_PATH --group res_10 --workers 64
rslearn dataset ingest --root $DATASET_PATH --group res_10 --workers 64 --no-use-initial-job
rslearn dataset materialize --root $DATASET_PATH --group res_10 --workers 64 --no-use-initial-job
```

For Google Satellite Embedding v1, use the commands below. It will start Google Earth
Engine export tasks which should take ~20 seconds each, and can be monitored from the
GCP console. The prepare step will download a CSV containing the geometry of all of the
Images in the GEE ImageCollection.

```
cp data/rslearn_dataset_configs/config_google_satembedding.json $DATASET_PATH/config.json
rslearn dataset prepare --root $DATASET_PATH --group res_10 --workers 64
rslearn dataset materialize --root $DATASET_PATH --workers 32 --load-workers 128 --group res_10 --no-use-initial-job --ignore-errors
```

### Parallelizing Materialization

For Sentinel-1 and Sentinel-2 L2A, it is helpful to parallelize the materialization
jobs. You can build a Docker image for this purpose:

```
docker build -f olmoearth_pretrain/dataset_creation/Dockerfile -t olmoearth-dataset-creation .
```

### Sentinel-2 L1C

Sentinel-2 L1C is supported in the code but not used in the final dataset. We obtain it
from a public GCS bucket which does not support random access. We provide a GCP Batch
job launcher to launch many jobs for materializing the data. The windows need to be
copied to GCS, and the data needs to be copied back after conversion to OlmoEarth format.

First copy the res_10 windows to GCS, along with rtree index:

```
cp data/rslearn_dataset_configs/config_sentinel2.json $DATASET_PATH/config.json
rslearn dataset prepare --root $DATASET_PATH --group res_10 --workers 64
# Replace BUCKET_NAME with your GCS bucket.
export GCS_DATASET_PATH=gs://BUCKET_NAME/
gsutil -m rsync -r -x '.*layers' $DATASET_PATH/windows/res_10/ $GCS_DATASET_PATH/windows/res_10/
gsutil cp $DATASET_PATH/cache/sentinel2/rtree_index.idx $GCS_DATASET_PATH/cache/sentinel2/rtree_index.idx
gsutil cp $DATASET_PATH/cache/sentinel2/rtree_index.dat $GCS_DATASET_PATH/cache/sentinel2/rtree_index.dat
gsutil cp $DATASET_PATH/cache/sentinel2/rtree_index.done $GCS_DATASET_PATH/cache/sentinel2/rtree_index.done
```

Build the Docker image:

```
# The tag is just an example, and needs to be adjusted so the image is pushed to GCP.
docker build -f olmoearth_pretrain/dataset_creation/Dockerfile -t us-west1-docker.pkg.dev/GCP_PROJECT/olmoearth/olmoearth-sentinel2-l1c .
docker image push us-west1-docker.pkg.dev/GCP_PROJECT/olmoearth/olmoearth-sentinel2-l1c
```

Launch the jobs:
```
gsutil cp data/rslearn_dataset_configs/config_sentinel2.json $GCS_DATASET_PATH/config.json
# Test with 1 job first.
python -m olmoearth_pretrain.dataset_creation.scripts.sentinel2_l1c.launch_jobs --ds_path $GCS_DATASET_PATH --image us-west1-docker.pkg.dev/GCP_PROJECT/olmoearth/olmoearth-sentinel2-l1c --project GCP_PROJECT --region us-west1 --max_jobs 1 --workers 128
# Then if it works run all the jobs.
python -m olmoearth_pretrain.dataset_creation.scripts.sentinel2_l1c.launch_jobs --ds_path $GCS_DATASET_PATH --image us-west1-docker.pkg.dev/GCP_PROJECT/olmoearth/olmoearth-sentinel2-l1c --project GCP_PROJECT --region us-west1 --workers 128
```

## Convert Data to OlmoEarth Format

Now convert the data from the rslearn dataset to OlmoEarth format.

```
export OLMOEARTH_PATH=./olmoearth_dataset
python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.cdl --ds_path $DATASET_PATH --olmoearth_path $OLMOEARTH_PATH
python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.era5_10 --ds_path $DATASET_PATH --olmoearth_path $OLMOEARTH_PATH
python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.landsat --ds_path $DATASET_PATH --olmoearth_path $OLMOEARTH_PATH
python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.naip_10 --ds_path $DATASET_PATH --olmoearth_path $OLMOEARTH_PATH
python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.openstreetmap --ds_path $DATASET_PATH --olmoearth_path $OLMOEARTH_PATH
python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.sentinel1 --ds_path $DATASET_PATH --olmoearth_path $OLMOEARTH_PATH
python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.sentinel2_l2a --ds_path $DATASET_PATH --olmoearth_path $OLMOEARTH_PATH
python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.srtm --ds_path $DATASET_PATH --olmoearth_path $OLMOEARTH_PATH
python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.worldcereal --ds_path $DATASET_PATH --olmoearth_path $OLMOEARTH_PATH
python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.worldcover --ds_path $DATASET_PATH --olmoearth_path $OLMOEARTH_PATH
python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.wri_canopy_height_map --ds_path $DATASET_PATH --olmoearth_path $OLMOEARTH_PATH
# The modalities below are not used in our final dataset but supported in this code.
python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.era5 --ds_path $DATASET_PATH --olmoearth_path $OLMOEARTH_PATH
python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.gse --ds_path $DATASET_PATH --olmoearth_path $OLMOEARTH_PATH
python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.naip --ds_path $DATASET_PATH --olmoearth_path $OLMOEARTH_PATH
python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.sentinel2 --ds_path $GCS_DATASET_PATH --olmoearth_path $OLMOEARTH_PATH
python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.worldpop --ds_path $DATASET_PATH --olmoearth_path $OLMOEARTH_PATH
```

The conversions yield individual metadata CSV files for each window. Concatenate them
into the per-modality CSVs:

```
python -m olmoearth_pretrain.dataset_creation.make_meta_summary --olmoearth_path $OLMOEARTH_PATH --modality cdl
python -m olmoearth_pretrain.dataset_creation.make_meta_summary --olmoearth_path $OLMOEARTH_PATH --modality era5_10 --time_span two_week
python -m olmoearth_pretrain.dataset_creation.make_meta_summary --olmoearth_path $OLMOEARTH_PATH --modality era5_10 --time_span year
python -m olmoearth_pretrain.dataset_creation.make_meta_summary --olmoearth_path $OLMOEARTH_PATH --modality landsat --time_span two_week
python -m olmoearth_pretrain.dataset_creation.make_meta_summary --olmoearth_path $OLMOEARTH_PATH --modality landsat --time_span year
python -m olmoearth_pretrain.dataset_creation.make_meta_summary --olmoearth_path $OLMOEARTH_PATH --modality naip_10
python -m olmoearth_pretrain.dataset_creation.make_meta_summary --olmoearth_path $OLMOEARTH_PATH --modality openstreetmap
python -m olmoearth_pretrain.dataset_creation.make_meta_summary --olmoearth_path $OLMOEARTH_PATH --modality sentinel1 --time_span two_week
python -m olmoearth_pretrain.dataset_creation.make_meta_summary --olmoearth_path $OLMOEARTH_PATH --modality sentinel1 --time_span year
python -m olmoearth_pretrain.dataset_creation.make_meta_summary --olmoearth_path $OLMOEARTH_PATH --modality sentinel2_l2a --time_span two_week
python -m olmoearth_pretrain.dataset_creation.make_meta_summary --olmoearth_path $OLMOEARTH_PATH --modality sentinel2_l2a --time_span year
python -m olmoearth_pretrain.dataset_creation.make_meta_summary --olmoearth_path $OLMOEARTH_PATH --modality srtm
python -m olmoearth_pretrain.dataset_creation.make_meta_summary --olmoearth_path $OLMOEARTH_PATH --modality worldcereal
python -m olmoearth_pretrain.dataset_creation.make_meta_summary --olmoearth_path $OLMOEARTH_PATH --modality worldcover
python -m olmoearth_pretrain.dataset_creation.make_meta_summary --olmoearth_path $OLMOEARTH_PATH --modality wri_canopy_height_map
# The modalities below are not used in our final dataset but supported in this code.
python -m olmoearth_pretrain.dataset_creation.make_meta_summary --olmoearth_path $OLMOEARTH_PATH --modality era5 --time_span two_week
python -m olmoearth_pretrain.dataset_creation.make_meta_summary --olmoearth_path $OLMOEARTH_PATH --modality era5 --time_span year
python -m olmoearth_pretrain.dataset_creation.make_meta_summary --olmoearth_path $OLMOEARTH_PATH --modality gse
python -m olmoearth_pretrain.dataset_creation.make_meta_summary --olmoearth_path $OLMOEARTH_PATH --modality worldpop
python -m olmoearth_pretrain.dataset_creation.make_meta_summary --olmoearth_path $OLMOEARTH_PATH --modality naip
python -m olmoearth_pretrain.dataset_creation.make_meta_summary --olmoearth_path $OLMOEARTH_PATH --modality sentinel2 --time_span two_week
python -m olmoearth_pretrain.dataset_creation.make_meta_summary --olmoearth_path $OLMOEARTH_PATH --modality sentinel2 --time_span year
```

We use a rasterized version of the OpenStreetMap vector data for pre-training, produced
by this script:

```
python -m olmoearth_pretrain.dataset_creation.rslearn_to_olmoearth.rasterize_openstreetmap --olmoearth_path $OLMOEARTH_PATH
```

## Create H5s

We can now create the H5 files used during training from the OlmoEarth dataset. The H5s
are better optimized for reading during training. Note that here we split up each
256x256 example in the dataset into four 128x128 examples.

```
python -m olmoearth_pretrain.internal.run_h5_conversion --tile_path=$OLMOEARTH_PATH --supported_modality_names='[cdl,era5_10,landsat,naip_10,openstreetmap_raster,sentinel1,sentinel2_l2a,srtm,worldcereal,worldcover,wri_canopy_height_map]' --compression=zstd --compression_opts=3 --tile_size=128
```
