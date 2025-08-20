Create Windows
--------------

The first step is to create windows in an rslearn dataset based on one or more sampling
strategies.

Make a new folder for the dataset and copy the dataset configuration files that is used
for initializing the dataset (it has NAIP and Sentinel-2 data sources configured, which
are used to pick the window timestamp and filter out windows that don't have Sentinel-2
coverage).

    mkdir dataset/
    cp data/rslearn_dataset_configs/config_init.json dataset/config.json

The NAIP data source derives data from an AWS bucket, and so AWS credentials must be
set (i.e. the `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` environment variables).

There are two ways to create windows: randomly sampling longitude/latitude pairs, or
creating windows based on locations in a JSON file containing a list of [lon, lat]
sub-lists. The latter method is used for both the WorldCover-based sampling and for the
OSM-based sampling datasets, see `helios/dataset_creation/scripts/osm_sampling.go` and
`helios.dataset_creation.scripts.osm_tiles_by_category_to_lonlats`.

For creating windows with random sampling:

    python -m helios.dataset_creation.create_windows.random --ds_path dataset/ --count 1000

For creating windows from locations in a JSON file:

    python -m helios.dataset_creation.create_windows.from_lon_lat_list --ds_path dataset/ --fname list.json


Materialize Data
----------------

Now we use rslearn to materialize the data.

Sentinel-1 and Sentinel-2 L2A are ingested from Microsoft Planetary Computer which
supports random access so it is relatively fast. For 10K+ windows, it may be helpful to
parallelize the materialize commands.

For Sentinel-1:

    # Set DATASET_PATH to the location of your rslearn dataset from the previous step.
    export DATASET_PATH=/weka/dfive-default/helios/dataset_creation/X/
    cp data/rslearn_dataset_configs/config_sentinel1.json $DATASET_PATH/config.json
    rslearn dataset prepare --root $DATASET_PATH --group res_10 --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60 --jobs-per-process 16
    rslearn dataset materialize --root $DATASET_PATH --group res_10 --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60

For Sentinel-2 L2A:

    cp data/rslearn_dataset_configs/config_sentinel2_l2a.json $DATASET_PATH/config.json
    rslearn dataset prepare --root $DATASET_PATH --group res_10 --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60 --jobs-per-process 16
    rslearn dataset materialize --root $DATASET_PATH --group res_10 --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60

NAIP is also from Planetary Computer, but it only applies to the 0.625 m/pixel windows:

    cp data/rslearn_dataset_configs/config_naip.json $DATASET_PATH/config.json
    rslearn dataset prepare --root $DATASET_PATH --group res_0.625 --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60 --jobs-per-process 16
    rslearn dataset materialize --root $DATASET_PATH --group res_0.625 --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60

For 10 m/pixel window/tiling NAIP data:

    cp data/rslearn_dataset_configs/config_naip_10.json $DATASET_PATH/config.json
    rslearn dataset prepare --root $DATASET_PATH --group res_10 --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60 --jobs-per-process 16
    rslearn dataset materialize --root $DATASET_PATH --group res_10 --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60

OpenStreetMap can be processed on one machine. We use 16 workers for preparing and
ingesting since processing the PBF can use a lot of memory:

    cp data/rslearn_dataset_configs/config_openstreetmap.json $DATASET_PATH/config.json
    rslearn dataset prepare --root $DATASET_PATH --group res_10 --workers 16
    rslearn dataset ingest --root $DATASET_PATH --group res_10 --workers 16 --no-use-initial-job
    rslearn dataset materialize --root $DATASET_PATH --group res_10 --workers 64 --no-use-initial-job

WorldCover, SRTM, CDL, and WorldPop can also be processed on one machine:

    cp data/rslearn_dataset_configs/config_{worldcover,srtm,cdl,worldpop}.json $DATASET_PATH/config.json
    rslearn dataset prepare --root $DATASET_PATH --group res_10 --workers 64
    rslearn dataset ingest --root $DATASET_PATH --group res_10 --workers 64 --no-use-initial-job
    rslearn dataset materialize --root $DATASET_PATH --group res_10 --workers 64 --no-use-initial-job

For Google Satellite Embedding v1, use the commands below. It will start Google Earth
Engine export tasks which should take ~20 seconds each, and can be monitored from the
GCP console. The prepare step will download a CSV containing the geometry of all of the
Images in the GEE ImageCollection.

    cp data/rslearn_dataset_configs/config_google_satembedding.json $DATASET_PATH/config.json
    rslearn dataset prepare --root $DATASET_PATH --group res_10 --workers 64
    rslearn dataset materialize --root $DATASET_PATH --workers 32 --load-workers 128 --group res_10 --no-use-initial-job --ignore-errors

WorldCereal can also be processed on one machine. As with OpenStreetMap we use 16 workers.

    cp data/rslearn_dataset_configs/config_worldcereal.json $DATASET_PATH/config.json
    rslearn dataset prepare --root $DATASET_PATH --group res_10 --workers 64
    rslearn dataset ingest --root $DATASET_PATH --group res_10 --workers 16 --no-use-initial-job
    rslearn dataset materialize --root $DATASET_PATH --group res_10 --workers 16 --no-use-initial-job

ERA5 can also be processed on one machine:

    cp data/rslearn_dataset_configs/config_era5.json $DATASET_PATH/config.json
    rslearn dataset prepare --root $DATASET_PATH --group res_160 --workers 64
    rslearn dataset ingest --root $DATASET_PATH --group res_160 --workers 64 --no-use-initial-job
    rslearn dataset materialize --root $DATASET_PATH --group res_160 --workers 64 --no-use-initial-job

For 10 m/pixel window/tiling ERA5 data:

    cp data/rslearn_dataset_configs/config_era5_10.json $DATASET_PATH/config.json
    rslearn dataset prepare --root $DATASET_PATH --group res_10 --workers 64
    rslearn dataset ingest --root $DATASET_PATH --group res_10 --workers 64 --no-use-initial-job
    rslearn dataset materialize --root $DATASET_PATH --group res_10 --workers 64 --no-use-initial-job



The steps above can be performed in a Beaker session:

```
beaker session create --budget ai2/d5 --workspace ai2/earth-systems --priority high --gpus 1 --shared-memory 128GiB --bare --mount src=weka,ref=dfive-default,dst=/weka/dfive-default
```

For Sentinel-1 and Sentinel-2 L2A, it is helpful to parallelize the materialization
jobs. Create a Docker image and start the jobs:

    docker build -f helios/dataset_creation/scripts/Dockerfile -t helios-dataset-creation .
    beaker image create --name helios-dataset-creation helios-dataset-creation
    python -m helios.dataset_creation.scripts.beaker_launcher --ds_path $DATASET_PATH --modality sentinel1 --image_name favyen/helios-dataset-creation --hosts jupiter-cs-aus-130.reviz.ai2.in,jupiter-cs-aus-131.reviz.ai2.in,jupiter-cs-aus-132.reviz.ai2.in,jupiter-cs-aus-133.reviz.ai2.in

This will start a CPU-only Beaker job on each of the specified hosts.


Convert Data
------------

Now we convert the data to Helios format.

    export HELIOS_PATH=/weka/dfive-default/helios/dataset/X/
    python -m helios.dataset_creation.rslearn_to_helios.naip --ds_path $DATASET_PATH --helios_path $HELIOS_PATH
    python -m helios.dataset_creation.rslearn_to_helios.naip_10 --ds_path $DATASET_PATH --helios_path $HELIOS_PATH
    python -m helios.dataset_creation.rslearn_to_helios.openstreetmap --ds_path $DATASET_PATH --helios_path $HELIOS_PATH
    python -m helios.dataset_creation.rslearn_to_helios.sentinel1 --ds_path $DATASET_PATH --helios_path $HELIOS_PATH
    python -m helios.dataset_creation.rslearn_to_helios.sentinel2_l2a --ds_path $DATASET_PATH --helios_path $HELIOS_PATH
    python -m helios.dataset_creation.rslearn_to_helios.srtm --ds_path $DATASET_PATH --helios_path $HELIOS_PATH
    python -m helios.dataset_creation.rslearn_to_helios.worldcover --ds_path $DATASET_PATH --helios_path $HELIOS_PATH
    python -m helios.dataset_creation.rslearn_to_helios.gse --ds_path $DATASET_PATH --helios_path $HELIOS_PATH
    python -m helios.dataset_creation.rslearn_to_helios.worldcereal --ds_path $DATASET_PATH --helios_path $HELIOS_PATH
    python -m helios.dataset_creation.rslearn_to_helios.cdl --ds_path $DATASET_PATH --helios_path $HELIOS_PATH
    python -m helios.dataset_creation.rslearn_to_helios.worldpop --ds_path $DATASET_PATH --helios_path $HELIOS_PATH
    python -m helios.dataset_creation.rslearn_to_helios.era5 --ds_path $DATASET_PATH --helios_path $HELIOS_PATH
    python -m helios.dataset_creation.rslearn_to_helios.era5_10 --ds_path $DATASET_PATH --helios_path $HELIOS_PATH


Landsat
-------

Landsat 8/9 data is from an AWS bucket and should be materialized on an AWS machine.
Then the data can be transferred back after converting to Helios format. This minimizes
the egress fee.

First copy the res_10 windows to the AWS machine:

    rsync -av --exclude layers --exclude items.json $DATASET_PATH/windows/res_10/ ubuntu@X:/mnt/rslearn_dataset/windows/res_10/

Then materialize the data on the AWS machine:

    export DATASET_PATH=/mnt/rslearn_dataset
    cp data/rslearn_dataset_configs/config_landsat.json $DATASET_PATH/config.json
    rslearn dataset prepare --root $DATASET_PATH --group res_10 --workers 64 --no-use-initial-job
    rslearn dataset materialize --root $DATASET_PATH --group res_10 --workers 64 --no-use-initial-job --ignore-errors --retry-max-attempts 4 --retry-backoff-seconds 1

Convert the data:

    python -m helios.dataset_creation.rslearn_to_helios.landsat --ds_path $DATASET_PATH --helios_path $HELIOS_PATH


Sentinel-2 L1C
--------------

Sentinel-2 L1C does not support random access. A special GCP Batch job launcher exists
to launch many jobs for materializing the data. The windows need to be copied to GCS,
and the data needs to be copied back to Weka after conversion to Helios format.

First copy the res_10 windows to GCS, along with rtree index:

    cp data/rslearn_dataset_configs/config_sentinel2.json $DATASET_PATH/config.json
    rslearn dataset prepare --root $DATASET_PATH --group res_10 --workers 64
    export GCS_DATASET_PATH=gs://ai2-helios/dataset_creation/X
    gsutil -m rsync -r -x '.*layers' $DATASET_PATH/windows/res_10/ $GCS_DATASET_PATH/windows/res_10/
    gsutil cp $DATASET_PATH/cache/sentinel2/rtree_index.idx $GCS_DATASET_PATH/cache/sentinel2/rtree_index.idx
    gsutil cp $DATASET_PATH/cache/sentinel2/rtree_index.dat $GCS_DATASET_PATH/cache/sentinel2/rtree_index.dat
    gsutil cp $DATASET_PATH/cache/sentinel2/rtree_index.done $GCS_DATASET_PATH/cache/sentinel2/rtree_index.done

Build the Docker image:

    cd /path/to/helios/
    docker build -f helios/dataset_creation/scripts/Dockerfile -t us-west1-docker.pkg.dev/earthsystem-dev-c3po/helios/helios-sentinel2-l1c .
    docker image push us-west1-docker.pkg.dev/earthsystem-dev-c3po/helios/helios-sentinel2-l1c

Launch the jobs:

    gsutil cp data/rslearn_dataset_configs/config_sentinel2.json $GCS_DATASET_PATH/config.json
    # Test with 1 job first.
    python -m helios.dataset_creation.scripts.sentinel2_l1c.launch_jobs --ds_path $GCS_DATASET_PATH --image us-west1-docker.pkg.dev/earthsystem-dev-c3po/helios/helios-sentinel2-l1c --project earthsystem-dev-c3po --region us-west1 --max_jobs 1 --workers 128
    # Then if it works run all the jobs.
    python -m helios.dataset_creation.scripts.sentinel2_l1c.launch_jobs --ds_path $GCS_DATASET_PATH --image us-west1-docker.pkg.dev/earthsystem-dev-c3po/helios/helios-sentinel2-l1c --project earthsystem-dev-c3po --region us-west1 --workers 128

Convert it to Helios format:

    python -m helios.dataset_creation.rslearn_to_helios.sentinel2 --ds_path $GCS_DATASET_PATH --helios_path $HELIOS_PATH


Concatenated CSVs
-----------------

The conversions yield individual metadata CSV files for each window. Concatenate them
into the per-modality CSVs:

    python -m helios.dataset_creation.make_meta_summary --helios_path $HELIOS_PATH --modality landsat --time_span two_week
    python -m helios.dataset_creation.make_meta_summary --helios_path $HELIOS_PATH --modality landsat --time_span year
    python -m helios.dataset_creation.make_meta_summary --helios_path $HELIOS_PATH --modality naip
    python -m helios.dataset_creation.make_meta_summary --helios_path $HELIOS_PATH --modality naip_10
    python -m helios.dataset_creation.make_meta_summary --helios_path $HELIOS_PATH --modality openstreetmap
    python -m helios.dataset_creation.make_meta_summary --helios_path $HELIOS_PATH --modality sentinel1 --time_span two_week
    python -m helios.dataset_creation.make_meta_summary --helios_path $HELIOS_PATH --modality sentinel1 --time_span year
    python -m helios.dataset_creation.make_meta_summary --helios_path $HELIOS_PATH --modality sentinel2 --time_span two_week
    python -m helios.dataset_creation.make_meta_summary --helios_path $HELIOS_PATH --modality sentinel2 --time_span year
    python -m helios.dataset_creation.make_meta_summary --helios_path $HELIOS_PATH --modality sentinel2_l2a --time_span two_week
    python -m helios.dataset_creation.make_meta_summary --helios_path $HELIOS_PATH --modality sentinel2_l2a --time_span year
    python -m helios.dataset_creation.make_meta_summary --helios_path $HELIOS_PATH --modality srtm
    python -m helios.dataset_creation.make_meta_summary --helios_path $HELIOS_PATH --modality worldcover
    python -m helios.dataset_creation.make_meta_summary --helios_path $HELIOS_PATH --modality gse
    python -m helios.dataset_creation.make_meta_summary --helios_path $HELIOS_PATH --modality worldcereal
    python -m helios.dataset_creation.make_meta_summary --helios_path $HELIOS_PATH --modality cdl
    python -m helios.dataset_creation.make_meta_summary --helios_path $HELIOS_PATH --modality worldpop
    python -m helios.dataset_creation.make_meta_summary --helios_path $HELIOS_PATH --modality era5 --time_span two_week
    python -m helios.dataset_creation.make_meta_summary --helios_path $HELIOS_PATH --modality era5 --time_span year
    python -m helios.dataset_creation.make_meta_summary --helios_path $HELIOS_PATH --modality era5_10 --time_span two_week
    python -m helios.dataset_creation.make_meta_summary --helios_path $HELIOS_PATH --modality era5_10 --time_span year


Create H5s
----------

```
python -m helios.internal.run_h5_conversion --tile_path=$HELIOS_PATH --supported_modality_names='[sentinel2_l2a,sentinel1,worldcover,srtm,landsat,openstreetmap_raster,gse,cdl,worldpop]' --compression=zstd --compression_opts=3 --tile_size=128
```
