Create Windows
--------------

The first step is to create windows in an rslearn dataset based on one or more sampling
strategies.

Currently only random sampling is supported. Currently the sampling is tied to
populating the rslearn dataset with windows, but we may separate it so the sampling
creates a CSV of locations and there is a different script to create the windows from
that CSV.

Make a new folder for the dataset and copy the dataset configuration files that is used
for initializing the dataset (it has NAIP and Sentinel-2 data sources configured, which
are used to pick the window timestamp and filter out windows that don't have Sentinel-2
coverage).

    mkdir dataset/
    cp data/rslearn_dataset_configs/config_init.json dataset/config.json

Run the random sampling strategy:

    python -m helios.dataset_creation.create_windows.random --ds_path dataset/ --count 1000


Materialize Data
----------------

Now we use rslearn to materialize the data.

Each modality has a separate dataset configuration file, so that they can be ingested
in independent Beaker jobs in the future, but for the sample dataset for now that means
we need to swap in the configuration files one by one.

Sentinel-2:

    cp data/rslearn_dataset_configs/config_sentinel2.json dataset/config.json
    rslearn dataset prepare --root dataset/ --workers 64
    rslearn dataset ingest --root dataset/ --workers 32 --no-use-initial-job --jobs-per-process 1
    rslearn dataset materialize --root dataset/ --workers 32 --no-use-initial-job

For NAIP, we only materialize in the "naip" group (which contains the windows created
using `create_windows.naip`):

    cp data/rslearn_dataset_configs/config_naip.json dataset/config.json
    rslearn dataset prepare --root dataset/ --group naip --workers 64
    rslearn dataset ingest --root dataset/ --group naip --workers 32 --no-use-initial-job --jobs-per-process 1
    rslearn dataset materialize --root dataset/ --group naip --workers 32 --no-use-initial-job

OpenStreetMap:

    cp data/rslearn_dataset_configs/config_openstreetmap.json dataset/config.json
    rslearn dataset prepare --root dataset/
    rslearn dataset ingest --root dataset/
    rslearn dataset materialize --root dataset/ --workers 32 --no-use-initial-job

WorldCover:

    cp data/rslearn_dataset_configs/config_worldcover.json dataset/config.json
    rslearn dataset prepare --root dataset/ --workers 64
    rslearn dataset ingest --root dataset/ --workers 32 --no-use-initial-job --jobs-per-process 1
    rslearn dataset materialize --root dataset/ --workers 32 --no-use-initial-job


Convert Data
------------

Now we convert the data to Helios format. In the future this will also run in different
Beaker jobs for different regions.

    python -m helios.dataset.rslearn_to_helios.naip --ds_path dataset/ --helios_path gs://ai2-helios/data/.../
    python -m helios.dataset.rslearn_to_helios.openstreetmap --ds_path dataset/ --helios_path gs://ai2-helios/data/.../
    python -m helios.dataset.rslearn_to_helios.sentinel2 --ds_path dataset/ --helios_path gs://ai2-helios/data/.../
    python -m helios.dataset.rslearn_to_helios.worldcover --ds_path dataset/ --helios_path gs://ai2-helios/data/.../

These conversions yield individual metadata CSV files for each window. Concatenate them
into the per-modality CSVs:

    python make_meta_summary.py --helios_path gs://ai2-helios/data/.../ --modality naip
    python make_meta_summary.py --helios_path gs://ai2-helios/data/.../ --modality openstreetmap
    python make_meta_summary.py --helios_path gs://ai2-helios/data/.../ --modality sentinel2_monthly
    python make_meta_summary.py --helios_path gs://ai2-helios/data/.../ --modality sentinel2_freq
    python make_meta_summary.py --helios_path gs://ai2-helios/data/.../ --modality worldcover

Finally create the overall index CSV that lists the examples and which modalities are
available for each example:

    python make_index.py --helios_path gs://ai2-helios/data/.../


Beaker
------

```
beaker session create --budget ai2/d5 --workspace ai2/earth-systems --priority high --gpus 1 --shared-memory 128GiB --bare --mount weka://dfive-default=/dfive-default
```
