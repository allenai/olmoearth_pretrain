Create Windows
--------------

The first step is to create windows in an rslearn dataset based on one or more sampling
strategies.

The sampling strategies will be developed outside of the dataset creation code, but for
now we create windows based on some hand-chosen locations.

Make a new folder for the dataset and copy one of the dataset configuration files:

    mkdir dataset/
    cp data/rslearn_dataset_configs/config_sentinel2.json dataset/config.json

Run one or more of the window creation scripts:

    python -m helios.dataset.create_windows.random --ds_path dataset/
    python -m helios.dataset.create_windows.naip --ds_path dataset/

`create_windows.random` creates windows with a random timestamp, while
`create_windows.naip` creates them based on the timestamp of a NAIP image.


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
