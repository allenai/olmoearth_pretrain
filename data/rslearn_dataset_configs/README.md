These are rslearn dataset configuration files for different modalities in OlmoEarth Pretrain.

Some notes:
- `config_sentinel2.json`: this specifies tile store at `/tmp/rslearn_helios_tile_store`
  because materialization is meant to be run from many Beaker jobs. Use
  `python -m olmoearth_pretrain.dataset_creation.scripts.sentinel2_l1c.launch_jobs` to launch the
  ingestion/materialization jobs. Each job will process one batch (default 10) of
  windows.
