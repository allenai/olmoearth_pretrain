## Internal Documentation

A Beaker session can be used to run most of the window creation and data
materialization steps:

```
beaker session create --budget ai2/es-platform --workspace ai2/earth-systems --priority high --gpus 1 --shared-memory 128GiB --bare --mount src=weka,ref=dfive-default,dst=/weka/dfive-default
```

The only exception is for Sentinel-1 and Sentinel-2 L2A, where it may be desirable to
run multiple `rslearn dataset materialize` commands in parallel. You can use the
[Beaker Data Materialization in rslearn_projects](https://github.com/allenai/rslearn_projects/tree/master/rslp/common#beaker-data-materialization)
for this purpose. That link shows how to build the Beaker image, and you can
materialize like this:

```
python -m rslp.main common launch_data_materialization_jobs --image BEAKER_IMAGE_NAME --ds_path /weka/path/to/rslearn/dataset --hosts+=jupiter-cs-aus-134.reviz.ai2.in --hosts+=jupiter-cs-aus-135.reviz.ai2.in --command '["rslearn", "dataset", "materialize", "--root", "/weka/path/to/rslearn/dataset", "--group", "res_10", "--workers", "64", "--no-use-initial-job", "--retry-max-attempts", "8", "--retry-backoff-seconds", "60", "--ignore-errors"]'
```

These scripts were previously used but may no longer be relevant:
- `olmoearth_pretrain.dataset_creation.scripts.get_dataset_neighbors`: this was used
  to get the longitude/latitude JSON for creating the `presto_neighbor` dataset, which
  contains the neighboring tiles to the `presto` dataset.
- `olmoearth_pretrain.dataset_creation.scripts.remove_duplicate_lonlats`: for each
  successive dataset, we used this script to remove duplicates that were already in any
  of the previous datasets. It takes in a longitude/latitude JSON along with a list of
  existing dataset paths, and outputs a pruned longitude/latitude JSON.
