# Studio Dataset Ingestion

> **Internal use only.** Requires AI2 Weka access and internal rslearn datasets.
> See [`docs/Adding-Eval-Datasets.md`](../../../docs/Adding-Eval-Datasets.md) for the full tutorial.

This module ingests rslearn datasets from Weka/GCS into the OlmoEarth eval registry,
computing splits and band stats so datasets can be used as linear probe evaluations
in training loops.

## Quick reference

```bash
# Ingest a dataset
OLMOEARTH_INGEST_WORKERS=16 NAME=my_task \
  SOURCE=/weka/dfive-default/rslearn-eai/datasets/my_task \
  CONFIG=/weka/dfive-default/henryh/helios/olmoearth_projects/olmoearth_run_data/my_task
OLMOEARTH_INGEST_WORKERS=16 nohup python -m olmoearth_pretrain.evals.studio_ingest.cli ingest \
  --name "$NAME" --source "$SOURCE" --olmoearth-run-config-path "$CONFIG" \
  --register --overwrite > "${NAME}_ingest.out" 2>&1 &

# List registered datasets
python -m olmoearth_pretrain.evals.studio_ingest.cli list

# Inspect a registered dataset
python -m olmoearth_pretrain.evals.studio_ingest.cli info --name my_task
```

## Existing datasets

| Name | Task | Modalities |
|------|------|-----------|
| `tolbi_crop` | segmentation | sentinel2_l2a |
| `canada_wildfire_sat_eval_split` | segmentation | sentinel2_l2a |
| `yemen_crop` | segmentation | sentinel2_l2a |
| `geo_ecosystem_annual_test` | segmentation | sentinel2_l2a, sentinel1, landsat |
| `forest_loss_driver` | classification | sentinel2_l2a, sentinel1, landsat |
| `nigeria_settlement` | segmentation | sentinel2_l2a, sentinel1, landsat |
| `nandi_crop_map` | segmentation | sentinel2_l2a, sentinel1, landsat |
| `awf_lulc_map` | segmentation | sentinel2_l2a, sentinel1, landsat |

See the full tutorial for how to add a new one.
