# Checkpoint Eval Curves GPU Tests

Short checklist for validating checkpoint eval curves on a GPU before launching a full sweep.

Set these once:

```bash
export CLUSTER=local
export BASE_MODULE=scripts/official/base.py
export LARGE_MODULE=scripts/official/large.py
export BASE_CHECKPOINT_DIR=/path/to/official_base/checkpoints
export LARGE_CHECKPOINT_DIR=/path/to/official_large/checkpoints
export H5PY_DIR=/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828
export STEPS=10000
```

Use one small checkpoint step first. Increase `STEPS` only after these pass.

## 1. Command Builder Smoke Test

This validates command generation without running model eval.

```bash
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --checkpoint_dir="$BASE_CHECKPOINT_DIR" \
  --steps="$STEPS" \
  --cluster="$CLUSTER" \
  --module_path="$BASE_MODULE" \
  --model_name=base_gpu_smoke \
  --project_name=checkpoint_eval_curves_smoke \
  --defaults_only \
  --dry_run
```

Check that the printed command uses `checkpoint_sweep_evals.py`, includes `CHECKPOINT_DIR`, and includes `CHECKPOINT_STEPS`.

## 2. Embedding Diagnostics Only

This is the fastest real GPU test. It should log `eval_embed_diagnostics/pretrain_subset/*` to wandb.

```bash
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --checkpoint_dir="$BASE_CHECKPOINT_DIR" \
  --steps="$STEPS" \
  --cluster="$CLUSTER" \
  --module_path="$BASE_MODULE" \
  --model_name=base_embed_diag_smoke \
  --project_name=checkpoint_eval_curves_smoke \
  --defaults_only \
  --embedding_diagnostics_only
```

Pass criteria:
- Loads the checkpoint.
- Runs only the `pretrain_subset` diagnostics task.
- Logs effective rank, norms, cosine stats, and spatial diagnostics.

## 3. Low-Label Fanout

This checks that label percentages create separate runs and override train partitions.

```bash
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --checkpoint_dir="$BASE_CHECKPOINT_DIR" \
  --steps="$STEPS" \
  --cluster="$CLUSTER" \
  --module_path="$BASE_MODULE" \
  --model_name=base_low_label_smoke \
  --project_name=checkpoint_eval_curves_smoke \
  --defaults_only \
  --label_percentages=0.01,0.1 \
  --task-skip-names=pretrain_worldcover_probe,pretrain_osm_probe,pretrain_srtm_regression
```

Pass criteria:
- Creates two runs with `_label0.01x` and `_label0.1x` in their names.
- The `0.01x_train` and `0.10x_train` partitions are used.
- At least one regular downstream eval completes.

## 4. Pretrain Auxiliary Probe Smoke Tests

Run each new pretrain target separately first. These are the most likely to catch label-shape or target-modality issues.

### WorldCover

```bash
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --checkpoint_dir="$BASE_CHECKPOINT_DIR" \
  --steps="$STEPS" \
  --cluster="$CLUSTER" \
  --module_path="$BASE_MODULE" \
  --model_name=base_worldcover_probe_smoke \
  --project_name=checkpoint_eval_curves_smoke \
  --defaults_only \
  --trainer.callbacks.downstream_evaluator.tasks_to_run='["pretrain_worldcover_probe"]' \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_worldcover_probe.h5py_dir="$H5PY_DIR" \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_worldcover_probe.pretrain_train_samples=32 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_worldcover_probe.pretrain_valid_samples=32 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_worldcover_probe.pretrain_test_samples=32 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_worldcover_probe.epochs=2
```

### OSM Raster

```bash
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --checkpoint_dir="$BASE_CHECKPOINT_DIR" \
  --steps="$STEPS" \
  --cluster="$CLUSTER" \
  --module_path="$BASE_MODULE" \
  --model_name=base_osm_probe_smoke \
  --project_name=checkpoint_eval_curves_smoke \
  --defaults_only \
  --trainer.callbacks.downstream_evaluator.tasks_to_run='["pretrain_osm_probe"]' \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_osm_probe.h5py_dir="$H5PY_DIR" \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_osm_probe.pretrain_train_samples=32 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_osm_probe.pretrain_valid_samples=32 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_osm_probe.pretrain_test_samples=32 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_osm_probe.epochs=2
```

### SRTM Regression

```bash
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --checkpoint_dir="$BASE_CHECKPOINT_DIR" \
  --steps="$STEPS" \
  --cluster="$CLUSTER" \
  --module_path="$BASE_MODULE" \
  --model_name=base_srtm_regression_smoke \
  --project_name=checkpoint_eval_curves_smoke \
  --defaults_only \
  --trainer.callbacks.downstream_evaluator.tasks_to_run='["pretrain_srtm_regression"]' \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_srtm_regression.h5py_dir="$H5PY_DIR" \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_srtm_regression.pretrain_train_samples=32 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_srtm_regression.pretrain_valid_samples=32 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_srtm_regression.pretrain_test_samples=32 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_srtm_regression.epochs=2
```

Pass criteria:
- WorldCover and OSM log segmentation metrics such as `miou`, `macro_f1`, and `micro_f1`.
- SRTM logs `mae`, `rmse`, `neg_rmse`, and `r2`.
- No shape mismatch between spatial embeddings and labels.

## 5. Large Model Parity Smoke Test

After base passes, repeat the fastest diagnostics test on the large checkpoint.

```bash
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --checkpoint_dir="$LARGE_CHECKPOINT_DIR" \
  --steps="$STEPS" \
  --cluster="$CLUSTER" \
  --module_path="$LARGE_MODULE" \
  --model_name=large_embed_diag_smoke \
  --project_name=checkpoint_eval_curves_smoke \
  --defaults_only \
  --embedding_diagnostics_only
```

If this OOMs, lower the embedding batch size for the diagnostics task:

```bash
--trainer.callbacks.downstream_evaluator.tasks.pretrain_subset.embedding_batch_size=1
```

## 6. Two-Step Curve Test

Once single-step smoke tests pass, run two nearby checkpoints to verify wandb curves use `checkpoint_step`.

```bash
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --checkpoint_dir="$BASE_CHECKPOINT_DIR" \
  --steps=10000,20000 \
  --cluster="$CLUSTER" \
  --module_path="$BASE_MODULE" \
  --model_name=base_two_step_curve_smoke \
  --project_name=checkpoint_eval_curves_smoke \
  --defaults_only \
  --trainer.callbacks.downstream_evaluator.tasks_to_run='["pretrain_srtm_regression"]' \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_srtm_regression.pretrain_train_samples=32 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_srtm_regression.pretrain_valid_samples=32 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_srtm_regression.pretrain_test_samples=32 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_srtm_regression.epochs=2
```

Pass criteria:
- One wandb run contains both checkpoint steps.
- Metrics are plotted against `checkpoint_step`, not wall-clock eval step.
