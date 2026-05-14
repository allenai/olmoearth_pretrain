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
export ALL_STEPS=50000,100000,150000,200000,250000,300000,350000,400000,450000,500000,550000,600000,650000
export WANDB_PROJECT=2026_05_phase2_eval_curves
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

## 3. Low-Label Smoke Test

This checks that `label_fraction` creates one low-label run and maps through the
dataset-specific low-label implementation. Standard eval datasets use their
existing train partitions; pretrain probes scale only their train sample count,
leaving valid/test counts unchanged.

```bash
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --checkpoint_dir="$BASE_CHECKPOINT_DIR" \
  --steps="$STEPS" \
  --cluster="$CLUSTER" \
  --module_path="$BASE_MODULE" \
  --model_name=base_low_label_smoke \
  --project_name=checkpoint_eval_curves_smoke \
  --defaults_only \
  --label_fraction=0.1 \
  --task-names=m_eurosat,m_bigearthnet
```

Pass criteria:
- Creates a run with `_label0.1x` in the name.
- The generated command contains `label_fraction=0.1`, not `partition=...`.
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
  --task-names=pretrain_worldcover_probe_s2 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_worldcover_probe_s2.h5py_dir="$H5PY_DIR" \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_worldcover_probe_s2.pretrain_train_samples=32 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_worldcover_probe_s2.pretrain_valid_samples=32 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_worldcover_probe_s2.pretrain_test_samples=32 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_worldcover_probe_s2.epochs=2
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
  --task-names=pretrain_osm_probe_s2 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_osm_probe_s2.h5py_dir="$H5PY_DIR" \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_osm_probe_s2.pretrain_train_samples=32 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_osm_probe_s2.pretrain_valid_samples=32 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_osm_probe_s2.pretrain_test_samples=32 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_osm_probe_s2.epochs=2
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
  --task-names=pretrain_srtm_regression_s2 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_srtm_regression_s2.h5py_dir="$H5PY_DIR" \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_srtm_regression_s2.pretrain_train_samples=32 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_srtm_regression_s2.pretrain_valid_samples=32 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_srtm_regression_s2.pretrain_test_samples=32 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_srtm_regression_s2.epochs=2
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
  --task-names=pretrain_srtm_regression_s2 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_srtm_regression_s2.pretrain_train_samples=32 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_srtm_regression_s2.pretrain_valid_samples=32 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_srtm_regression_s2.pretrain_test_samples=32 \
  --trainer.callbacks.downstream_evaluator.tasks.pretrain_srtm_regression_s2.epochs=2
```

Pass criteria:
- One wandb run contains both checkpoint steps.
- Metrics are plotted against `checkpoint_step`, not wall-clock eval step.
- Metrics stream to W&B after each evaluator completes, rather than waiting for
  all tasks at the checkpoint to finish.

## 7. Production Sweep: S2 Pretrain Probes

Dry-run first, then remove `--dry_run` to launch. Run base and large separately.

```bash
export PRETRAIN_PROBE_TASKS=pretrain_worldcover_probe_s2,pretrain_osm_probe_s2,pretrain_srtm_regression_s2,pretrain_canopy_regression_s2,pretrain_cdl_probe_s2,pretrain_worldcereal_probe_s2,pretrain_worldcover_probe_geo_s2,pretrain_osm_probe_geo_s2,pretrain_srtm_regression_geo_s2,pretrain_canopy_regression_geo_s2,pretrain_cdl_probe_geo_s2,pretrain_worldcereal_probe_geo_s2

python -m olmoearth_pretrain.internal.full_eval_sweep \
  --checkpoint_dir="$BASE_CHECKPOINT_DIR" \
  --steps="$ALL_STEPS" \
  --cluster=ai2/saturn \
  --module_path="$BASE_MODULE" \
  --model_name=phase2_base_pretrain_probes_s2 \
  --project_name="$WANDB_PROJECT" \
  --defaults_only \
  --task-names="$PRETRAIN_PROBE_TASKS" \
  --dry_run

python -m olmoearth_pretrain.internal.full_eval_sweep \
  --checkpoint_dir="$LARGE_CHECKPOINT_DIR" \
  --steps="$ALL_STEPS" \
  --cluster=ai2/saturn \
  --module_path="$LARGE_MODULE" \
  --model_name=phase2_large_pretrain_probes_s2 \
  --project_name="$WANDB_PROJECT" \
  --defaults_only \
  --task-names="$PRETRAIN_PROBE_TASKS" \
  --dry_run
```

## 8. Production Sweep: Other Tasks at 10% Labels

This runs all non-pretrain-probe tasks, including embedding diagnostics, with
`label_fraction=0.1`. The fraction affects train labels only; validation and
test splits remain full size.

```bash
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --checkpoint_dir="$BASE_CHECKPOINT_DIR" \
  --steps="$ALL_STEPS" \
  --cluster=ai2/saturn \
  --module_path="$BASE_MODULE" \
  --model_name=phase2_base_other_tasks_label0.1 \
  --project_name="$WANDB_PROJECT" \
  --defaults_only \
  --label_fraction=0.1 \
  --task-skip-names="$PRETRAIN_PROBE_TASKS" \
  --dry_run

python -m olmoearth_pretrain.internal.full_eval_sweep \
  --checkpoint_dir="$LARGE_CHECKPOINT_DIR" \
  --steps="$ALL_STEPS" \
  --cluster=ai2/saturn \
  --module_path="$LARGE_MODULE" \
  --model_name=phase2_large_other_tasks_label0.1 \
  --project_name="$WANDB_PROJECT" \
  --defaults_only \
  --label_fraction=0.1 \
  --task-skip-names="$PRETRAIN_PROBE_TASKS" \
  --dry_run
```

Always commit and push before launching; Beaker checks out the branch from git,
so local-only changes will not be included.
