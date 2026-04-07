# Handoff Notes: EMA + Supervision Diagnostics

## Context
We're investigating why EMA (exponential moving average target encoder) is unstable when combined with supervision on decode-only modalities. This branch adds embedding quality diagnostics and a "No Maps EMA" experiment that excludes decode-only modalities from the EMA target encoder.

## What was built

### 1. Embedding Diagnostics Module
**File**: `olmoearth_pretrain/evals/embedding_diagnostics.py`

Pure functions operating on `[N, D]` embedding tensors:
- `effective_rank()` — Shannon entropy of SVD singular values. 1 = collapsed, D = fully spread.
- `per_dim_variance_stats()` — mean/min/max/std of per-dim variance + dead dim count.
- `uniformity()` — Wang & Isola 2020. More negative = more uniform. Near 0 = crowding.
- `pairwise_cosine_stats()` — mean/std/min/max of pairwise cosine similarities.
- `embedding_norm_stats()` — L2 norm distribution.
- `compute_embedding_diagnostics()` — runs all of the above, returns flat dict.

**Tests**: `tests/unit/eval/test_embedding_diagnostics.py` — 16 tests, all passing.

### 2. EvalMode.EMBEDDING_DIAGNOSTICS
**File**: `olmoearth_pretrain/train/callbacks/evaluator_callback.py`

New eval mode that only computes embedding diagnostics (no kNN/probe). Configure as:
```python
DownstreamTaskConfig(
    dataset="m-eurosat",  # or "pretrain_subset"
    eval_mode=EvalMode.EMBEDDING_DIAGNOSTICS,
    eval_interval=Duration.steps(2000),
)
```

Existing eval modes (KNN, LINEAR_PROBE) can also run diagnostics by setting `run_embedding_diagnostics=True`.

**Logging**: All metrics go to `eval_embed_diag/{task_name}/{metric}` in both trainer metrics and wandb.

### 3. Pretrain Subset Dataset
**File**: `olmoearth_pretrain/evals/datasets/pretrain_subset.py`

Wraps `OlmoEarthDataset` as an eval dataset. Returns `(MaskedOlmoEarthSample, dummy_label)`.

Config fields on `DownstreamTaskConfig`:
- `h5py_dir` — path to training h5py data (required for pretrain_subset)
- `pretrain_max_samples` — max samples to load (default 512)

Registered in `configs.py` as `"pretrain_subset"` and wired into `get_eval_dataset()`.

### 4. `unmask_excluding()` on MaskedOlmoEarthSample
**File**: `olmoearth_pretrain/datatypes.py`

New method that unmasks all modalities EXCEPT those in the exclude list. Used to prevent decode-only (MAP) modalities from being passed through the EMA target encoder.

### 5. `unmask_exclude_modalities` on Train Modules
**Files**: `olmoearth_pretrain/train/train_module/contrastive_latentmim.py`, `latent_mim.py`

New config field `unmask_exclude_modalities: list[str]`. When set, `model_forward` calls `batch.unmask_excluding(...)` instead of `batch.unmask()` before the target encoder forward.

### 6. Experiment Script
**File**: `scripts/vnext/single_bandset_band_dropout/base_band_dropout_no_s1_drop_random_time_supervision_no_maps_ema.py`

Based on the supervision script with:
- `ema_decay=(0.996, 1.0)` — re-enables EMA
- `unmask_exclude_modalities=ONLY_DECODE_MODALITIES` — MAP tokens excluded from target encoder
- `pretrain_embed_diag` task — embedding diagnostics on training data subset every 2k steps
- `run_embedding_diagnostics=True` on m-eurosat KNN task

## What still needs doing

### Must do before launch
1. **Verify the script dry-runs**: `python3 scripts/vnext/.../base_..._no_maps_ema.py dry_run test_run local`
2. **Check `PretrainSubsetDataset` works with the h5py path** — the h5py dir is hardcoded for the weka cluster; make sure it's accessible from the training nodes.
3. **Run integration tests** to make sure the modified `model_forward` still works: `pytest -vv tests/integration/`

### Design decisions to revisit
- **Diagnostics on val vs train embeddings**: For `run_embedding_diagnostics=True` on downstream tasks, diagnostics currently run on val embeddings. Could switch to train embeddings for consistency with `pretrain_subset`. The diagnostics measure encoder properties so either split should surface the same issues.
- **`PretrainSubsetDataset` uses fixed patch_size=4 and hw_p=8**: These may not match the actual training distribution. The training dataloader samples variable patch sizes and hw_p values. For diagnostics this is fine — we want a consistent measurement — but note it doesn't match the exact training regime.
- **Uniformity subsampling**: `MAX_UNIFORMITY_SAMPLES=2048` caps the O(N²) cost. If pretrain_max_samples is 512, all samples are used (no subsampling).

### Collapse modes this setup detects
| Metric | Failure mode | Healthy signal |
|--------|-------------|----------------|
| effective_rank ↓ | Dimensional collapse | Should be >> 1, ideally 100+ for D=768 |
| num_dead_dims ↑ | Dimensional collapse | Should be 0 |
| uniformity → 0 | Crowding / non-uniformity | Should be negative (e.g. -2 to -4) |
| cosine_sim_mean → 1 | Full collapse | Should be near 0 for random-like |
| norm_std → 0 | Norm collapse | Should have healthy variance |

### Failure modes NOT yet detected
- **Student-teacher divergence**: Would need comparing student vs target encoder outputs. Not implemented as an eval (would need both encoders at eval time).
- **Prediction head collapse**: Diagnostics only measure encoder embeddings. If the projector/predictor collapses but the backbone is fine, we'd miss it. Could add diagnostics on `latent_projected_and_pooled` in a future iteration.
