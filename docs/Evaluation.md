# OlmoEarth Evaluation Guide

This guide explains how we launch evaluations for OlmoEarth checkpoints and baseline models, including KNN, linear probing, and finetuning jobs.

---

## Choose Your Evaluation Path

> **🏢 AI2 Researchers (Internal):**
> You have access to Beaker/Weka clusters and shared checkpoints. Skim [Setup-Internal.md](Setup-Internal.md) for environment details, then follow the launch instructions below.

> **🌍 External Users:**
> You can run these workflows on local/cloud GPUs. You will need the datasets referenced in [Dataset Setup](Pretraining.md#dataset-setup).

---

## Table of Contents

1. [Evaluation Overview](#evaluation-overview)
2. [Datasets & Model Checkpoints](#datasets--model-checkpoints)
3. [Quick Start](#quick-start)
4. [KNN / Linear Probing](#knn--linear-probing)
5. [Finetune](#finetune-sweep)
6. [Embedding Diagnostics](#embedding-diagnostics)
7. [Tiling Artifact Diagnostics](#tiling-artifact-diagnostics)
8. [Monitoring & Outputs](#monitoring--outputs)
9. [Helpful Files](#helpful-files)
10. [Adding New Eval Datasets (Internal)](#adding-new-eval-datasets-internal)

---

## Evaluation Overview

We run evaluations through the same `olmoearth_pretrain/internal/experiment.py` entrypoint used for pretraining. The helper scripts below build the underlying launch commands:

- `olmoearth_pretrain/internal/full_eval_sweep.py` runs KNN (classification) and linear probing (segmentation) sweeps for OlmoEarth checkpoints or baseline models, with optional sweeps over learning rate, pretrained / dataset normalizers, and pooling (mean or max).
- `olmoearth_pretrain/internal/full_eval_sweep_finetune.py` runs fine-tuning sweeps for OlmoEarth checkpoints or baseline models, with optional sweeps over learning rate and pretrained / dataset normalizers.

Both scripts use:
- [`olmoearth_pretrain/internal/all_evals.py`](../olmoearth_pretrain/internal/all_evals.py) for the task registry (`EVAL_TASKS` for KNN and linear probing, and `FT_EVAL_TASKS` for fine-tuning).
- [`olmoearth_pretrain/evals`](../olmoearth_pretrain/evals) for dataset and model wrappers.

Every launch uses one of the evaluation subcommands in `experiment.py`:
- `dry_run_evaluate` prints the config (no execution) for quick checks.
- `evaluate` runs the job locally.
- `launch_evaluate` submits the job to Beaker.

The sweep scripts set `TRAIN_SCRIPT_PATH` automatically and select `torchrun` for local runs and `python3` for Beaker jobs.

### Prerequisites

- Python environment configured as described in [Pretraining.md](Pretraining.md#environment-setup).
- One 80 GB GPU (A100 or H100 recommended). If you see OOM errors when running some tasks, consider reducing the batch size, e.g., use the override `--TASK_NAME.ft_batch_size` to adjust batch size for fine-tuning.

### Supported Models

- **OlmoEarth models:** Nano, Tiny, Base, and Large size.
- **Others:** Supported baseline models are defined in `olmoearth_pretrain/evals/models/__init__.py`, which includes Galileo, Satlas, Terramind, Prithvi v2, Panopticon, CROMA, AnySat etc. Multi-size variants (if available) are also supported.

---

## Datasets & Model Checkpoints

- **Evaluation datasets**
  - *Internal*: All datasets live on Weka, the defaults in [`evals/datasets/paths.py`](../olmoearth_pretrain/evals/datasets/paths.py) point to shared mounts.
  - *External*: Follow the download instructions in [Pretraining.md](Pretraining.md#evaluation-datasets).
- **OlmoEarth checkpoints**
  - *Internal*: All checkpoints (distributed weights) live on Weka. The paths are defined below:
    ```
    Nano: /weka/dfive-default/helios/checkpoints/joer/nano_lr0.001_wd0.002/step370000
    Tiny: /weka/dfive-default/helios/checkpoints/joer/tiny_lr0.0002_wd0.02/step360000
    Base: /weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02/step667200
    Large: /weka/dfive-default/helios/checkpoints/joer/phase2.0_large_lr0.0001_wd0.002/step560000
    ```
  - *External*: Clone the release repos from Hugging Face, e.g.:
    ```bash
    git clone git@hf.co:allenai/OlmoEarth-v1-Nano
    git clone git@hf.co:allenai/OlmoEarth-v1-Tiny
    git clone git@hf.co:allenai/OlmoEarth-v1-Base
    git clone git@hf.co:allenai/OlmoEarth-v1-Large
    ```
  - Pass the desired checkpoint directory via `--checkpoint_path` and the corresponding `--module_path` (scripts can be found in `olmoearth_pretrain/scripts/official`) when running the evaluation sweeps.

- **Baselines**: When using `--model=<name>`, some models (e.g., AnySat, Panopticon, Terramind) will automatically download checkpoints from Hugging Face or TorchHub. Others models require manually downloading their checkpoints and set the model path in the config (for example, set `load_directory` for Satlas model as defined in `olmoearth_pretrain/evals/models/satlas/satlas.py`).

---

## Quick Start

### 1. Activate your environment

```bash
source .venv-olmoearth_pretrain/bin/activate
```

If you would like to evaluate the models against the Breizhcrops dataset, breizhcrops must be explicitly imported into the codebase. You can do this by running `uv pip install breizhcrops==0.0.4.1`.

### 2. Run a dry run to inspect the commands

```bash
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --cluster=local \
  --checkpoint_path=/your/path/to/OlmoEarth-v1-Base \
  --module_path=scripts/official/base.py \
  --defaults_only \
  --dry_run
```

This prints the exact command without executing it.

### 3. Launch for real

Remove `--dry_run` once the command looks correct. Pick the launch target you need:

- **Local GPUs (`--cluster=local`)**

  ```bash
  python -m olmoearth_pretrain.internal.full_eval_sweep \
    --cluster=local \
    --checkpoint_path=/your/path/to/OlmoEarth-v1-Base \
    --module_path=scripts/official/base.py \
    --project_name=olmoearth_evals \
    --defaults_only
  ```

- **Beaker (`--cluster=<ai2 cluster>`, internal only)**

  ```bash
  python -m olmoearth_pretrain.internal.full_eval_sweep \
    --cluster=ai2/ceres \
    --checkpoint_path=/weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02/step667200 \
    --module_path=scripts/official/base.py \
    --project_name=olmoearth_evals \
    --defaults_only
  ```

---

## KNN / Linear Probing

Use `olmoearth_pretrain/internal/full_eval_sweep.py` for KNN and linear probing tasks.

### Required flags

- `--cluster`: Cluster identifier (*External Users*: set it to `local`).
- `--module_path`: Override the launch module (defaults to the model-specific launcher).
- Exactly one of:
  - `--checkpoint_path`: Passing OlmoEarth checkpoint.
  - `--model=<baseline_name>` or `--model=all`: Evaluate baseline models defined in [`evals/models`](../olmoearth_pretrain/evals/models).

### Common optional flags

- `--project_name`: W&B project (defaults to `EVAL_WANDB_PROJECT`).
- `--defaults_only`: Run a single command using the default lr / normalization / pooling.
- `--lr_only`: Sweep learning rates but keep normalization + pooling at defaults.
- `--all_sizes` or `--size=<variant>`: Evaluate every published size for multi-size baselines.
- `--model-skip-names=a,b`: Skip a subset when using `--model=all`.
- `--select_best_val`: Uses validation metric to pick the best epoch before reporting test metrics.
- `--dry_run`: Print commands without launching.

When `--model=all`, the script automatically switches to the correct launcher for each model and constructs run names like `<checkpoint>_lr1e-3_norm_dataset_pool_mean`.

### Example: Local run for OlmoEarth
```bash
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --cluster=local \
  --checkpoint_path=/your/path/to/OlmoEarth-v1-Nano \
  --module_path=scripts/official/nano.py \
  --project_name=olmoearth_evals \
  --select_best_val
  --trainer.callbacks.downstream_evaluator.run_on_test=True
  --trainer.callbacks.downstream_evaluator.tasks_to_run=\[m_eurosat\]
  --defaults_only
```

### Example: Beaker run for OlmoEarth
```bash
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --cluster=ai2/saturn \
  --checkpoint_path=/weka/dfive-default/helios/checkpoints/joer/nano_lr0.001_wd0.002/step370000 \
  --module_path=scripts/official/nano.py \
  --project_name=olmoearth_evals \
  --select_best_val
  --trainer.callbacks.downstream_evaluator.run_on_test=True
  --trainer.callbacks.downstream_evaluator.tasks_to_run=\[m_eurosat\]
  --defaults_only
```

### Example: Local run for Galileo
```bash
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --cluster=local \
  --model=galileo \
  --all_sizes \
  --select_best_val \
  --project_name=baselines_evals \
  --trainer.callbacks.downstream_evaluator.run_on_test=True
  --trainer.callbacks.downstream_evaluator.tasks_to_run=\[m_eurosat\]
  --defaults_only
```

### Example: Beaker run for Galileo
```bash
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --cluster=ai2/saturn \
  --model=galileo \
  --all_sizes \
  --select_best_val \
  --project_name=baselines_evals \
  --trainer.callbacks.downstream_evaluator.run_on_test=True
  --trainer.callbacks.downstream_evaluator.tasks_to_run=\[m_eurosat\]
  --defaults_only
```

---

## Finetune Sweep

Use `olmoearth_pretrain/internal/full_eval_sweep_finetune.py` for fine-tuning tasks.

### Required flags

- `--cluster`: Cluster identifier (*External Users*: set it to `local`).
- One of:
  - `--checkpoint_path`: Fine-tune an OlmoEarth checkpoint.
  - `--model=<preset_key>`: Use a baseline preset (choices listed in the script’s help).

### Fine-tune specific flags

- `--defaults_only`: Run only the first learning rate in `FT_LRS`.
- `--module_path`: Override the launch script (defaults to the preset’s launcher).
- `--use_dataset_normalizer`: Force dataset statistics even when a preset has its own pretrained normalizer. Leave unset to keep the pretrained normalizer.
- `--finetune_seed`: Set a random base seed for running the downstream tasks.
- `--dry_run`: Print commands without launching.

### Example: Local run for OlmoEarth
```bash
python -m olmoearth_pretrain.internal.full_eval_sweep_finetune \
  --cluster=local \
  --checkpoint_path=/your/path/to/OlmoEarth-v1-Base \
  --module_path=scripts/official/base.py \
  --project_name=olmoearth_evals \
  --defaults_only \
```

### Example: Beaker run for OlmoEarth with lr sweep
```bash
python -m olmoearth_pretrain.internal.full_eval_sweep_finetune \
  --cluster=ai2/ceres \
  --checkpoint_path=/weka/dfive-default/helios/checkpoints/joer/phase2.0_base_lr0.0001_wd0.02/step667200 \
  --module_path=scripts/official/base.py \
  --project_name=olmoearth_evals \
  --finetune_seed=1234 \
```

### Example: Local run for Terramind using dataset normalizer
```bash
python -m olmoearth_pretrain.internal.full_eval_sweep_finetune \
  --cluster=local \
  --model=terramind \
  --project_name=baseline_evals \
  --use_dataset_normalizer \
  --defaults_only
```

### Example: Beaker run for Terramind Large
```bash
python -m olmoearth_pretrain.internal.full_eval_sweep_finetune \
  --cluster=ai2/ceres \
  --model=terramind_large \
  --project_name=baseline_evals \
  --defaults_only
```

---

## Embedding Diagnostics

Embedding diagnostics measure the geometric quality of encoder representations without requiring labeled data. They detect common self-supervised pretraining failure modes such as dimensional collapse, representation crowding, and patch uniformity loss.

### What it measures

| Metric | Healthy range | What it detects |
|--------|--------------|-----------------|
| `effective_rank` | > 0.7 × D | Dimensional collapse (few active SVD components) |
| `uniformity` | < -2.0 | How uniformly embeddings cover the hypersphere |
| `cosine_sim_mean` | < 0.3 | Representation crowding (all embeddings similar) |
| `intra_cosine_sim_mean` | < 0.5 | Patch collapse within images (bad for segmentation) |

For spatial (patch-level) embeddings, metrics are reported with three prefixes:
- `global_*` — all patches flattened together
- `inter_*` — mean-pooled per image, then compared across images
- `intra_*` — patch diversity within each image

### Running during training (in-loop)

Embedding diagnostics are included in the default `build_trainer_config` in `scripts/official/script.py`. They run on a fixed subset of pretrain data (`pretrain_subset_128`) at the interval specified:

```python
DownstreamTaskConfig(
    dataset="pretrain_subset_128",
    eval_mode=EvalMode.EMBEDDING_DIAGNOSTICS,
    embedding_batch_size=4,
    eval_interval=Duration.steps(20000),
    h5py_dir=H5PY_DIR,
    pretrain_max_samples=256,
    input_modalities=[Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name, Modality.LANDSAT.name],
)
```

Metrics are logged to W&B under `eval_embed_diagnostics/<task_name>/<metric>`.

### Running on saved checkpoints

Use `checkpoint_sweep_evals.py` with the `EMBEDDING_DIAGNOSTICS_ONLY` env var:

```bash
EMBEDDING_DIAGNOSTICS_ONLY=1 \
TRAIN_SCRIPT_PATH=scripts/official/base.py \
CHECKPOINT_DIR=/weka/.../checkpoints/my_run \
torchrun olmoearth_pretrain/internal/checkpoint_sweep_evals.py \
    evaluate my_run_embed_diag local
```

Or via `all_evals.py`:

```bash
EMBEDDING_DIAGNOSTICS_ONLY=1 \
TRAIN_SCRIPT_PATH=scripts/official/base.py \
python3 olmoearth_pretrain/internal/all_evals.py \
    launch_evaluate my_run_embed_diag ai2/saturn-cirrascale
```

### Interpreting results

- **`effective_rank` dropping** → model is collapsing to fewer dimensions. Often happens with too-high learning rate or missing stop-gradient.
- **`cosine_sim_mean` near 1.0** → all embeddings point the same direction. Complete collapse.
- **`intra_cosine_sim_mean` near 1.0** → patches within images are identical. The model cannot distinguish spatial locations, so segmentation tasks will fail.

---

## Tiling Artifact Diagnostics

Tiling diagnostics detect spatial tiling and striping artifacts in encoder embeddings (see GitHub issue #499). These artifacts appear as periodic grid patterns when the model's spatial representation has systematic biases aligned with patch boundaries.

### What it measures

| Metric | Healthy value | Artifact signal |
|--------|--------------|-----------------|
| `row_col_var_ratio` | ~1.0 | Far from 1.0 → directional stripes (horizontal if >1, vertical if <1) |
| `fft_axis_energy_frac` | ~0.12 | > 0.25 → periodic grid artifacts |
| `fft_dominant_period_px` | — | Period of the strongest artifact in pixels |

Additionally, a **PCA RGB image** is logged to W&B, showing the first 3 PCA components of a sample's spatial embeddings as an RGB image. Healthy embeddings look like a smooth, spatially-varying color map. Tiling artifacts appear as a visible grid or stripe pattern.

### How it works

1. **Row/column variance ratio**: Averages embeddings along rows and columns separately, then compares their variances. Isotropic embeddings have a ratio near 1.0; directional stripes cause large deviations.
2. **FFT axis energy**: Projects all patch embeddings to their first PCA component, computes a 2D FFT per sample, and measures what fraction of spectral energy lies on the horizontal and vertical frequency axes (excluding DC and the k=1 gradient). High axis energy means periodic grid patterns exist.
3. **PCA RGB**: Fits PCA on a single sample's [H, W, D] embeddings and maps the first 3 components to RGB channels. Logged as a `wandb.Image`.

### Running during training (in-loop)

Tiling diagnostics are included in the default `build_trainer_config` in `scripts/official/script.py` for 64px and 128px spatial sizes:

```python
DownstreamTaskConfig(
    dataset="pretrain_subset_128",  # or pretrain_subset_64
    eval_mode=EvalMode.TILING_DIAGNOSTICS,
    embedding_batch_size=32,
    eval_interval=Duration.steps(20000),
    h5py_dir=H5PY_DIR,
    pretrain_max_samples=128,
    patch_size=4,
    input_modalities=[Modality.SENTINEL2_L2A.name],
)
```

Metrics appear in W&B under `eval_embed_diagnostics/tiling_64px/*` and `eval_embed_diagnostics/tiling_128px/*`.

### Running on saved checkpoints

Use `checkpoint_sweep_evals.py` with the `TILING_DIAGNOSTICS_ONLY` env var:

```bash
TILING_DIAGNOSTICS_ONLY=1 \
TRAIN_SCRIPT_PATH=scripts/official/base.py \
CHECKPOINT_DIR=/weka/.../checkpoints/my_run \
torchrun olmoearth_pretrain/internal/checkpoint_sweep_evals.py \
    evaluate my_run_tiling_diag local
```

Or launch on Beaker:

```bash
TILING_DIAGNOSTICS_ONLY=1 \
TRAIN_SCRIPT_PATH=scripts/official/base.py \
CHECKPOINT_DIR=/weka/.../checkpoints/my_run \
python3 olmoearth_pretrain/internal/checkpoint_sweep_evals.py \
    launch_evaluate my_run_tiling_diag ai2/saturn-cirrascale
```

### Interpreting results

- **`fft_axis_energy_frac` > 0.25**: Likely tiling artifacts. Check the PCA RGB image for visible grid lines.
- **`row_col_var_ratio` far from 1.0**: Directional striping. Values > 5 suggest horizontal stripes; values < 0.2 suggest vertical stripes.
- **`fft_dominant_period_px` matches patch size multiples**: The artifact period aligning with the patch size (e.g. 16px for patch_size=4 at 4-patch intervals) confirms the artifact comes from the patch embedding or positional encoding.
- **PCA RGB image shows grid lines**: Visual confirmation. Compare early vs. late checkpoints — artifacts that persist or worsen indicate a systematic architecture issue rather than an early-training transient.

### Relevant source files

- [`evals/embedding_diagnostics.py`](../olmoearth_pretrain/evals/embedding_diagnostics.py) — Metric computation (`compute_tiling_artifact_metrics`, `pca_rgb_image`)
- [`evals/datasets/configs.py`](../olmoearth_pretrain/evals/datasets/configs.py) — `pretrain_subset_64` / `pretrain_subset_128` dataset configs
- [`train/callbacks/evaluator_callback.py`](../olmoearth_pretrain/train/callbacks/evaluator_callback.py) — `_val_tiling_diagnostics()` callback method
- [`internal/all_evals.py`](../olmoearth_pretrain/internal/all_evals.py) — `TILING_DIAG_TASKS` and `EMBED_DIAG_TASKS` task registries

---

## Monitoring & Outputs

- **W&B logging:** Both scripts default to `EVAL_WANDB_PROJECT`. Override with `--project_name` or disable W&B via `--trainer.callbacks.wandb.enabled=False`.
- **Inspecting results:** Use [`scripts/tools/get_max_eval_metrics_from_wandb.py`](../scripts/tools/get_max_eval_metrics_from_wandb.py) to pull the best metric per task across runs.

---

## Helpful Files

- [`evals/models`](../olmoearth_pretrain/evals/models): Baseline models and their launchers.
- [`evals/eval_wrapper.py`](../olmoearth_pretrain/evals/eval_wrapper.py): Eval wrapper contract to be able to run evals on various models.
- [`evals/datasets`](../olmoearth_pretrain/evals/datasets/): Dataset loaders and shared dataset utils.
- [`evals/datasets/configs.py`](../olmoearth_pretrain/evals/datasets/configs.py): Dataset definitions (paths, splits, normalization) used to build commands.

---

## Adding New Eval Datasets (Internal)

> **AI2 internal only** — requires Weka access.

See **[Adding-Eval-Datasets.md](Adding-Eval-Datasets.md)** for the full step-by-step guide covering:
- Running the ingest pipeline on a new rslearn dataset
- What fields in `DownstreamTaskConfig` require user judgment
- Common errors and how to fix them
- What a PR adding a new dataset needs to include
