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
6. [Monitoring & Outputs](#monitoring--outputs)
7. [Helpful Files](#helpful-files)

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
python -m olmoearth_pretrain/internal/full_eval_sweep \
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

## Monitoring & Outputs

- **W&B logging:** Both scripts default to `EVAL_WANDB_PROJECT`. Override with `--project_name` or disable W&B via `--trainer.callbacks.wandb.enabled=False`.
- **Inspecting results:** Use [`scripts/get_max_eval_metrics_from_wandb.py`](../scripts/get_max_eval_metrics_from_wandb.py) to pull the best metric per task across runs.

---

## Helpful Files

- [`evals/models`](../olmoearth_pretrain/evals/models): Baseline models and their launchers.
- [`evals/eval_wrapper.py`](../olmoearth_pretrain/evals/eval_wrapper.py): Eval wrapper contract to be able to run evals on various models.
- [`evals/datasets`](../olmoearth_pretrain/evals/datasets/): Dataset loaders and shared dataset utils.
- [`evals/datasets/configs.py`](../olmoearth_pretrain/evals/datasets/configs.py): Dataset definitions (paths, splits, normalization) used to build commands.
