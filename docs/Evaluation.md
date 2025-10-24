# OlmoEarth Evaluation Guide

This guide explains how we launch evaluation sweeps for OlmoEarth checkpoints and baseline models, including KNN, linear probing, and finetuning jobs.

---

## Choose Your Evaluation Path

> **🏢 AI2 Researchers (Internal):**
> You have access to Beaker/Weka clusters and shared checkpoints. Skim [Setup-Internal.md](Setup-Internal.md) for environment details, then follow the launch instructions below.

> **🌍 External Users:**
> You can run a reduced version of these workflows on local/cloud GPUs. You will need the datasets referenced in [Dataset Setup](Pretraining.md#dataset-setup).

---

## Table of Contents

1. [Evaluation Overview](#evaluation-overview)
2. [Quick Start](#quick-start)
3. [KNN / Linear Probing](#knn--linear-probing)
4. [Finetune Sweep](#finetune-sweep)
5. [Monitoring & Outputs](#monitoring--outputs)
6. [Helpful Files](#helpful-files)

---

## Evaluation Overview

We run evaluations through the same `experiment.py` entrypoint used for training. The helper scripts below build the underlying launch commands and fan out the learning-rate, normalization, and pooling sweeps we use in papers.

- `olmoearth_pretrain/internal/full_eval_sweep.py` launches frozen-feature evaluations (KNN, linear probes, zero-shot) against a checkpoint or a supported baseline model.
- `olmoearth_pretrain/internal/full_eval_sweep_finetune.py` launches downstream fine-tuning evaluations, including optional sweeps over pretrained normalizers.

Both scripts rely on:
- [`olmoearth_pretrain/internal/all_evals.py`](../olmoearth_pretrain/internal/all_evals.py) for the task registry.
- [`olmoearth_pretrain/evals`](../olmoearth_pretrain/evals) for dataset/model wrappers.
- `EVAL_LAUNCH_PATH` in [`internal/constants.py`](../olmoearth_pretrain/internal/constants.py) to point at the shared evaluation launcher module.

### Prerequisites

- Python environment configured as described in [Pretraining.md](Pretraining.md#environment-setup).
- Access to evaluation datasets (see [`evals/datasets/paths.py`](../olmoearth_pretrain/evals/datasets/paths.py) for expected locations).
- W&B API key (`WANDB_API_KEY`) if you want metrics to stream automatically.
- For AI2 infra: valid Beaker cluster name (`ai2/saturn`, `ai2/titan`, etc.).

---

## Quick Start

### 1. Activate your environment

```bash
source .venv-olmoearth_pretrain/bin/activate
```

### 2. Run a dry run to inspect the planned commands

```bash
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --cluster=local \
  --checkpoint_path=/path/to/checkpoint/step123000 \
  --defaults_only \
  --dry_run
```

This prints the exact `torchrun`/`python3` command that will be executed for each task and hyperparameter combination.

### 3. Launch for real

Remove `--dry_run` once the command looks correct. On local GPUs the helper scripts will call `torchrun`; on Beaker they call `python3` with the launch module defined by `EVAL_LAUNCH_PATH`.

---

## KNN / Linear Probing

Use this script for frozen-feature evaluations (KNN, linear probes, zero-shot). Invoke it either through `python -m olmoearth_pretrain.internal.full_eval_sweep` or by running the file directly.

### Required flags

- `--cluster`: Cluster identifier (`local` for on-box runs).
- Exactly one of:
  - `--checkpoint_path=/path/to/checkpoint/stepXXXX`: Evaluate an OlmoEarth checkpoint.
  - `--model=<baseline_name>` or `--model=all`: Evaluate published baseline models defined in [`evals/models`](../olmoearth_pretrain/evals/models).

### Common optional flags

- `--module_path`: Override the launch module (defaults to the model-specific launcher).
- `--project_name`: W&B project (defaults to `EVAL_WANDB_PROJECT`).
- `--defaults_only`: Run a single command using the default lr / normalization / pooling.
- `--lr_only`: Sweep learning rates but keep normalization + pooling at defaults.
- `--all_sizes` or `--size=<variant>`: Evaluate every published size for multi-size baselines.
- `--model-skip-names=a,b`: Skip a subset when using `--model=all`.
- `--select_best_val`: Uses validation MIoU to pick the best epoch before reporting test metrics.
- `--dry_run`: Print commands without launching.
- Extra CLI arguments (e.g. `--trainer.max_duration.unit=epochs`) are forwarded to the underlying train module.

### Example: Frozen evaluation against a checkpoint (local debug)

```bash
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --cluster=local \
  --checkpoint_path=/data/checkpoints/phase2_base/step667200 \
  --module_path=scripts/2025_10_02_phase2/base.py \
  --defaults_only
```

### Example: Launch baseline sweep on Beaker

```bash
python -m olmoearth_pretrain.internal.full_eval_sweep \
  --cluster=ai2/saturn-cirrascale \
  --model=dino_v3 \
  --project_name=2025_10_eval_comparison \
  --lr_only
```

When `--model=all`, the script automatically switches to the correct launcher for each model and constructs run names like `<checkpoint>_lr1e-3_norm_dataset_pool_mean`.

---

## Finetune Sweep

Use `olmoearth_pretrain/internal/full_eval_sweep_finetune.py` for downstream fine-tuning tasks. It shares many flags with the frozen sweep but adds fine-tune–specific knobs.

### Required flags

- `--cluster`: Cluster identifier.
- One of:
  - `--checkpoint_path=/path/to/olmoearth/stepXXXX`: Fine-tune an OlmoEarth checkpoint.
  - `--model=<preset_key>`: Use a baseline preset (choices listed in the script’s help).

### Fine-tune specific flags

- `--defaults_only`: Run only the first learning rate in `FT_LRS`.
- `--sweep_normalizer`: For models with pretrained normalizers, run both dataset stats and pretrained normalizer variants.
- `--module_path`: Override the launch script (defaults to the preset’s launcher).
- Extra CLI arguments append to every command (e.g. `--trainer.max_duration.value=50000`).
- `--dry_run`: Preview commands.

The script sets `FINETUNE=1` in the environment before launching so downstream code enables fine-tuning heads automatically.

### Example: Checkpoint fine-tune sweep (Beaker)

```bash
python -m olmoearth_pretrain.internal.full_eval_sweep_finetune \
  --cluster=ai2/titan \
  --checkpoint_path=/weka/.../phase2.0_base_lr0.0001_wd0.02/step667200 \
  --project_name=2025_10_08_phase2_finetune \
  --defaults_only
```

### Example: Baseline fine-tune with normalizer sweep

```bash
python -m olmoearth_pretrain.internal.full_eval_sweep_finetune \
  --cluster=ai2/saturn-cirrascale \
  --model=galileo \
  --sweep_normalizer
```

---

## Monitoring & Outputs

- **W&B logging:** Both scripts default to `EVAL_WANDB_PROJECT`. Override with `--project_name` or disable W&B via `--trainer.callbacks.wandb.enabled=False`.
- **Checkpoints:** Evaluation launches set `--trainer.no_checkpoints=True` for baseline models so runs do not write new checkpoints. OlmoEarth checkpoints keep checkpointing enabled by default.
- **Run names:** Generated from the checkpoint directory (`<run>/<step>`) or baseline name plus the swept hyperparameters to simplify aggregation.
- **Inspecting results:** Use [`scripts/get_max_eval_metrics_from_wandb.py`](../scripts/get_max_eval_metrics_from_wandb.py) to pull the best MIoU/accuracy per task across runs.
- **Dry run safety:** Always start with `--dry_run` when editing sweeps or passing overrides—command strings can be long and the dry run verifies the generated arguments.

---

## Helpful Files

- [`internal/all_evals.py`](../olmoearth_pretrain/internal/all_evals.py): Lists frozen and fine-tune tasks, feature extractor settings, and metric names.
- [`evals/models`](../olmoearth_pretrain/evals/models): Launcher modules and wrappers for baseline models.
- [`evals/datasets/configs.py`](../olmoearth_pretrain/evals/datasets/configs.py): Dataset configs used when constructing evaluation commands.
- [`docs/Pretraining.md`](Pretraining.md): Shared environment setup; refer back if you need to rebuild Docker images or install dependencies.

Happy evaluating! Let the team know in `#olmoearth` if new baselines or tasks need presets added to the sweep scripts.
