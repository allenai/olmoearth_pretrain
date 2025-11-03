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
2. [Assets & Paths](#assets--paths)
3. [Quick Start](#quick-start)
4. [KNN / Linear Probing](#knn--linear-probing)
5. [Finetune](#finetune-sweep)
6. [Monitoring & Outputs](#monitoring--outputs)
7. [Helpful Files](#helpful-files)

---

## Evaluation Overview

We run evaluations through the same `olmoearth_pretrain/internal/experiment.py` entrypoint used for pretraining. The helper scripts below build the underlying launch commands:

- `olmoearth_pretrain/internal/full_eval_sweep.py` runs KNN (classification) and linear probing (segmentation) sweeps for OlmoEarth checkpoints or baseline models, with optional sweeps over learning rate, pretrained/dataset normalizers, and pooling (mean or max).
- `olmoearth_pretrain/internal/full_eval_sweep_finetune.py` runs fine-tuning sweeps for OlmoEarth checkpoints or baseline models, with optional sweeps over learning rate, and pretrained/dataset normalizers.

Both scripts rely on:
- [`olmoearth_pretrain/internal/all_evals.py`](../olmoearth_pretrain/internal/all_evals.py) for the task registry (`EVAL_TASKS` for KNN and learning probing, and `FT_EVAL_TASKS` for fine-tuning).
- [`olmoearth_pretrain/evals`](../olmoearth_pretrain/evals) for dataset/model wrappers.

Every launch uses the evaluation subcommands from `experiment.py`:
- `dry_run_evaluate` prints the config (no execution) for quick checks.
- `evaluate` runs evaluation job locally.
- `launch_evaluate` submits evaluation job to Beaker.

The sweep scripts set `TRAIN_SCRIPT_PATH` automatically and choose `torchrun` for local runs and `python3` for Beaker jobs.

### Prerequisites

- Python environment configured as described in [Pretraining.md](Pretraining.md#environment-setup).
- One 80 GB GPU (A100 or H100 recommended). If you see OOM errors, lower the batch size can help.
- W&B API key (`WANDB_API_KEY`) if you want metrics to stream automatically.

### Supported Models

- **OlmoEarth models:** Nano, Tiny, Base, and Large size.
- **Others:** `dino_v3`, `panopticon`, `galileo`, `satlas`, `croma`, `copernicusfm`, `presto`, `anysat`, `tessera`, `prithvi_v2`, `terramind`, `clay`. Multi-size variants (e.g. `croma_large`, `galileo_large`, `terramind_large`) are handled automatically by the sweep scripts when requested.

---

## Assets & Paths

- **Evaluation datasets**
  - *Internal*: All datasets live on Weka; the defaults in [`evals/datasets/paths.py`](../olmoearth_pretrain/evals/datasets/paths.py) point to shared mounts.
  - *External*: Download from `gs://ai2-olmoearth-projects-public-data/research_benchmarks` (e.g. with `gsutil -m rsync`). Update the same `paths.py` file or override the environment variables (`GEOBENCH_DIR`, `CROPHARVEST_DIR`, etc.) so the loaders can resolve local copies.
- **OlmoEarth checkpoints**
  - Clone the release repos from Hugging Face, e.g.:
    ```bash
    git clone git@hf.co:allenai/OlmoEarth-v1-Nano
    git clone git@hf.co:allenai/OlmoEarth-v1-Tiny
    git clone git@hf.co:allenai/OlmoEarth-v1-Base
    git clone git@hf.co:allenai/OlmoEarth-v1-Large
    ```
  - Pass the desired checkpoint directory via `--checkpoint_path` when invoking the sweeps (or set it inside your finetune/LP launch scripts).
- **Baselines**: When using `--model=<name>`, the sweeps download/load checkpoints through the model wrappers; no extra setup required beyond dataset paths.

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
  --checkpoint_path=~/Downloads/OlmoEarth-v1-Base \
  --defaults_only \
  --dry_run
```

This prints the exact command that would run inside `experiment.py dry_run_evaluate`.

### 3. Launch for real

Remove `--dry_run` once the command looks correct. The helper picks the right subcommand for you:

- **Local GPUs (`--cluster=local`)**

  ```bash
  python -m olmoearth_pretrain.internal.full_eval_sweep \
    --cluster=local \
    --checkpoint_path=~/Downloads/OlmoEarth-v1-Base \
    --module_path=scripts/2025_10_02_phase2/base.py \
    --defaults_only
  ```

- **Beaker (`--cluster=<ai2 cluster>`)**

  ```bash
  python -m olmoearth_pretrain.internal.full_eval_sweep \
    --cluster=ai2/saturn-cirrascale \
    --model=dino_v3 \
    --project_name=2025_10_eval_comparison \
    --lr_only
  ```

---

## KNN / Linear Probing

Use this script for KNN and linear probing evaluations. Invoke it either through `python -m olmoearth_pretrain.internal.full_eval_sweep` or by running the file directly.

### Required flags

- `--cluster`: Cluster identifier (`local` for on-box runs).
- Exactly one of:
  - `--checkpoint_path=~/Downloads/OlmoEarth-v1-Base`: Evaluate an OlmoEarth checkpoint.
  - `--model=<baseline_name>` or `--model=all`: Evaluate published baseline models defined in [`evals/models`](../olmoearth_pretrain/evals/models).

### Common optional flags

- `--module_path`: Override the launch module (defaults to the model-specific launcher).
- `--project_name`: W&B project (defaults to `EVAL_WANDB_PROJECT`).
- `--defaults_only`: Run a single command using the default lr / normalization / pooling.
- `--lr_only`: Sweep learning rates but keep normalization + pooling at defaults.
- `--all_sizes` or `--size=<variant>`: Evaluate every published size for multi-size baselines.
- `--model-skip-names=a,b`: Skip a subset when using `--model=all`.
- `--select_best_val`: Uses validation MIoU to pick the best epoch before reporting test metrics.
- `--dry_run`: Print commands without launching (`dry_run_evaluate`).
- Extra CLI arguments (e.g. `--trainer.max_duration.unit=epochs`) are forwarded to the underlying train module.

When `--model=all`, the script automatically switches to the correct launcher for each model and constructs run names like `<checkpoint>_lr1e-3_norm_dataset_pool_mean`.

---

## Finetune Sweep

Use `olmoearth_pretrain/internal/full_eval_sweep_finetune.py` for downstream fine-tuning tasks. It shares many flags with the KNN and linear probing sweep but adds fine-tune–specific knobs.

### Required flags

- `--cluster`: Cluster identifier.
- One of:
  - `--checkpoint_path=~/Downloads/OlmoEarth-v1-Base`: Fine-tune an OlmoEarth checkpoint.
  - `--model=<preset_key>`: Use a baseline preset (choices listed in the script’s help).

### Fine-tune specific flags

- `--defaults_only`: Run only the first learning rate in `FT_LRS`.
- `--module_path`: Override the launch script (defaults to the preset’s launcher).
- `--use_dataset_normalizer`: Force dataset statistics even when a preset ships its own pretrained normalizer. Leave unset to keep the pretrained normalizer.
- `--finetune_seed`: Apply a single base seed across every downstream task (otherwise each task chooses its own default).
- Extra CLI arguments append to every command (e.g. `--trainer.max_duration.value=50000`).
- `--dry_run`: Preview commands (`dry_run_evaluate`).

The script sets `FINETUNE=1` in the environment before launching so downstream code enables fine-tuning heads automatically.

Launch behavior mirrors the linear-probe sweep: `--cluster=local` runs `evaluate`, any other cluster uses `launch_evaluate`.

### Example: Local OlmoEarth sanity check

```bash
python -m olmoearth_pretrain.internal.full_eval_sweep_finetune \
  --cluster=local \
  --checkpoint_path=~/Downloads/OlmoEarth-v1-Base \
  --module_path=scripts/2025_10_02_phase2/base.py \
  --project_name=2025_11_15_local_sanity \
  --defaults_only \
  --dry_run
```

### Example: Beaker run with seeding

```bash
python -m olmoearth_pretrain.internal.full_eval_sweep_finetune \
  --cluster=ai2/titan \
  --checkpoint_path=~/Downloads/OlmoEarth-v1-Base \
  --module_path=scripts/2025_10_02_phase2/base.py \
  --project_name=2025_11_15_phase2_finetune \
  --finetune_seed=42
```

### Example: Terramind preset with dataset stats

```bash
python -m olmoearth_pretrain.internal.full_eval_sweep_finetune \
  --cluster=ai2/saturn-cirrascale \
  --model=terramind \
  --project_name=2025_11_15_baseline_comparison \
  --use_dataset_normalizer \
  --defaults_only
```

For a local sanity check, add `--cluster=local --dry_run` first, then drop `--dry_run` to execute on your workstation.

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
