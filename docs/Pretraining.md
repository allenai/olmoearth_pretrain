# OlmoEarth Pretraining Guide

This guide walks you through setting up and running pretraining jobs for OlmoEarth.

---

## Choose Your Setup Path

> **🏢 AI2 Researchers (Internal):**
> You have access to Beaker, Weka, and AI2 infrastructure. See **[Setup-Internal.md](Setup-Internal.md)** for Beaker configuration, sessions, and internal workflows, then return here for training instructions.

> **🌍 External Users:**
> Continue reading below for local/cloud setup instructions.

---

## Table of Contents

1. [Environment Setup](#environment-setup) - External users start here
2. [Dataset Setup](#dataset-setup) - Required for external users
3. [Launching Scripts](#launching-scripts) - All users
4. [Overrides and Experiments](#overrides-and-experiments) - All users
5. [Gotchas and Troubleshooting](#gotchas-and-troubleshooting) - All users
6. [Additional Resources](#additional-resources)

---

## Environment Setup

> **AI2 Researchers:** You can skip this section. See [Setup-Internal.md](Setup-Internal.md) for Beaker-specific setup.

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (recommended: 40GB+ VRAM)
- Linux/macOS environment

### Installation

1. **Create a virtual environment:**
   ```bash
   python3.12 -m venv .venv-olmoearth_pretrain
   source .venv-olmoearth_pretrain/bin/activate
   ```

2. **Install the package:**
   ```bash
   cd /path/to/olmoearth_pretrain
   pip install -e '.[all]'
   ```

3. **Set up pre-commit hooks (optional but recommended):**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

### Required Environment Variables

#### 1. W&B API Key (for logging)

```bash
export WANDB_API_KEY="your_wandb_api_key_here"
```

**OR** disable W&B in your config:
```bash
--trainer.callbacks.wandb.enabled=False
```

See the [Reference Guide](Reference.md#troubleshooting-guide) for more details.

#### 2. Evaluation Dataset Paths (Optional)

If you want to run evaluations, override the default evaluation dataset paths:

```bash
export GEOBENCH_DIR="/your/path/to/geobench"
export CROPHARVEST_DIR="/your/path/to/cropharvest"
export BREIZHCROPS_DIR="/your/path/to/breizhcrops"
export MADOS_DIR="/your/path/to/mados"
export FLOODS_DIR="/your/path/to/floods"
export PASTIS_DIR="/your/path/to/pastis"
export SICKLE_DIR="/your/path/to/sickle"
export NANDI_DIR="/your/path/to/nandi"
export AWF_DIR="/your/path/to/awf"
```

You only need to set those you want to override; others will use their defaults (which point to AI2 internal paths).

---

## Dataset Setup

> **AI2 Researchers:** Training datasets are already available on Weka at `/weka/dfive-default/helios/dataset/`. See the [README](../README.md#olmoearth-pretrain-dataset) for specific paths. You can skip the rest of this section.

### Training Dataset Requirements

Your training data must be in **H5 format**. The dataset can be stored:
- **Locally:** `/path/to/h5data/num_samples`
- **Google Cloud Storage:** `gs://bucket_path/to/h5data/num_samples`
- **Other cloud storage:** Supported via appropriate filesystem libraries

### Creating an H5 Dataset

To create an H5 dataset from tiles:

```bash
python3 -m olmoearth_pretrain.internal.run_h5_conversion
```

See `olmoearth_pretrain/data/dataset.py` for the expected H5 schema and format details.

### Dataset Path Configuration

The dataset path **must be overridden** when launching scripts:

```bash
--dataset.h5py_dir=/your/path/to/h5data/num_samples
```

### Evaluation Datasets

Evaluation datasets have default paths set in [`olmoearth_pretrain/evals/datasets/paths.py`](../olmoearth_pretrain/evals/datasets/paths.py).

**For external users:** These defaults point to AI2 internal infrastructure. To use evaluations:

1. Download/prepare the evaluation datasets locally
2. Set environment variables (see [Environment Setup](#environment-setup))
3. Or disable evaluations you don't have:
   ```bash
   --trainer.callbacks.downstream_evaluator.tasks_to_run='["mados","pastis_sentinel2"]'
   ```

---

## Launching Scripts

### Main Training Script

The main training entry point is `scripts/official/base.py`, which is launched via `torchrun`.

#### Basic Command Structure

```bash
torchrun [TORCHRUN_OPTIONS] scripts/official/base.py train <RUN_NAME> <CLUSTER> [OVERRIDES...]
```

**Arguments:**
- `train`: The subcommand (always "train" for training)
- `<RUN_NAME>`: A descriptive name for your run (e.g., `debug_pretrain`)
- `<CLUSTER>`: Cluster identifier
  - **External users:** Use `local`
  - **AI2 researchers:** Use `local` for sessions or see [Setup-Internal.md](Setup-Internal.md) for Beaker clusters
- `[OVERRIDES...]`: CLI overrides for config parameters (see [Overrides](#overrides-and-experiments))

#### Example: Local Single-GPU Training

```bash
torchrun \
  --nproc_per_node=1 \
  scripts/official/base.py train debug_pretrain local \
  --dataset.h5py_dir=/path/to/your/h5data/num_samples \
  --data_loader.num_workers=4 \
  --train_module.rank_microbatch_size=16 \
  --data_loader.global_batch_size=64
```

#### Example: Multi-GPU Training

```bash
torchrun \
  --nproc_per_node=8 \
  --nnodes=1 \
  scripts/official/base.py train multi_gpu_run local \
  --dataset.h5py_dir=/path/to/your/h5data/num_samples \
  --data_loader.num_workers=4 \
  --data_loader.global_batch_size=512
```

**Note:** For cloud storage (GCS):
```bash
--dataset.h5py_dir=gs://your-bucket/path/to/h5data/num_samples
```

### Launching Ablations

Ablation scripts are located in dated directories under `scripts/`. To run an ablation:

1. Navigate to the appropriate script directory (e.g., `scripts/2025_10_02_phase2/`)
2. Identify the ablation script (e.g., `train_cross_random.py`)
3. Launch with `torchrun`:
   ```bash
   torchrun --nproc_per_node=8 scripts/2025_10_02_phase2/your_ablation.py train ablation_name local [OVERRIDES]
   ```

**Tip:** Most ablation scripts follow the same pattern as `base.py` but with different default configurations.

---

## Overrides and Experiments

### How Overrides Work

The experiment framework uses a builder pattern with override capabilities. You can override any configuration parameter via CLI arguments using dotted notation.

### Common Overrides

#### Dataset Configuration

```bash
--dataset.h5py_dir=/path/to/h5py/data
--dataset.training_modalities='["SENTINEL2_L2A","SENTINEL1","LANDSAT"]'
```

#### Data Loader Configuration

```bash
--data_loader.num_workers=16
--data_loader.global_batch_size=512
--data_loader.token_budget=2250
--data_loader.prefetch_factor=4
--data_loader.seed=3622
```

#### Training Module Configuration

```bash
--train_module.rank_microbatch_size=32
--train_module.max_grad_norm=1.0
--train_module.optim_config.lr=0.0001
--train_module.optim_config.weight_decay=0.02
--train_module.scheduler.warmup_steps=8000
```

#### Trainer Configuration

```bash
--trainer.max_duration.epochs=300
--trainer.metrics_collect_interval=10
--trainer.save_interval=5000
```

#### Model Configuration

Override model architecture (requires understanding the model config structure):

```bash
--model.encoder_config.depth=24
--model.encoder_config.num_heads=16
--model.encoder_config.embedding_size=768
--model.decoder_config.depth=8
```

### Example: Full Custom Run

```bash
torchrun --nproc_per_node=8 scripts/official/base.py train custom_experiment local \
  --dataset.h5py_dir=/your/path/to/data \
  --data_loader.global_batch_size=256 \
  --data_loader.num_workers=8 \
  --train_module.rank_microbatch_size=8 \
  --train_module.optim_config.lr=0.0002 \
  --train_module.scheduler.warmup_steps=5000 \
  --trainer.max_duration.epochs=100
```

For more override patterns and examples, see the [Reference Guide](Reference.md#override-patterns).

---

## Gotchas and Troubleshooting

### 1. Dataset Path Must Be Overridden

**Problem:** The default dataset path in `scripts/official/script.py` points to a specific path that may not exist in your environment.

**Solution:** Always override `--dataset.h5py_dir` when launching:
```bash
--dataset.h5py_dir=/your/actual/dataset/path
```

### 2. Some Evaluations Don't Support Remote Filesystems

**Problem:** Certain evaluation datasets (e.g., GeoBench) may not work with GCS paths.

**Solution:**
- Copy evaluation datasets to local storage
- Override the evaluation dataset paths via environment variables
- Or disable problematic evaluations:
  ```bash
  --trainer.callbacks.downstream_evaluator.tasks_to_run='["mados","pastis_sentinel2"]'
  ```

### 3. W&B Errors

**Problem:** Training fails with W&B authentication errors.

**Solution:** Either:
- Set `WANDB_API_KEY` environment variable
- OR disable W&B in the config:
  ```bash
  --trainer.callbacks.wandb.enabled=False
  ```

### 4. Out of Memory (OOM) Errors

**Problem:** GPU runs out of memory during training.

**Solution:**
- Reduce microbatch size: `--train_module.rank_microbatch_size=8`
- Reduce global batch size: `--data_loader.global_batch_size=128`
- Reduce token budget: `--data_loader.token_budget=1500`

### 5. Slow Data Loading

**Problem:** Training is bottlenecked by data loading.

**Solution:**
- Increase workers: `--data_loader.num_workers=16`
- Increase prefetch: `--data_loader.prefetch_factor=8`
- Use faster storage (local SSD > network storage > cloud storage)
- Consider using 128x128 tile datasets (better for some bottleneck scenarios)

### 6. Multi-Node Training

**Problem:** Running multi-node distributed training requires additional configuration.

**Solution:** See the [olmo-core documentation](https://github.com/allenai/olmo-core) for multi-node torchrun setup with proper `--nnodes`, `--node_rank`, and `--master_addr` parameters.

For more troubleshooting, see the [Reference Guide](Reference.md#troubleshooting-guide).

---

## Additional Resources

### Documentation

- **[Setup-Internal.md](Setup-Internal.md)** - AI2 researchers: Beaker, sessions, internal infrastructure
- **[Reference.md](Reference.md)** - Deep configuration reference, troubleshooting, helpful files
- **[README.md](../README.md)** - Project overview, datasets, evaluation suite

### Key Files

- `scripts/official/base.py` - Main entry point and model config
- `scripts/official/script.py` - Component builders (dataset, dataloader, trainer)
- `olmoearth_pretrain/evals/datasets/paths.py` - Evaluation dataset paths
- `olmoearth_pretrain/data/dataset.py` - Dataset implementation

### External Resources

- [olmo-core documentation](https://github.com/allenai/olmo-core) - Underlying training framework
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html) - Multi-GPU/multi-node training

---

## Quick Reference

### Minimal Working Example

```bash
# 1. Set up environment
export WANDB_API_KEY="your_key"  # or use --trainer.callbacks.wandb.enabled=False

# 2. Launch training
torchrun \
  --nproc_per_node=1 \
  scripts/official/base.py train test_run local \
  --dataset.h5py_dir=/your/path/to/h5data/num_samples \
  --data_loader.global_batch_size=64 \
  --train_module.rank_microbatch_size=16
```

### Getting Help

- **Open an issue:** [GitHub Issues](https://github.com/allenai/olmoearth_pretrain/issues)
- **Check documentation:** See [Reference.md](Reference.md) for detailed troubleshooting
- **AI2 researchers:** See internal Slack channels or [Setup-Internal.md](Setup-Internal.md)
