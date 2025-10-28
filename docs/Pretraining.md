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
2. [Launching Scripts](#launching-scripts) - All users
3. [Dataset Setup](#dataset-setup) - Required for external users
4. [Experiment Tracking](#experiment-tracking) - All users
5. [Official Training Scripts](#official-training-scripts) - All users
6. [Ablations](#ablations) - All users
7. [Overrides and Experiments](#overrides-and-experiments) - All users
8. [Helpful Files for Understanding](#helpful-files-for-understanding) - All users

---

## Environment Setup

> **Note for Ai2 Researchers:** You can skip this section. See [Setup-Internal.md](Setup-Internal.md) for Beaker-specific setup.

### Python Environment Setup

**Prerequisites:** Python 3.11 or higher (Python 3.12 recommended)

Follow the setup instructions in the [main README](../README.md#general-setup) to install dependencies using `uv`.

Once setup is complete, activate the virtual environment before running training scripts:
```bash
source .venv/bin/activate
```

### Running on Docker

We run our training scripts using the `olmo-core-tch271cu128-2025-09-15` Docker image published by [ai2-olmo-core](https://github.com/allenai/OLMo-core/blob/main/README.md).

**Important Notes:**
- The code from this repository is **not included** in the Docker image to aid in active development. The code is mounted or copied at runtime.
- This Docker image may not work on your own cluster if you have different hardware or driver/CUDA versions. The image is built for CUDA 12.8 with PyTorch 2.7.1.
- **For adaptation:** See the Dockerfile in [ai2-olmo-core](https://github.com/allenai/OLMo-core/blob/main/README) to understand how to build an image compatible with your hardware and CUDA setup

## Launching Scripts

### Basic Command Structure

All training scripts configure builder functions used in the main entrypoint: [`main` in `olmoearth_pretrain/internal/experiment.py`](../olmoearth_pretrain/internal/experiment.py#L364).

**General Usage:**
```bash
python scripts/official/<SCRIPT>.py <SUBCOMMAND> <RUN_NAME> <CLUSTER> [OVERRIDES...]
```

For multi-gpu training, use `torchrun`:
```bash
torchrun [TORCHRUN_OPTIONS] scripts/official/<SCRIPT>.py <SUBCOMMAND> <RUN_NAME> <CLUSTER> [OVERRIDES...]
```

#### Available Subcommands

The most commonly used subcommands are:
- **`train`**: Run distributed training (use with `torchrun` for multi-GPU)
- **`launch`**: Submit a Beaker job (AI2 researchers only)
- **`dry_run`**: Validate configuration without running training

For a complete list of subcommands and their usage, see the [`SubCmd` enum](../olmoearth_pretrain/internal/experiment.py#L280-L294) in the source code, or run any script without arguments to see the help message:
```bash
python scripts/official/base.py
```

#### Command Structure Examples

**Example 1: Single-GPU Training for Debugging**
```bash
torchrun scripts/official/nano.py train my_debug_run local \
  --dataset.h5py_dir=/path/to/data \
  --data_loader.global_batch_size=64
```

**Example 2: Multi-GPU Training with torchrun**
```bash
torchrun --nproc_per_node=4 scripts/official/base.py train my_pretrain_run local \
  --dataset.h5py_dir=/path/to/data \
  --data_loader.global_batch_size=512
```

**Example 3: Validate Configuration Without Training**
```bash
python scripts/official/base.py dry_run my_config_test local \
  --trainer.max_duration.value=100
```

**Example 4: Submitting a Beaker Job (AI2 Only)**
```bash
python scripts/official/large.py launch my_large_model ai2/saturn \
  --trainer.max_duration.value=300
```

## Dataset Setup

> **Note for AI2 Researchers:** Training datasets are already available on Weka at `/weka/dfive-default/helios/dataset/`. See the [README](../README.md#olmoearth-pretrain-dataset) for specific paths. You can skip the rest of this section.

### Training Dataset Setup

The pretraining dataset can be found here: https://huggingface.co/datasets/allenai/olmoearth_pretrain_dataset . \

1. Download dataset `hf downloadallenai/olmoearth_pretrain_dataset --repo-type dataset --local-dir /path/to/save/location --include  "h5py_data/*" `
2. Extract dataset from tar files
  ```
  export H5_DIR=/path/to/extraction/location/
  export TAR_DIR=/path/to/save/location/

  mkdir -p "$H5_DIR"
  cd "$TAR_DIR"

  # run one tar per core; tune -P if disk gets saturated
  find . -maxdepth 1 -name '*.tar' -print0 \
  | xargs -0 -n1 -P"$(nproc)" -I{} \
    tar -xf "{}" -C "$H5_DIR" \
      --no-same-owner --numeric-owner --delay-directory-restore

  echo "Dataset downloaded at ${H5_DIR}h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/1138828"
  ```


### Dataset Path Configuration

External users must specify the dataset path when launching training scripts:

```bash
--dataset.h5py_dir=/your/path/to/h5data/num_samples
```

### Evaluation Datasets

Evaluation datasets have default paths set in [`olmoearth_pretrain/evals/datasets/paths.py`](../olmoearth_pretrain/evals/datasets/paths.py).

**For external users:** These defaults point to AI2 internal infrastructure. To use evaluations:

1. Download/prepare the evaluation datasets locally (TODO: Add instructions once on HF)
2. Set environment variables for each dataset path to override defaults in [`olmoearth_pretrain/evals/datasets/paths.py`](../olmoearth_pretrain/evals/datasets/paths.py)

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
3. If you wish to only use a subset of the evaluations, add the following override:

   For example, to only run mados and pastis_sentinel2 evals add the following override:
   ```bash
   --trainer.callbacks.downstream_evaluator.tasks_to_run=\[mados,pastis_sentinel2\]
   ```
   The task names correspond to the user-chosen names specified in the training configuration
4. If you do not want to run **any** evaluations during training, add the following override to your command:
   ```bash
   --trainer.callbacks.downstream_evaluator.enabled=False
   ```

---

## Experiment Tracking

#### 1. W&B API Key (For Logging)

```bash
export WANDB_API_KEY="your_wandb_api_key_here"
```

Alternatively, you can disable W&B logging in your configuration:
```bash
--trainer.callbacks.wandb.enabled=False
```

## Official Training Scripts

> **🏢 AI2 Researchers - Choose Your Launch Method:**
>
> **For Beaker Batch Jobs (Pre-emptible):**
> Use `python3` with the `launch` subcommand:
> ```bash
> python3 scripts/official/base.py launch base_run ai2/saturn --launch.num_gpus=8
> ```
> Replace `ai2/saturn` with your target cluster.
> To launch on multiple clusters, specify any cluster and append the following override to your command:
> ```bash
> --launch.clusters=\[ai2/saturn,ai2/jupiter\]
> ```
> **⚠️ Remember:** Commit and push your code before launching and give your run a memorable name!
>
> **For Beaker Sessions (Interactive Debugging):**
> Use the `torchrun` commands shown in the table below, just like external users:
> ```bash
> torchrun --nproc_per_node=8 scripts/official/base.py train base_run local
> ```
>
> See [Setup-Internal.md](Setup-Internal.md#launch-methods) for more details.

All Official release scripts can be found at [`scripts/official/`](../scripts/official/).
Below is a table demonstrating how to launch various model sizes using `torchrun` (for external users and AI2 sessions). Adjust the dataset path and configuration overrides as needed for your setup.

| Model Size | Script | Hardware | Example Command |
|------------|--------|----------|-----------------|
| **Nano** | [`scripts/official/nano.py`](../scripts/official/nano.py) | 4x GPUs (16GB+ VRAM each) | `torchrun --nproc_per_node=4 scripts/official/nano.py train nano_run local` |
| **Tiny** | [`scripts/official/tiny.py`](../scripts/official/tiny.py) | 4-8x GPUs (24GB+ VRAM each) | `torchrun --nproc_per_node=4 scripts/official/tiny.py train tiny_run local` |
| **Base** | [`scripts/official/base.py`](../scripts/official/base.py) | 8x GPUs (40GB+ VRAM each) | `torchrun --nproc_per_node=8 scripts/official/base.py train base_run local` |
| **Large** | [`scripts/official/large.py`](../scripts/official/large.py) | 8x GPUs (80GB VRAM each) | `torchrun --nproc_per_node=8 scripts/official/large.py train large_run local` |

> **⚠️ Hardware Adaptation Note:**
> You may need to adapt parameters depending on your available hardware:
> - **Limited VRAM:** Reduce `--data_loader.global_batch_size` and/or `--train_module.rank_microbatch_size`
> - **Fewer GPUs:** Adjust `--nproc_per_node` and scale batch size accordingly
> - **Different GPU types:** Monitor memory usage and adjust batch sizes to avoid OOM errors
> - **CPU constraints:** Adjust `--data_loader.num_workers` based on available CPU cores

**External Users - Specifying Dataset Path:**

External users must specify the path to their HDF5 dataset directory by adding `--dataset.h5py_dir=/path/to/h5py_dir` to any command above.

**Example:**
```bash
torchrun --nproc_per_node=4 scripts/official/nano.py train nano_run local \
  --dataset.h5py_dir=/path/to/h5py_dir
```

**Example with Base model:**
```bash
torchrun --nproc_per_node=8 scripts/official/base.py train base_run local \
  --dataset.h5py_dir=/path/to/your/h5data/num_samples
```

> **💾 Checkpoint Saving Note:**
> When using `local` as the cluster argument, checkpoints are automatically saved to `./local_output`. You can override this location with `--common.save_folder=path/to/savefolder`.

## Ablations

Ablation studies isolate the impact of specific components in the base model configuration. All ablations can be launched similarly to the official training scripts (see the table in [Official Training Scripts](#official-training-scripts)).

### Available Ablations

The bash script [`scripts/official/ablations/base_launch_ablations.sh`](../scripts/official/ablations/base_launch_ablations.sh) contains all base model ablations. These are grouped into two categories:

#### Loss & Training Strategy Ablations

These ablations modify the training objective or strategy:

- **No contrastive loss** - Disables the contrastive loss component (sets weight to 0.0)
- **Random masking** - Uses random masking instead of structured masking strategy
- **MAE (Masked Autoencoder)** - Switches to a pure MAE training approach
- **Random target init** - Reinitializes target projections randomly instead of using pretrained weights
- **Original patch disc loss** - Uses the legacy patch discrimination loss implementation
- **EMA active** - Re-enables exponential moving average for target encoder

#### Modality Ablations

These ablations progressively remove modalities to measure their contribution:

- **No ag maps** - Removes agricultural map modalities (WorldCereal, CDL)
- **No maps** - Removes all map modalities (ag maps + WorldCover, OpenStreetMap, canopy height)
- **No decode modalities** - Removes all decode-only modalities (maps + SRTM)
- **No Landsat** - Removes Landsat imagery
- **S2 only** - Sentinel-2 only (removes Sentinel-1 as well)

### Running Ablations

**For AI2 Researchers:** Launch all ablations at once using the bash script:
```bash
bash scripts/official/ablations/base_launch_ablations.sh
```

**For External Users or Individual Ablations:** Run specific ablations with `torchrun`:

Example - No contrastive loss ablation:
```bash
torchrun --nproc_per_node=8 scripts/official/base.py train base_no_contrastive local \
  --train_module.contrastive_config.loss_config.weight=0.0 \
  --dataset.h5py_dir=/path/to/data
```

Example - S2-only modality ablation:
```bash
torchrun --nproc_per_node=8 scripts/official/base.py train base_s2_only local \
  --common.training_modalities='[sentinel2_l2a]' \
  --train_module.masking_config.strategy_config.only_decode_modalities='[]' \
  --dataset.h5py_dir=/path/to/data
```

> **💡 Tip:** Check [`scripts/official/ablations/base_launch_ablations.sh`](../scripts/official/ablations/base_launch_ablations.sh) for the exact override parameters used in each ablation.

---

## Overrides and Experiments

### How Overrides Work

The experiment framework uses a builder pattern with override capabilities. Launch scripts can be edited to change the configuration or you can override any configuration parameter via CLI arguments using dotted notation.

### Example: Custom Training Run with Multiple Overrides

```bash
torchrun --nproc_per_node=8 scripts/official/base.py train custom_experiment local \
  --data_loader.global_batch_size=256 \
  --data_loader.num_workers=8 \
  --train_module.rank_microbatch_size=8 \
  --train_module.optim_config.lr=0.0002 \
  --train_module.scheduler.warmup_steps=5000 \
  --trainer.max_duration.epochs=100
  # Optionally --dataset.h5py_dir=/your/path/to/data \
```


## Dataset Directory and File Structure

The H5 dataset follows a hierarchical directory structure (see [`set_h5py_dir` in convert_to_h5py.py](../olmoearth_pretrain/dataset/convert_to_h5py.py)):

```
<tile_path>/
  h5py_data_w_missing_timesteps[_compression_settings][_tilesize_x_numsubtiles]/
    <sorted_modality_names>[_required_<required_mods>]/
      <num_samples>/
        sample_0.h5
        sample_1.h5
        ...
        sample_metadata.csv
        latlon_distribution.npy
        compression_settings.json
```

**Example path:**
```
/path/to/data/h5py_data_w_missing_timesteps_gzip_9_shuffle_256_x_16/era5_10_naip_sentinel2/4096/
```

#### Core Files in Each Dataset

1. **`sample_{index}.h5`** - Individual sample files containing:
   - **`latlon`**: Float32 array `[lat, lon]` - geographic coordinates
   - **`timestamps`**: Integer array `[T, 3]` where T=time steps, columns are `[day, month, year]`
   - **Modality datasets**: Named by modality (e.g., `"sentinel2"`, `"era5_10"`, `"naip"`, `"landsat"` - see all available modalities in [`constants.py`](../olmoearth_pretrain/data/constants.py))
     - Spatial modalities: Shape `[H, W, T, C]` or `[H, W, C]` depending on temporal variation
     - Non-spatial modalities: Shape `[T, C]`
   - **`missing_timesteps_masks/`** group: Boolean masks per modality (shape `[T]`) indicating which timestamps from the longest timestamp array are present for that specific modality (see [`_create_missing_timesteps_masks` in convert_to_h5py.py](../olmoearth_pretrain/dataset/convert_to_h5py.py))

2. **`sample_metadata.csv`** - CSV with columns `sample_index, <modality1>, <modality2>...` where values are 1 (present) or 0 (absent), tracking which modalities exist in each sample (see [`save_sample_metadata` in convert_to_h5py.py](../olmoearth_pretrain/dataset/convert_to_h5py.py))

3. **`latlon_distribution.npy`** - NumPy array `[N, 2]` of all sample lat/lons for dataset statistics (see [`save_latlon_distribution` in convert_to_h5py.py](../olmoearth_pretrain/dataset/convert_to_h5py.py))

4. **`compression_settings.json`** - Stores compression algorithm, compression level options, and shuffle filter settings used for all H5 files (see [`save_compression_settings` in convert_to_h5py.py](../olmoearth_pretrain/dataset/convert_to_h5py.py))

**Key Invariant:** All H5 files follow the same schema with `latlon`, `timestamps`, modality datasets, and `missing_timesteps_masks` group structure, ensuring consistency across the entire dataset.


## Helpful Files for Understanding

### Configuration Files
- [`scripts/official/base.py`](../scripts/official/base.py) - Main entry point, model config
- [`scripts/official/script.py`](../scripts/official/script.py) - All component builders (dataset, dataloader, trainer, callbacks)
- [`olmoearth_pretrain/evals/datasets/paths.py`](../olmoearth_pretrain/evals/datasets/paths.py) - Evaluation dataset path configuration

### Dataset Files
- [`olmoearth_pretrain/data/dataset.py`](../olmoearth_pretrain/data/dataset.py) - Dataset implementation and configuration
- [`olmoearth_pretrain/data/dataloader.py`](../olmoearth_pretrain/data/dataloader.py) - Dataloader implementation and configuration
- [`olmoearth_pretrain/data/constants.py`](../olmoearth_pretrain/data/constants.py) - Modality definitions and constants

### Training Files
- [`olmoearth_pretrain/internal/experiment.py`](../olmoearth_pretrain/internal/experiment.py) - Core experiment orchestration
- [`olmoearth_pretrain/train/train_module/`](../olmoearth_pretrain/train/train_module/) - Training module implementations
- [`olmoearth_pretrain/train/masking.py`](../olmoearth_pretrain/train/masking.py) - Masking strategy implementations

---
