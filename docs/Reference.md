# OlmoEarth Pretrain Reference

This document provides detailed reference information for configuration, troubleshooting, and understanding the codebase.

---

## Helpful Files for Understanding

### Configuration Files
- `scripts/official/base.py` - Main entry point, model config
- `scripts/official/script.py` - All component builders (dataset, dataloader, trainer, callbacks)
- `olmoearth_pretrain/evals/datasets/paths.py` - Evaluation dataset path configuration
- `beaker_config_example.yaml` - Example Beaker launch configuration

### Dataset Files
- `olmoearth_pretrain/data/dataset.py` - Dataset implementation and configuration
- `olmoearth_pretrain/data/constants.py` - Modality definitions and constants
- `olmoearth_pretrain/convert_dataset_to_studio_format.py` - Dataset conversion utilities

### Training Files
- `olmoearth_pretrain/internal/experiment.py` - Core experiment orchestration
- `olmoearth_pretrain/train/train_module/` - Training module implementations

---

## Configuration Deep Dive

### Dataset Configuration Options

```python
dataset = OlmoEarthDataset(
    h5py_dir="/path/to/h5py/data",  # Required: path to H5 dataset
    training_modalities=["SENTINEL2_L2A", "SENTINEL1", "LANDSAT"],
    # ... additional options
)
```

**Key Parameters:**
- `h5py_dir` - Path to H5 dataset (local, Weka, or GCS)
- `training_modalities` - List of modalities to use for training
- See `olmoearth_pretrain/data/dataset.py` for complete options

### Data Loader Configuration Options

```python
data_loader_config = {
    "num_workers": 16,          # Number of data loading workers
    "global_batch_size": 512,   # Total batch size across all GPUs
    "token_budget": 2250,       # Token budget per sample
    "prefetch_factor": 4,       # Prefetch factor for data loading
    "seed": 3622,               # Random seed for data loading
}
```

### Training Module Configuration Options

```python
train_module_config = {
    "rank_microbatch_size": 32,       # Microbatch size per GPU
    "max_grad_norm": 1.0,             # Gradient clipping norm
    "optim_config": {
        "lr": 0.0001,                 # Learning rate
        "weight_decay": 0.02,         # Weight decay
    },
    "scheduler": {
        "warmup_steps": 8000,         # Warmup steps
    },
}
```

### Trainer Configuration Options

```python
trainer_config = {
    "max_duration": {"epochs": 300},   # Maximum training duration
    "metrics_collect_interval": 10,    # How often to collect metrics
    "save_interval": 5000,             # Checkpoint save interval (steps)
}
```

### Model Configuration Options

```python
model_config = {
    "encoder_config": {
        "depth": 24,              # Number of encoder layers
        "num_heads": 16,          # Number of attention heads
        "embedding_size": 768,    # Embedding dimension
    },
    "decoder_config": {
        "depth": 8,               # Number of decoder layers
    },
}
```

---

## Override Patterns

### CLI Override Syntax

Use dotted notation to override nested configuration:

```bash
--config.nested.option=value
```

### List and Dict Overrides

For lists, use JSON syntax:
```bash
--dataset.training_modalities='["SENTINEL2_L2A","SENTINEL1"]'
```

For nested dicts:
```bash
--train_module.optim_config.lr=0.0001
```

### Common Override Combinations

#### Small Debug Run
```bash
--data_loader.global_batch_size=64 \
--train_module.rank_microbatch_size=8 \
--trainer.max_duration.epochs=10
```

#### High-Performance Training
```bash
--data_loader.num_workers=16 \
--data_loader.global_batch_size=512 \
--data_loader.prefetch_factor=4 \
--train_module.rank_microbatch_size=32
```

#### Custom Architecture
```bash
--model.encoder_config.depth=12 \
--model.encoder_config.num_heads=12 \
--model.encoder_config.embedding_size=768 \
--model.decoder_config.depth=4
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue: "W&B authentication failed"

**Solution 1:** Set the environment variable
```bash
export WANDB_API_KEY="your_wandb_api_key"
```

**Solution 2:** Disable W&B in config
```bash
--trainer.callbacks.wandb.enabled=False
```

#### Issue: "Dataset path not found"

**Solution:** Always override the dataset path
```bash
--dataset.h5py_dir=/your/actual/dataset/path
```

#### Issue: "Out of memory"

**Solutions:**
- Reduce microbatch size: `--train_module.rank_microbatch_size=8`
- Reduce global batch size: `--data_loader.global_batch_size=128`
- Reduce token budget: `--data_loader.token_budget=1500`

#### Issue: "Data loading is slow"

**Solutions:**
- Increase workers: `--data_loader.num_workers=16`
- Increase prefetch: `--data_loader.prefetch_factor=8`
- Use 128x128 tile datasets (better for GB/s bottlenecks)
- Check if dataset is on fast storage (local SSD > Weka > GCS)

#### Issue: "Evaluation datasets not found"

**Solution:** Override evaluation dataset paths via environment variables:
```bash
export GEOBENCH_DIR="/your/path/to/geobench"
export MADOS_DIR="/your/path/to/mados"
# ... etc
```

Or disable evaluations you don't need:
```bash
--trainer.callbacks.downstream_evaluator.tasks_to_run='["mados","pastis_sentinel2"]'
```

#### Issue: "Some evaluations don't support remote filesystems (GCS)"

**Solution:**
- Copy evaluation datasets to local/Weka storage
- Override paths to point to local copies
- Or disable remote-incompatible evaluations

#### Issue: "Multi-node training not working"

**Solution:** See [olmo-core documentation](https://github.com/allenai/olmo-core) for proper multi-node torchrun setup with:
- `--nnodes=<number_of_nodes>`
- `--node_rank=<current_node_rank>`
- `--master_addr=<master_node_address>`

---

## Performance Tips

### Data Loading Optimization

1. **Use appropriate num_workers**: Start with `num_workers = num_cpus / num_gpus`
2. **Tune prefetch_factor**: Higher values (4-8) can help with throughput
3. **Consider dataset tile size**: 128x128 tiles may be faster than 256x256 on some systems
4. **Use local/fast storage**: Local SSD > Weka > Cloud Storage

### Training Optimization

1. **Batch size tuning**:
   - Larger batch sizes = better GPU utilization
   - But may require more warmup steps and adjusted learning rate
2. **Gradient accumulation**: Use microbatch size to fit in memory, global batch for effective batch size
3. **Mixed precision**: Enabled by default, provides speedup with minimal quality impact

### Memory Optimization

1. **Reduce microbatch size** if OOM
2. **Reduce token budget** to process fewer tokens per sample
3. **Use gradient checkpointing** (if supported in model config)
4. **Monitor GPU memory** with `nvidia-smi` or W&B profiling

---

## File Formats and Conventions

### H5 Dataset Format

Training datasets must be in H5 format with the following structure:
- One H5 file per sample
- Organized in directory: `/path/to/h5data/num_samples/`
- See `olmoearth_pretrain/data/dataset.py` for expected schema

### Checkpoint Format

Checkpoints are saved in PyTorch format with:
- Model state dict
- Optimizer state dict
- Scheduler state
- Training metadata

### Modality Names

Standard modality names (see `olmoearth_pretrain/data/constants.py`):
- `SENTINEL2_L2A` - Sentinel-2 Level 2A
- `SENTINEL1` - Sentinel-1 SAR
- `LANDSAT` - Landsat imagery
- `SRTM` - SRTM elevation data
- `WORLDCOVER` - ESA WorldCover land cover
- `OPENSTREETMAP_RASTER` - OpenStreetMap raster data

---

## Additional Resources

- **Main Documentation:** [Pretraining.md](Pretraining.md)
- **Internal Setup:** [Setup-Internal.md](Setup-Internal.md)
- **Project README:** [README.md](../README.md)
- **olmo-core Documentation:** [GitHub](https://github.com/allenai/olmo-core)

---

## Contributing

When modifying the codebase:

1. **Follow pre-commit hooks** - Run `pre-commit install`
2. **Test locally first** - Use small debug runs
3. **Document changes** - Update relevant documentation
4. **Check linting** - Ensure code passes linting checks

For questions or issues, open a GitHub issue or contact the team.
