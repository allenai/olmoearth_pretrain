# Hybrid Patch Size Training

This directory contains scripts for training models with hybrid patch sizes: processing at patch size 8 for speed, but outputting embeddings at patch size 1 for fine-grained representations.

## Overview

The hybrid patch size approach provides:
- **Fast Training**: Process at patch size 8 (~64x faster attention than ps=1)
- **Fine-grained Outputs**: Output embeddings at patch size 1 for better downstream task performance
- **Memory Efficient**: Only upsamples tokens at the end, not during attention

## Quick Start

### Option 1: Using the Shell Script (Recommended)

```bash
# Basic usage with defaults
./scripts/piperw/train_hybrid_patch_size.sh

# Custom run name and cluster
./scripts/piperw/train_hybrid_patch_size.sh my_experiment ai2/ceres-cirrascale

# Full customization
./scripts/piperw/train_hybrid_patch_size.sh my_experiment ai2/jupiter-cirrascale-2 16 urgent
```

### Option 2: Direct Python Command

```bash
# Basic launch
python scripts/piperw/nano.py launch nano_hybrid_ps8_to_ps1 ai2/jupiter-cirrascale-2 \
  --launch.num_gpus=8 \
  --train_module.processing_patch_size=8 \
  --train_module.output_patch_size=1

# With custom settings
python scripts/piperw/nano.py launch my_run_name ai2/jupiter-cirrascale-2 \
  --launch.num_gpus=16 \
  --launch.priority=urgent \
  --trainer.callbacks.wandb.project=my_project \
  --train_module.processing_patch_size=8 \
  --train_module.output_patch_size=1
```

### Option 3: Local Training (for testing)

```bash
# Train locally (useful for debugging)
python scripts/piperw/nano.py train test_run local \
  --train_module.processing_patch_size=8 \
  --train_module.output_patch_size=1
```

## Configuration

The hybrid patch size settings are configured in `scripts/piperw/script.py`:

```python
processing_patch_size=8,  # Process at ps=8 for speed
output_patch_size=1,      # Output at ps=1 for fine-grained embeddings
```

You can override these via command line:

```bash
# Use different patch sizes
python scripts/piperw/nano.py launch my_run ai2/jupiter \
  --train_module.processing_patch_size=4 \
  --train_module.output_patch_size=1
```

## How It Works

1. **Encoder Processing**: The encoder processes input at `processing_patch_size=8`
   - Attention operations are ~64x faster than at ps=1
   - Most computation happens at this resolution

2. **Upsampling**: After attention, tokens are upsampled to `output_patch_size=1`
   - Uses bilinear interpolation for tokens
   - Uses nearest-neighbor for masks
   - Only happens at the end, not during attention

3. **Target Encoder**: Always uses `output_patch_size=1`
   - Ensures consistent random projections
   - Targets computed at fine-grained resolution

4. **Projection & Loss**: Both happen at ps=1
   - Final embeddings are at fine-grained resolution
   - Loss computed between ps=1 predictions and targets

## Architecture Details

The implementation includes:

- **PatchUpsampler** (`flexi_vit.py`): Module that upsamples tokens from larger to smaller patch sizes
- **Encoder Updates**: Supports `output_patch_size` parameter for upsampling
- **Training Module**: Manages dual patch sizes throughout training
- **Target Encoder**: Always uses output patch size for consistency

## Monitoring

Training progress can be monitored via:
- WandB: Check the project specified in `--trainer.callbacks.wandb.project`
- Logs: Check the work directory specified in the trainer config

## Troubleshooting

### Memory Issues
If you encounter OOM errors:
- Reduce batch size: `--dataloader.global_batch_size=256`
- Reduce number of GPUs: `--launch.num_gpus=4`
- Use gradient checkpointing (if available)

### Speed Issues
If training is slower than expected:
- Verify `processing_patch_size=8` is being used
- Check that upsampling only happens at the end
- Monitor GPU utilization

### Validation
To verify the patch sizes are working correctly:
- Check logs for "Hybrid patch size configuration" message
- Monitor token counts in WandB (should see ps=8 during processing, ps=1 at output)
- Verify target encoder always uses ps=1

## Files

- `train_hybrid_patch_size.sh`: Shell script for easy launching
- `nano.py`: Main training script (uses config from `script.py`)
- `script.py`: Contains the hybrid patch size configuration
- `patch_size_hybrid_plan.md`: Detailed implementation plan

## References

See `patch_size_hybrid_plan.md` for detailed implementation notes and design decisions.

