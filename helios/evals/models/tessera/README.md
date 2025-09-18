# Tessera Model Integration

This directory contains the integration of the Tessera foundation model for satellite imagery into the Helios evaluation framework.

## Overview

Tessera is a foundation model for Earth observation data, primarily trained on Sentinel-2 satellite imagery. This integration allows Tessera to be evaluated alongside other models like DINOv3, Galileo, and Panopticon in the Helios benchmarking suite.

## Files

- `tessera.py` - Main model wrapper that adapts Tessera to work with Helios data structures
- `tessera_launch.py` - Launch configuration for running Tessera evaluations
- `__init__.py` - Module initialization

## Usage

To run evaluations with Tessera, use the `full_eval_sweep.py` script with the `--tessera` flag:

```bash
python helios/internal/full_eval_sweep.py --cluster=ai2/saturn-cirrascale --tessera --checkpoint_path=/path/to/tessera/checkpoint --defaults_only
```

## Current Implementation Status

### ✅ Completed Features

- **Model wrapper structure**: Follows the same pattern as other models in the evaluation framework
- **Data preprocessing**: Handles Sentinel-2 band selection, resizing, and normalization
- **Evaluation integration**: Full integration with `full_eval_sweep.py`
- **Argument parsing**: Support for `--tessera` flag
- **Normalization handling**: Appropriate normalization strategy for satellite imagery

### 🚧 Pending Implementation

- **Checkpoint loading**: Currently uses a placeholder model - needs actual Tessera model loading logic
- **Architecture details**: Model architecture specifics need to be updated based on actual Tessera implementation
- **Band configuration**: May need adjustment based on Tessera's exact band requirements
- **Performance optimization**: Batch sizes and other parameters may need tuning

## Model Configuration

The Tessera wrapper currently expects:
- **Input modalities**: Sentinel-2 L2A imagery
- **Input size**: 224x224 pixels (configurable)
- **Bands**: Uses 10 Sentinel-2 bands (B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12)
- **Normalization**: STANDARDIZE method with dataset statistics
- **Pooling**: Supports mean and max pooling across temporal dimensions

## Adding the Checkpoint

When the Tessera checkpoint becomes available, update the `_load_tessera_checkpoint` method in `tessera.py`:

```python
def _load_tessera_checkpoint(self, checkpoint_path: str) -> nn.Module:
    """Load actual Tessera model from checkpoint."""
    # Replace this with actual Tessera loading logic
    model = TesseraModel.from_pretrained(checkpoint_path)
    return model
```

## Integration Points

The Tessera model is integrated into:
1. `helios/evals/models/__init__.py` - Model registration
2. `helios/internal/full_eval_sweep.py` - Evaluation sweep support
3. Normalization and preprocessing pipeline

## Notes

- The current implementation uses a placeholder model for testing the integration
- Checkpoint loading is optional - the model will use the placeholder if no checkpoint is provided
- Normalization parameters may need adjustment based on Tessera's training data statistics
- The model follows the same interface as other evaluation models for consistency