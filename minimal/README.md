# OlmoEarth Pretrain - Minimal Package

This is a minimal subset of the `olmoearth_pretrain` package containing only the essential files needed to initialize and load OlmoEarth models. This package is designed for inference-only use cases where you don't need the full training, evaluation, or dataset creation functionality.

## What's Included

This minimal package includes:

- **Model Loading**: Functions to load pre-trained models from HuggingFace or local paths
- **Model Architecture**: Core model classes (LatentMIM, Encoder, Decoder)
- **Configuration**: Standalone Config class that works without olmo-core
- **Data Structures**: Type definitions and data structures needed for model forward passes

## What's NOT Included

The following are excluded to keep the package minimal:

- Training code (`train/`, `internal/`)
- Evaluation code (`evals/`)
- Dataset creation (`dataset/`, `dataset_creation/`)
- Data loading utilities
- Other model architectures (Galileo, MAE, etc.)

## Installation

### From Source

```bash
cd minimal
pip install -e .
```

### Dependencies

The minimal package requires:
- `torch>=2.7,<2.8`
- `einops>=0.7.0`
- `numpy>=1.26.4`
- `huggingface_hub`
- `universal-pathlib>=0.2.5`

**Note**: `olmo-core` is NOT required for inference-only use. The package includes a standalone Config class that works without olmo-core.

## Usage

### Basic Model Loading

```python
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id

# Load a pre-trained model from HuggingFace
model = load_model_from_id(ModelID.OLMOEARTH_V1_NANO)

# Load without weights (random initialization)
model = load_model_from_id(ModelID.OLMOEARTH_V1_NANO, load_weights=False)
```

### Available Models

- `ModelID.OLMOEARTH_V1_NANO` - Nano model (1.4M encoder params)
- `ModelID.OLMOEARTH_V1_TINY` - Tiny model (6.2M encoder params)
- `ModelID.OLMOEARTH_V1_BASE` - Base model (89M encoder params)
- `ModelID.OLMOEARTH_V1_LARGE` - Large model (308M encoder params)

### Loading from Local Path

```python
from olmoearth_pretrain.model_loader import load_model_from_path

# Load from a local directory containing config.json and weights.pth
model = load_model_from_path("/path/to/model")
```

### Example Script

See `example_initialize_model.py` for a complete example showing different ways to initialize the model.

```bash
python example_initialize_model.py
```

## Model Structure

The OlmoEarth model consists of:

- **Encoder**: Processes masked input samples into token representations
- **Decoder/Predictor**: Generates predictions from encoded tokens
- **Reconstructor** (optional): For auto-encoding tasks

The model uses a flexible patch embedding system that supports multiple modalities (Sentinel-2, Sentinel-1, Landsat, etc.) and variable patch sizes.

## Configuration

The minimal package includes a standalone `Config` class that can deserialize model configurations from JSON files. This works without `olmo-core`, making it suitable for inference-only deployments.

If `olmo-core` is installed, the package will automatically use the full-featured Config from olmo-core instead.

## Limitations

This minimal package is designed for:

- ✅ Loading and initializing pre-trained models
- ✅ Running inference (forward passes)
- ✅ Model inspection and analysis

It does NOT support:

- ❌ Training new models
- ❌ Full evaluation pipelines
- ❌ Dataset creation and processing
- ❌ Advanced configuration features (OmegaConf, CLI overrides)

For full functionality including training, install the complete `olmoearth-pretrain` package with training dependencies:

```bash
pip install olmoearth-pretrain[training]
```

## File Structure

```
minimal/
├── README.md                    # This file
├── PLAN.md                      # Detailed plan document
├── pyproject.toml               # Package configuration
├── example_initialize_model.py  # Example usage script
└── olmoearth_pretrain/          # Minimal package
    ├── __init__.py
    ├── config.py                # Standalone Config class
    ├── model_loader.py          # Model loading functions
    ├── types.py                 # Type aliases
    ├── datatypes.py             # Data structures
    ├── data/
    │   ├── __init__.py
    │   └── constants.py         # Modality specifications
    └── nn/
        ├── __init__.py
        ├── latent_mim.py        # Main model class
        ├── flexi_vit.py         # Encoder/Decoder
        ├── attention.py         # Attention blocks
        ├── encodings.py         # Position encodings
        ├── flexi_patch_embed.py # Patch embedding
        └── utils.py             # Utility functions
```

## License

Same as the main olmoearth_pretrain package.

## References

- Main package: [olmoearth_pretrain](https://github.com/allenai/olmoearth_pretrain)
- Model weights: Available on [HuggingFace](https://huggingface.co/allenai)

