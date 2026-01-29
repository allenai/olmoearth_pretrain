# Plan for Minimal OlmoEarth Pretrain Repo

This document outlines the plan for creating a minimal repository containing only the files needed to initialize the OlmoEarth model.

## Goal

Create a minimal subset of the `olmoearth_pretrain` package that contains only the essential files needed to:
- Load model configurations from JSON
- Initialize model instances (with or without weights)
- Support inference-only use cases

## Directory Structure

```
olmoearth_pretrain/minimal/
├── README.md                    # Documentation for the minimal repo
├── PLAN.md                      # This file
├── olmoearth_pretrain/          # Minimal package
│   ├── __init__.py
│   ├── config.py                # Config class (standalone version)
│   ├── types.py                 # Type aliases
│   ├── datatypes.py             # MaskedOlmoEarthSample and related types
│   ├── model_loader.py          # Main entry point for loading models
│   ├── data/
│   │   ├── __init__.py
│   │   └── constants.py         # ModalitySpec, Modality, BandSet, etc.
│   └── nn/
│       ├── __init__.py
│       ├── utils.py             # DistributedMixins, unpack_encoder_output, etc.
│       ├── attention.py         # Block class for attention
│       ├── encodings.py         # Position encoding functions
│       ├── flexi_patch_embed.py # FlexiPatchEmbed, FlexiPatchReconstruction
│       ├── flexi_vit.py         # Encoder, Predictor, Reconstructor, CompositeEncodings
│       └── latent_mim.py        # LatentMIM model class and config
└── pyproject.toml               # Minimal dependencies
```

## Files to Include

### Core Files

1. **`config.py`** - Standalone Config class that works without olmo-core
   - Contains `_StandaloneConfig` class
   - Handles JSON deserialization with `_CLASS_` field resolution
   - Essential for loading model configs from JSON

2. **`model_loader.py`** - Main entry point
   - `ModelID` enum for model identifiers
   - `load_model_from_id()` and `load_model_from_path()` functions
   - Downloads configs/weights from HuggingFace or loads from local paths

3. **`types.py`** - Type aliases
   - `ArrayTensor` type alias

4. **`datatypes.py`** - Data structures
   - `MaskValue` enum
   - `MaskedOlmoEarthSample` NamedTuple
   - Required for model forward pass signatures

### Data Module

5. **`data/constants.py`** - Modality specifications
   - `ModalitySpec` dataclass
   - `BandSet` dataclass
   - `Modality` class with all modality definitions
   - `get_modality_specs_from_names()` function
   - Constants like `BASE_GSD`, `IMAGE_TILE_SIZE`, etc.

### Neural Network Module

6. **`nn/utils.py`** - Utility functions
   - `DistributedMixins` class (for FSDP support, optional but may be in configs)
   - `unpack_encoder_output()` function
   - `get_cumulative_sequence_lengths()` function

7. **`nn/attention.py`** - Attention blocks
   - `Block` class (transformer block)
   - Required by encoder/decoder

8. **`nn/encodings.py`** - Position encodings
   - `get_1d_sincos_pos_encoding()`
   - `get_2d_sincos_pos_encoding_with_resolution()`
   - `get_month_encoding_table()`
   - Used by CompositeEncodings

9. **`nn/flexi_patch_embed.py`** - Patch embedding
   - `FlexiPatchEmbed` class
   - `FlexiPatchReconstruction` class
   - Required for tokenization

10. **`nn/flexi_vit.py`** - Core model architecture
    - `Encoder` class and `EncoderConfig`
    - `Predictor` class and `PredictorConfig` (decoder)
    - `Reconstructor` class and `ReconstructorConfig`
    - `CompositeEncodings` class
    - `TokensAndMasks` NamedTuple
    - All the core model logic

11. **`nn/latent_mim.py`** - Main model class
    - `LatentMIM` class (the main model)
    - `LatentMIMConfig` class
    - Combines encoder, decoder, and optional reconstructor

## Dependencies

### Required (minimal inference)
- `torch>=2.7,<2.8` - PyTorch
- `einops>=0.7.0` - Tensor operations
- `numpy>=1.26.4` - Array operations
- `huggingface_hub` - For downloading models from HuggingFace
- `universal-pathlib>=0.2.5` - Path handling (UPath)

### Optional (for training, not needed for minimal)
- `olmo-core` - Only needed for training, not for inference

## Files to Exclude

The following are NOT needed for model initialization:

1. **Training code**
   - `train/` directory
   - `internal/` directory (experiment code)
   - `scripts/` directory

2. **Evaluation code**
   - `evals/` directory

3. **Dataset creation**
   - `dataset/` directory
   - `dataset_creation/` directory

4. **Data loading**
   - `data/dataset.py`
   - `data/dataloader.py`
   - `data/normalize.py`
   - `data/transform.py`
   - `data/concat.py`
   - `data/visualize.py`
   - `data/utils.py`
   - `data/norm_configs/` (not needed for model init)

5. **Inference benchmarking**
   - `inference_benchmarking/` directory

6. **Other model architectures**
   - `nn/galileo.py`
   - `nn/mae.py`
   - `nn/st_model.py`
   - `nn/pooled_modality_predictor.py`

7. **Compatibility/helpers**
   - `_compat.py`
   - `decorators.py`
   - `helios/` directory (if not needed)

## Implementation Steps

1. **Create directory structure**
   - Create `olmoearth_pretrain/minimal/` directory
   - Create subdirectories: `olmoearth_pretrain/data/`, `olmoearth_pretrain/nn/`

2. **Copy core files**
   - Copy the essential files listed above
   - Ensure all `__init__.py` files are created

3. **Update imports**
   - Check that all imports within the minimal files are valid
   - Remove or stub out any imports that reference excluded modules
   - Ensure relative imports work correctly

4. **Create minimal pyproject.toml**
   - Include only minimal dependencies
   - Set up package structure

5. **Create README.md**
   - Document what's included
   - Show usage examples
   - Explain what's excluded and why

6. **Test model loading**
   - Verify that `load_model_from_id()` works
   - Test loading configs from JSON
   - Ensure model can be instantiated

## Usage Example

```python
from olmoearth_pretrain.model_loader import ModelID, load_model_from_id

# Load a pre-trained model
model = load_model_from_id(ModelID.OLMOEARTH_V1_NANO)

# Or load without weights (random initialization)
model = load_model_from_id(ModelID.OLMOEARTH_V1_NANO, load_weights=False)

# Or load from a local path
from olmoearth_pretrain.model_loader import load_model_from_path
model = load_model_from_path("/path/to/model")
```

## Notes

- The minimal repo should work independently or can be used as a subdirectory
- All model configs should deserialize correctly using the standalone Config
- The model should be able to forward pass (though actual data loading is excluded)
- FSDP/distributed features may not work fully without olmo-core, but model init should work

## Future Considerations

- Could further minimize by removing FSDP-related code if not needed
- Could create a separate "ultra-minimal" version that only loads weights without building from config
- Consider if `flexihelios.py` (deprecated) needs to be included for backward compatibility with old checkpoints

