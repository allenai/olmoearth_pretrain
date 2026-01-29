"""Example script showing how to initialize the OlmoEarth model.

This script demonstrates different ways to load and initialize the OlmoEarth model:
1. Loading a pre-trained model from HuggingFace by ID
2. Loading a model from a local path
3. Loading a model without weights (random initialization)
"""

import torch

from olmoearth_pretrain.model_loader import ModelID, load_model_from_id, load_model_from_path


def main():
    """Demonstrate model initialization."""
    print("=" * 60)
    print("OlmoEarth Model Initialization Examples")
    print("=" * 60)

    # Example 1: Load a pre-trained model from HuggingFace
    print("\n1. Loading OlmoEarth-v1-Nano from HuggingFace...")
    print("   (This will download the model config and weights if not cached)")
    try:
        model = load_model_from_id(ModelID.OLMOEARTH_V1_NANO)
        print(f"   ✓ Model loaded successfully!")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")

    # Example 2: Load a model without weights (random initialization)
    print("\n2. Loading model config only (no weights)...")
    try:
        model_no_weights = load_model_from_id(
            ModelID.OLMOEARTH_V1_NANO, load_weights=False
        )
        print(f"   ✓ Model initialized with random weights!")
        print(f"   Model type: {type(model_no_weights).__name__}")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")

    # Example 3: Load from a local path (if you have a model saved locally)
    print("\n3. Loading from local path...")
    print("   (Uncomment and modify the path below if you have a local model)")
    # local_model_path = "/path/to/your/model"
    # try:
    #     model_local = load_model_from_path(local_model_path)
    #     print(f"   ✓ Model loaded from local path!")
    # except Exception as e:
    #     print(f"   ✗ Error loading model: {e}")

    # Example 4: List available model IDs
    print("\n4. Available model IDs:")
    for model_id in ModelID:
        print(f"   - {model_id.value} ({model_id.repo_id()})")

    # Example 5: Basic model usage (forward pass with dummy data)
    print("\n5. Testing model forward pass with dummy data...")
    print("   Note: This requires actual modality data. The model expects at least")
    print("   one modality (e.g., sentinel2_l2a) to be present in the sample.")
    try:
        model = load_model_from_id(ModelID.OLMOEARTH_V1_NANO, load_weights=False)
        model.eval()

        # Create dummy input data
        # Note: In practice, you would use proper data loading utilities
        batch_size = 1
        num_timesteps = 12
        patch_size = 8
        height, width = 16, 16  # Small spatial dimensions for dummy data

        # Create dummy timestamps [B, T, 3] where 3 = [day, month, year]
        timestamps = torch.zeros(batch_size, num_timesteps, 3, dtype=torch.long)
        timestamps[:, :, 1] = torch.arange(num_timesteps) % 12  # months (0-11)
        timestamps[:, :, 2] = 2024  # year

        # Create dummy sentinel2_l2a data [B, H, W, T, C]
        # Sentinel2 has 13 bands (3 band sets: 4 + 6 + 3)
        sentinel2_data = torch.randn(batch_size, height, width, num_timesteps, 13)
        sentinel2_mask = torch.zeros(batch_size, height, width, num_timesteps, 3, dtype=torch.long)
        # Set all to ONLINE_ENCODER (value 0)
        from olmoearth_pretrain.datatypes import MaskValue
        sentinel2_mask.fill_(MaskValue.ONLINE_ENCODER.value)

        # Create dummy sample with sentinel2 data
        from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample

        sample = MaskedOlmoEarthSample(
            timestamps=timestamps,
            sentinel2_l2a=sentinel2_data,
            sentinel2_l2a_mask=sentinel2_mask,
        )

        # Forward pass
        with torch.no_grad():
            output = model(sample, patch_size=patch_size)
            print(f"   ✓ Forward pass successful!")
            print(f"   Output type: {type(output)}")
            if isinstance(output, tuple) and len(output) > 0:
                print(f"   First output element type: {type(output[0])}")
    except Exception as e:
        print(f"   ✗ Error in forward pass: {e}")
        print("   (This is expected if the model requires specific data formats)")
        # Don't print full traceback for expected errors
        if "non-empty list" not in str(e):
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

