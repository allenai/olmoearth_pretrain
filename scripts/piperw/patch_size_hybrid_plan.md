# Plan: Hybrid Patch Size Training (ps=8 processing, ps=1 output)

## Goal
Train the model using patch size 8 for fast processing, but output embeddings at patch size 1 for fine-grained representations. The target random projection should always use patch size 1.

## Current Architecture Understanding

1. **Encoder**: Processes input at a given `patch_size`, outputs tokens at that resolution
2. **Target Encoder**: Processes unmasked input at `patch_size`, outputs target embeddings
3. **Decoder**: Takes encoder tokens and predicts at `patch_size`
4. **ProjectAndAggregate**: Projects and pools tokens for contrastive loss

## Implementation Approaches

### Approach 1: Dual Patch Size Forward Pass (Recommended)
**Concept**: Process at ps=8 internally, but upsample to ps=1 before final projection and target comparison.

**Changes Needed**:

1. **Add Upsampling Layer** (`olmoearth_pretrain/nn/flexi_vit.py`):
   - Create a new `PatchUpsampler` module that converts ps=8 tokens to ps=1 tokens
   - Input: `[B, H/8, W/8, D]` tokens at ps=8
   - Output: `[B, H, W, D]` tokens at ps=1
   - Method: Use learned or bilinear upsampling to expand spatial dimensions

2. **Modify Encoder Forward** (`olmoearth_pretrain/nn/flexi_vit.py::Encoder.forward`):
   - Add optional `output_patch_size` parameter
   - If `output_patch_size != patch_size`, apply upsampling before `project_and_aggregate`
   - Keep internal processing at `patch_size` for speed

3. **Modify Training Module** (`olmoearth_pretrain/train/train_module/contrastive_latentmim.py`):
   - Add config parameter: `processing_patch_size` (default: 8) and `output_patch_size` (default: 1)
   - In `model_forward`:
     - Encoder processes at `processing_patch_size=8`
     - Before projection, upsample to `output_patch_size=1`
     - Target encoder always uses `output_patch_size=1`

4. **Modify LatentMIM Forward** (`olmoearth_pretrain/nn/latent_mim.py`):
   - Accept both `processing_patch_size` and `output_patch_size`
   - Pass `processing_patch_size` to encoder
   - Upsample encoder output to `output_patch_size` before decoder/target comparison

### Approach 2: Separate Processing and Output Patch Sizes
**Concept**: Explicitly separate the patch size used for processing vs output.

**Changes Needed**:

1. **Modify All Forward Methods**:
   - Change signature from `patch_size: int` to `processing_patch_size: int, output_patch_size: int`
   - Use `processing_patch_size` for all internal operations
   - Use `output_patch_size` only for final output and target comparison

2. **Add Upsampling in Encoder**:
   - After attention layers, upsample from `processing_patch_size` to `output_patch_size`
   - This happens before `project_and_aggregate`

### Approach 3: Post-Processing Upsampling Layer
**Concept**: Add a lightweight upsampling layer after the encoder that converts ps=8 to ps=1.

**Changes Needed**:

1. **New Module**: `PatchSizeUpsampler` in `flexi_vit.py`
   ```python
   class PatchSizeUpsampler(nn.Module):
       """Upsamples tokens from one patch size to another."""
       def forward(self, tokens: Tensor, from_ps: int, to_ps: int) -> Tensor:
           # Upsample spatial dimensions by factor (to_ps / from_ps)
   ```

2. **Integrate into Encoder**:
   - Add as optional final layer in Encoder
   - Only active when `output_patch_size < processing_patch_size`

## Recommended Implementation: Approach 1 (Hybrid)

### Step-by-Step Implementation

#### Step 1: Create Patch Upsampler Module
**File**: `olmoearth_pretrain/nn/flexi_vit.py`

```python
class PatchUpsampler(nn.Module):
    """Upsamples tokens from a larger patch size to a smaller one.
    
    Converts tokens from [B, H/ps_large, W/ps_large, D] to [B, H/ps_small, W/ps_small, D]
    where ps_small < ps_large.
    """
    def __init__(
        self,
        embedding_size: int,
        interpolation: str = "bilinear",
        learnable: bool = False,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.interpolation = interpolation
        if learnable:
            # Could use transposed conv or learned upsampling
            self.upsample = nn.ConvTranspose2d(embedding_size, embedding_size, 
                                               kernel_size=2, stride=2)
        else:
            self.upsample = None
    
    def forward(
        self, 
        tokens: Tensor, 
        from_patch_size: int, 
        to_patch_size: int
    ) -> Tensor:
        """Upsample tokens from from_patch_size to to_patch_size.
        
        Args:
            tokens: [B, H/from_ps, W/from_ps, D] or [B, H/from_ps, W/from_ps, T, D]
            from_patch_size: Current patch size
            to_patch_size: Target patch size (must be < from_patch_size)
        """
        # Implementation: reshape, upsample spatial dims, reshape back
```

#### Step 2: Modify Encoder to Support Output Patch Size
**File**: `olmoearth_pretrain/nn/flexi_vit.py::Encoder`

- Add `output_patch_size` parameter to `forward()`
- After `apply_attn`, if `output_patch_size != patch_size`:
  - Apply `PatchUpsampler` to upsample tokens
  - Update mask shapes accordingly

#### Step 3: Update Training Module
**File**: `olmoearth_pretrain/train/train_module/contrastive_latentmim.py`

- Add config parameters:
  - `processing_patch_size: int = 8`
  - `output_patch_size: int = 1`
- Modify `model_forward()`:
  - Pass `processing_patch_size` to encoder
  - Pass `output_patch_size` to encoder for upsampling
  - Target encoder always uses `output_patch_size=1`

#### Step 4: Update Script Configuration
**File**: `olmoearth_pretrain/scripts/piperw/nano.py`

- Add configuration for hybrid patch sizes
- Set `processing_patch_size=8`, `output_patch_size=1`

### Training Considerations

1. **Memory**: Upsampling from ps=8 to ps=1 increases token count by 64x (8²)
   - Only upsample at the end, not during attention
   - Consider gradient checkpointing

2. **Speed**: Processing at ps=8 is ~64x faster than ps=1 for attention
   - Most computation happens at ps=8
   - Only final projection/upsampling at ps=1

3. **Target Encoder**: Always use ps=1 for target computation
   - Ensures consistent target embeddings
   - Random projection computed at ps=1

4. **Loss Computation**: 
   - Decoder predictions at ps=8 need to be upsampled to ps=1 for comparison
   - Or compute loss at ps=8 and upsample targets

### Alternative: Gradual Transition

Instead of fixed ps=8 → ps=1, could implement:
- Start training at ps=8 throughout
- Gradually transition: ps=8 → ps=4 → ps=2 → ps=1
- Or use curriculum learning: increase output resolution over time

## Testing Plan

1. **Unit Tests**:
   - Test `PatchUpsampler` with various patch size ratios
   - Test encoder forward with different processing/output patch sizes
   - Verify token shapes match expected dimensions

2. **Integration Tests**:
   - Test full forward pass with ps=8 processing, ps=1 output
   - Verify target encoder uses ps=1
   - Check loss computation works correctly

3. **Training Tests**:
   - Small-scale training run to verify gradients flow correctly
   - Check memory usage vs pure ps=1 training
   - Verify speedup vs pure ps=1 training

## Implementation Priority

1. **Phase 1**: Implement `PatchUpsampler` module
2. **Phase 2**: Modify encoder to support output patch size
3. **Phase 3**: Update training module configuration
4. **Phase 4**: Update script and test
5. **Phase 5**: Full training run validation

## Existing Code Patterns

The codebase already has similar patterns we can leverage:

1. **`FlexiPatchReconstruction`** (`flexi_patch_embed.py:247-304`):
   - Uses `F.interpolate` to resize between patch sizes
   - Pattern: reshape → interpolate → reshape back
   - Can adapt this pattern for token upsampling

2. **`FlexiPatchEmbed`** (`flexi_patch_embed.py:132-143`):
   - Resizes input before patch embedding
   - Uses interpolation with antialiasing

**Recommended Pattern for Token Upsampling**:
```python
# Similar to FlexiPatchReconstruction._resize
# Input: [B, H/8, W/8, D] tokens
# 1. Reshape to [B, D, H/8, W/8]
# 2. Interpolate to [B, D, H, W] (upsample by 8x)
# 3. Reshape back to [B, H, W, D]
```

## Simplified Implementation (MVP)

For a first implementation, consider:

1. **Minimal Changes**: Only modify the encoder output before `project_and_aggregate`
2. **Use Existing Interpolation**: Leverage `F.interpolate` (no learned upsampling initially)
3. **Keep Decoder at ps=8**: Only upsample for target comparison, not for decoder input
4. **Memory Efficient**: Only upsample the tokens needed for projection, not all tokens

**Simplified Flow**:
```
Input → Encoder (ps=8) → [B, H/8, W/8, D]
                         ↓
                    Upsample to ps=1
                         ↓
                    [B, H, W, D] → ProjectAndAggregate → Target Comparison (ps=1)
```

## Questions to Resolve

1. Should upsampling be learned or fixed (bilinear)? **Recommend starting with bilinear**
2. How to handle masks during upsampling? **Upsample masks using nearest-neighbor**
3. Should decoder also output at ps=1 or stay at ps=8? **Recommend keeping decoder at ps=8 for speed**
4. Memory budget: can we afford 64x token increase at output? **Only upsample tokens going to projection, not all tokens**
5. Should we upsample before or after attention? **After attention, before projection (recommended)**

