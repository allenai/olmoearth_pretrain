# APT (Adaptive Patch Tokenization) Architecture

## Overview

APT replaces uniform patching with content-adaptive patching. Complex regions get
more tokens (small patches), homogeneous regions get fewer (large patches).

## Data Flow

```
Raw Image [B, H, W, T, C]
│
│  ┌─────────────────────────────────────────────────────────┐
│  │ APTEncoderWrapper.forward()                             │
│  │                                                         │
│  │  1. Force patch_size = base_patch_size (e.g. 4 or 1)    │
│  │  2. Force fast_pass = False (APT needs masking)         │
│  │                                                         │
│  │  ┌───────────────────────────────────────────┐          │
│  │  │ SCORING + PARTITIONING (CPU, per-sample)  │          │
│  │  │                                           │          │
│  │  │  For each sample, for each timestep:      │          │
│  │  │    frame = image[:,:,t,:].numpy()          │          │
│  │  │    patches = partitioner.partition(frame)  │          │
│  │  │                                           │          │
│  │  │  QuadtreePartitioner._partition_recursive: │          │
│  │  │    Start at coarsest scale (e.g. 8px)      │          │
│  │  │    score = EntropyScorer(patch_region)     │          │
│  │  │    if score < threshold:                   │          │
│  │  │      → keep as large patch (1 token)       │          │
│  │  │    else:                                   │          │
│  │  │      → subdivide into 4 children           │          │
│  │  │      → recurse at next finer scale         │          │
│  │  │                                           │          │
│  │  │  Output: list[PatchDescriptor] per sample  │          │
│  │  │    (y, x, scale, size, score, timestep)    │          │
│  │  └───────────────────────────────────────────┘          │
│  │                          │                              │
│  │                          ▼                              │
│  │  ┌───────────────────────────────────────────┐          │
│  │  │ APTEncoder.forward()                      │          │
│  └──┤                                           │          │
      │                                           │          │
      ▼                                           │
```

### Inside APTEncoder.forward()

```
Raw Image [B, H, W, T, C]
│
▼
┌──────────────────────────────────────────────────────────────┐
│ Step 1: STANDARD PATCH EMBEDDING (FlexiPatchEmbed)           │
│                                                              │
│   patch_embeddings.forward(x, patch_size)                    │
│   → Patchify ALL modalities at base_patch_size               │
│   → Output: dict of per-modality tokens                      │
│     S2:  [B, H/p, W/p, T, num_bandsets, D]                  │
│     other modalities: [B, H', W', T', ...]                   │
│                                                              │
│   ** This is where the raw pixels become D-dim tokens **     │
│   ** via Conv2d or Linear projection (same as non-APT) **    │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 2: COMPOSITE ENCODINGS (positional + temporal + channel)│
│                                                              │
│   composite_encodings.forward(tokens, timestamps, ...)       │
│   → Adds sinusoidal positional, temporal, channel encodings  │
│   → Output: same shape, but with encodings added             │
│                                                              │
│   ** Still at uniform base-patch resolution here **          │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 3: ADAPTIVE TOKEN MERGING (AdaptivePatchEmbed)          │
│                                                              │
│   Only for the APT modality (e.g. sentinel2_l2a):            │
│                                                              │
│   For each sample, for each PatchDescriptor:                 │
│     if scale == 0 (base):                                    │
│       token = base_tokens[y, x, t]  →  just take it          │
│     if scale > 0 (coarser):                                  │
│       block = base_tokens[y:y+2^s, x:x+2^s, t]              │
│       token = ConvDownsample(block)                          │
│         → stack of 2x2 stride-2 convs                        │
│         → reduces 2^s × 2^s grid to single token             │
│         → init: avg pooling (identity + 1/4 spatial avg)     │
│                                                              │
│   ** This is where token count is REDUCED **                 │
│   ** N_apt tokens per sample (variable length) **            │
│                                                              │
│   Output: list of [N_i, D] tensors (variable per sample)     │
│           list of [N_i, 4] position tensors                  │
│                                                              │
│   Other modalities: unchanged (still uniform tokens)         │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 4: COLLAPSE + COMBINE + PAD                             │
│                                                              │
│   collapse_and_combine_hwtc_apt():                           │
│     APT modality: pad variable-length to max_across_samples  │
│       → [B, max_apt_tokens, D] with MISSING mask on padding  │
│     Other modalities: flatten H,W,T,C → [B, N_other, D]     │
│     Concatenate all modalities → [B, N_total, D]             │
│     Concatenate all masks     → [B, N_total]                 │
│                                                              │
│   ** All samples now have same sequence length **            │
│   ** Padding positions marked as MISSING **                  │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 5: REMOVE MASKED + PACK (for attention)                 │
│                                                              │
│   _maybe_remove_masked_tokens():                             │
│     Sort tokens so real tokens come first, masked last       │
│     Trim to max real tokens across batch                     │
│     → seq_lengths per sample, max_seqlen                     │
│                                                              │
│   if use_flash_attn:                                         │
│     cu_seqlens = get_cumulative_sequence_lengths(seq_lengths) │
│     tokens = pack_tokens(tokens, mask)                       │
│       → [total_real_tokens, D]  (no padding)                 │
│   else:                                                      │
│     build attn_mask for SDPA                                 │
│       → [B, N, N] mask (THIS IS WHAT OOMs)                  │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 6: TRANSFORMER BLOCKS                                   │
│                                                              │
│   for blk in self.blocks:                                    │
│     tokens = blk(x=tokens, cu_seqlens=..., max_seqlen=...)   │
│                                                              │
│   if use_flash_attn:                                         │
│     flash_attn_varlen_func (no N×N mask materialized)        │
│   else:                                                      │
│     F.scaled_dot_product_attention (SDPA, needs N×N mask)    │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 7: UNPACK + NORM + SCATTER BACK                         │
│                                                              │
│   if use_flash_attn:                                         │
│     tokens = unpack_tokens(tokens, mask, og_shape)           │
│                                                              │
│   tokens = self.norm(tokens)                                 │
│   tokens = _maybe_add_removed_tokens(...)                    │
│                                                              │
│   split_and_expand_per_modality():                           │
│     APT modality: SCATTER tokens back to spatial grid        │
│       For each patch position (y, x, size, t):               │
│         if size == 1: output[y, x, t] = token                │
│         if size > 1:  output[y:y+s, x:x+s, t] = broadcast   │
│       → [B, H, W, T, num_bandsets, D]                        │
│     Other modalities: standard reshape                       │
│                                                              │
│   ** Coarse tokens are BROADCAST to all grid positions **    │
│   ** they cover — same embedding at every sub-position **    │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
              TokensAndMasks output
```

## Scale Examples

### 2-scale (base_patch_size=4, num_scales=2)

```
Patch sizes: [4px, 8px]
Thresholds:  [0.8]  (1 threshold for 8→4 split)

8×8 region with score < 0.8  →  1 token  (saves 3 tokens vs uniform)
8×8 region with score >= 0.8 →  4 tokens (subdivide to 4×4, same as uniform)
```

### 4-scale (base_patch_size=1, num_scales=4)

```
Patch sizes: [1px, 2px, 4px, 8px]
Thresholds:  [0.3, 0.5, 0.8]
  thresholds[2]=0.8: 8px→4px split
  thresholds[1]=0.5: 4px→2px split
  thresholds[0]=0.3: 2px→1px split

8×8 region:
  score < 0.8  →  1 token
  score >= 0.8 →  subdivide to 4×4:
    each 4×4 sub-region:
      score < 0.5  →  1 token
      score >= 0.5 →  subdivide to 2×2:
        each 2×2 sub-region:
          score < 0.3  →  1 token
          score >= 0.3 →  4 tokens (1×1 each)
```

## Key Design Points

1. **Scoring happens on CPU** using raw pixel values (numpy), before any GPU ops
2. **Patch embedding happens at base resolution** — ALL pixels are embedded at the
   smallest patch size first, then coarse tokens are created by merging base tokens
3. **Positional/temporal encodings are added BEFORE merging** — so the conv
   downsample aggregates tokens that already have positional information baked in
4. **ConvDownsample** with `avg` init starts as simple spatial averaging, then
   learns to weight sub-patches during finetuning
5. **Scatter-back broadcasts** coarse tokens — every grid position covered by a
   coarse patch gets the same embedding (no interpolation)
6. **Flash attention** avoids materializing the N×N attention mask, which is
   critical when APT produces many tokens (4-scale with aggressive thresholds)
