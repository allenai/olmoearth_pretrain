# Token Merging (ToMe) Experiment

Based on [Token Merging: Your ViT But Faster (Bolya et al., 2023)](https://arxiv.org/abs/2210.09461).

## Overview

Token Merging (ToMe) progressively merges similar tokens during the encoder forward pass, reducing the number of tokens that subsequent transformer blocks need to process. After encoding, merged tokens are **unmerged** back to their original positions so downstream components (decoder, loss, masking) see the full token count unchanged.

## Configuration

Defined in `olmoearth_pretrain/nn/token_merging.py` via `ToMeConfig`:

```python
tome_config=ToMeConfig(
    enabled=True,
    merge_layers=[3, 6, 9],   # Merge after blocks 3, 6, 9
    r_ratio=0.2,              # Reduce 20% of tokens per merge step
)
```

- `merge_layers`: Which encoder block indices trigger a merge step.
- `r_ratio`: Fraction of (non-protected) tokens to remove at each merge step. Alternative: `r_per_layer` for a fixed count.

## How Token Merging Works

### Algorithm (`bipartite_soft_matching`)

1. **Partition** tokens into two sets: `dst` (even indices) and `src` (odd indices).
2. **Cosine similarity** between all `(dst, src)` pairs.
3. **Select** the `r` most similar `src` tokens.
4. **Merge** each selected `src` into its best-matching `dst` by averaging.
5. **Output** = `[dst (with merges absorbed), unmerged src]` → `N - r` tokens total.

Protected tokens (register tokens) at the start of the sequence are never merged.

### Unmerge (`unmerge_tokens`)

Reverses the merge by duplicating merged representations back to original positions, restoring the full `N`-token sequence. Applied in reverse order of the merge stack.

## Impact on Each Component

### Online Encoder

**Where:** `FlexiViTEncoder.apply_attn()` in `flexi_vit.py`

The encoder is the only component where ToMe actually reduces computation:

1. Tokens enter the transformer blocks at full count.
2. After each layer in `merge_layers` (e.g., blocks 3, 6, 9), `bipartite_soft_matching` merges `r` tokens.
3. All subsequent blocks process fewer tokens → **faster attention and FFN**.
4. Attention masks are truncated to match the reduced sequence length.
5. After all blocks + layer norm, tokens are **unmerged** back to full count.
6. The per-modality split and output projection see the original token count.

With `r_ratio=0.2` and 3 merge layers, the token count through the encoder is approximately:
- Blocks 0–3: `N` tokens
- Blocks 4–6: `~0.8N` tokens
- Blocks 7–9: `~0.64N` tokens
- Blocks 10–11: `~0.51N` tokens
- Output: `N` tokens (after unmerge)

**Constraint:** ToMe is only applied when NOT using flash attention (`use_flash_attn=False`), since flash attention requires `cu_seqlens` bookkeeping that isn't compatible with mid-sequence token count changes.

### Target Encoder

**Where:** `ContrastiveLatentMIMTrainModule.model_forward()` in `contrastive_latentmim.py`

The **same** `tome_config` is passed to the target encoder's forward call:

```python
output_dict = self.model.target_encoder.forward(
    batch.unmask(),
    patch_size=patch_size,
    token_exit_cfg=token_exit_cfg,
    tome_config=self.tome_config,
)
```

The target encoder runs inside `torch.no_grad()` and processes **unmasked** input. ToMe applies identically — merge during blocks, unmerge at the end — so the target encoder output has the same shape as without ToMe. This means the target representations used for loss computation are unaffected in dimensionality, but the intermediate computation is faster.

Note: Since the target encoder processes different input (unmasked) than the online encoder (masked), the bipartite matching will select different token pairs to merge.

### Decoder

**Where:** `LatentMIM.forward()` in `latent_mim.py`

The decoder is **not affected** by ToMe. By the time encoder output reaches the decoder, all tokens have been unmerged back to original positions. The decoder receives the same `TokensAndMasks` structure it would without ToMe. No `tome_config` is passed to the decoder.

### Masking

**Where:** `MaskingConfig` in `train/masking.py`

Masking is **not affected** by ToMe. The masking strategy runs before encoding (in the dataloader / train module) and determines which tokens are visible vs. masked. ToMe operates *inside* the encoder on the already-masked subset of tokens. Since tokens are unmerged before leaving the encoder, the mask shapes remain consistent.

The masking config uses `modality_cross_random` with `encode_ratio=0.5` and `decode_ratio=0.5`, with certain modalities restricted to decode-only. This is orthogonal to ToMe.

## Key Design Decisions

1. **Merge-then-unmerge**: ToMe only speeds up intermediate encoder blocks. The output dimensionality is preserved, so no downstream changes are needed.
2. **Register token protection**: Register tokens (if any) are excluded from merging via the `protected` parameter.
3. **No flash attention support**: ToMe is skipped when `use_flash_attn=True` because changing sequence length mid-forward would break `cu_seqlens` tracking.
4. **Applied to both encoders**: Both online and target encoder use the same ToMe config, ensuring structural consistency while independently selecting which tokens to merge based on their respective inputs.

## Files Changed

| File | Change |
|------|--------|
| `olmoearth_pretrain/nn/token_merging.py` | **New file.** `ToMeConfig`, `ToMeMergeInfo`, `bipartite_soft_matching`, `unmerge_tokens`, `apply_tome_unmerge_stack` |
| `olmoearth_pretrain/nn/flexi_vit.py` | Added ToMe merge/unmerge logic inside `apply_attn()`. Merge after specified blocks, update attn masks, unmerge after norm. |
| `olmoearth_pretrain/nn/latent_mim.py` | Thread `tome_config` through `LatentMIM.forward()` → `encoder()` call |
| `olmoearth_pretrain/train/train_module/contrastive_latentmim.py` | Store `tome_config` on train module, pass to both online encoder (via `self.model()`) and target encoder (via `self.model.target_encoder.forward()`) |
| `scripts/experimental/2026_02_03_token_merge/base.py` | Experiment script with `ToMeConfig(enabled=True, merge_layers=[3,6,9], r_ratio=0.2)` |
