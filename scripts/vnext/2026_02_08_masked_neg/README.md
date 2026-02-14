# Single Bandset / Merged Bandset Experiments

Experiments investigating S2 bandset configurations and cross-spectral learning strategies with masked-negatives loss.

## Experiments

| # | Key | Description |
|---|-----|-------------|
| 1 | `single_bandset_cross_random_masked_neg` | modality_cross_random + single bandset S2 (all 12) / Landsat |
| 2 | `single_bandset_random_decode_masked_neg` | random_with_decode + single bandset S2 (all 12) / Landsat |
| 3 | `single_bandset_no60m_cross_random_masked_neg` | modality_cross_random + single bandset S2 (no 60m: 10 bands) / Landsat |
| 4 | `single_bandset_10m_only_cross_random_masked_neg` | modality_cross_random + single bandset S2 (10m only: 4 bands) / Landsat |
| 5 | `single_bandset_band_dropout_cross_random_masked_neg` | modality_cross_random + single bandset S2 + band dropout 0.3 |
| 6 | `single_bandset_band_dropout_0.5_cross_random_masked_neg` | modality_cross_random + single bandset S2 + band dropout 0.5 |
| 7 | `single_bandset_era5_decode_only_masked_neg` | random_with_decode + single bandset S2/Landsat + ERA5 decode-only |
| 8 | `two_bandset_cross_random_masked_neg` | modality_cross_random + 2 bandsets S2 (10m+20m) / Landsat single |
| 9 | `merged_bandsets_cross_random_masked_neg` | 3 bandsets S2 / 2 bandsets Landsat, merge before transformer, unmerge in decoder |
| 10 | `midlayer_merged_bandsets_cross_random_masked_neg` | 3 bandsets S2 / 2 bandsets Landsat, merge after layer 3, unmerge in decoder |
| 11 | `default_bandsets_via_tokenconfig_cross_random_masked_neg` | default 3 bandsets S2 / 2 bandsets Landsat via TokenizationConfig (sanity check) |
| 12 | `two_bandset_midlayer_merged_cross_random_masked_neg` | 2 bandsets S2 (10m+20m) / Landsat single, merge after layer 3, unmerge in decoder |
| 13 | `single_bandset_no60m_random_band_dropout_cross_random_masked_neg` | single bandset S2 (no 60m: 10 bands) + random band dropout ~ Uniform(0, 0.3) |
| 14 | `single_bandset_all12_random_band_dropout_cross_random_masked_neg` | single bandset S2 (all 12) + random band dropout ~ Uniform(0, 0.3) |

## Launch Commands

Run all commands from the **repo root**. Commit and push before launching — Beaker pulls code from the repo.

### Dry run (always do this first)

```bash
EXPERIMENT=<key> \
  python scripts/vnext/2026_02_08_masked_neg/single_bandset_masked_neg.py dry_run run_name local
```

### Experiment 9: Pre-transformer merge

```bash
EXPERIMENT=merged_bandsets_cross_random_masked_neg \
  python scripts/vnext/2026_02_08_masked_neg/single_bandset_masked_neg.py launch \
  merged_bandsets_cross_random_masked_neg_fix ai2/jupiter \
  launch.num_gpus=8 \
  'launch.clusters=[ai2/jupiter,ai2/ceres,ai2/titan]' \
  trainer.callbacks.wandb.project=2026_02_08_masked_neg
```

### Experiment 10: Mid-layer merge (merge after layer 3)

```bash
EXPERIMENT=midlayer_merged_bandsets_cross_random_masked_neg \
  python scripts/vnext/2026_02_08_masked_neg/single_bandset_masked_neg.py launch \
  midlayer_merged_bandsets_cross_random_masked_neg_fix ai2/jupiter \
  launch.num_gpus=8 \
  'launch.clusters=[ai2/jupiter,ai2/ceres,ai2/titan]' \
  trainer.callbacks.wandb.project=2026_02_08_masked_neg
```

### Experiment 12: Two bandsets S2 + mid-layer merge (combines exp 8 + 10)

```bash
EXPERIMENT=two_bandset_midlayer_merged_cross_random_masked_neg \
  python scripts/vnext/2026_02_08_masked_neg/single_bandset_masked_neg.py launch \
  two_bandset_midlayer_merged_cross_random_masked_neg ai2/jupiter \
  launch.num_gpus=8 \
  'launch.clusters=[ai2/jupiter,ai2/ceres,ai2/titan]' \
  trainer.callbacks.wandb.project=2026_02_08_masked_neg
```

### Experiment 13: Single bandset S2 (no 60m) + random band dropout

```bash
EXPERIMENT=single_bandset_random_band_dropout_cross_random_masked_neg \
  python scripts/vnext/2026_02_08_masked_neg/single_bandset_masked_neg.py launch \
  single_bandset_no60m_random_band_dropout_cross_random_masked_neg ai2/jupiter \
  launch.num_gpus=8 \
  'launch.clusters=[ai2/jupiter,ai2/ceres,ai2/titan]' \
  trainer.callbacks.wandb.project=2026_02_08_masked_neg
```

### Experiment 14: Single bandset S2 (all 12) + random band dropout

```bash
EXPERIMENT=single_bandset_all12_random_band_dropout_cross_random_masked_neg \
  python scripts/vnext/2026_02_08_masked_neg/single_bandset_masked_neg.py launch \
  single_bandset_all12_random_band_dropout_cross_random_masked_neg ai2/jupiter \
  launch.num_gpus=8 \
  'launch.clusters=[ai2/jupiter,ai2/ceres,ai2/titan]' \
  trainer.callbacks.wandb.project=2026_02_08_masked_neg
```

## Architecture: Bandset Merge/Unmerge

### Motivation

Switching S2 from 3 bandsets to 1 caused EuroSat performance drop while other tasks were unaffected. The hypothesis is that a single `Conv2d` over all 12 bands loses resolution-aware spatial structure (10m/20m/60m bands have different native resolutions and kernel sizes). The merge approach preserves per-resolution patch embeddings while reducing sequence length.

### Files changed

| File | What |
|------|------|
| `olmoearth_pretrain/nn/bandset_merge.py` | New `BandsetMerge` and `BandsetUnmerge` modules |
| `olmoearth_pretrain/nn/flexi_vit.py` | `Encoder`: `merge_bandsets`, `merge_after_layer` params + mid-layer merge logic; `Predictor`: `unmerge_bandsets` param |
| `olmoearth_pretrain/nn/latent_mim.py` | Sets `target_encoder.merge_enabled = False` after deepcopy |
| `tests/unit/nn/test_bandset_merge.py` | Unit tests for merge/unmerge modules, encoder, predictor, LatentMIM |

---

### Exp 9: Pre-transformer merge

```
Patch Embed (3 bandsets) → Composite Encodings → BandsetMerge (3→1) → Transformer (all 12 layers) → Output
```

**Config:** `EncoderConfig(merge_bandsets=True)` (default `merge_after_layer=-1`)

**Merge happens in `Encoder.apply_attn`**, after composite encodings and before `collapse_and_combine_hwtc`:
1. For each modality with >1 bandset (S2: 3, Landsat: 2), `BandsetMerge` is applied
2. `modalities_to_dims_dict` is updated to reflect the new 1-bandset shape
3. Tokens are then collapsed into a flat sequence and fed to the transformer

**BandsetMerge forward (`bandset_merge.py`):**

Given tokens `[B, P_H, P_W, T, num_bandsets, D]` and mask `[B, P_H, P_W, T, num_bandsets]`:

1. **Zero non-ENCODER bandsets**: `tokens *= (mask == ONLINE_ENCODER)` — prevents leakage of DECODER tokens under `modality_cross_random` masking
2. **Compute scale**: `scale = num_bandsets / num_active_bandsets` — compensates for zeroed bandsets so magnitude is consistent
3. **Concatenate and project**: `flatten(tokens) → Linear(num_bandsets*D, D)` — learned projection
4. **Rescale**: `merged *= scale`
5. **Merge mask**: `ONLINE_ENCODER` if any bandset was ENCODER, else `min(mask)` across bandsets

**Weight initialization**: `Linear` weight is set to `[I/n | I/n | ... | I/n]` (block-diagonal identity / n), bias=0. At init this is exactly mean pooling. The model learns to deviate over training.

**Concrete example (S2, one spatial position, mask = [ENCODER, DECODER, ENCODER]):**
- token_0 (10m): kept, token_1 (20m): zeroed, token_2 (60m): kept
- At init: `proj(cat(t0, 0, t2)) * (3/2) = (t0/3 + 0 + t2/3) * 1.5 = (t0 + t2) / 2` — mean of active bandsets

---

### Exp 10: Mid-layer merge

```
Patch Embed (3 bandsets) → Composite Encodings → Transformer layers 0–3 → BandsetMerge (3→1) → Transformer layers 4–11 → Output
```

**Config:** `EncoderConfig(merge_bandsets=True, merge_after_layer=3)`

**Why this is better than exp 9:** The first 4 transformer layers see all 3 bandset tokens per spatial position, so cross-spectral interactions (e.g. NDVI-like NIR/Red combinations across bandsets) are learned via full self-attention — not just a shallow linear projection. By the time we merge, each token is already contextualized by attending to other bandsets AND other modalities.

**Mid-layer merge implementation (`Encoder.apply_attn`, inside the transformer loop):**

After block `merge_after_layer`, the code performs a 10-step unpack → merge → repack:

```
1.  Pop register tokens (if any)
2.  Unpack flash-attention packing (if flash_attn)
3.  Restore removed non-ENCODER tokens to full [B, N, D] sequence
4.  Split flat sequence back to per-modality tensors using modalities_to_dims_dict
5.  Apply _apply_bandset_merge() — same BandsetMerge as exp 9
6.  Re-collapse per-modality tensors to flat [B, N', D] (N' < N after merge)
7.  Re-remove non-ENCODER tokens
8.  Re-pack for flash attention (if flash_attn)
9.  Rebuild attention mask for new sequence length
10. Re-add register tokens (if any)
```

The remaining transformer blocks then run on the shorter merged sequence.

---

### Decoder (both exp 9 and 10)

**Config:** `PredictorConfig(unmerge_bandsets=True)`

The decoder receives merged tokens `[..., 1, D]` and needs to produce per-bandset predictions to match the target encoder's output.

**BandsetUnmerge forward (`bandset_merge.py`):**
1. `Linear(D, num_bandsets * D)` — expands 1 token to `num_bandsets` tokens
2. `reshape → [..., num_bandsets, D]`
3. Uses **original pre-merge per-bandset masks** (passed via `output_dict["original_bandset_masks"]`) so decoder knows which bandsets are ENCODER (context) vs DECODER (to predict)

The original masks are saved in `Encoder.forward` before `apply_attn` and flow through `unpack_encoder_output → decoder_kwargs → Predictor.forward(original_bandset_masks=...)`.

---

### Target encoder

- `LatentMIM.__init__` does `deepcopy(encoder)` then sets `target_encoder.merge_enabled = False`
- The merge modules are kept in the target encoder (same `nn.ModuleDict`) so EMA parameter updates stay aligned between online and target encoders
- The target encoder always processes the full multi-bandset sequence (3 tokens per S2 position) — it provides granular per-bandset targets

---

### Masking interaction

Both experiments use `modality_cross_random` masking, which can assign different mask values (ENCODER/DECODER) to individual bandsets within the same (h, w, t) position. This interacts with the merge as follows:

- **Online encoder**: Non-ENCODER bandsets are zeroed before merge projection; rescaling compensates for missing bandsets
- **Target encoder**: No merge — processes all bandsets normally, producing per-bandset targets
- **Decoder**: Gets the original per-bandset masks, so loss is computed correctly per bandset

---

### Parameters added

For S2 (D=768, 3 bandsets):
- `BandsetMerge.proj`: `Linear(2304, 768)` = ~1.77M params
- `BandsetUnmerge.proj`: `Linear(768, 2304)` = ~1.77M params

For Landsat (D=768, 2 bandsets):
- `BandsetMerge.proj`: `Linear(1536, 768)` = ~1.18M params
- `BandsetUnmerge.proj`: `Linear(768, 1536)` = ~1.18M params

Total added: ~5.9M params (small relative to base model)
