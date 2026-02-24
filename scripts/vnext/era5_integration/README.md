# ERA5 Integration Experiments

Making the encoder climate-aware using ERA5 without the conflicting reconstruction objective.

## Motivation

Exp 15 (`single_bandset_all12_random_band_dropout_era5_random_decode_masked_neg`) showed that using ERA5 as a per-token **reconstruction target** causes InfoNCE loss to increase with longer training. The root cause: ERA5's ~10km resolution makes nearby tokens have near-identical targets, pushing the encoder toward spatial uniformity — directly conflicting with the contrastive objective.

These experiments explore alternative approaches that use ERA5 to enrich the representation without per-token reconstruction pressure. All use `random_with_decode` masking to match exp 15. In all approaches, ERA5 does NOT flow through the main encoder/decoder — it's a separate signal on the pooled representation only.

## Experiments

| # | Key | Script | Approach | Temporal handling | Description |
|---|-----|--------|----------|-------------------|-------------|
| 1 | `era5_decode_no_mask_neg` | `era5_decode_no_mask_neg.py` | Decode target | per-token | ERA5 as decode-only target with `modality_patch_discrimination` (no masked negatives) |
| 2 | `era5_clip_mean` | `era5_clip_alignment.py` | CLIP contrastive | mean pool | ERA5 encoder: mean→MLP(6→256→768), InfoNCE alignment |
| 3 | `era5_reg_mean` | `era5_regression.py` | Regression | mean pool | Linear(768→6), MSE on time-averaged ERA5 |
| 4 | `era5_clip_conv1d` | `era5_clip_alignment.py` | CLIP contrastive | 1D conv | ERA5 encoder: Conv1d(6→256,k=3)→pool→Linear(256→768), InfoNCE alignment |
| 5 | `era5_reg_timeseries` | `era5_regression.py` | Regression | flatten | Linear(768→72), MSE on full ERA5 time series (B,12,6) flattened |

**Baseline:** Exp 15 (`single_bandset_all12_random_band_dropout_era5_random_decode_masked_neg`) — ERA5 as per-token decode target with `random_with_decode` masking.

---

## CLIP-style Contrastive Alignment (`era5_clip_alignment.py`)

A separate ERA5 encoder produces a climate embedding. InfoNCE aligns the pooled encoder output to the ERA5 embedding. Constrains the **full 768-dim** pooled representation. Batch-composition-sensitive.

### Exp 2: `era5_clip_mean` — mean-pooled ERA5 encoder

```python
# ERA5 encoder: mean across time, then MLP
class ERA5EncoderMean(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256, output_dim=768):
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim),
        )
    def forward(self, era5):  # (B, T, 6)
        return self.mlp(era5.mean(dim=1))  # (B, D)
```

### Exp 4: `era5_clip_conv1d` — 1D conv ERA5 encoder

Captures local temporal patterns (trends, seasonality) before pooling.

```python
# ERA5 encoder: 1D conv over time, then pool + project
class ERA5EncoderConv1d(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256, output_dim=768, kernel_size=3):
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.LayerNorm(hidden_dim),  # applied per-timestep
            nn.GELU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim),
        )
    def forward(self, era5):  # (B, T, 6)
        x = self.conv(era5.transpose(1, 2))  # (B, 256, T)
        x = x.mean(dim=-1)                    # (B, 256) — pool over time
        return self.proj(x)                    # (B, D)
```

### Code changes needed
- **ERA5 encoder** (`olmoearth_pretrain/nn/era5_encoder.py`): Both encoder variants
- **Train module** (`contrastive_latentmim.py`): `era5_alignment_config` + ERA5 encoder, alignment loss in `train_batch`

---

## Regression (`era5_regression.py`)

A linear head predicts ERA5 values from pooled encoder output. MSE loss on normalized ERA5 values (already in ~[0,1]). No batch sensitivity. Simpler — no separate ERA5 encoder.

### Exp 3: `era5_reg_mean` — predict time-averaged ERA5 (768→6)

Constrains ~6 dimensions. Captures location-level climate state.

```python
era5_target = batch.era5_10.mean(dim=1)  # (B, T, 6) → (B, 6)
era5_pred = self.era5_head(pooled)        # (B, 768) → (B, 6)
loss += F.mse_loss(era5_pred, era5_target) * weight
```

### Exp 5: `era5_reg_timeseries` — predict full ERA5 time series (768→72)

Constrains ~72 dimensions. Captures temporal dynamics (temperature trajectory, seasonal patterns). Requires missing timestep masking in the loss.

```python
era5_target = batch.era5_10.flatten(1, 2)  # (B, T, 6) → (B, 72)
era5_pred = self.era5_head(pooled)          # (B, 768) → (B, 72)
# Mask missing timesteps (filled with normalized MISSING_VALUE)
valid = (era5_target > -1.0)  # after normalization, missing values are very negative
loss += F.mse_loss(era5_pred[valid], era5_target[valid]) * weight
```

### Code changes needed
- **Train module** (`contrastive_latentmim.py`): `era5_regression_config`, `nn.Linear` head, MSE in `train_batch`

---

## Key metrics to track

- **InfoNCE loss**: Should NOT increase with training (unlike exp 15)
- **Patch discrimination loss**: Should remain stable
- **Downstream eval** (m-eurosat, m-so2sat, mados, pastis): Climate-aware representations should improve geospatial tasks
- **ERA5 regression MSE** (Exp 3, 5): Per-band prediction accuracy
- **ERA5 alignment loss** (Exp 2, 4): Should decrease over training

## Implementation priority

1. **Exp 1** (`era5_decode_no_mask_neg`) — no code changes needed, ready to run
2. **Exp 3** (`era5_reg_mean`) — simplest new approach: just `nn.Linear(768, 6)` + MSE
3. **Exp 5** (`era5_reg_timeseries`) — same but `nn.Linear(768, 72)` + masked MSE
4. **Exp 2** (`era5_clip_mean`) — needs ERA5 encoder MLP
5. **Exp 4** (`era5_clip_conv1d`) — needs ERA5 encoder with Conv1d
