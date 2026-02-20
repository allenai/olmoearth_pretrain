# Training Speed Optimizations

Scripts in this folder are the starting point for new runs that use the full set of
training speed improvements. Use these instead of `scripts/official/` for any new
experiment that doesn't need to resume an old checkpoint.

## What's enabled

| Optimization | Config knob | Impact |
|---|---|---|
| Flash Attention | `encoder_config.use_flash_attn=True`, `decoder_config.use_flash_attn=True` | ~20-30% attention speedup on A100/H100 |
| Linear patch embed | `encoder_config.use_linear_patch_embed=True` | Replaces Conv2d with `nn.Linear` (reshape + cuBLAS GEMM). Faster for small channel counts on all GPU types. **Checkpoints from this path are not compatible with `use_linear_patch_embed=False`.** |
| torch.compile | `train_module.compile_model=True` | Fuses kernels across transformer blocks. Best gains on H100/B200. |
| Vectorized loss | `modality_patch_discrimination_new_vec` | Eliminates per-sample Python loops and GPU→CPU syncs in the discrimination loss. |
| Sync-free unmask | (always on) | `Encoder.unmask` uses scatter ops instead of boolean indexing, removing a GPU sync per step. |

## Checkpoint compatibility

Old checkpoints (trained with `scripts/official/`) used `nn.Conv2d` for patch
projection. The weight layouts differ, so they **cannot be loaded directly** into a
model built from this script.

To convert an old Conv2d patch embed weight to the Linear layout:
```python
# conv_weight: [out_chans, in_chans, p_h, p_w]
linear_weight = conv_weight.permute(0, 2, 3, 1).reshape(out_chans, -1)
```

To resume an old checkpoint without migrating weights, use the official script with
`--model.encoder_config.use_linear_patch_embed=False` (already set as default there).

## Scripts

| Script | Train module | Views/step | Contrastive |
|---|---|---|---|
| `base_speedup.py` | `ContrastiveLatentMIM` | 2 | Yes (InfoNCE) |
| `base_speedup_latent_mim.py` | `LatentMIM` | 1 | No |

`base_speedup_latent_mim.py` is ~2x cheaper per step. Use it to ablate the value of the
contrastive term, or when GPU memory is tight.

## Launch

```bash
# Dry run first
python scripts/vnext/speedups/base_speedup.py dry_run base_speedup_test local

# Contrastive (2-view)
python scripts/vnext/speedups/base_speedup.py launch base_speedup ai2/jupiter \
    --launch.num_gpus=8 \
    --launch.clusters=[ai2/ceres,ai2/jupiter,ai2/titan] \
    --trainer.callbacks.wandb.project=YYYY_MM_DD_speed_optimizations

# Non-contrastive (1-view, single forward pass)
python scripts/vnext/speedups/base_speedup_latent_mim.py launch base_speedup_latentmim ai2/jupiter \
    --launch.num_gpus=8 \
    --launch.clusters=[ai2/ceres,ai2/jupiter,ai2/titan] \
    --trainer.callbacks.wandb.project=YYYY_MM_DD_speed_optimizations
```
