# Lighthouse inference for the register-bottleneck encoder (v1.3 RC, rc_pix512)

Lighthouse replaces tiled inference (16 px crops at overlap 4) with one pass over a
large chunk in which every query attends within its OWN 16 px field of view, centred
on its cell and sliding one cell at a time. Code: `olmoearth_pretrain/nn/lighthouse_rc.py`
(switch: `encoder.lighthouse = RCLighthouseSettings(fov_px=16)`), driver
`scripts/tools/lighthouse_aoi_inference.py` (`lh_ps{P}_h{halo}_c{core}[_q{t}]` configs),
eval option `DownstreamTaskConfig.lighthouse_fov_px` / `lighthouse_retile_px`
(task `pastis_year_aligned_lh16_ps1_sentinel1_sentinel2_landsat`).
Results page: https://claude.ai/artifact/Uoxs4tta9muVHmPnnZHoMh (Lighthouse sections).

## What it does

* ViT self-attention, Perceiver reads and latent self-attention are each restricted
  to the W x W-cell box `[clip(r - W//2, 0, n - W), + W)` around the query's cell
  (W = 16 / patch size). Works for any token patch size and latent stride (v1.3:
  latents at the patch stride; rc_pix512: one latent per pixel).
* MISSING tokens are dropped from the sequence (masks respected for free). In
  production, missingness is per scene (whole timesteps), never per pixel.
* A one-window domain reproduces the stock forward exactly (CPU tests, GPU parity
  cos >= 0.99999). Exact chunk halo = (vit_depth + 1 + latent_depth) * (W//2) * ps +
  max_patch_size = 144 px for the 4-layer Perceiver (128 for 2 layers); in practice a
  16 px halo leaves only a faint chunk edge (ratio 1.08).
* FlexAttention block tables come from the layout geometry. **They must be padded to
  the KV-block width** (`pad_index_width`, default): narrow partial + full tables gave
  wrong outputs on torch 2.9.1 (`scripts/tools/lighthouse_rc_flex_check.py`).

## Results (v1.3 release / random-init rc_pix512, 2026-10-02/03)

**Quality.** Sundarbans (321 km2, ps1): tile-seam ratio 1.84 (tiled) -> 1.01
(Lighthouse), no periodic artifact left. The tile-quantized variant (`fov_quantum=8`,
all-full blocks) just moves the artifact to 8 px steps (1.65-1.77): not useful.

**Accuracy.** PASTIS LP (lr 0.05; the curve is flat 1e-3..0.5): tiled ws16 0.5838 val /
0.5681 test, whole-sample Lighthouse 0.5832 / 0.5635. Null.

**Speed** (s/km2 of output, model forward; tiled incl. 1.77x overlap, Lighthouse incl.
16 px halo; tiled fast_pass is the fair baseline since missing timesteps can be dropped):

| GPU | ps | tiled | Lighthouse flex | Lighthouse + NATTEN (projected) |
|---|---|---|---|---|
| H100 | 4 | 0.075 | 0.109 | 0.091 |
| H100 | 2 | 0.265 | 0.628 | 0.352 |
| H100 | 1 | 1.60 | 4.86 | - |
| A100 | 4 | 0.136 | 0.154 | 0.157 |
| A100 | 2 | 0.492 | 0.862 | 0.881 |
| A100 | 1 | 2.91 | 6.48 | - |

**Why.** torch.profiler (`scripts/tools/lighthouse_rc_torch_profile.py`): outside
attention, Lighthouse costs the same per token as tiling and the GPU is ~95% busy.
Window attention costs 15-20x flash per token; that is the whole gap. NATTEN 0.21.5
(`natten==0.21.5+torch290cu128`, whl.natten.org) is exact for even kernels but 2.4-8.8x
slower than flash at these shapes (worst at ps4); its strided 2D layout is faster on
A100 but half a cell off. No existing kernel fits "small 2D window over cells of 36-72
tokens".

**Ceiling.** With window attention at flash efficiency, Lighthouse would be ~1.45x
faster than tiled at ps4 and ~1.27x at ps2. Getting there needs a custom
FlashAttention-style kernel that walks each window as a few contiguous cell runs.

## Recommendation

Not worth more work unless seam-free maps become a product requirement. If they do,
Lighthouse at ps4 with flex is usable today (+13% on A100, +45% on H100; ~+20% on H100
once NATTEN is wired in for the ViT and latent self-attention).

## Scripts

* `lighthouse_rc_gpu_checks.py` - parity / flex-vs-dense / exact-halo gates + chunk scan
* `lighthouse_rc_flex_check.py` - pure-attention flex vs dense correctness matrix
* `lighthouse_rc_profile.py` - per-phase and kernel-bound timings (v1.3 checkpoint)
* `lighthouse_rc_bench_arch.py` - random-init architecture: tiled vs flex vs NATTEN projection
* `lighthouse_rc_torch_profile.py` - torch.profiler per kernel category and per token
* `lighthouse_natten_check.py`, `lighthouse_natten_sweep.py`, `lighthouse_natten_strided.py`
* `lighthouse_rc_analysis.py`, `lighthouse_rc_render.py` - seams / agreement / viewer panels
