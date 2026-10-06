# Searchlight inference

Searchlight is an inference-only way of running the v1.3 encoder (ViT + Perceiver,
including the per-pixel-latent `pix512` models) over a large area without tiling
artifacts. Code: `olmoearth_pretrain/nn/searchlight.py`.

## Why

Tiled inference cuts an area into 16 px windows (the training size), embeds each
window on its own and stitches the results. A pixel's context therefore jumps at
every window seam, which shows up as a grid in the embeddings.

Searchlight runs a whole area in one forward and gives every query its own window:
the box of `16 / patch_size` cells it would see if a 16 px training window were
centred on it, sliding one cell at a time and shifted inward (not shrunk) at the
area's edge. The rule applies to all three attentions of the encoder:

- ViT self-attention: a token attends every token whose cell is in its box;
- Perceiver read: a latent attends every token in the box around its cell;
- latent self-attention: a latent attends every latent whose cell is in that box.

An area exactly one window wide reproduces the stock forward. Larger areas differ
from tiled inference by design: no pixel sits at a seam.

## How to run

```python
from olmoearth_pretrain.nn.searchlight import SearchlightSettings, embed_domain

encoder = model.encoder.cuda().eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    emb = embed_domain(
        encoder,
        sample,                 # MaskedOlmoEarthSample, batch size 1, the whole area
        patch_size=2,
        latent_patch_size=1,    # one embedding per pixel (pix512 models)
        core_px=448,
        halo_px=16,
        settings=SearchlightSettings(compile=True),
    )                           # [H, W, D] student embeddings (output_key)
```

`embed_domain` cuts the area into cores of `core_px`, runs one Searchlight forward
per core plus `halo_px` on every side, and keeps only the cores. One forward is the
ordinary encoder call with one extra argument:

```python
out = encoder(sample, patch_size=2, input_res=10, latent_patch_size=1,
              searchlight=SearchlightSettings())
```

**Input.** The usual `MaskedOlmoEarthSample` of raw bands with batch size 1:
`sentinel2_l2a` `[1, H, W, T, C]` (likewise S1 / Landsat and their masks) and
`timestamps` `[1, T, 3]`. `H` and `W` are any multiples of the patch size, at least
16 px. Batching is replaced by size: one 480 px forward at ps2 is ~2M tokens.

**Missing data** must be whole timesteps, per modality: every cell must keep the
same number of tokens. A timestep missing in S1 but present in S2 is fine; a
timestep missing in only part of the area raises an error. rslearn exports drop
whole timesteps, so this holds for them.

**Chunk sizes.**

- `halo_px`: the exact reach of an output is `searchlight_reach_px(...)` (120 px for
  12 ViT + 2 Perceiver layers at ps2), but 16 px is indistinguishable in practice
  (measured on v1.3 at ps1: per-pixel cosine p01 0.9999 vs the exact halo; 32 px is
  identical to five decimals). Each chunk processes `(core + 2 * halo)^2 / core^2` of its area, ~1.15x
  at 448 / 16.
- `core_px`: as large as memory allows; at 448 px the original implementation
  peaked at ~38 GB at ps2 with S1 + S2 + Landsat x 12 months.

**Settings** (`SearchlightSettings`): `neighborhood_attention_size_px` (16, the training window),
`compile` (compile the projection / MLP math, ~1.3x), `backend` (see below),
`tokens_per_call` (memory only) and `origin_px` (set by `embed_domain`).

## Setup

The attention kernel depends on the GPU (`backend="auto"`):

- **H100 (Hopper and newer): NATTEN.** Not a declared dependency; install the
  prebuilt wheel matching your torch and CUDA from https://whl.natten.org, e.g.
  `natten==0.21.5+torch290cu128` for the locked torch 2.9.1. The module docstring
  of `searchlight.py` lists the constraints we hit (upgrade torch and torchvision
  together; our H100 nodes' driver 570 cannot load CUDA 13 builds).
- **A100 and older: FlexAttention**, built into torch; NATTEN is not needed (its
  Ampere kernels are slower than flex).
- **CPU:** a dense masked reference, for tests and small checks only.

## Results

All numbers are for `v1_3_rc_ld2_pix512` at step 320k (2 Perceiver layers, mid
training), patch size 2, latent patch size 1, `core_px=448`, `halo_px=16`,
`compile=True`, bf16 autocast, unless stated otherwise. Speeds are model forward
only (no data loading or writing), per km^2 of output.

### Correctness

Checked against a real checkpoint on AOI windows of 1536-1792 px (H100:
Sundarbans, Georgia pine, Seattle; A100: Georgia pine, Seattle):

| Check | H100 (NATTEN) | A100 (FlexAttention) |
|---|---|---|
| One 16 px window vs the stock forward, per-pixel cosine (min) | 0.99998 | 0.99997 |
| Whole windows vs the original Searchlight implementation, int8 cosine (mean / min) | 0.99996 / 0.9988 | 0.99994 / 0.9984 |

The residual difference is bf16 rounding (different attention kernels and
summation order); in fp32 on CPU the one-window forward matches the stock forward
to 1e-5 (`tests/integration/nn/test_searchlight.py`).

### Speed

| GPU | Window | Searchlight (this code) | Original Searchlight implementation |
|---|---|---|---|
| H100 | Sundarbans | 0.201 s/km^2 | 0.207 |
| H100 | Georgia pine | 0.163 | 0.168 |
| H100 | Seattle | 0.204 | 0.210 |
| A100 | Seattle | 0.660 | 0.674 (same node) |
| A100 | Georgia pine | 0.424 | 0.426 (same node) |

Over all 16 AOI windows on H100 the original implementation averaged 0.203 s/km^2
for this model, 0.187 for the 1-layer `ld1_pixtgtpool`.

For reference, on the Seattle window with the 2-layer architecture: tiled
inference at ps2 (16 px windows, overlap 4, whole-encoder `torch.compile`) runs at
0.137 s/km^2 on H100 (0.122 with torch 2.11) and 0.290 on A100, so Searchlight costs
~1.5x tiled at ps2 on H100 (0.204 vs 0.137) and ~2.3x on A100. The production v1.3 run (rslp worker
`patrickj/global-2025-v2`, Beaker job 01M3XSS3: ps1, overlap 4, compiled, batch
128, H100) takes ~1.06 s/km^2 to predict, of which ~0.93 is the encoder forward.

### Quality

- **Seams:** the tile-seam ratio (mean |difference| between neighbouring pixels at
  seams over the same at comparable non-seam positions; 1.00 = no seam) is 1.51
  (median over 16 AOIs) for production tiled v1.3 and 1.00-1.01 for every
  Searchlight run.
- **PASTIS** (v1.3 RC, ps1, val mIoU, original implementation): tiled 0.5838,
  Searchlight 0.5832, a null.
- **Token lattice at ps2:** with 2 px tokens a periodic 2 px pattern remains (the
  token grid, not the tiles), measured the same way as the seam ratio: median 1.74
  (`ld2_pix512`) down to 1.27 (`pixtgt`) at step 320k. Pixel MIM targets reduce it in every
  window.

The AOI comparison (panels, per-window seam and lattice ratios, timings) is at
https://claude.ai/artifact/6U4sRmWiHqPnCxLD26wLxJ (made before the rename, so it
calls Searchlight "Lighthouse").

## Limitations

- Batch size 1 (one area per forward).
- Missing data per timestep, not per pixel (see above).
- Encoder register tokens are not supported (they are global within a window).
- NATTEN is an out-of-lock dependency on H100.
- Searchlight outputs differ from tiled outputs (different context per pixel); a
  downstream model trained on tiled embeddings should be checked on Searchlight
  ones.
