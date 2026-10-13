# Searchlight inference

Searchlight is an inference-only way of running the v1.3 encoder over a large area
without tiling artifacts. Code: `olmoearth_pretrain/nn/searchlight.py`.

## Why

Tiled inference cuts an area into 16 px windows (the training size), embeds each
window on its own and stitches the results. A pixel's context therefore jumps at
every window seam, which shows up as a grid in the embeddings.

Searchlight runs a whole area in one forward and gives every query its own window:
the box of `16 / patch_size` cells it would see if a 16 px training window were
centred on it, sliding one cell at a time and shifted inward (not shrunk) at the
area's edge. The rule applies to all three attentions of the encoder:

- ViT self-attention: a token attends every token whose cell is in its box;
- AttentionPooler read: a location query attends every token in the box around its cell;
- query self-attention: a location query attends every query whose cell is in that box.

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
        location_query_patch_size=1,    # one embedding per pixel
        crop_px=480,
        overlap_px=32,
        settings=SearchlightSettings(compile=True),
    )                           # [H, W, D] student embeddings (output_key)
```

`embed_domain` runs the area as overlapping crops, with the same meaning as
rslearn's sliding-window inference: each forward covers a `crop_px` square,
neighbouring crops share `overlap_px` pixels, and half of the overlap is dropped
from each side of a crop. One forward is the ordinary encoder call with one extra
argument:

```python
out = encoder(sample, patch_size=2, input_res=10, location_query_patch_size=1,
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

**Crop sizes.**

- `overlap_px`: an output depends on inputs up to `searchlight_reach_px(...)` away
  (112 px for v1.3's 12 ViT layers + 1 AttentionPooler layer at ps2), so an overlap
  of twice that is exact. A smaller overlap is an approximation near crop
  boundaries; to choose one, compare it against the exact overlap on your area.
  Each forward computes `crop^2 / (crop - overlap)^2` times the area it keeps,
  ~1.15x at 480 / 32.
- `crop_px`: as large as memory allows.

**Settings** (`SearchlightSettings`): `neighborhood_attention_size_px` (16, the training window),
`compile` (compile the projection / MLP math) and `backend` (see below).

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

## Correctness

An area exactly one 16 px window wide reproduces the stock forward: in fp32 on CPU
the outputs match to 1e-5 (`tests/integration/nn/test_searchlight.py`). On GPU
under bf16 autocast the remaining difference is rounding from the different
attention kernels and summation order.

## Limitations

- Batch size 1 (one area per forward).
- Missing data per timestep, not per pixel (see above).
- Encoder register tokens are not supported (they are global within a window).
- NATTEN is an out-of-lock dependency on H100.
- Searchlight outputs differ from tiled outputs (different context per pixel); a
  downstream model trained on tiled embeddings should be checked on Searchlight
  ones.
