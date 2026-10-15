# clipearth

Align text with OlmoEarth v1.3 embeddings, so a text query can be matched to locations by
cosine similarity.

## Data

Molmo's open-set segmentation set (`open_set_seg` in mm_olmo's remote-sensing mixture):
48,975 train and 2,655 val windows from 248 source datasets. Each is an rslearn window of
128x128 px at 10 m over one year, with monthly Sentinel-2 L2A (12 bands), Sentinel-1 and
Landsat, plus a per-pixel class mask. Text per class: its name and the short and detailed
visual descriptions in `molmoearth_categories.json` (`seg_open_set_visdesc`).

## Pipeline

1. **Embed** (GPU, Beaker): `scripts/clipearth/launch_embed.sh` runs `clipearth.embed`.
   Each window is cut into 16 px tiles (v1.3's training window), encoded, and the
   AttentionPooler location tokens (768-d) and student tokens (128-d) are pooled into
   - one embedding per labeled class region (the CLIP training pairs), and
   - one embedding per 16 px tile with its lat/lon (the search index).

   The checkpoint is the v1.3 release candidate converted to the current
   `AttentionPooler` layout with `scripts/official/v1_3/convert_legacy_checkpoint.py`.
2. **Train** (CPU is enough): `python -m olmoearth_pretrain.clipearth.train --emb_dir
   <embeddings> --out_dir <dir>`. Needs `sentence-transformers` installed in the project
   environment (`uv pip install sentence-transformers`). Trains a projection on each side
   with a frozen text encoder and reports zero-shot accuracy and retrieval mAP on Molmo's
   val windows and on held-out source datasets.

## Notes

- `embed.py` replaces the encoder's `_maybe_get_attn_mask` on its instance. The stock
  method drops the attention mask in eval, so tiles with different numbers of missing
  tokens in one batch attend to padding and their embeddings depend on their batch.
- Lower-resolution band groups (Sentinel-2 20/40 m, Landsat 20 m) are upsampled to 10 m
  by repeating pixels. Sentinel-1 is converted from linear to dB.
