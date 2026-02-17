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

### Experiment 14: Single bandset S2 (all 12) + random band dropout

```bash
EXPERIMENT=single_bandset_all12_random_band_dropout_cross_random_masked_neg \
  python scripts/vnext/2026_02_08_masked_neg/single_bandset_masked_neg.py launch \
  single_bandset_all12_random_band_dropout_cross_random_masked_neg ai2/jupiter \
  launch.num_gpus=8 \
  'launch.clusters=[ai2/jupiter,ai2/ceres,ai2/titan]' \
  trainer.callbacks.wandb.project=2026_02_08_masked_neg
```

<!-- ## Architecture: Bandset Merge/Unmerge

### Motivation

Switching S2 from 3 bandsets to 1 caused EuroSat performance drop while other tasks were unaffected. The hypothesis is that a single `Conv2d` over all 12 bands loses resolution-aware spatial structure (10m/20m/60m bands have different native resolutions and kernel sizes). The merge approach preserves per-resolution patch embeddings while reducing sequence length.

### Files changed

| File | What |
|------|------|
| `olmoearth_pretrain/nn/bandset_merge.py` | New `BandsetMerge` and `BandsetUnmerge` modules |
| `olmoearth_pretrain/nn/flexi_vit.py` | `Encoder`: `merge_bandsets`, `merge_after_layer` params + mid-layer merge logic; `Predictor`: `unmerge_bandsets` param |
| `olmoearth_pretrain/nn/latent_mim.py` | Sets `target_encoder.merge_enabled = False` after deepcopy |
| `tests/unit/nn/test_bandset_merge.py` | Unit tests for merge/unmerge modules, encoder, predictor, LatentMIM | -->

---

## Exp 14: Masking with single bandset

**Setup:** Single bandset S2 (all 12) + Landsat (all 11) + `modality_cross_random` + band dropout ~Uniform(0, 0.3)

### How `modality_cross_random` works with single bandset

The strategy picks `(modality, bandset_idx)` tuples to encode or decode. With single bandset, the encodable pool is just 3 atomic units:

- `(s2, 0)` — all 12 bands
- `(s1, 0)` — VV, VH
- `(landsat, 0)` — all 11 bands

(6 decode-only modalities: worldcover, srtm, osm, wri_canopy, cdl, worldcereal)

With `min_encoded_bandsets=2` (default), the strategy randomly encodes **2 or 3** of the 3 encodable units per sample. With `allow_encoding_decoding_same_bandset=True`, decoded units are drawn from the full pool of 9.

### What changes vs multi-bandset

| | 3 bandsets S2 / 2 Landsat | 1 bandset S2 / 1 Landsat |
|---|---|---|
| Encodable units | 6 | 3 |
| Encode set size | random 2–6 | random 2–3 |
| Intra-S2 spectral prediction | Yes (e.g. predict 20m from 10m) | **No** — S2 is all-or-nothing |
| Band dropout compensates? | N/A | Partially — forces spectral robustness but unstructured |

### Band dropout & target encoder

Band dropout (zeroing random bands before Conv2d) only applies to the **online encoder**. The target encoder has `band_dropout_rate=0.0` (set in `LatentMIM.__init__`), so it always sees full spectral info. The model learns to match the target's representations despite missing bands. At inference, the online encoder also sees all bands — matching the target encoder's view.

The dropout rate is a per-band drop **probability**, not a fixed proportion — each band is independently dropped with probability `rate`. With `random_band_dropout=True`, `rate ~ Uniform(0, 0.3)` is sampled once per forward call (shared across the batch). This means the model frequently sees near-full bands (when rate≈0) and occasionally heavy dropout (rate≈0.3), covering the inference regime (rate=0) during training. At least 1 band is always kept per sample. Applies to any modality with >1 band (S2, Landsat, S1).
