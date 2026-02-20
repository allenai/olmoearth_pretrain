# Single Bandset Experiments

Experiments investigating S2 bandset configurations and cross-spectral learning strategies with masked-negatives loss.

## Experiments

| # | Key | Description |
|---|-----|-------------|
| 1 | `single_bandset_cross_random_masked_neg` | modality_cross_random + single bandset S2 (all 12) / Landsat |
| 2 | `single_bandset_random_decode_masked_neg` | random_with_decode + single bandset S2 (all 12) / Landsat |
| 5 | `single_bandset_band_dropout_0.3_cross_random_masked_neg` | modality_cross_random + single bandset S2 + band dropout 0.3 |
| 6 | `single_bandset_band_dropout_0.5_cross_random_masked_neg` | modality_cross_random + single bandset S2 + band dropout 0.5 |
| 8 | `two_bandset_cross_random_masked_neg` | modality_cross_random + 2 bandsets S2 (10m+20m) / Landsat single |
| 13 | `single_bandset_random_band_dropout_cross_random_masked_neg` | single bandset S2 (no 60m: 10 bands) + random band dropout ~ Uniform(0, 0.3) |
| 14 | `single_bandset_all12_random_band_dropout_cross_random_masked_neg` | single bandset S2 (all 12) + random band dropout ~ Uniform(0, 0.3) |
| 15 | `single_bandset_all12_random_band_dropout_era5_random_decode_masked_neg` | single bandset S2 (all 12) + random band dropout ~ Uniform(0, 0.3) + ERA5 decode-only + random_with_decode masking |
| 16 | `single_bandset_all12_random_band_dropout_era5_cross_random_masked_neg` | single bandset S2 (all 12) + random band dropout ~ Uniform(0, 0.3) + ERA5 decode-only + modality_cross_random masking |
| 17 | `single_bandset_all12_random_band_dropout_random_decode_masked_neg` | single bandset S2 (all 12) + random band dropout ~ Uniform(0, 0.3) + random_with_decode masking |

## Launch Commands

### Experiment 16: Single bandset S2 (all 12) + random band dropout + ERA5 decode-only + modality_cross_random

```bash
EXPERIMENT=single_bandset_all12_random_band_dropout_era5_cross_random_masked_neg \
  python scripts/vnext/2026_02_08_masked_neg/single_bandset_masked_neg.py launch \
  single_bandset_all12_random_band_dropout_era5_cross_random_masked_neg ai2/jupiter \
  launch.num_gpus=8 \
  'launch.clusters=[ai2/jupiter,ai2/ceres,ai2/titan]' \
  trainer.callbacks.wandb.project=2026_02_08_masked_neg
```

### Experiment 15: Single bandset S2 (all 12) + random band dropout + ERA5 decode-only + random_with_decode

```bash
EXPERIMENT=single_bandset_all12_random_band_dropout_era5_random_decode_masked_neg \
  python scripts/vnext/2026_02_08_masked_neg/single_bandset_masked_neg.py launch \
  single_bandset_all12_random_band_dropout_era5_random_decode_masked_neg ai2/jupiter \
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
