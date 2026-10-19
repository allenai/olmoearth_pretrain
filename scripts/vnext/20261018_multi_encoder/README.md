# Multi-encoder arms (2026-10-18)

Does one shallow encoder **tower per input modality** (S2, S1, Landsat), fused through the
v1.3 AttentionPooler, match the single shared encoder, while making the model extensible
(a new modality = a new tower against a frozen location grid)?

Model code: `olmoearth_pretrain/nn/multi_encoder.py` (`MultiEncoder`, `MultiEncoderConfig`),
masking: `modality_dropout` in `olmoearth_pretrain/train/masking.py`.

`base.py` is the v1.3 release recipe rebuilt on main's API (the v1.3 scripts on
`gabi/allcap-slot-narrow` predate the Perceiver → AttentionPooler rename). All arms share
it and differ only in the encoder and, for `*_moddrop`, the masking.

| file | encoder | masking | notes |
|---|---|---|---|
| `control.py` | shared 12 blocks | v1.2 `random_time_with_decode` | the v1.3 recipe |
| `towers_t4_f8.py` | 3 × 4-block towers → 8 shared | v1.2 | equal depth per token (20 block-params) |
| `towers_t4_f0.py` | 3 × 4-block towers → pooler only | v1.2 | equal params (12), fusion only in the reads |
| `towers_t8_f4.py` | 3 × 8-block towers → 4 shared | v1.2 | equal depth per token, 324M; mid-point of t_k+f_(12−k) |
| `control_moddrop.py` | shared 12 blocks | + whole-modality dropout p=0.3 | isolates the masking effect |
| `towers_t4_f8_moddrop.py` | 3 × 4 → 8 | + whole-modality dropout p=0.3 | towers with the structured asymmetry |
| `smoke_nano.py` | nano, 3 × 2 → 2 | + dropout | 1-GPU functional check, not an arm |

Open questions these arms answer: (1) does early cross-modal attention matter (`t4_f8` vs
`t4_f0` vs control), (2) does whole-modality dropout help or hurt, and (3) with towers,
does it replace fine masking. The post-hoc-addition test (pretrain without Landsat, add a
Landsat tower against a frozen grid) is the follow-up once an arm is chosen.

## Running

```bash
# dry run (prints the config)
python scripts/vnext/20261018_multi_encoder/towers_t4_f8.py dry_run me_t4f8 local

# Beaker (clean tree, pushed)
python scripts/vnext/20261018_multi_encoder/towers_t4_f8.py launch me_t4f8 ai2/jupiter \
    --launch.num_gpus=8 --launch.priority=urgent --launch.clusters='[ai2/jupiter,ai2/ceres]'
```

Per the v1.3 convention the architecture is baked into each file (`MODULE_PATH`), not passed
as CLI overrides.

## Compute (MACs, base size, `flops.py`, 2026-10-18)

Training forward = online encoder + pooler + decoder + heads + projection-only target, under the
v1.3 masking (which encodes 2 of the 3 input bandsets at 50% and holds the third out: 1/3 of the
S1+S2+L8 tokens are visible, and one tower is empty for every sample). A step is about 3x this.

| shape (patch grid, T, p) | control | towers t4+f8 | towers t4+f0 |
|---|---|---|---|
| hw9 T12 p4 (2916 tok) | 172 G | 169 G (98%) | 103 G (60%) |
| hw16 T4 p4 (3072 tok) | 239 G | 236 G (99%) | 165 G (69%) |
| hw32 T1 p4 (3072 tok) | 551 G | 547 G (99%) | 476 G (87%) |
| hw4 T12 p4 (576 tok) | 31 G | 31 G (100%) | 20 G (63%) |

Inference = unmasked encoder on S1+S2+L8 (location tokens + patch tokens):

| shape | control | towers t4+f8 | towers t4+f0 |
|---|---|---|---|
| 16x16 T12 p1 / 64x64 T12 p4 (9216 tok) | 2449 G | 2101 G (86%) | 536 G (22%) |
| 128x128 T1 p4 (3072 tok) | 538 G | 500 G (93%) | 210 G (39%) |
| 32x32 T1 p4 (192 tok) | 22 G | 22 G (99%) | 10.5 G (48%) |

The control's 2449 G at 16x16/T12/p1 is the number recorded for the v1.3 RC, so the counter is
anchored. Per block over N visible tokens: linear 12·N·d², attention 2·N²·d; a tower layer costs
the same linear work as a shared layer and 1/3 of its attention.
