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
