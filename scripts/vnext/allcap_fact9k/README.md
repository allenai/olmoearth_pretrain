# allcap_fact9k

OlmoEarth v1.2 Base trained on the every-capture ("allcap") corpus with factorized
encoder attention and a 9000-real-token budget (joer, started 2026-09-30). The
experiment log, with results and decisions, is
`/root/repos/experimentor/logs/pretrain/2026-09-30-allcap-fact9k.md`.

Branch `joer/allcap-fact9k`, branched from main at fc76d8fef (2026-09-30). Main has
moved on since (38 commits by 2026-10-15); nothing here has been merged.

## Scripts

| Script | What |
|---|---|
| `base.py` | The run: `v1_2/base_faster.py` plus factorized attention, the allcap corpus, the real-token budget and the time-range menu, landsat_l2, in-loop m-eurosat + PASTIS every 5k, 600k steps, every permanent checkpoint kept. |
| `cloudmask.py` | Fork of `base.py` started from the baseline's step 75000: cloudy S2 decoder targets become MISSING. It ran 75k to 100k (2026-10-06 to 10-08) with no measurable effect. |
| `profile_step.py` | `torch.profiler` on a few steps at global batch 64 (1 GPU). Do not rename it `profile.py`, which shadows the stdlib module. |
| `compute_norm_stats.py` | Normalization stats for `landsat_l2` (B1-B7 converted from the S2 L2A computed stats, B10 measured). |

Launch or resume (the same run name resumes the latest checkpoint and the same W&B run):

```bash
cd /root/repos/oe_allcap_fact9k
PYTHONPATH=. /weka/dfive-default/joer/venvs/oe_allcap_fact9k/bin/python \
  scripts/vnext/allcap_fact9k/base.py launch base_allcap_fact9k_1 ai2/jupiter \
  --launch.num_gpus=8 --launch.priority=urgent "--launch.clusters=[ai2/jupiter,ai2/ceres]"
```

The venv was built with `uv sync --locked --extra all-no-flash`, with the uv cache,
Python install and environment all on Weka (`/weka/dfive-default/joer/venvs/`). A
venv in the session scratchpad disappears when the session restarts.

## Code changes (vs fc76d8fef)

- **Every-capture data** (`data/dataset.py`, `per_modality_timestamps=True`):
  - Each h5 stores each modality's own captures with `timestamps_<modality>`.
  - A sample is a random crop, then a random time range (menu of 7 to 1825 days, drawn per microbatch), then every capture in that range.
  - Captures are merged into a union timeline (same-day captures share a step). The 9000-token budget is applied to real tokens only, as a contiguous run of steps.
- **Compact per-modality layout** (4888329c):
  - S2, S1 and Landsat each keep only their own captures, plus `{modality}_time_index` (`[B, T_m]`, -1 means padding) into the union `timestamps` (`datatypes.py`).
  - Masking maps between slots and union steps (`train/masking.py`, `random_time_with_decode`).
  - Month encodings and 3D RoPE read each modality's own dates (`MaskedOlmoEarthSample.modality_timestamps()`, `nn/flexi_vit.py`).
  - Collate pads each modality separately (`data/collate.py`), and `split_masked_batch` trims time padding per microbatch (`train/utils.py`).
  - `test_compact_layout_is_equivalent_to_union_layout` checks that masks, outputs and loss match the old layout.
- **Factorized encoder attention** (`EncoderConfig.attention_mode="factorized"`, `nn/flexi_vit.py`, `nn/attention.py`):
  - Even blocks attend within a location; odd blocks attend within a (modality, bandset, timestep) slot.
  - Packed tokens are regrouped with integer `index_select`/`index_copy` (`TokenGroups`), with per-sample drop path.
- **Speed and memory:**
  - The masked-negatives vec loss scores only decoder tokens, with one host sync (`train/loss.py`).
  - Loader options: `pin_memory=False`, `uint8_masks`, 4 workers x prefetch 1 (`data/dataloader.py`).
  - `expandable_segments` is set in `base.py`.
- **Cloud flags** (b0cb268d):
  - `OlmoEarthDataset(load_s2_cloud_mask=True)` reads `sentinel2_scl` and returns per-pixel flags (SCL 3/8/9/10) as `OlmoEarthSample.sentinel2_l2a_cloud`, which is not a modality.
  - SCL is matched to S2 captures by date, because about 0.3% of files have one extra SCL capture.
  - The collators turn S2 decoder targets that are more than 50% cloud into MISSING.
- **W&B resume across Beaker experiments** (`train/callbacks/wandb.py`): `allow_val_change` on the Beaker config keys.

Tests: `tests/unit/data/test_per_modality_timestamps.py` and `tests/unit/nn/test_factorized_attention.py`.

## Gotchas

- **Permanent checkpoints aren't kept by default.** olmo-core 2.6 keeps only the last 3 permanent checkpoints a job saved (`max_checkpoints=3`). `base.py` sets it to None.
- **Forks can't keep `load_path` set.** The train module reads `<load_path>/config.json` on every save and load, so `cloudmask.py` sets it only on the first start.
- **A crash mid-save leaves a partial `stepN/`.** The resumed job then dies with FileExistsError when it reaches that step. Rename the partial directory first.
- **Microbatch 16 runs out of memory** in decoder attention, so stay at 8.
- **All-timestep PASTIS** is evaluated outside this repo, with earthy branch `joer/oe-eval-s1` (`scripts/goal_diagnostics_olmoearth.sh`). See the experiment log's operating notes.
