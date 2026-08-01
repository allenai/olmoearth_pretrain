# v1.2 official runs

Everything here trains the **v1.2 recipe** — v1.1 (hidden patch-embed projection) plus
the winning mixed 3D RoPE config from the temporal-RoPE sweep (`rope_3d_mixed`,
`rope_mixed_base=10`, `rope_temporal_coordinate_scale=1/30` ≈ months). Those knobs are
baked into `base.py`; nothing below overrides them.

Two largely separate efforts share the directory:

- **Size sweep** (`nano`/`tiny`/`small`/`large` + `base`) → wandb
  `2026_06_12_v1_2_size_sweep`.
- **Spatial register bottleneck** (everything named `regbtl_*`) → wandb
  `2026_07_02_perceiver`. This is the active program and the bulk of this README.

## Conventions

- **One script per run.** Architecture is baked into the script rather than passed as a
  CLI override, because the in-loop eval Beaker jobs re-import `MODULE_PATH` to
  reconstruct the matching model. A CLI-only override would give the eval job a
  different model than the one being trained. Dataloader-only knobs are safe to override
  in principle, but we still bake them in so a run is fully described by its script.
- **`MODULE_PATH` must match the file's own path.** It is what the eval job imports; a
  stale value silently evaluates the wrong architecture.
- **In-loop evals run as separate non-blocking Beaker jobs** via
  `add_loop_eval_beaker_job`, and log to a companion wandb run suffixed
  `_loop_evals`. Eval metrics live there, *not* in the training run.
- **Launch is a shell script per experiment batch**, and requires a clean tree pushed to
  the remote (`allow_dirty=False`; the Beaker job clones `$GIT_REF`). Commit and push
  before launching.
- **wandb run names carry a `_vN` suffix** when a run was relaunched after a crash or an
  OOM (`newsamp_v3` … `newsamp_v6`). The script is unchanged across those; only
  memory/launch settings moved. Prefer the highest `_vN` when reading results.

## Decoding the names

`regbtl_v1_2_gdyn_d128_il_pdproj_noic_lsa_wideread_regsup_ndvi_latlon_w0p1_tanchor` is
built from these fragments:

| fragment | meaning |
|---|---|
| `regbtl` | spatial register bottleneck (Perceiver-style read into a register grid) |
| `small` | encoder/decoder size preset `small_shallow_decoder` (384-d, depth 12, 6 heads) at the size sweep's LR (2e-4); absent means the v1.2 base encoder |
| `gdyn` | dynamic single-latent register grid (`register_grid_size=0`) |
| `il` | interleaved reads — `[read → self] × 4` |
| `pdproj` | per-depth read projections |
| `d128` … `d768` | `register_dim` (register storage width) |
| `ic` / `noic` | InfoNCE instance-contrastive loss on / off |
| `lsa` / `nolsa` | bottleneck latent self-attention on / off |
| `1fwd` | one forward pass per batch instead of two (valid once contrastive is off) |
| `fusedadamw` | fused AdamW kernel |
| `faster` | all validated speedups: 1fwd + fused AdamW + projection-only target + replicated DP + bf16 |
| `wideread` | bottleneck attention decoupled to **encoder** width (`register_attn_dim` = `embedding_size`, 768 at base / 384 at small); `register_dim` becomes pure storage width |
| `regsup` | register-grid supervision (auxiliary decode heads on the register grid) |
| `latlon` | latlon regression arm — a supervised **target**, never a model input |
| `ndvi` | time-conditioned NDVI arm: MLP on `[register_cell ; φ(day_of_year)]` regresses each cell's NDVI at every observed timestep |
| `w0p1` | supervision `base_weight=0.1` (10× the original 0.01) |
| `tanchor` | `register_temporal_anchor="year_start"` — reads use axial 3D RoPE anchored at Jan 1 of the sample's first observation year; the register grid itself stays a time-free 2D map |
| `newsampling` / `newsamp` | decorrelated (grid, timestep) shape sampler — see below |
| `psuniform` / `ps1heavy` / `ps1only` | patch-size distribution variants of the newsampling recipe — P(ps=1) = 0.125 / 0.70 / 1.00 |
| `b<N>` | `token_budget = N` (e.g. `b1536`); absent means the committed 3072 |
| `tb<N>` | `temporal_bias = N` (e.g. `tb0`, `tb6`); absent means the committed 2.75 |

Note that older scripts spell out `il_pdproj_noic_lsa_wideread` while the newsampling-era
scripts abbreviate to just `wideread`. They mean the same architecture.

## The program, in the order it happened

### 1. Bottleneck architecture sweep — `launch_regbtl_v1_2_sweep.sh`

**Motivation:** frozen multimodal fusion. Establish whether a Perceiver-style register
bottleneck helps, and pick its shape.

- `regbtl_v1_2_gdyn_d768_il_pdproj_{ic,noic}_{lsa,nolsa}.py` — the 2×2 over instance
  contrastive and latent self-attention. `noic_lsa` won.
- `..._noic_lsa_1fwd.py`, `..._1fwd_fusedadamw.py` — throughput only; with contrastive
  off the second forward pass is dead work. Loss-equivalent to `noic_lsa`.
- `regbtl_v1_2_gdyn_d{128,256,512}_il_pdproj_noic_lsa_faster.py` — register-width sweep
  with all speedups. Tests how much register storage width actually matters.
- `regbtl_v1_2_gdyn_d{128,256,512}_il_pdproj_noic_lsa_wideread.py` — same widths, but
  with read attention at encoder width. **Result: `wideread` makes narrow registers
  competitive**, so `d128 wideread` becomes the workhorse — cheap storage, full-width
  reads.

### 2. Register supervision — same launch script

**Motivation:** the bottleneck's registers were not obviously learning a useful spatial
map; add auxiliary decode heads to force content into them.

- `..._wideread_regsup.py`, `..._wideread_regsup_latlon.py` — supervision on / with the
  latlon arm.
- `regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_regsup{,_latlon}.py` — full-width twins.
- `w0p1` variants raise `base_weight` 0.01 → 0.1 after the original weight proved too
  weak to change anything.

> **Gotcha:** the plain old-sampling `w0p1` baselines — wandb runs `w0p1` /
> `latlon_w0p1`, which most later A/Bs compare against — have **no script in this
> directory**. They were added by `dcc131a93` and then deleted by `3935725ae`
> ("Remove the srtm terrain … and their sweep scripts") as collateral. Recover them from
> `git show 2037272d8:scripts/official/v1_2/regbtl_v1_2_gdyn_d128_il_pdproj_noic_lsa_wideread_regsup{,_latlon}_w0p1.py`
> if you need to relaunch or reconstruct that arm. Only the `_tanchor` and
> `_newsampling` w0p1 scripts survive here. (Stale `__pycache__/*.pyc` files for the
> deleted scripts are still on disk — they are not usable configs.)

### 3. Temporal anchor — `launch_regbtl_v1_2_tanchor.sh`

**Motivation:** the frozen ps=1 PASTIS embedding evals looked bottlenecked on
phenology-defined classes, and the register read was time-blind (the temporal coordinate
was simply sliced off). `tanchor` gives the *read* temporal geometry so heads can learn
season-selective read patterns, while keeping the register grid a time-free 2D map so
the decoder, regsup, and eval contracts are unchanged.

Four runs — `{regsup, regsup+latlon} × {old sampling, newsampling}`:

- `regbtl_v1_2_gdyn_d128_il_pdproj_noic_lsa_wideread_regsup{,_latlon}_w0p1_tanchor.py`
- `regbtl_v1_2_gdyn_d128_wideread_regsup{,_latlon}_w0p1_tanchor_newsampling.py`

### 4. Time-conditioned NDVI — `launch_regbtl_v1_2_tanchor_ndvi.sh`

**Motivation:** a stronger version of the same idea. Rather than only giving the read
temporal geometry, *force* each time-free register cell to store its own NDVI
trajectory, decodable given a time query — exactly the property a frozen ps=1 phenology
probe needs. NDVI is derived in the dataset from raw S2 L2A B04/B08 and is decode-only,
never a model input.

- `..._wideread_regsup_ndvi{,_latlon}_w0p1_tanchor.py` (old sampling — note NDVI *does*
  count against the token budget here)
- `..._wideread_regsup_ndvi{,_latlon}_w0p1_tanchor_newsampling.py` (newsampling excludes
  it automatically via `exclude_only_decode_from_budget`)

### 5. Sampling — `launch_regbtl_v1_2_newsampling.sh`

**Motivation:** the old sampler derived timestep count as "whatever fits the budget for
the sampled grid", perfectly anti-correlating grid size with sequence length, so
large-grid × full-year shapes never occurred. The new sampler decorrelates the two axes.
All knobs live in `regbtl_v1_2_newsampling_common.py`; it changes ~6 things at once:

1. timesteps sampled independently of the grid (`time_priority_prob=0.5`)
2. biased toward full sequences (`temporal_bias=2.75`)
3. token floor `min_tokens_per_instance=228` (drops degenerate tiny shapes)
4. **ps=1 oversampled to 0.40** (vs the uniform 0.125 default)
5. token budget 2250 → 3072, decode-only maps excluded from the budget
6. larger grids reachable (`sampled_hw_p` up to 32)

- `regbtl_v1_2_gdyn_d128_wideread_regsup{,_latlon}_w0p1_newsampling.py`

### 6. NDVI without the anchor — `launch_regbtl_v1_2_ndvi_newsampling.sh`

**Motivation:** every NDVI run so far also carried `tanchor`, so the two were
confounded. These isolate the NDVI arm under newsampling.

- `regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi{,_latlon}_w0p1_newsampling.py`

### 7. Patch-size sweep — `launch_regbtl_v1_2_ps_sweep.sh`

**Motivation:** see the findings below — the newsampling gain looks like patch-size
reallocation rather than temporal exposure. These runs plus the committed `_newsampling`
run form a four-point sweep in P(ps=1) with everything else fixed:

| script | P(ps=1) | in-loop evals |
|---|---|---|
| `..._w0p1_newsampling_psuniform.py` | 0.125 (uniform, the dataloader default) | standard |
| `..._w0p1_newsampling.py` | 0.40 (committed newsampling) | standard |
| `..._w0p1_newsampling_ps1heavy.py` | 0.70 | standard |
| `..._w0p1_newsampling_ps1only.py` | 1.00 | **ps=1 only** |

`ps1only` is deliberately asymmetric. It trains at ps=1 exclusively, so it calls
`set_ps1_only_loop_evals` instead of `add_loop_eval_beaker_job`: the eval set is
*replaced* with the 18 ps=1 tasks (2 PASTIS ps=1 exports + 8 AEF supplemental datasets ×
{linear probe, kNN}) and the shared ps=4 catalog is dropped, since those evals would be
probing a resolution the model never trained at. **Consequence: `ps1only` is comparable
to the other three arms only on the ps=1 metrics** — it has no ps=4 numbers at all, by
construction.

Restricting evals means editing `evaluator.tasks` in the training script's
`build_trainer_config`; there is no CLI or env-var filter for the in-loop path. The eval
job reads that dict back out via `checkpoint_sweep_evals.get_train_run_eval_tasks`.

### 8. Shape sweep: token budget × temporal bias — `launch_regbtl_v1_2_shape_sweep.sh`

**Motivation:** with patch-size oversampling ruled out (§7 results below) and budget 6144
worth only ~+0.005, the temporal knobs are the last untested part of the newsampling
bundle. A 2×2 on the v6 + uniform-patch-size base:

| | bias 0 | bias 2.75 | bias 6 |
|---|---|---|---|
| **budget 1536** | — | `..._psuniform_b1536` (~45h) | `..._psuniform_b1536_tb6` (~45h) |
| **budget 3072** | `..._psuniform_tb0` (~90h) | `..._psuniform` *(running)* | `..._psuniform_tb6` (~90h) |
| **budget 6144** | — | `newsamp_v5_b32` *(running, ps=0.40)* | — |

Two arms are free: the 3072/2.75 centre is the running `psuniform` run, and budget 6144
is already covered by `newsamp_v5_b32`, which pairs cleanly with `newsamp_v6` (identical
`patch_size_probs`) to isolate the budget effect. That measured only ~+0.005, which is why
no uniform-ps 6144 arm is launched — it would cost ~7.5 days.

**Why a 2×2 and not two independent arms.** The axes interact. `token_budget` sets how
wide the feasible timestep window is at each grid size; `temporal_bias` only skews the
draw *within* that window, so it cannot create tokens — at fixed budget it trades spatial
extent for temporal extent. With the token floor at 228 and decode-only maps excluded
(cost `3·hw²·t`):

| budget | full year (t=12) reachable at | hw=16 — the ws16 ps=1 **eval** shape |
|---|---|---|
| 1536 | hw ≤ 6 | t ≤ 2 |
| 3072 | hw ≤ 9 | t ≤ 4 |
| 6144 | hw ≤ 13 | t ≤ 8 |

So `b1536_tb6` — the cheap-and-fast corner, and the one with a real payoff since bias is
free while budget is linear in cost — may lose *specifically* because it never trains
near the large-grid-and-long-sequence regime the frozen ps=1 probes evaluate at. That
failure mode would itself explain the small 6144 gain, so a negative result is nearly as
informative as a positive one. `b1536` exists only to decompose that corner: without it,
a good result can't be attributed to the cheap budget being harmless versus the bias
compensating.

**Duration is deliberately untouched.** `epochs(300)` = 662,700 steps at *every* budget,
because an epoch is a fixed number of *instances* (`total_batches = instances /
global_batch_size`) and the sampler changes each instance's shape, not how many there
are. Every arm therefore shares one LR schedule and stays comparable to every committed
300-epoch run. Do **not** switch to a steps- or tokens-based duration to "compute-match":
step counts would diverge from the committed runs, and `Duration.tokens` is unusable here
because `Trainer.tokens_per_batch` returns `global_batch_size` (512 *instances*, not
tokens).

## Findings snapshot (2026-07-25, preliminary — single seeds)

**Newsampling's gain is concentrated on ps=1 evals and is mildly negative elsewhere.**
Mean delta (newsampling − old sampling) across the four matched `tanchor`/`ndvi` arms at
matched checkpoints:

| eval | eval patch size | mean Δ |
|---|---|---|
| PASTIS ws16 ps1 (S2) | 1 | **+0.025** |
| PASTIS in-loop | 4 | +0.008 |
| mados | 4 | +0.014 |
| yemen_crop | 4 | +0.011 |
| eurosat | 4 | −0.009 |
| geo_ecosystem | 4 | −0.011 |
| fifty_cities (S2) | 4 | −0.013 |
| so2sat | 4 | −0.021 |

The cleanest control is that PASTIS is evaluated at both resolutions — same dataset,
same labels, same time series, both mean-pooled linear probes. The ps=1 export gains
+0.025 while the ps=4 probe moves +0.008. If full-sequence exposure (`temporal_bias`)
were the driver, the ps=4 probe should have gained too. `patch_size=4` is the
`DownstreamTaskConfig` default, so every in-loop eval except the `ws16_ps1` exports runs
at ps=4.

Mechanism that fits: ps=1 went 0.125 → 0.40 (3.2× more exposure) and ps=4 went
0.125 → 0.10. The eval deltas track that reallocation.

**Corollary worth taking seriously:** at matched sampling, the architectural arms do
almost nothing on ps=1 PASTIS. At 140k, no-tanchor 0.5221 vs tanchor 0.5225; with latlon
0.5296 vs 0.5305. NDVI adds ~+0.003, within noise. The metric that motivated both
`tanchor` and the NDVI arm is being moved by a dataloader knob, not by the read's
temporal geometry.

## Update (2026-07-27) — two hypotheses above are now settled

**1. Patch-size oversampling is NOT what buys the newsampling gain.** The §7 sweep
returned. At ckpt 260k, PASTIS ps=1 (S2):

| P(ps=1) | 0.125 | 0.40 | 0.70 | 1.00 |
|---|---|---|---|---|
| ps=1 (S2) | 0.5484 | 0.5521 | 0.5488 | 0.5350 |
| ps=4 | 0.5110 | 0.5023 | 0.4967 | *(evals restricted)* |

Flat from 0.125 to 0.70, dropping at 1.00, while ps=4 degrades monotonically as P(ps=1)
rises. **Uniform is the better default** — tied-best on ps=1, clearly best on ps=4 —
which is why `UNIFORM_PATCH_SIZE_PROBS` exists and why §8 sweeps on the uniform base.
Note `ps1only` is the *worst* ps=1 arm: patch-size diversity is useful regularization even
for the deployment resolution. This refutes the "ps=1 bias owns the gain" hypothesis
stated in the `_psuniform` docstring and above.

**2. tanchor and NDVI are null; both were demoted to `high` priority.** Pooled across 8
matched pairs (both samplers × both latlon settings), over every overlapping checkpoint:

| metric | pooled Δ | per-pair range |
|---|---|---|
| PASTIS ps=1 (S2) | −0.0003 | −0.008 .. +0.005 |
| PASTIS ps=1 (S1+S2) | −0.0016 | −0.009 .. +0.004 |
| PASTIS ps=4 | −0.0068 | −0.019 .. +0.001 |

The sign flips between the latlon and no-latlon versions of the *same* comparison
(tanchor+newsampling is −0.0055 without latlon, +0.0048 with), which is a noise signature
rather than a small effect; magnitudes sit inside the checkpoint-to-checkpoint scatter
(sd 0.004–0.008). The one genuinely non-null result is **tanchor under old sampling, which
is harmful**: −0.019 on ps=4 with 0 of 31 checkpoints positive.

**3. Token budget: real but poor value.** `newsamp_v5_b32` (6144) vs `newsamp_v6` (3072),
identical `patch_size_probs`, at matched steps: +0.005 on ps=1 and +0.005 on ps=4 — and
notably it lifts *both* resolutions rather than trading them, unlike the patch-size axis.
But 6144 uses exactly 2.00× tokens/step (196,608 vs 98,304 per device) at half the
throughput (BPS 1.03 vs 2.04), so it is +0.005 for 2× compute. Beware comparing arms at
matched *tokens* (6144@N vs 3072@2N): with a shared `epochs(300)` horizon those points sit
at different places in the cosine decay, which flatters the longer-step run.

**4. Longer training doesn't help.** `ep600` vs its true 300-epoch twin at the same
supervision weight: +0.005 (regsup) and +0.003 (latlon) on ps=4 — flat for 2× compute.
(Comparing `ep600` against `w0p1` instead makes it look actively harmful, but that is
confounded by supervision weight.) Since tokens = epochs × instances × tokens-per-instance
and tokens-per-instance scales with budget, **3072 @ 600 epochs and 6144 @ 300 epochs cost
the same** — so this is the reference point for whether richer context per sample beats
more samples.

What remains unexplained: the temporal knobs (`temporal_bias`, `time_priority_prob`) plus
the budget bump 2250 → 3072. That is what §8 tests.

## Shared modules (not runnable)

| file | contents |
|---|---|
| `base.py` | v1.2 baseline config: RoPE/temporal knobs, `token_budget=2250`, patch sizes 1–8 |
| `base_faster.py` | `base` + all validated speedups (~1.35×) |
| `regbtl_v1_2_common.py` | register-bottleneck model builder; `add_loop_eval_beaker_job` (merges fifty_cities + PASTIS ps=1 into the catalog) and `set_ps1_only_loop_evals` (replaces the catalog with the 18 ps=1 tasks) |
| `regbtl_v1_2_faster_common.py` | `wideread` model builder + faster train module |
| `regbtl_v1_2_regsup_common.py` | register supervision heads; latlon and time-conditioned NDVI arms; extra-decode dataset/dataloader plumbing |
| `regbtl_v1_2_newsampling_common.py` | all newsampling knobs + `SUPERVISION_BASE_WEIGHT`; `apply_uniform_patch_sizes` and `apply_shape_sweep` (the budget × bias overrides, applied *after* `apply_new_sampling`) |

Note that `regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd.py` doubles as a module: most
newer scripts import `build_common_components`, `build_dataset_config`,
`build_dataloader_config`, and `build_visualize_config` from it.

## Reading results

Eval metrics are on the `*_loop_evals` wandb runs, keyed `eval/<task>` for the primary
metric and `eval_other/<task>/<metric>` for the rest, with `checkpoint_step` giving the
training step the checkpoint came from (`_step` is the eval run's own counter, not the
training step — always group by `checkpoint_step`). The ps=1 PASTIS keys are
`eval/pastis_ws16_ps1_sentinel2_pretrain_export` and
`eval/pastis_ws16_ps1_sentinel1_sentinel2_pretrain_export`; the ps=4 in-loop probe is
`eval/pastis`.
