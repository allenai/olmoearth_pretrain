# v1.2 official runs

Everything here trains the **v1.2 recipe** — v1.1 (hidden patch-embed projection) plus
the winning mixed 3D RoPE config from the temporal-RoPE sweep (`rope_3d_mixed`,
`rope_mixed_base=10`, `rope_temporal_coordinate_scale=1/30` ≈ months). Those knobs are
baked into `base.py`; nothing below overrides them.

Two efforts share the directory:

- **Size sweep** (`nano`/`tiny`/`small`/`large` + `base`) → wandb
  `2026_06_12_v1_2_size_sweep`.
- **Spatial register bottleneck** (everything named `regbtl_*`) — the lineage of the
  v1.3 release candidate, `mlpgram1`. Only that lineage is kept here; the ablation arms
  that led to it live in git history (`git log -- scripts/official/v1_2`).

## Conventions

- **One script per run.** Architecture is baked into the script rather than passed as a
  CLI override, because the in-loop eval Beaker jobs re-import `MODULE_PATH` to
  reconstruct the matching model. A CLI-only override would give the eval job a
  different model than the one being trained.
- **`MODULE_PATH` must match the file's own path.** It is what the eval job imports; a
  stale value silently evaluates the wrong architecture.
- **In-loop evals run as separate non-blocking Beaker jobs** and log to a companion
  wandb run suffixed `_loop_evals`. Eval metrics live there, *not* in the training run.
- **Launch is a shell script per run**, and requires a clean tree pushed to the remote
  (the Beaker job clones `$GIT_REF`). Commit and push before launching.

## The release-candidate lineage

`regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsampling_psuniform_stunorm_mlpgram1`
(wandb `2026_08_26_student_norm`) decodes as:

| fragment | meaning |
|---|---|
| `regbtl` | spatial register bottleneck (Perceiver-style read into a register grid) |
| `gdyn` | dynamic single-latent register grid (`register_grid_size=0`) |
| `d768` | `register_dim` of the teacher (primary) bottleneck |
| `proj128lin` | detached linear student, Matryoshka dims `[128, 64]` — the SHIPPED embedding |
| `sup768` | register-grid supervision heads on the d768 registers only |
| `w1` | supervision `base_weight=1.0` |
| `newsampling` / `newsamp` | decorrelated (grid, timestep) shape sampler |
| `psuniform` | uniform patch-size sampling (P(ps=1) = 0.125) |
| `stunorm` | `register_projection_output_norm=True`: LayerNorm on the student output |
| `mlpgram1` | flat Gram weight 1.0 (the default) + 2-layer MLP back-projection head, H=256 |

Also implied, by construction: interleaved reads with per-depth projections, no
instance-contrastive loss, latent self-attention on, one forward pass per batch, fused
AdamW, projection-only target, replicated DP + bf16, and `wideread` (bottleneck attention
at encoder width, so `register_dim` is purely the storage width).

Scripts, in import order:

| file | contents |
|---|---|
| `base.py` | v1.2 baseline config: RoPE/temporal knobs, patch sizes 1–8 |
| `base_faster.py` | `base` + all validated speedups (~1.35×) |
| `regbtl_v1_2_common.py` | register-bottleneck model builder + in-loop eval task sets |
| `regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd.py` | the d768 bottleneck run; doubles as the module most scripts import `build_common_components` / `build_dataset_config` / `build_dataloader_config` / `build_visualize_config` from |
| `regbtl_v1_2_gdyn_d768_il_pdproj_noic_lsa_1fwd_fusedadamw.py` | + fused AdamW train module |
| `regbtl_v1_2_faster_common.py` | `wideread` model builder + faster (ddp/bf16) train module |
| `regbtl_v1_2_regsup_common.py` | register supervision heads (+ the latlon arm and its extra-decode plumbing) |
| `regbtl_v1_2_newsampling_common.py` | the newsampling knobs, `apply_uniform_patch_sizes`, `apply_microbatch` |
| `regbtl_v1_2_proj_common.py` | the detached student: model builder, student LR group, in-loop evals on both student widths |
| `..._sup768_w1_newsampling_psuniform_stunorm.py` | the (flat gram, linear head) base cell — `launch_regbtl_v1_2_proj128_stunorm.sh` |
| `..._stunorm_mlpgram1.py` | the release candidate — `launch_regbtl_v1_2_proj128_stunorm_gram_head.sh` |

## Reading results

Eval metrics are on the `*_loop_evals` wandb runs, keyed `eval/<task>` for the primary
metric and `eval_other/<task>/<metric>` for the rest, with `checkpoint_step` giving the
training step the checkpoint came from (`_step` is the eval run's own counter, not the
training step — always group by `checkpoint_step`). The student is scored under the
`_proj128` / `_proj64` task suffixes.
