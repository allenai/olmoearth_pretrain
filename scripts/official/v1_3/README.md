# v1.3 official runs

OlmoEarth v1.3 is the v1.2 recipe (`../v1_2/base.py`: hidden patch-embed projection,
mixed 3D RoPE, decode-only map modalities) plus two additions:

- **Aggregation** — a Perceiver-style spatial register bottleneck (`register_dim=768`,
  attention at encoder width, `[read -> self-attn] x 4`, 2D RoPE on the reads and the
  decoder), with per-modality supervision heads on the register grid.
- **Compaction** — a detached linear `[128, 64]` Matryoshka student on the registers,
  ending in a LayerNorm, distilled from the teacher with a cosine term (through a
  2-layer MLP back-projection) and a Gram term. The student is the shipped embedding.

Everything is in `base.py`, which imports the v1.2 config rather than copying it.

| file | contents |
|---|---|
| `base.py` | the release recipe: model, sampler, train module, in-loop evals |
| `ablations/no_supervision.py` | `base.py` with `supervision_head_config = None` |
| `ablations/query_token_compaction.py` | native d128 registers with supervision, no student |
| `experiments/pixreg_pixrecon.py` | pixel-resolution d128 registers + per-pixel raw-band reconstruction (see below) |

## Experiments

`experiments/` holds post-release arms that build on `base.py` but are not part of the
v1.3 report.

- **`pixreg_pixrecon.py`** -- the `query_token_compaction` shape (native d128
  registers, no student) with the register grid laid at **pixel** resolution
  (`register_pixel_grid`: one register per pixel, whatever patch size the trunk runs
  at, at pixel-center RoPE coordinates), the latent self-attention narrowed to the
  register width (`register_latent_attn_dim=128`, 2 x 64 heads, affine-free block
  norms) while the reads keep the wideread shape, patch sizes restricted to 1..4 and
  grids to hw_p <= 24 (worst case 96 x 96 = 9k registers), map supervision at one
  value per cell (`spatial_unfold=1`, base weight 0.1), and two **time-conditioned**
  reconstruction heads (an MLP on `[register_cell ; phi(day_of_year)]`) that regress
  the normalized S2 L2A and S1 inputs per (pixel, timestep) at weight 0.05 each.
  Re-bases `regbtl_v1_2_..._ps14_pixreg_pixrecon` (W&B `2026_08_19_pixel_branch`)
  onto the v1.3 stack, without that run's temporally anchored read or NDVI head.
  Evals score the register grid at 40k-step intervals.

## Conventions

- **Architecture is baked into the script**, not passed as a CLI override. The in-loop
  eval Beaker jobs re-import `MODULE_PATH` to reconstruct the model, so a CLI-only
  override would give the eval job a different model than the one being trained.
- **`MODULE_PATH` must match the file's own path.** A stale value silently evaluates the
  wrong architecture.
- **In-loop evals run as separate non-blocking Beaker jobs** and log to a companion
  wandb run suffixed `_loop_evals`. Eval metrics live there, *not* in the training run.
  The eval set replaces the shared catalog: the AEF balanced-trial datasets (kNN twins,
  which carry the AEF protocol) plus year-aligned PASTIS on S1+S2+Landsat, scored on
  the student at 128 and 64 dims (`_proj128` / `_proj64`), or on the register grid for
  the query-token ablation.
- **Launch requires a clean tree pushed to the remote** (the Beaker job clones
  `$GIT_REF`). Commit and push before launching.

## The runs behind the report

These scripts were consolidated from a chain of `regbtl_v1_2_*` modules after the runs
finished; the configs they build are identical to what was trained (verified field by
field), but the W&B run names keep the old scheme. All in project
`2026_08_26_student_norm`:

| script | trained as |
|---|---|
| `base.py` | `regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform_stunorm_mlpgram1` |
| `ablations/no_supervision.py` | `regbtl_v1_2_nosup_gdyn_d768_proj128lin_newsamp_psuniform_stunorm_mlpgram1` |
| `ablations/query_token_compaction.py` | `regbtl_v1_2_qtc_gdyn_d128_wideread_regsup_w1_newsamp_psuniform` |

The old run name decodes as: `regbtl` register bottleneck; `gdyn` dynamic single-latent
grid; `d768` teacher width; `proj128lin` linear `[128, 64]` student; `sup768` heads on the
teacher only; `w1` supervision base weight 1.0; `newsamp` decorrelated shape sampler;
`psuniform` uniform patch sizes; `stunorm` LayerNorm on the student; `mlpgram1` Gram
weight 1.0 + MLP back-projection head. The exploratory arms that led here live in git
history (`git log -- scripts/official/v1_2`).

## Reading results

Eval metrics are on the `*_loop_evals` wandb runs, keyed `eval/<task>` for the primary
metric and `eval_other/<task>/<metric>` for the rest, with `checkpoint_step` giving the
training step the checkpoint came from (`_step` is the eval run's own counter, not the
training step — always group by `checkpoint_step`).
