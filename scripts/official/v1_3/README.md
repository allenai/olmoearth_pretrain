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
| `open_set_base.py` | `base.py` + the open-set supervised probe on the d768 register grid (see below) |
| `open_set_only.py` | from scratch on the open-set supervised dataset only |
| `open_set_osm.py` | from scratch on osm_sampling + open-set |
| `open_set_post_train.py` | the release checkpoint, post-trained on osm_sampling + open-set in two in-run phases |
| `open_set_only_mlp.py` | `open_set_only` with a shared 2-layer MLP probe trunk |
| `open_set_only_text.py` | `open_set_only` with CLIP-style cosine logits against frozen class text embeddings |
| `open_set_only_dsbal.py` | `open_set_only` with per-dataset temperature-balanced (tau 0.5) supervised loss |
| `launch_open_set.sh` | launcher for the three open-set runs |

## Open-set supervised runs

`open_set_base.py` adds an `OpenSetProbe` (linear classification + regression heads,
`olmoearth_pretrain/train/open_set_probe.py`) that reads the same d768 register grid the
map supervision heads read, driven by the `open_set` / `open_set_regression` label layers
of the open-set dataset (`olmoearth_pretrain/open_set_segmentation_data/README.md`). The
labels are loaded as decode-only modalities so the encoder never tokenizes them and they
do not consume the token budget; the loss is per-sample balanced and weighted 0.1x.

Paired pre/post **change** samples carry an `open_set_change_boundary` (the split date).
The temporal crop and the time-masking branch keep at least one before and one after
timestep visible to the encoder, and change labels are only supervised when both sides
were seen.

Probe variants (all `OpenSetProbeConfig` switches passed through
`open_set_base.build_model_config(**probe_overrides)`, `open_set_only.py` is the control):

- `head_type="mlp"`: a shared `Linear(768, 768) -> GELU` trunk before the linear heads.
- `head_type="text"`: a shared `Linear -> GELU -> Linear(768)` trunk scored by scaled
  cosine similarity against frozen class text embeddings (`all-mpnet-base-v2` on
  "class name; dataset name"). They live on weka
  (`open_set_base.CLASS_TEXT_EMBEDDINGS_PATH`, under the label bank's
  `open_set_segmentation/class_text_embeddings/`), generated once in a Beaker session with
  `uv pip install sentence-transformers && python -m olmoearth_pretrain.open_set_segmentation_data.embed_class_names`;
  the `.json` sidecar pins the class-mapping hash, verified at model build.
- `dataset_balance="dataset_temperature"`, `balance_temperature=0.5`: weight each labeled
  sample by its dataset size `n_d^(tau-1)` (mean 1 under natural sampling), emulating
  `p_d ~ n_d^tau` dataset sampling without touching the loader.

### High-quality subset (`open_set_hq/`)

`open_set_hq/select_h5_indices.py` recovers which label-bank dataset each open-set H5
sample came from by reading its label layer: `open_set` class ids and
`open_set_regression` dataset ids both map to a dataset slug through
`class_mapping.json`. (Matching by location does not work: many datasets have several
samples at the same window, e.g. one per year.) It writes the H5 indices of
`HIGH_QUALITY_SLUGS` (33 manually curated datasets, roughly one per concept) as a
`.npy` plus a full `sample_index, slug, labeled_pixels, other_slugs` table.
`open_set_hq/open_set_only_hq.py` is `open_set_only.py` with that file as
`OlmoEarthDatasetConfig.filter_idx_file`: same H5s, same class mapping, no rebuild.

The indices are positions in one H5 build, so the filter file is named after the
build's sample count (`filters/open_set_hq_v1_<num_samples>.npy`, following
`open_set_base.OPEN_SET_H5_DIR`) and must be regenerated whenever that directory
changes.

```bash
python scripts/official/v1_3/open_set_hq/select_h5_indices.py        # once per H5 build, weka mounted
python scripts/official/v1_3/open_set_hq/open_set_only_hq.py launch open_set_only_hq \
    ai2/jupiter --launch.num_gpus=8
```

`open_set_post_train.py` runs a single job with two phases
(`OpenSetLatentMIMTrainModule.freeze_backbone_until_step`): first only the probe trains
against the frozen checkpoint (encoder-only forward), then the backbone trains at a low LR
on the full v1.3 objective plus the supervised loss through the frozen probe. The release
checkpoint is initialized via `init_weights_path` from a `convert_legacy_checkpoint.py`
output (see the script docstring); `init_weights_allow_missing=["open_set_probe."]` lets
the probe keep its fresh init.

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
