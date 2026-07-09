# Set-Latent Perceiver (SLP) encoder — session handoff

Context for a fresh session picking up the SLP-encoder port into
`olmoearth_pretrain`. Read this first, then `docs/perceiver_encoder_spec.md` for
the full design + integration map.

## TL;DR

- **What:** a new self-supervised encoder for `olmoearth_pretrain`, ported from
  the review-hardened `SetLatentSSLModel` in the `earthy` project. Native-res
  token soup → nested-K latent funnel → metadata-only Perceiver-IO readout →
  soft-InfoNCE against frozen random-projection targets. Positionless latents, no
  RoPE, no fixed spatial/temporal budget, single-timestep supported.
- **Where:** git worktree `/root/repos/olmoearth_pretrain-perceiver`, branch
  **`joer/perceiver-encoder`** (off `origin/main`). Two commits; **not pushed**.
- **Status:** implemented, self-verified, all tests green, single-batch overfit
  works. NOT yet run on multi-GPU or real corpus data (out of the first pass's
  scope). See "Next steps".

## Where the work lives

Added on `joer/perceiver-encoder`:

| path | what |
| --- | --- |
| `olmoearth_pretrain/nn/set_latent_perceiver.py` | `SetLatentPerceiver` + `SetLatentPerceiverConfig` — the whole model (tokenize → funnel → mask → readout → loss), the `MaskedOlmoEarthSample` data bridge, `encode`/`encode_global`, `apply_fsdp`/`apply_compile` |
| `olmoearth_pretrain/train/train_module/set_latent_perceiver.py` | thin SSL train module (rank-free K seed, per-rank mask seed, DDP grad anchor, no EMA/target-encoder forward) |
| `olmoearth_pretrain/evals/eval_wrapper.py` | `SetLatentPerceiverEvalWrapper` + an `isinstance` branch in `get_eval_wrapper` |
| `scripts/vnext/perceiver/{base.py,nano.py,launch_perceiver.sh}` | ViT-B + nano launch configs (config-driven, `main()`-wired) |
| `tests/unit/nn/test_set_latent_perceiver.py` (22) + `tests/integration/nn/test_set_latent_perceiver.py` (2) | invariants ported from earthy |
| `docs/perceiver_encoder_spec.md` | full design spec + integration map + §11 implementation decisions |

**Reference implementation (source of truth, READ-ONLY):** the `earthy` repo —
`/root/repos/earthy/src/earthy/perceiver.py` (`SetLatentSSLModel`),
`/root/repos/earthy/src/earthy/ssl.py` (loss helpers, `ModalityGroupSpec`,
time features), `/root/repos/earthy/PLAN_encoder_v2.md` (design record),
`/root/repos/earthy/KNOWN_ISSUES.md` (the "Historical bugs fixed" section is a
do-not-reintroduce list), `/root/repos/earthy/tests/test_perceiver.py`.

## Design in one paragraph

Every `(modality, band set)` is one group with its own conv patchifier. All tokens
from all modalities/timesteps form one set; a fixed pool of **learned latent
queries** (K_max=1024, d=768, 12 heads ≈ ViT-B) cross-attends the set and is
refined by self-attention (the "funnel"), compressing to a fixed latent rep
regardless of input size. Readout is via **metadata-only** output queries
(gps + abs/rel time + group + extent, no content). Geometry/time enter as
additive continuous encodings, two-scale: **global GPS + absolute time are
droppable** (cond-dropout, and eval uses the trained null state), **local Fourier
geometry + relative time are always kept**. Trained by predicting frozen
random-projection targets of masked tokens (mask families: random-token /
temporal / spatial-block) with a per-group global-pool soft-label InfoNCE loss.
The 10 review-hardened invariants (§6 of the spec) are all preserved — verify any
change against them.

## Verification done (self-checked, not just self-reported)

- `uv run pytest tests/{unit,integration}/nn/test_set_latent_perceiver.py` → **24 passed**.
- CPU single-batch overfit (60 steps): loss **3.91 → 0.54**, InfoNCE top-1 **0.06 → 0.98**; grads reach all trainable params; `encode`→(B,H,W,D), `encode_global`→(B,D); eval wrapper routes classification vs segmentation correctly.
- ViT-B param counts: **~89 M encoder + ~14 M decoder** (matches spec). `nano` (8.3 M) and `base` (103.9 M) configs build; `python scripts/vnext/perceiver/nano.py dry_run …` wires end-to-end.
- ruff / mypy / bandit / interrogate clean (pre-commit passed on commit).

## Key integration decisions (full list: spec §11)

1. **Group = (modality, band set)**, tokenized on the modality's stored grid
   (band sets are co-registered there), `patch_px` chosen per modality to hit an
   ~80 m token extent; inputs padded up to a multiple of `patch_px`.
2. **Global GPS is sample-level** (`latlon`), broadcast; per-token spatial variation
   is the always-kept local Fourier geometry. Missing latlon → trained null vector
   (never fabricated — the eurosat lesson).
3. **Validity from `MISSING_VALUE`** (`isfinite & x > MISSING_VALUE/2`); the loader
   pre-normalizes, so there is no `GroupInputNorm`.
4. **Masking is internal** to `forward` (deliberate divergence from
   `MASKING_STRATEGY_REGISTRY`); dataloader masking is a trivial ignored config.
5. **Eval:** the model exposes no `.encoder`, so `get_eval_wrapper` routes to
   `SetLatentPerceiverEvalWrapper` (classification/window → `encode_global`;
   segmentation/per-pixel → `encode`).

## Deferrals / NOT done (pick up here)

- **ERA5 absent from the v1.2 corpus** → launch configs use **S2 / S1 / Landsat**
  only. Add `era5_10` to both `training_modalities` and `supported_modality_names`
  when a corpus provides it (the model class already supports it).
- **Cloud masking is OFF** — no per-token cloud/QA modality in the corpus.
  `_apply_cloud_mask` is implemented + unit-tested with synthetic grids; wire it up
  when a cloud modality exists.
- **Not yet run on multi-GPU DDP or real corpus data.** Next: a small real-data run.
- **Branch not pushed** to `origin` (awaiting the go-ahead).

## How to run

```bash
cd /root/repos/olmoearth_pretrain-perceiver          # branch joer/perceiver-encoder
uv run pytest tests/unit/nn/test_set_latent_perceiver.py tests/integration/nn/test_set_latent_perceiver.py -q
python scripts/vnext/perceiver/nano.py dry_run <run-name> <cluster>   # inspect the wired config
# launch (Beaker): scripts/vnext/perceiver/launch_perceiver.sh   # see scripts/official/v1_2 for the pattern
```

## Next steps (suggested order)

1. Push `joer/perceiver-encoder` (if desired) and/or open a draft PR.
2. Small real-data run on the v1.2 corpus (S2/S1/Landsat) — confirm it trains,
   loss/top-1 move, downstream eval (GEO-Bench/PASTIS) via the callback works.
3. Multi-GPU DDP smoke (grad-anchor coverage with a modality absent from a batch).
4. Sweep the SLP knobs the design flags: `num_input_reads {1,2,3}`, nested-K,
   masking family probs; compare vs the FlexiViT/LatentMIM baselines.
5. When a corpus adds ERA5 / a cloud modality, wire them (see Deferrals).

## Related context (the earthy side)

The SLP originated in `earthy` (from-scratch RS SSL library, `PYTHONPATH=src`,
GitHub `pjreddie/earthy`). There it is the `perceiver` encoder, trained on a
30-tile packed dataset; a live 8×H100 run and a data-loading-efficiency effort
(128 px ZSTD repack + loader hotspot fixes) are ongoing there — unrelated to this
port except as the reference. The earthy `KNOWN_ISSUES.md` + `PLAN_encoder_v2.md`
are the authoritative design/rationale record.
