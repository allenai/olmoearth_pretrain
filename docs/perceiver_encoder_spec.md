# Set-Latent Perceiver (SLP) Encoder — Implementation Spec for olmoearth_pretrain

**Goal.** Port the Set-Latent Perceiver self-supervised encoder — developed and
review-hardened in the `earthy` project — into `olmoearth_pretrain`, following
this codebase's model/data/config conventions. This is a *new* encoder option; do
not remove or regress existing models.

**Reference implementation (source of truth).** The working, tested, and
adversarially-reviewed implementation lives in the `earthy` repo. Read it
directly; port its *behavior* and *invariants*, not its exact class layout:

- `/root/repos/earthy/src/earthy/perceiver.py` — `SetLatentSSLModel` (the whole
  encoder: tokenization, latent funnel, readout, masking, forward/loss, eval).
- `/root/repos/earthy/src/earthy/ssl.py` — shared pieces the perceiver imports:
  `ModalityGroupSpec`, `prediction_loss`, `soft_target_contrastive_loss`,
  `temporal_contrastive_loss`, `time_features`, `years_since_2020`,
  `REFERENCE_TIME` (2020-01-01), `MEAN_YEAR_DAYS`, `GroupInputNorm`,
  `PatchTokenizer`, `TwoLayerEncoding`.
- `/root/repos/earthy/PLAN_encoder_v2.md` — the design record and rationale.
- `/root/repos/earthy/KNOWN_ISSUES.md` — the review ledger (accepted trade-offs,
  deferred work, and ~30 fixed bugs). **The "Historical bugs fixed" section is a
  do-not-reintroduce list.**
- `/root/repos/earthy/tests/test_perceiver.py` — the invariants encoded as tests;
  port these.

You have read access to the earthy repo. When in doubt about earthy behavior,
read the code — do not guess.

---

## 1. What the encoder is (one paragraph)

Every modality/resolution is tokenized at its **native GSD** with one small conv
patchifier per group (no common-grid resampling; ERA5 = 1 token/timestep). All
tokens from all modalities and timesteps form **one set** ("token soup"). A fixed
pool of learned **latent queries** cross-attends the set and is refined by
self-attention (a "funnel"), compressing the variable-size set to a fixed latent
representation independent of how many modalities/timesteps/pixels were fed.
Anything is read back out with **metadata-only output queries** (Perceiver-IO):
SSL targets, dense downstream grids at any GSD, or a global pooled feature. There
is **no RoPE and no fixed spatial/temporal budget** — geometry and time enter as
additive continuous encodings on each token; the latents are positionless by
design. Trained self-supervised by predicting frozen random-projection targets of
masked tokens with a global-pool soft-label InfoNCE loss.

## 2. Tokenization (native resolution, metadata-carrying)

One `Conv2d(in=len(assets), out=dim, kernel=patch_px, stride=patch_px)` per group.
Patch sizes (`PATCH_PX` in perceiver.py) harmonize token ground-extents to
~80–120 m for fine sensors, 1 native pixel for coarse:

| group | patch px | gsd | assets |
| --- | --- | --- | --- |
| sentinel1_10m | 8 | 10 m | vv, vh |
| sentinel2_10m | 8 | 10 m | B02,B03,B04,B08 |
| sentinel2_20m | 4 | 20 m | B05,B06,B07,B8A,B11,B12 |
| sentinel2_60m | 2 | 60 m | B01,B09 |
| landsat8_30m | 3 | 30 m | 8 OLI bands |
| sentinel3_300m | 1 | 300 m | 8 SYN reflectance |
| modis_500m | 1 | 500 m | 7 surface-reflectance |
| era5_25km | 1 | 25 km | 5 atmospheric vars |

(QA/SCL bands are **not** data channels — categorical codes used only for cloud
masking.) Each token embedding = content + **two-scale** metadata, added:

```
content = conv(patch)                                  # (b,t,gh,gw,d)
meta =  mlp(gps unit-sphere of patch center)           # GLOBAL position — DROPPABLE
      + linear(fourier(Δeast, Δnorth from window origin))   # LOCAL geometry — ALWAYS KEPT
      + mlp(time feats: years-since-2020 + annual sin/cos)  # ABSOLUTE time — DROPPABLE
      + linear(fourier(Δdays from sample-median date))      # RELATIVE time — ALWAYS KEPT
      + group_embedding                                # modality identity — kept
      + mlp(log ground-extent of the token)            # scale — kept
```

`content` and `meta` are kept **separate**: `meta` alone (no content) is the
decoder query. Local geometry uses **deterministic Fourier features** (~16
log-spaced freqs, 40 m–4 km) not an MLP — spectral bias means an MLP on raw
coords cannot resolve the ~1e-5 unit-sphere difference between adjacent tokens,
and window-relative geometry is load-bearing for spatial-block reconstruction and
dense decode. Relative-time Fourier max wavelength is 4000 d (covers multi-year
samples without aliasing). Absolute time outside the training year range
(`trained_years`) is nulled to the "unknown" state rather than extrapolated.

## 3. Encoder (latent funnel, nested capacity)

- Latent pool `K_max = 1024` learned queries, `d = 768`, `heads = 12` (ViT-B
  scale: ~88 M encoder + ~14 M decoder).
- **Nested-K**: each training step samples `K ∈ {128,256,512,1024}` and uses the
  prefix `latent_pool[:K]`; inference picks K freely (compute/fidelity dial). K
  must be **identical across DDP ranks per step** (rank-free `k_seed`) or every
  step stalls on whichever rank drew K=1024.
- **Funnel:** `[cross(K←inputs) → self×4] × num_input_reads` → `cross(K/4 ←
  latents)` downsample → `self×2` → LayerNorm. Cross-attend weights are
  **shared across input reads** (original-Perceiver style, no extra params). The
  second read is load-bearing: after the first read + self-attn the latent
  queries become input-conditioned, enabling adaptive addressing.
- Invalid/masked tokens are removed via `key_padding_mask` (never fed as content).

## 4. Readout (Perceiver-IO output queries)

Query = **metadata-only** embedding (gps + abs/rel time + group + extent, **no
content**) → `decoder_depth` cross-attn layers over final latents → `head`
Linear → prediction.

- **SSL:** query each masked token's metadata, predict its frozen
  random-projection target.
- **Dense downstream (`encode`):** query a grid at 60 m (`TOKEN_GSD_M`) → (b,h,w,d).
  The grid queries must mirror a **trained** configuration exactly: anchor
  group identity + its extent + gps (real or null) + **null absolute time** +
  **rel_time(0)**. (An earlier eval-only learned readout token got zero gradient
  and was removed; a dead-weight-parameter test guards this.)
- **Global downstream (`encode_stage2`):** mean over latents, broadcast to grid —
  a fully-trained global-feature baseline (no decoder). Log both readouts.

## 5. Masking & loss

**Mask families**, sampled per-sample (probs `(0.3, 0.4, 0.3)`): random-token
drop / whole-timestep-or-modality / **spatial block** (rect covering 25–50% of
the window, dropped across all modalities & timesteps — this subsumes a separate
spatial-MIM stage). Cloudy tokens are excluded from **targets** (any-cloudy over
the token's exact footprint via integral image) but stay **encoder-visible**.
Always guarantee ≥1 visible and ≥1 target token per sample (`_ensure_nonempty`,
searching *all* groups — the first group is fully-invalid on ~48% of S1 windows).

**Targets:** frozen random-projection tokenizer (a twin `PatchTokenizer`,
`requires_grad=False`) applied to the raw patch — content only, no metadata.
data2vec-style but with a fixed random teacher (cannot collapse). **No EMA
teacher** (joer: permanently ruled out for this architecture).

**Loss:** per-group **global-pool InfoNCE** with exact-duplicate dedup + **soft
labels** (`label distribution = softmax(target-similarity / label_temperature)`,
`label_temperature=0.05`): near-duplicate targets (uniform fields, static scenes)
share label mass instead of being false negatives; distinct tokens stay hard.
Per-group averaging (S2 token counts dwarf ERA5). `temporal` scope and
`cosine`/`smooth_l1` regression remain available as flags. **DDP grad anchor:**
`loss += 0.0 * sum(p.sum() for trainable p)` so every parameter participates in
backward on every rank even when a group is absent in a batch.

*Caveat to document, not "fix":* the soft-label loss value has a
scene-uniformity-dependent entropy floor — track accuracy (top-1) and downstream
eval, not raw loss.

## 6. Review-hardened invariants — MUST preserve (do not reintroduce fixed bugs)

These were each a real bug found and fixed in earthy review cycles. Port them:

1. **Metadata/content separation** — the decoder query is `meta` only; feeding
   content would leak the answer. There is a no-leak test: spatially-block-masked
   tokens must be invisible to their own reconstruction queries.
2. **Absolute time dropped on BOTH encoder tokens AND decoder queries** — relative
   time alone identifies the target timestep; this trains the null-abs-time query
   that eval uses.
3. **cond-dropout nulls only global GPS + absolute time** (never local/relative
   geometry) so a location-unknown sample still reconstructs/decodes spatially.
4. **Never fabricate GPS** at eval — unknown/placeholder georef → explicit null
   (zero) vector, which is a trained state. A constant placeholder near a training
   tile poisons features with that place's prior (eurosat kNN 0.24 vs 0.66).
5. **Nested-K K is rank-synced**; mask seed is per-rank (distinct data → distinct
   masks) but K seed is rank-free.
6. **Eval readout queries mirror a trained configuration** (no eval-only learned
   params); a dead-weight-parameter test asserts every parameter receives grad.
7. **Local Fourier (deterministic), not learned MLP,** for local geometry.
8. **Frozen (not trainable) random-projection targets**; a test asserts
   `target_tokenizers` have no grad.
9. **Per-group loss averaging** and the **DDP grad anchor**.
10. **`_ensure_nonempty` searches all groups** and recomputes visibility after the
    target fallback (one-valid-token samples must not become fully key-masked).

## 7. Deliverables

1. The SLP encoder as a new model in olmoearth_pretrain (see §8 for where), config-
   selectable alongside existing encoders, not replacing them.
2. Config(s) for a ViT-B-scale run following this repo's config conventions.
3. Unit tests mirroring `earthy/tests/test_perceiver.py`: token counts &
   metadata, nested-K prefixes, each mask family, no-leak, T=1 fallback,
   soft-target duplicate handling, cloud any-pooling, dead-weight-parameter,
   grad-anchor coverage with a missing group, eval grid readout shape.
4. A short design note under `docs/` (this file, kept updated) describing the
   model and how it plugs into the repo.
5. Everything on branch `joer/perceiver-encoder` (already created off `main`).

## 8. Integration into olmoearth_pretrain

Follow the repo's model/data/config conventions; the earthy code is the
behavioral reference, not a drop-in. The repo is config-driven (dataclass
`Config` subclasses with `build()`, resolved by `_CLASS_` reflection — there is
**no model name registry**). Package root: `olmoearth_pretrain/`.

### 8.1 Integration philosophy — self-contained model, do NOT force it into FlexiViT

The SLP is a different architecture shape from the repo's `FlexiVitBase`
encoders. Those return `{"tokens_and_masks": TokensAndMasks}` — **per-modality
grid tokens** — and the `LatentMIM`/eval machinery pools/decodes those. The SLP
instead compresses everything to **positionless latents**, and its masking,
metadata-only readout, and soft-InfoNCE-against-frozen-random-targets loss are
*integral to the architecture*. Do **not** try to emit `TokensAndMasks` from the
SLP or reuse `LatentMIM`. Instead implement the SLP as a **self-contained model
wrapper** (like `nn/latent_mim.py` is a wrapper) that internally does
tokenize→funnel→mask→readout→loss, and wrap it with a thin train module and a
custom eval wrapper. This keeps every SLP invariant in §6 intact while reusing
the repo's data corpus, config system, trainer, launch, and eval datasets.

### 8.2 Files to add / touch

1. **`olmoearth_pretrain/nn/set_latent_perceiver.py`** (new) — port earthy's
   `SetLatentSSLModel` here as `SetLatentPerceiver(nn.Module, DistributedMixins)`
   (`DistributedMixins` from `nn/utils.py:52`). Bring over the earthy blocks
   (`CrossBlock`, `SelfBlock`, `PatchTokenizer`, `TwoLayerEncoding`,
   `GroupInputNorm`, `fourier_features`) — or reuse repo equivalents where they
   match exactly. Implement `forward(sample, patch_size) -> loss+metrics`,
   `encode(sample) -> (B,H,W,D)`, `encode_global(sample) -> (B,D)`,
   `apply_fsdp(...)`, `apply_compile()`, and class attr
   `supports_multiple_modalities_at_once = True`. Add `SetLatentPerceiverConfig(Config)`
   in the same file: all hyperparameters as dataclass fields **with defaults** (so
   old checkpoints deserialize), a `validate()`, and `build() -> SetLatentPerceiver`
   mirroring `EncoderConfig.build` (flexi_vit.py:2644 — `as_dict(exclude_none=True,
   recurse=False)` auto-forwards fields whose names match constructor kwargs).

2. **Data bridge (the main porting effort).** The SLP must consume the repo's
   sample type — `MaskedOlmoEarthSample` / `OlmoEarthSample` (`datatypes.py`),
   per-modality `[B,H,W,T,bands]` (+ `[B,T,bands]` for era5), `timestamps
   [B,T,3]`, `latlon [B,2]`, `MISSING_VALUE=-99999` — **not** earthy's nested
   dict. Map earthy's per-resolution groups to the repo's `Modality`/`BandSet`
   structure (`data/constants.py`): S2 has band sets by resolution (10/20/60 m),
   plus `SENTINEL1`, `LANDSAT`, `ERA5_10`, etc. Derive SLP metadata from the
   sample: global GPS = unit-sphere of `latlon`; absolute time = years-since-2020
   + annual sin/cos from `timestamps` (reuse the repo's `timestamps_to_days` /
   month-encoding in `nn/encodings.py` if convenient, but keep the SLP's
   two-scale scheme); relative time = Fourier of Δdays from the sample-median
   date; local geometry = Fourier of patch-center Δeast/Δnorth (derive from
   patch index × token extent; you do not need true UTM — window-relative offsets
   suffice, exactly as earthy does). Validity/`MISSING_VALUE` → the SLP's
   valid-token threshold. Cloud masking: if the repo exposes a QA/cloud modality
   use it for target exclusion; otherwise make cloud masking optional/off and
   note it.

3. **Masking — internal to the SLP (a deliberate divergence).** The SLP's
   mask-family mixture (random-token / temporal / spatial-block) is integral and
   runs inside `forward`. Do **not** route it through
   `MASKING_STRATEGY_REGISTRY`; feed the SLP an unmasked sample (or one with all
   tokens `ONLINE`) and let it mask internally. Document this divergence in the
   design note. (If a reviewer later wants it registered, that's a follow-up.)

4. **Loss — internal.** Port earthy's `soft_target_contrastive_loss` /
   `prediction_loss` / `temporal_contrastive_loss` (earthy `ssl.py`) into the SLP
   module (or a small `nn/` helper). The frozen targets are the SLP's own
   `target_tokenizers` (frozen random-projection twin patchifiers), **not** a
   deepcopied `target_encoder` like `LatentMIM`, and there is **no EMA**. You do
   not need to register in `LOSS_REGISTRY` since the loss is internal to forward.

5. **Train module** — `olmoearth_pretrain/train/train_module/set_latent_perceiver.py`
   (new), `SetLatentPerceiverTrainModule(OlmoEarthTrainModule)` +
   `...Config(OlmoEarthTrainModuleConfig)`. Model it on
   `train/train_module/latent_mim.py` but much thinner: microbatch → `loss,
   metrics = model(batch)` → backward; no separate target-encoder forward
   (targets are internal); `update_target_encoder()` is a no-op. Keep the DDP
   grad-anchor (§6.9) and rank-synced K seed vs per-rank mask seed (§6.5). Log
   loss, top-1 accuracy, and per-group valid-token fraction.

6. **Eval wrapper** — add a `SetLatentPerceiverEvalWrapper` in
   `evals/eval_wrapper.py` and an `isinstance(model, SetLatentPerceiver)` branch
   in `get_eval_wrapper` (:481). It should call `model.encode(sample)` for dense
   features and `model.encode_global(sample)` for a pooled vector, returning what
   the downstream probes expect (see `OlmoEarthEvalWrapper.__call__` :142 and
   `nn/pooling.py`). The pretraining-time `DownstreamEvaluatorCallback` extracts
   `model.encoder` if present — either expose `.encoder` returning something the
   default wrapper handles, or ensure the callback path reaches your wrapper.
   **Never fabricate GPS** at eval (§6.4): unknown location → null vector.

7. **Launch config** — `scripts/vnext/perceiver/base.py` (+ `nano.py`/size
   presets and a `launch_*.sh`), copied from `scripts/official/v1_2/base.py`:
   swap the model config to `SetLatentPerceiverConfig`, the train-module config to
   `SetLatentPerceiverTrainModuleConfig`, keep `build_common_components`,
   dataloader, dataset, trainer (with `DownstreamEvaluatorCallbackConfig`), and
   the `main(...)` wiring (`internal/experiment.py`). ViT-B preset: d=768,
   heads=12, K=1024.

8. **Tests** — `tests/unit/nn/test_set_latent_perceiver.py` (+ an integration
   test under `tests/integration/nn/`) porting `earthy/tests/test_perceiver.py`:
   token counts/metadata, nested-K prefixes, each mask family, no-leak, T=1
   fallback, soft-target duplicate handling, dead-weight-parameter, grad-anchor
   coverage with a missing modality, eval grid-readout shape. Follow the repo's
   pytest layout (`tests/unit` vs `tests/integration`, google-style docstrings,
   ruff). `tests_minimal_deps/` is inference-only — put SLP tests under `tests/`.

### 8.3 Reference files in this repo to copy patterns from

- `nn/st_model.py` — the best example of a *second, distinct* encoder + config.
- `nn/latent_mim.py` — model-wrapper pattern (encoder+decoder+frozen target).
- `train/train_module/latent_mim.py` — the SSL train loop (thin your version).
- `scripts/official/v1_2/base.py` — a config assembled end-to-end.
- `datatypes.py`, `data/constants.py`, `data/dataloader.py`, `nn/encodings.py` —
  the sample format, modality/band specs, loader/masking, encodings you'll bridge.
- `evals/eval_wrapper.py` (`get_eval_wrapper`), `nn/pooling.py` — eval contract.

### 8.4 olmo-core / distributed notes

Training pulls `ai2-olmo-core==2.3.0` (the `[training]` extra); inference works
without it. Use FSDP via `torch.distributed.fsdp.fully_shard` in `apply_fsdp`
(see existing encoders). New config fields **must have defaults** so old
checkpoints deserialize (`_strip_unknown_fields` / `patch_legacy_encoder_config`
handle unknowns). Run `uv` for env; ruff + pydocstyle(google) for lint.

## 9. Verification milestones (port from earthy PLAN §8)

1. Unit tests green (the list in §7).
2. Single-batch overfit: loss → ~0, InfoNCE top-1 → ~1.
3. Multi-GPU DDP smoke (a few steps) incl. grad-anchor coverage with a group
   absent from the batch.
4. A short pretraining run that logs loss, top-1 accuracy, per-group valid-token
   fraction, and a first-batch dead-group warning; confirm it trains and evals.

## 10. When to ask joer vs decide yourself

- **Decide yourself:** naming, file placement, following repo conventions,
  wiring config plumbing, test structure, anything the earthy reference already
  settles.
- **Ask (via the managing agent):** anything that would *change the SLP design*
  (loss, masking, metadata scheme, funnel structure), any olmoearth_pretrain
  convention that has two equally-valid options with downstream consequences, or
  any conflict between the earthy reference and this repo's data format that you
  cannot resolve by reading code. Record blocking questions clearly; the managing
  agent will get answers.
