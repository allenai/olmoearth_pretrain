# Open-set text-conditioned binary segmentation

A minimum-viable text-conditioned segmentation head built on top of a frozen
OlmoEarth encoder. Given a satellite tile and a class name (e.g. `"maize"`),
the model produces a binary mask. The same model can answer for class names
it has never seen during training, opening the door to zero-shot
generalization to held-out classes.

## Motivation

Users still need labels to train models. We want to move toward a model that
needs none — at least for the simpler tasks (binary classification or
single-value regression on segmentation outputs). The bet is that the
pretraining label sources we already ingest (OSM, CDL, WorldCover,
EuroCrops, ...) give enough semantic coverage that a text-conditioned head
trained against them can extrapolate to *new* classes via the structure
that a vision-language text encoder (SigLIP) provides.

## Architecture

```
"maize"  ─▶  SigLIP text encoder  ─▶  text tokens + pooled
                                            │
                                            ▼
satellite tile  ─▶  OlmoEarth encoder (frozen)  ─▶  image tokens
                                            │
                                            ▼
              cross-attention decoder (image Q, text KV)
                                            │
                                            ▼
              dot-product head → per-pixel binary logits
```

The decoder reuses `nn.attention.Block(cross_attn=True)` — no new attention
implementation. Each refined image token predicts a `max_patch_size²` grid
of pixel embeddings; those unfold to `(P_H × max_patch_size, P_W ×
max_patch_size)` and dot-product with the text-class embedding. When the
actual patch size equals `max_patch_size` the output is at image
resolution natively (no interpolation); when smaller it gets bilinearly
downsampled to the OSM raster's resolution (this matches the supervision
pattern on `gabi/supervision`).

## Forward pass

For a batch:

1. Encoder runs **once** → flat `[B, N, D_enc]` token sequence + per-modality shape dict.
2. Sampler picks `K_pos` present + `K_neg` absent classes per image.
3. Class union deduplicated across the batch into `C` text queries.
4. Decoder is replicated `C` times via `[C, B, ...]` reshaping — one forward.
5. Logits come out `[C, B, H_osm, W_osm]`; per-(image, class) BCE losses are summed and one `.backward()` runs.

## Reasons to be optimistic

- OSM and CDL alone cover many real use cases — rich semantic supervision
  for free.
- The decoder is small relative to the encoder; we get to reuse a powerful
  pretrained backbone without paying its training cost again.
- The approach extends naturally to unseen classes via SigLIP's text
  geometry (subject to the limits below).

## Reservations

- **Conditioning cost.** The decoder runs once per (image, class) query. We
  cache text embeddings on disk so SigLIP only fires once per class. The
  encoder forward is shared across all class queries in a batch.
- **Gradient signal.** Cross-attention conditioning (rather than a
  hypernetwork) keeps the decoder weights fixed across queries — gradients
  flow as in any transformer decoder. Avoided one of the failure modes we
  worried about up front.
- **SigLIP wasn't trained on overhead views.** "Maize" in SigLIP's image
  space is mostly ground-level corn. The image encoder side of SigLIP isn't
  used here, but the *text geometry* still reflects the web; whether that
  geometry transfers to satellite-relevant similarity is the empirical
  question the held-out-class eval is meant to answer.
- **Binary negative framing leaks.** Sampling negatives randomly from the
  same source means "not maize" includes wheat, urban, water, etc. — much
  easier than "maize vs. sorghum from above". Hard-negative mining (sample
  semantically similar absent classes) is the planned next step; the
  `ClassSampler` Protocol is general enough to add it without touching
  call sites.
- **OSM/CDL coverage is uneven.** OSM is global but skewed; CDL is
  US-only. Generalization beyond the represented classes is bounded by
  this.

## Where things live

```
open_set/
├── catalog/                     ClassEntry + extractors + per-source enumeration
│   ├── registry.py              ClassEntry, ClassExtractor protocol, ClassRegistry
│   └── osm.py                   30 OSM entries from Modality.OPENSTREETMAP_RASTER
├── text/                        Generic TextEncoder + SigLIP wrapper + on-disk cache
│   ├── base.py                  TextEncoder Protocol, TextEncoding NamedTuple
│   ├── siglip_encoder.py        Lazy HF wrapper (transformers is an optional dep)
│   └── embedding_cache.py       Disk-backed cache keyed by encoder name
├── data/                        Mask extraction, K_pos+K_neg sampler, modality subsampler
│   ├── mask_extractor.py        Per-(image, class) binary mask materialization
│   ├── sampler.py               RandomNegativeSampler — hard-negative TODO
│   └── modality_subsample.py    Optional input-modality dropping
├── model/                       Frozen encoder + cross-attn decoder + dot-product head
│   ├── encoder_wrapper.py       Distributed checkpoint loader + frozen wrapper
│   ├── cross_attn_decoder.py    Stacks nn.attention.Block(cross_attn=True/False)
│   └── open_set_model.py        OpenSetSegmenter + OpenSetModelConfig
└── train/                       olmo-core TrainModule + entrypoint
    └── train_module.py          OpenSetTrainModule subclass of OlmoEarthTrainModule
```

## Status / TODO

- ✅ MVP catalog (OSM only — 30 classes from band names).
- ✅ SigLIP text cache, generic over text encoder.
- ✅ Random-negative sampler.
- ✅ Modality subsampling.
- ✅ Frozen-encoder loader (distributed + consolidated checkpoint formats).
- ✅ Cross-attention decoder (reuses existing `nn.attention.Block`).
- ✅ Dot-product head with `max_patch_size`-aware sub-patch prediction.
- ✅ olmo-core `TrainModule` implementation.
- ⬜ CDL / WorldCover / EuroCrops catalogs (the value-equality extractor is ready).
- ⬜ Hard-negative sampling — the headline improvement once it's wired in.
- ⬜ Held-out-class evaluation — the result that justifies the approach.
- ⬜ Cross-source transfer (CDL → EuroCrops) and out-of-catalog evals (PASTIS, MADOS).
