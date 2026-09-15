# CH4Net (methane super-emitter plumes) — REJECTED (needs-credential: gated HF repo)

- **Slug**: `ch4net_methane_super_emitter_plumes`
- **Name**: CH4Net (methane super-emitter plumes)
- **Source**: AMT paper (Vaughan et al. 2024, https://doi.org/10.5194/amt-17-2583-2024); data/code DOI https://doi.org/10.57967/hf/2117
  -> Hugging Face dataset `av555/ch4net`.
- **Family / region**: plume / Turkmenistan (23 super-emitter sites), 2017-2021.
- **Label type (manifest)**: dense_raster, single class `methane plume`, manual annotation.
- **License**: manifest says CC-BY-4.0; **HF repo is tagged `cc-by-nc-nd-4.0`** (see below).
- **Status**: **rejected** — `needs-credential`.
- **Primary rejection reason**: the only distribution is a **gated** Hugging Face repo that
  requires authentication + granted access we do not have.

## What CH4Net is

A hand-annotated methane-plume segmentation dataset on Sentinel-2 imagery for automated
super-emitter monitoring. It covers 23 known super-emitter locations in Turkmenistan and
comprises **925 hand-annotated plume masks** plus **9,121 plume-free scenes** (10,046 images
total, 2017-2021). The HF repo is organized as `{train,val,test}/{label,s2,mbmp}/{i}.npy`
(8,255 train / 255 val / 2,473 test triples), where `label` is the plume mask, `s2` the
Sentinel-2 patch, and `mbmp` the multi-band multi-pass methane product. The paper describes
patches as 0.01 deg x 0.01 deg (200 x 200 px) centered on the emitter sites.

## Why it is rejected (needs-credential)

The AMT data/code-availability statement gives a **single** location — "Code and
hand-annotated masks are available at https://doi.org/10.57967/hf/2117" — which resolves to `av555/ch4net`. That repo is
**gated**: every attempted file download (label/s2/mbmp `.npy` and `quickstart.ipynb`)
returns `GatedRepoError` / HTTP 401: *"Access to dataset av555/ch4net is restricted. You
must have access to it and be authenticated to access it."* Only the (empty) `README.md`
metadata is fetchable. Accessing the data requires a Hugging Face account, accepting the
dataset's access terms to be granted access, and an `HF_TOKEN` — none available here. The
paper lists **no Zenodo, GitHub, or other ungated mirror**. This is a persistent access
gate, not a transient outage, so it is `rejected` (needs-credential), not
`temporary_failure`.

## Secondary concerns (for the user)

1. **License mismatch.** The manifest records `CC-BY-4.0`, but the HF dataset is tagged
   **`cc-by-nc-nd-4.0`** (NonCommercial-**NoDerivatives**). The paper *text* is CC-BY-4.0;
   the *dataset* is NC-ND. NoDerivatives is in tension with generating/redistributing
   derived label rasters for pretraining — confirm terms before use even once access is
   granted. This could independently support a "license forbids use" rejection.

2. **Georeferencing unverified.** Files are opaque running-index `.npy` arrays with no
   coordinate table visible in the gated listing. Per-patch geolocation is *plausibly*
   recoverable (23 known site centroids + a fixed 0.01 deg extent) IF the release includes a
   patch->site mapping, but this could not be confirmed because the arrays are inaccessible.
   If access is later granted, check for a site/coordinate index first; if the arrays are
   coordinate-free (like Landslide4Sense/LoveDA), reject instead on "no recoverable
   georeferencing".

## Intended recipe if accepted later

`dense_raster`, single foreground class `methane plume` (id 0). Positive-only phenomenon —
per spec 5, do NOT fabricate negatives (the 9,121 plume-free scenes already provide real
negatives; leave non-plume pixels as nodata/ignore for the positive scenes). Resample the
~5.5 m masks to 10 m with mode/nearest and tile to <=64x64 local-UTM patches; time_range =
the scene acquisition year (all 2017-2021, post-2016 Sentinel era). All splits are usable as
pretraining labels.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.ch4net_methane_super_emitter_plumes
```

This re-verifies the gate and re-writes this summary; it produces no dataset outputs. To
process once credentials exist: `huggingface-cli login` (or export `HF_TOKEN`), request
access to `av555/ch4net`, then implement the dense_raster path above.
