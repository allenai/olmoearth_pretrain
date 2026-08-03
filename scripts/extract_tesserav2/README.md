# Tessera v2 + OlmoEarth S1 export — Phase 0 recon

Read-only scripts that answer the questions the combined export plan depends
on, for the eight AEF supplemental eval datasets (`AEF_SUPPLEMENTAL_DATASETS`).
Neither script writes to a dataset. Run both on a Weka-mounted machine before
writing any fetch config.

**Why both exports at once.** The datasets currently carry Sentinel-2 only, so
OlmoEarth is scored single-sensor against AEF, which fuses S1 + S2 + more.
Fixing that needs 12 monthly S1 mosaics per window. Tessera v2 has no published
product, so evaluating it means running the v2 students ourselves
(`docs/TesseraV2Inference.md`), which needs every acquisition in the window's
calendar year via the `*_all` layers. Both fetches hit the same Planetary
Computer collections over the same windows, so they share one config, one
shard list, and one PR — but they are **separate `prepare`/`materialize`
invocations**, because the S1 layers are ~5% of the work and shouldn't wait
behind the v2 fetch to become evaluable.

## Order

```bash
# A. Structure — no network. Run on all eight.
python scripts/extract_tesserav2/dataset_shape.py --json_out shape.json

# B. Cost — hits the PC STAC API. Two or three datasets is enough; coverage
#    varies by geography, so pick ones that differ (e.g. ethiopia + lcmap).
python scripts/extract_tesserav2/fetch_cost_probe.py \
    --datasets ethiopia_crops,lcmap_lu --sample 15 --json_out cost.json
```

## What the output decides

| Finding | Consequence |
|---|---|
| **CASE A** — every window's time range is its calendar label year | Tessera v2's `*_all` layers ride the eval windows; no mirrored group needed |
| **CASE B** — time ranges are offset (Sep–Sep) or mixed | Generalize `pastis_tessera_v2.create_windows` to take the year per window (it hardcodes `PRODUCT_YEAR = 2019`) and fetch v2 in a second group |
| Item groups **chronological, ~monthly** | A second sensor aligns by group index; mirror the S2 layer's `query_config` (see below) |
| Item groups **reverse chronological** | Same, but the S1 layers must be listed in model.yaml in the same descending order. Also means the ascending timestamps synthesized by `rslearn_dataset.get_timestamps` currently label the series backwards |
| Item groups **not chronological** | S1 timestep *i* won't correspond to S2 group *i*; scope re-materializing S2 before adding a second sensor |
| Per-window label year found in `options` / the window name | That's the `year` argument replacing `PRODUCT_YEAR` |
| Layer shapes differ across the sample | Some windows are partly materialized; the shard list must tolerate it |
| Projected reads/files from the cost probe | The scope decision: full corpus vs a stratified v2 subsample. File count on Weka, not bytes, is the binding constraint |
| `>CAP` flags | That window-year has more acquisitions than the layer's `max_matches`, so v2 would see a truncated series. Raise the cap or accept it knowingly |

Zero ascending *or* descending S1 over a window is normal (orbit geometry) and
`build_dpixel_inputs` handles it. Zero **S2** is not — that window can't be
embedded by v2.

## Measured 2026-07-31

All eight datasets fetch S2 as **one layer, 12 groups**, via
`space_mode: PER_PERIOD_MOSAIC` + `period_duration: 30d` + `max_matches: 12`,
`sort_by: eo:cloud_cover` (which picks the scene *within* each period, not the
period order). The groups come back **reverse chronological** — group 0 is the
latest 30-day period, group 11 the earliest.

So the S1 layer should **mirror that `query_config` exactly** — one `sentinel1`
layer with the same `space_mode` / `period_duration` / `max_matches` — rather
than 12 hand-written `sentinel1_mo*` layers with `time_offset`s. Same period
boundaries, same ordering, alignment by construction, and it works unchanged
for the CASE B datasets whose windows start mid-year. Use the non-deprecated
`space_mode: MOSAIC` + `period_duration` spelling and verify on one window that
the group ordering matches S2 before scaling.

The CASE A/B split is 4/4: `africa_crop_mask`, `descals`, `glance`, `lcmap_lu`
are calendar-aligned; `canada_crops_coarse`, `canada_crops_fine`,
`ethiopia_crops`, `us_trees` carry per-window observation-dated ranges and need
mirrored fetch windows for Tessera v2.

## After Phase 0

Batch the config churn, since doing both exports together is what saves the
work: add the `sentinel1` layer **and** the `tessera_v2` layer
to each dataset's `config.json` in one edit (the v2 layer is inert until
something reads it), add both `model.yaml` inputs with `required: false`, add
both to the `Pad` transform's `image_selectors`, add both modalities to the
registry, then re-stamp `config_json_sha256` once with
`scripts/tools/backfill_eval_registry_provenance.py`.

The `*_all` fetch layers stay **out** of each dataset's `config.json` — put
them only in a git-tracked fetch config passed via `rslearn --config`. The v2
inference step reads those rasters through `window.get_raster_dir(...)`, not
through the dataset config, so they never need declaring. This also defuses the
landmine in `docs/TesseraV2Inference.md`: a `prepare`/`materialize` that forgets
`--enabled-layers` against a config containing `*_all` would fetch a year of
scenes for 160k windows.
