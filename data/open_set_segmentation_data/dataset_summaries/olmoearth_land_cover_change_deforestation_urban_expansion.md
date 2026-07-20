# OlmoEarth land-cover change (deforestation & urban expansion)

- **Slug:** `olmoearth_land_cover_change_deforestation_urban_expansion`
- **Status:** completed
- **Task type:** classification (binary change)
- **Samples:** 2000 (1000 negative + 1000 positive)
- **Label output:** per-window single-band uint8 GeoTIFFs (`locations/{id}.tif` + `.json`)

## Source

Two local rslearn eval datasets (`have_locally: true`, no download):

- `/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/lcc_deforestation`
- `/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/lcc_urban_expansion`

`raw/<slug>/SOURCE.txt` points at both (nothing copied).

Each dataset has **13,812 windows**. The two datasets share the **exact same window set**:
every `(group, name)` key exists in both with identical CRS, pixel bounds and time_range
(verified on the full join — 0 geometry mismatches; a 4,000-window cross-tab confirmed
matching geometry). They differ only in which change type each window is scored for.

Per window:
- `metadata.json` `options.category` ∈ {`positive`, `negative`} — a **per-tile binary
  change label** (the `label` vector layer is a full-window polygon carrying the same
  `category`).
- Projection = local UTM at 10 m; `bounds` = a **64×64** pixel tile (640 m); `time_range`
  = a **1-year "post" observation window**.
- Config: `pre_sentinel2` is materialized at `time_offset: -1095d` (~3 years before the
  window time_range), `post_sentinel2` at `0d`. The annotation compares the pre mosaic to
  the post mosaic.

Category distribution (full sets): deforestation 12,926 neg / 886 pos; urban 13,205 neg /
607 pos. Splits (both): 12,046 train / 1,766 val — **all splits used** (no filtering).

## Class scheme (unified, binary)

The two source datasets are combined into **one** dataset (spec §5) by joining on
`(group, name)` and taking the **union of positives**:

| id | name | meaning |
|----|----------|---------|
| 0 | negative | no change: negative in **both** deforestation and urban-expansion annotations |
| 1 | positive | deforestation **and/or** urban expansion occurred (union of the two positive sets) |

Rationale for the union rather than per-source processing: a deforestation-negative tile
can be urban-positive (and vice versa), so processing each source independently would emit
the same geographic tile with contradictory labels. Requiring a negative to be negative in
**both** sources keeps class 0 coherent, and the union makes class 1 = "any of these two
change types." This matches the manifest's binary `["negative","positive"]` scheme and its
"binary pre/post change classification" description.

Joined counts before balancing: 12,459 negative / 1,353 positive. (Of the positives,
~0.85% of windows are positive in *both* source datasets — genuine co-occurring
deforestation + urban expansion; `source_id` records `types=deforestation+urban_expansion`.)

## Label patches

Each label is a **coherent full-tile change annotation** (the source polygon covers the
whole 64×64 window), so labels are written as **dense single-band GeoTIFFs**, not a point
table. Every pixel of a tile carries the tile's class id (uniform 0 or uniform 1). Native
footprint 64×64 at 10 m in the window's own UTM CRS (reused directly). nodata = 255
(unused — tiles are uniform).

**Caveat — this is a tile-level (scene-level) change label, not a precise sub-tile change
mask.** A positive tile means "deforestation/urban expansion occurred somewhere in this
640 m tile"; the annotation marks the entire tile positive. Downstream should treat these
as coarse full-tile change labels.

## Time range & change handling (spec §5)

The change is defined by comparing a **post mosaic** against a **pre mosaic ~3 years
earlier** (`time_offset: -1095d`), so it is encoded under the **pre/post change scheme**:
each sample carries two independent six-month windows (each ≤ 183 days) with `time_range` =
**null**.

- `post_time_range` = a ~6-month window **centered in the post-observation year** (post
  years range 2020–2027, concentrated 2020–2025).
- `pre_time_range` = a ~6-month window **centered ~3 years earlier** (season-aligned to the
  post window).
- `change_time` = **midpoint** of the two windows (a reference only).

**Previously rejected; now resolved by pre/post windows.** The precise transition moment is
not resolvable to within ~1–2 months (the true change could have happened anywhere in the
~3-year span between the mosaics), which is why this dataset was originally **rejected** on
change-timing grounds. That imprecision is no longer a problem under the pre/post scheme:
the coarse timing simply sits in the gap between the two far-apart windows, so the dataset is
**completed / usable**.

## Sampling

`balance_by_class(records, "label", per_class=1000)` → 1000 negative + 1000 positive = 2000
(well under the 25k cap). Positives (1,353 available) are the scarcer class; 1000 kept.
No fabricated negatives, no dropped classes.

## Verification

- 2000 `.tif` + 2000 `.json`; opened samples: single band, uint8, UTM CRS at 10 m, 64×64,
  values ∈ {0} (neg) or {1} (pos).
- All 2000 JSONs: `time_range` = **null**, `pre_time_range` and `post_time_range` each ≤ 183
  days with `change_time` set between them (0 failures); `metadata.json` class ids {0,1}
  cover all raster values.
- Spatial sanity: place-named windows resolve correctly (Toronto ≈ −79.4/43.8, Abu Dhabi
  ≈ 54.5/24.3, matching the `source_id` city hints).
- Idempotent: re-running skips existing `{id}.tif`.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_land_cover_change_deforestation_urban_expansion
```
