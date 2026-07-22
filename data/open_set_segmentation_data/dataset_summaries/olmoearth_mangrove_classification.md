# OlmoEarth mangrove classification

- **slug:** `olmoearth_mangrove_classification`
- **status:** completed
- **task_type:** classification (sparse points → `points.geojson`, spec §2a)
- **num_samples:** 3000 (1000 per class)

## Source

Local rslearn eval dataset (`have_locally: true`), not copied — `raw/<slug>/SOURCE.txt`
points at it:
`/weka/dfive-default/rslearn-eai/datasets/mangrove/classification/20250626`

Labels are derived from **Global Mangrove Watch** (2020 baseline). The rslearn dataset has
two window groups, both used (spec §5 — all source splits are fair game):

- `reference` — 49,600 windows, `split=test`; classes present: Other (37,528), Mangrove (12,072).
- `sample_100K` — 100,000 windows, `split=train`/`val`; all three classes: Mangrove (45,850), Water (24,177), Other (29,973).

Total scanned: **149,600 windows**. Overall class availability: Mangrove 57,922, Water
24,177, Other 67,501 — every class has ≫1000, so all three truncate to 1000.

## Structure & label mapping

Each window is a 32×32 @10 m UTM patch carrying a **single uniform class** in
`metadata.json` `options.label` (duplicated as a one-category polygon covering the whole
window in the `label` vector layer, and as a uniform `label_raster`). A uniform-class
window is effectively a **sparse point**, matching the manifest `label_type: points`, so we
emit one dataset-wide GeoJSON point table (spec §2a), not per-window GeoTIFFs.

Class map (manifest order → id):

| id | name | description |
|----|------|-------------|
| 0 | Mangrove | Mangrove forest / mangrove-covered tidal wetland (per GMW). |
| 1 | Water | Open water: ocean, tidal channels, other permanent/standing water. |
| 2 | Other | Any non-mangrove, non-water land cover. |

nodata = 255 (uint8 class convention).

## Point location & time range

- **Location:** WGS84 center of each window's UTM `bounds` (computed via rslearn
  `STGeometry.to_projection`). Verified against the `{sample_id}_{lat}_{lon}` window-name
  encoding — agree to sub-pixel (~5 m).
- **Time range:** all source windows share a curated 30-day range
  (`2020-06-15 .. 2020-07-15`). Since these are static/annual GMW land-cover labels, each
  point is assigned the **spec §5 land-cover default: a 1-year window anchored on the
  labeled year (2020-01-01 .. 2021-01-01)**. See judgment call below.

## Sampling

`balance_by_class(per_class=1000, total_cap=25000)` → 1000/class × 3 = **3000 points**.
3 classes, so the 25k cap does not bind. No fabricated negatives (all three are real
classes; §5). No rare-class dropping needed.

## Outputs (on weka)

- `datasets/olmoearth_mangrove_classification/points.geojson` — 3000 Point features.
- `datasets/olmoearth_mangrove_classification/metadata.json`
- `datasets/olmoearth_mangrove_classification/registry_entry.json` (status=completed)
- `raw/olmoearth_mangrove_classification/SOURCE.txt`

## Verification

- points.geojson: FeatureCollection, count=3000, label counts {0:1000, 1:1000, 2:1000},
  3000 unique ids, single time_range, lon∈[-178.5,179.9], lat∈[-38.7,31.2] (tropical/
  subtropical coasts, as expected for a mangrove dataset).
- Spatial sanity: per-class sample coordinates land in known mangrove-belt coasts (Colombia
  Pacific, Thailand, Guinea-Bissau, Gabon, Tampa Bay, etc.). Georeferencing is inherited
  verbatim from a source rslearn dataset already matched to S2/S1 imagery in the OlmoEarth
  eval, so overlay alignment is trusted; no misalignment observed.

## Judgment calls

- **Time range 30-day → 1-year (2020).** The source eval fixed a 30-day June–July 2020
  window (for its imagery matching). For pretraining label co-location I used the spec §5
  land-cover default of a 1-year window anchored on the labeled year, giving the assembly
  step more S2/S1/Landsat imagery to pair with. Both are ≤1 year and valid; recorded here
  and in `metadata.json` notes. (Minor caveat: water/mangrove tidal boundaries can shift
  seasonally within a year, but GMW class assignment is annual.)
- **Both window groups used.** `reference` (test) and `sample_100K` (train/val) are both
  drawn from per spec §5 "use all splits". `reference` lacks the Water class; Water comes
  entirely from `sample_100K`.
- **Point vs GeoTIFF.** Windows are uniform-class 32×32, i.e. a single (location, class)
  pair — treated as sparse points per spec §2a/§4, avoiding thousands of redundant tiny
  tifs.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_mangrove_classification
```

Idempotent: re-running overwrites `points.geojson`/`metadata.json` atomically with the same
seeded selection (`balance_by_class` seed=42).
