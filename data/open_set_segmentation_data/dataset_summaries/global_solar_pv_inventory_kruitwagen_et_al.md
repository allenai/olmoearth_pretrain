# Global Solar PV Inventory (Kruitwagen et al.)

- **Slug:** `global_solar_pv_inventory_kruitwagen_et_al`
- **Status:** completed — classification, 1000 samples
- **Task type:** classification (single foreground class: `solar_pv`)
- **Family / region:** solar / global
- **License:** CC-BY-4.0

## Source

Kruitwagen, L. et al. "A global inventory of photovoltaic solar energy generating units",
*Nature* 598, 604-610 (2021). Data: Zenodo record **5005868** (CC-BY-4.0).

Accessed unauthenticated over HTTP from the Zenodo file API:
- `predicted_set.geojson` (274 MB, 68,661 polygons) — the **full predicted inventory**;
  carries per-feature `install_date` and `capacity_mw`. **This is the file we use.**
- `test_polygons.geojson` (6.4 MB, 7,263 polygons) — the manually photointerpreted test
  set; downloaded for provenance but **not used for labels** because it has geometry only
  (`aoi`/`id`), no dates or capacity, so time ranges cannot be assigned from it.

The inventory maps utility-scale PV generating units globally, detected from a 2016-2018
Sentinel-2 composite + SPOT 6/7 with a manually verified test set (model-derived labels).
Both files are WGS84 (EPSG:4326) polygons.

## Label mapping

Single-foreground-class **polygon** dataset, rasterized (spec §4 polygons) into footprint-
sized ≤64×64 local-UTM 10 m/pixel tiles:

| id  | name       | meaning |
|-----|------------|---------|
| 0   | background | non-PV land inside the tile (genuine surrounding land) |
| 1   | solar_pv   | PV generating-unit footprint (polygon rasterized, `all_touched`) |
| 255 | nodata     | not used here (no ignore pixels for these polygons) |

Each polygon → one tile centered on the polygon bbox center, sized to the footprint but
capped at 64×64 (640 m). ~3.4% of polygons exceed 640 m; those yield an all-solar 64×64
center tile (11 of 1000 selected tiles are all-solar, no background). Background pixels are
real surrounding land (spatially meaningful negatives, same convention as
`global_renewables_watch` / `olmoearth_solar_farm`), so **no separate/fabricated negative
tiles** are emitted (spec §5). Mean per-tile solar fraction ≈ 0.71.

## Time range handling

Solar farms are persistent once built, and every polygon is present in the 2018 detection
snapshot. `install_date` is one of: a concrete `YYYY-MM-...` (2016/2017/2018 commissioning
month), `<2016-...` (built before 2016), or empty (unknown). A **1-year window in which the
farm is fully present** is assigned:

- install year 2016 → **2017** window
- install year 2017 → **2018** window
- install year 2018 → **2019** window (first full year after commissioning)
- `<2016` or unknown → **2018** window (representative Sentinel-era snapshot; farm present)

All windows are post-2016, so no polygon is dropped on the pre-2016 rule. (Observed output
windows: 2017-2018, 2018-2019, 2019-2020, all ≤1 year.) `change_time` is null — treated as
a persistent label, not a dated change event.

## Sampling

Model-derived (not in-situ reference) labels, so we **prefer the higher-confidence
detections**: confidence **A or B** only (53,876 of 68,661 polygons; C/D dropped). Capped at
**1000 `solar_pv` tiles** (spec §5: up to 1000 locations per class), stratified across
install-year buckets `{2016, 2017, 2018, pre2016, unknown}` at 200 each for temporal +
geographic diversity. Selection is global (42 UTM zones in the output).

Class-tile counts: `solar_pv` present in 1000 tiles, `background` present in 989 tiles.

## Verification

- 1000 `.tif` + 1000 `.json`; every tif single-band uint8, UTM CRS at 10 m, size 2-64 px.
- Pixel values are exactly {0, 1}; nodata 255 declared, unused.
- All `time_range`s ≤ 1 year and post-2016.
- Spot round-trip: sample tile centers land on plausible global PV locations (e.g. tile
  000500 → 140.69°E, 35.82°N, Japan). Full S2 image overlay not performed (relied on the
  verified WGS84→UTM reprojection and the sibling solar-dataset recipe).

## Judgment calls / caveats

- Used `predicted_set.geojson` (not the manual `test_polygons.geojson`) because only it has
  dates/capacity needed for time ranges — as directed by the task note.
- Restricted to confidence A/B to raise label reliability for this derived product; ~14.8k
  C/D polygons excluded. Documented so it can be revisited if more samples are wanted.
- Time windows anchored to the first full year *after* commissioning so imagery in the
  window shows a fully-present farm (a refinement over anchoring on the build year itself).
- Single foreground class → only ~1000 samples (matches the per-class cap and the
  `global_renewables_watch` solar precedent); downstream assembly supplies negatives from
  other datasets.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_solar_pv_inventory_kruitwagen_et_al
```
Idempotent (skips already-written `locations/{id}.tif`). Outputs under
`/weka/dfive-default/helios/dataset_creation/open_set_segmentation/datasets/global_solar_pv_inventory_kruitwagen_et_al/`.
