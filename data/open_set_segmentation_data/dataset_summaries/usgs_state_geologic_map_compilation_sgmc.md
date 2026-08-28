# USGS State Geologic Map Compilation (SGMC)

- **Slug:** `usgs_state_geologic_map_compilation_sgmc`
- **Status:** completed
- **Task type:** classification (per-pixel generalized surface lithology)
- **Family:** geology · **label_type:** polygons · **region:** conterminous US
- **Samples written:** 15,767 label tiles · **Classes kept:** 31 (of 33 GENERALIZED_LITH values) · **Classes dropped:** 2

## Source

Horton, J.D., San Juan, C.A., and Stoeser, D.B. (2017), *The State Geologic Map Compilation
(SGMC) geodatabase of the conterminous United States*, USGS Data Series 1052
(doi:10.3133/ds1052; data release doi:10.5066/F7WH2N65). Public-domain ESRI file
geodatabase distributed by USGS ScienceBase (item `5888bf4fe4b05ccb964bab9d`) and referenced
at https://mrdata.usgs.gov/geology/state/ . No credential required.

- `USGS_SGMC_Geodatabase.zip` (~416 MB) — downloaded to
  `raw/usgs_state_geologic_map_compilation_sgmc/USGS_SGMC_Geodatabase.zip`, extracted to
  `raw/.../extracted/USGS_SGMC_Geodatabase/USGS_StateGeologicMapCompilation_ver1.1.gdb`.
- `USGS_SGMC_Tables_CSV.zip` (~1.5 MB) also fetched (schema inspection only; not used at
  runtime — the polygon layer carries the lithology attribute directly).

Note: a 2026 re-release exists (doi:10.5066/P1A3DQZK, newer USGS Geologic Map Schema). The
2017 ver 1.1 release used here is the standard, widely cited product and is fully sufficient
for generalized-lithology labels.

## Data structure and label mapping

The polygon feature class `SGMC_Geology` (313,732 MultiPolygons, USA Contiguous Albers
Equal-Area Conic ESRI:102039, metres) carries a **curated per-polygon `GENERALIZED_LITH`
field** — a standardized generalized-lithology category (33 distinct values). This is exactly
the "generalized lithology categories" the dataset advertises, so it is used directly as the
per-pixel class; **no CSV join is required** (the separate `SGMC_Lithology` LITH1–5 table is
a more verbose hierarchy of the same information).

- **Dropped (non-lithology):** `Unknown` (26 polygons) and `Dam` (7 polygons, an
  anthropogenic structure). Total dropped ≈ 33 polygons.
- **Kept:** the remaining **31 classes**, ids 0–30 assigned by **descending global polygon
  frequency**. Natural non-rock surface types `Water` (id 6) and `Ice` (id 30) are retained
  as legitimate, 10–30 m-observable surface units (same treatment as the GLiM sibling).
- Well under the 254-class uint8 cap; no frequency truncation needed.

**Geologic age** (also in the geodatabase, via the `Age` table) is **not encoded**: a
single-band per-pixel label can hold only one attribute, and surface lithology is the more
directly observable-at-10–30 m property. Age is left for a possible future age-labeled
variant.

## Processing recipe (mirrors `glim_global_lithological_map.py`)

SGMC is a generalized ~1:500,000-scale compilation, so — per spec §5 (large derived product)
— we **sample bounded homogeneous tiles from large polygons** rather than tracing boundaries:

1. Keep polygons with equal-area footprint `Shape_Area ≥ 1 km²` (~2.4× a 640 m tile) →
   145,138 candidate polygons across the 31 kept classes.
2. Sample up to 2,000 polygons/class (deterministic, seed 42), reproject those to WGS84 →
   31,525 candidate tasks.
3. Each seed polygon → one 64×64 (640 m) tile in local UTM at 10 m, centered on the
   polygon's interior representative point. Rasterize the seed lithology (all_touched);
   pixels **outside** the seed polygon = **255 (nodata/ignore)**, not a fabricated background
   class (positive-only foreground mask, spec §5 — every land pixel is *some* rock type and
   neighbours are intentionally left unresolved at this coarse scale). Downstream assembly
   supplies negatives from other datasets.
4. Drop candidates whose seed class covers < 0.5 of the tile → 29,736 homogeneous candidates.
5. Class-balanced selection (`balance_by_class`, total_cap=25,000): with 31 classes the
   effective per-class cap is `25000 // 31 = 806`. Frequent classes get 806 each; rarer
   classes keep all they have.

**Result:** 15,767 tiles; 7,917 carry an ignore border (tile straddles a polygon edge), the
rest are near-uniform single-class.

- **dtype/nodata:** uint8, nodata/ignore = 255.
- **Tile size:** 64×64, local UTM, 10 m/px, north-up.
- **Time range:** lithology is a **static** label with no per-polygon date → representative
  Sentinel-era 1-year window **2020-01-01 → 2021-01-01**; `change_time` = null.

## Samples per class (written)

Frequent classes at 806 (the total-cap-derived per-class ceiling): sedimentary_clastic,
unconsolidated_undifferentiated, sedimentary_undifferentiated, sedimentary_carbonate,
igneous_volcanic, igneous_intrusive, water, metamorphic_undifferentiated,
metamorphic_sedimentary_clastic, metamorphic_gneiss,
metamorphic_and_sedimentary_undifferentiated, igneous_and_sedimentary_undifferentiated,
unconsolidated_and_sedimentary_undifferentiated, metamorphic_schist,
igneous_and_metamorphic_undifferentiated.

Under-806 (fewer qualifying large/homogeneous polygons): metamorphic_volcanic 743,
igneous_undifferentiated 681, metamorphic_intrusive 437, metamorphic_amphibolite 342,
sedimentary_chemical 328, metamorphic_carbonate 321, metamorphic_serpentinite 216,
metamorphic_sedimentary 153, melange 117, tectonite_undifferentiated 108, sedimentary_iron_formation_undifferentiated 54, metamorphic_other 52, metamorphic_granulite 45, sedimentary_evaporite 68, **metamorphic_igneous 6, ice 6** (sparsest). Sparse
classes are kept per spec §5; downstream assembly discards any that fall below its minimum.

## Verification

- Opened multiple tifs: single band, uint8, 64×64, UTM CRS at 10 m, nodata 255, values are
  valid class ids in 0–30. Every tif has a matching `.json` with a 1-year `time_range` and
  `change_time=null`; `metadata.json` covers all 31 class ids.
- Spatial sanity: tile centers converted to lon/lat land inside the US state named in each
  `source_id` (SD, NC, TX, NV, MD, MT, AZ all correct) and classes are geologically
  plausible (e.g. Nevada → igneous, Appalachian NC → metasediments/metavolcanics).
- Idempotent: re-running skips existing `locations/{id}.tif`.

## Caveats

- SGMC is a generalized compilation; lithology is only *partially* inferable at 10–30 m via
  its influence on terrain/soil/vegetation, and polygon boundaries are approximate. Tiles are
  deliberately homogeneous single-class patches, not precise boundary segmentations.
- Positive-only foreground mask (no background class); tiles straddling polygon edges have a
  255 ignore border rather than a resolved neighbouring lithology.
- Conterminous US only (no AK/HI in this product).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.usgs_state_geologic_map_compilation_sgmc
```
Outputs: `datasets/usgs_state_geologic_map_compilation_sgmc/{metadata.json, locations/*.tif, locations/*.json}`
on weka. Tunable flags: `--min_area_m2`, `--per_class`, `--cand_per_class`, `--workers`.
