# OlmoEarth US tree genus (`olmoearth_us_tree_genus`)

## Source
Local rslearn eval dataset at
`/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/us_trees` (`have_locally: true`,
`source: olmoearth`, `license: internal`). Referenced via `raw/olmoearth_us_tree_genus/SOURCE.txt`
(not copied). Tree-genus reference points across the United States, derived from a tree
inventory. 45,382 windows total (`train`: 11,700, `test`: 33,682); all source splits used.

Each window is one point label. The genus name, lon/lat, split, and a ~1-year time range live
in the window `metadata.json` `options` block (`label`, `lon`, `lat`, `split`); the genus is
also stored as `properties.label` in the window's `label` vector layer (`data.geojson`). The
source additionally ships a rasterized `label_raster` layer and ingested Sentinel-2 imagery,
neither of which is needed here.

## Task type
Classification — sparse point segmentation. Each label is a single 10 m pixel carrying a
genus id, so per spec §2a we write **one dataset-wide `points.geojson`** (no per-point
GeoTIFFs), via `io.write_points_table`.

## Classes
39 plant genera. Well under the 254-class uint8 cap, so **no genus was dropped**. Class ids
are assigned by descending source frequency (spec §5 top-by-frequency rule; ties broken
alphabetically for determinism — the 12 most-common genera are tied at 1,596 windows each).
Common-name glosses were added to each class `description` in `metadata.json` (the source
stores only the latin genus).

Genera: quercus, pinus, acer, populus, picea, abies, juniperus, betula, fagus, salix, carya,
tsuga, pseudotsuga, liquidambar, prunus, ilex, cercis, yucca, cornus, elaeagnus, liriodendron,
prosopis, sassafras, diospyros, magnolia, ailanthus, aesculus, juglans, asimina, ulmus, thuja,
morus, gleditsia, maclura, triadica, sabal, taxodium, alnus, amelanchier.

## Sampling
`sampling.balance_by_class(records, "label", per_class=1000)` with the default
`total_cap=25000`. With 39 classes the effective per-class limit drops to
`25000 // 39 = 641`. Genera with fewer available windows contribute all they have. Result:
**24,536 points**, 503–641 per class. No rare-class dropping or negative fabrication (assembly
handles both downstream, spec §5).

Source raw genus-window counts range 503 (amelanchier) to 1,596 (12-way tie at top); all
selected class counts land at 503–641.

## Time range
Annual labels. Each point gets a 1-year window anchored on its labeled year via
`io.year_range(year)` where `year = int(window.time_range[0][:4])`. Labeled years span
2017–2022 — **all post-2016 (Sentinel era)**; nothing filtered on the pre-2016 rule. No
change labels (`change_time` null).

## Georeferencing
Point coordinates are passed through directly (WGS84 `[lon, lat]`) from the validated source
window options / label vector, so placement on the S2 grid is exact. A per-pixel Sentinel-2
overlay check is not meaningful for a single-tree point at 10 m; coordinates are trusted from
the source eval dataset.

## Outputs
- `datasets/olmoearth_us_tree_genus/points.geojson` — FeatureCollection, `count` = 24,536.
- `datasets/olmoearth_us_tree_genus/metadata.json` — 39 classes + counts, `nodata_value` 255.
- `datasets/olmoearth_us_tree_genus/registry_entry.json` — status `completed`.
- `raw/olmoearth_us_tree_genus/SOURCE.txt` — pointer to the local source path.

## Reproduce
```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_us_tree_genus
```
Idempotent: re-running rescans the source and atomically overwrites `points.geojson` /
`metadata.json` with identical (seeded) output.

## Caveats
- Genus (not species) resolution; a single point marks presence of a tree of that genus.
- Labels are inventory-derived reference points, not per-pixel species maps; downstream
  pretraining pairs them with imagery by geography/time.
- 12 top genera are frequency-tied at 1,596, so their id ordering is alphabetical, not
  strictly by count.
