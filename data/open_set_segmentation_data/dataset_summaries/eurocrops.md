# EuroCrops

- **Slug:** `eurocrops`
- **Status:** completed
- **Task type:** classification (crop type, per-pixel)
- **Samples:** 18,590 rasterized field-parcel tiles
- **Classes:** 254 HCAT classes (uint8; 255 = nodata/ignore)

## Source

EuroCrops — the largest harmonized open EU crop-type parcel dataset, built from national
LPIS / CAP farmer declarations and mapped to the shared **HCAT** (Hierarchical Crop and
Agriculture Taxonomy) via each parcel's `EC_hcat_c` 10-digit code.

- Zenodo record **10118572** (`https://zenodo.org/records/10118572`),
  GitHub `https://github.com/maja601/EuroCrops`.
- License: **CC-BY-4.0** (open, no credential required).
- Access: `download.download_zenodo("10118572", ...)` — plain HTTPS, unauthenticated.

## Access / bounded subset

EuroCrops is very large (dozens of country archives, tens of GB). We downloaded a
**bounded, geographically diverse subset of 8 countries** spanning the main European
biogeographic regions and crop mixes rather than attempting full coverage:

| Code   | Country / region            | Year | Parcels   | Source CRS |
|--------|-----------------------------|------|-----------|------------|
| PT     | Portugal (Mediterranean)    | 2021 | 100,000   | EPSG:4326  |
| ES_NA  | Spain / Navarra (Mediterr.) | 2020 | 996,679   | EPSG:25830 |
| AT     | Austria (Alpine)            | 2021 | 2,610,510 | EPSG:31287 |
| DK     | Denmark (Nordic lowland)    | 2019 | 587,461   | ETRS89/UTM32N |
| EE     | Estonia (Baltic)            | 2021 | 176,064   | EPSG:4326  |
| HR     | Croatia (Balkans)           | 2020 | 1,381,932 | EPSG:3765  |
| NL     | Netherlands (maritime NW)   | 2020 | 767,034   | EPSG:4326  |
| SE     | Sweden (Nordic)             | 2021 | 1,212,180 | EPSG:32633 |

Raw archives + `HCAT3.csv` are in `raw/eurocrops/` (see `SOURCE.txt`); shapefiles are
unzipped under `raw/eurocrops/unzip/{CODE}/`.

## Label construction

- **label_type = polygons.** Each selected field parcel is rasterized (via
  `rasterize.rasterize_shapes`, `all_touched=True`) into a **≤64×64 UTM 10 m** tile,
  sized to the parcel footprint and capped at 64 on each axis, centered on the parcel
  bounding box.
- Value inside the polygon = the parcel's crop **class id**; everything outside =
  **255 (nodata/ignore)**. EuroCrops only supplies a ground-truth crop label *inside*
  declared parcels, and inter-parcel land (roads/forest/water/buildings) has no crop
  label, so "outside" is an ignore region, **not** a background class. Geometries are
  reprojected to WGS84 and then to the tile's local UTM zone.
- 32 candidate parcels rasterized to zero pixels (sub-10 m slivers) and were skipped.

## Class mapping

- Classes are the **HCAT leaf codes present in the sampled parcels**. Names come from the
  repo mapping `data/eurocrops_hcat3_mapping.json` (equivalent to the record's `HCAT3.csv`).
- Classification labels are **uint8** → at most 254 classes. **262** distinct HCAT codes
  appeared across the 8 countries; we kept the **top 254 by global frequency** and dropped
  the 8 rarest (all with a handful of parcels): HCAT codes
  `3303020600, 3301061220, 3301061239, 3301061205, 3301210400, 3301080100, 3301083900,
  3301090206`.
- Class **ids are assigned 0..253 in descending global HCAT-code frequency** (id 0 =
  `pasture_meadow_grassland_grass`, 1 = `arable_crops`, 2 = `not_known_and_other`,
  3 = `tree_wood_forest`, 4 = `fallow_land_not_crop`, 5 = `winter_common_soft_wheat`, …).
  Full id↔name↔HCAT-code table is in `metadata.json` (`classes`).
- Note: a few high-frequency HCAT nodes are intermediate/aggregate categories
  (`arable_crops`, `not_known_and_other`); they are legitimate HCAT taxonomy entries and
  were kept as-is. The strong perennial-crop classes highlighted for EuroCrops are present
  (`vineyards_wine_vine_rebland_grapes`, `olive_plantations`, `orchards_fruits`).

## Sampling

- **Tiles-per-class balanced** with the hard 25k per-dataset cap:
  `balance_by_class(records, key="class_id", per_class=1000, total_cap=25000)`. With 254
  classes the effective per-class cap is `25000 // 254 = 98`.
- 145 of 254 classes reached the 98-tile cap; the remaining (rarer) classes contribute all
  available parcels (35 classes have <10 tiles; min = 1). Total = **18,590** (well under
  25k). Each tile counts toward the class of its center parcel.

## Time range

1-year window anchored on each country snapshot's year (e.g. AT/EE/PT/SE = 2021,
ES_NA/HR/NL = 2020, DK = 2019) via `io.year_range(year)`. No change labels.

## Verification

- Output tiles: single band, uint8, local UTM at 10 m/pixel, ≤64×64, nodata=255 — confirmed
  on a 300-tile sample; all class values in [0,253] ∪ {255}, max id seen 235.
- Every `.tif` has a matching `.json`; all sampled time ranges are exactly 1 year.
- `metadata.json` `class_counts` sum = 18,590 = num_samples; all 254 classes have ≥1 tile.
- Geographic sanity (500-tile sample): centroids span lon −9.4…27.6, lat 37.4…65.5, all
  within Europe, all 8 countries represented.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.eurocrops
```

Idempotent: re-running skips already-written `locations/{id}.tif`. Bounded country set,
class-id assignment, and 254-class cap are deterministic (frequency-ranked;
`balance_by_class` uses a fixed seed).

## Caveats

- Per-parcel tiles: outside the target parcel is ignore (255), so most tiles are a single
  crop class with an ignore surround; large parcels (>640 m) fill the tile uniformly.
  Neighboring parcels are **not** co-rasterized into the tile.
- Country LPIS declarations vary in taxonomic depth: ES_NA (10 codes) and HR (11 codes)
  record only coarse HCAT categories, while NL/DK/EE (~130 codes) are fine-grained. The
  merged class set reflects this heterogeneity.
- Some HCAT parent/aggregate and `not_known_and_other` codes are retained as classes.
