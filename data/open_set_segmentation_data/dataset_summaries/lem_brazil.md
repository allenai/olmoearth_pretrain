# LEM+ (Brazil) — crop-type field polygons

- **Slug:** `lem_brazil`
- **Status:** completed
- **Task type:** classification (per-pixel crop type)
- **Samples:** 3606 (rasterized crop-episode tiles) across 15 classes
- **Source:** "LEM: A dataset for crop type mapping" / LEM+ — Mendeley Data
  `vz6d7tw87f` v1, CC-BY-4.0. https://data.mendeley.com/datasets/vz6d7tw87f/1
- **Region / period:** western Bahia, Brazil (tropical Cerrado agriculture);
  agricultural year **Oct 2019 – Sep 2020**.

## Source

A single 1.2 MB zip (`LEM_dataset.zip`) containing an ESRI shapefile
(`LEM_dataset.shp`, WGS84 / EPSG:4326) of **1,854 field polygons**. Each polygon
carries **12 monthly crop/land-use labels** (columns `Oct_2019 … Sep_2020`) from a
manual field survey. Labels only — no imagery (pretraining supplies its own). Downloaded
directly from the Mendeley public-files endpoint (needs a browser User-Agent header);
no credentials required.

## Access

`download.download_http` of the public file, then `download.extract_zip`. Raw kept at
`raw/lem_brazil/` with `SOURCE.txt`.

## Label handling — crop episodes

This is a **double-cropping** region, so a field's label changes month to month
(e.g. `Uncultivated soil → Soybean → Uncultivated soil → Brachiaria`). To keep that
signal without contradictory supervision at one location, each polygon's 12-month
sequence is split into **crop episodes** = maximal runs of consecutive months sharing the
same label (6,307 episodes total). Each episode is one sample:

- **geometry**: the field polygon rasterized (`all_touched`) into a ≤64×64 local-UTM
  10 m tile centered on the polygon centroid — class id burned inside the polygon,
  **255 = nodata/ignore outside** (no background class; unlabeled land is ignore). Large
  fields (median ≈102 ha, up to 2,620 ha) yield a homogeneous 64×64 sub-window fully
  inside the field.
- **class**: the episode's crop label.
- **time_range**: first day of the episode's first month → first day of the month after
  its last, **clamped to ≤360 days**. Consecutive episodes at a field have disjoint month
  spans, so their windows do not overlap (no contradictory multi-label supervision).
  Transient crops get their true sub-year presence window; perennials / fallow that persist
  all year get the clamped ~360-day window. This realizes the intended "coherent 1-year
  window" (the 2019-10..2020-09 agricultural year) subdivided into the episodes the monthly
  ground truth records.

`"Not identified"` (198 monthly cells; annotator could not determine the crop) is treated
as **ignore** — it breaks an episode run and never becomes a class.

## Classes (id by descending global episode frequency) and written counts

| id | class | count |
|----|-------|------|
| 0 | Uncultivated soil | 1000 |
| 1 | Soybean | 999 |
| 2 | Millet | 432 |
| 3 | Brachiaria | 254 |
| 4 | Corn (maize) | 211 |
| 5 | Sorghum | 168 |
| 6 | Cerrado | 161 |
| 7 | Cotton | 139 |
| 8 | Pasture | 102 |
| 9 | Beans | 36 |
| 10 | Conversion area | 35 |
| 11 | Eucalyptus | 26 |
| 12 | Hay | 21 |
| 13 | Coffee | 20 |
| 14 | Crotalaria | 2 |

Class-balanced (`balance_by_class`, ≤1000/class, 25k cap; 15 classes → effective cap
1000). `Uncultivated soil` and `Soybean` are capped; all others kept in full, including
sparse classes (`Crotalaria`=2) which downstream assembly may drop. Soybean shows 999 (not
1000) because one tiny sliver polygon rasterized to an empty tile and was skipped.

## Caveats

- No background/negative class — outside-field pixels are nodata (255); the assembly step
  provides negatives from other datasets.
- Sparse classes (Crotalaria, Coffee, Hay, Eucalyptus, Conversion area, Beans) are retained
  per spec; downstream filtering removes too-small ones.
- Full georeferencing verified (tiles land in western Bahia UTM 23S at 10 m); a Sentinel-2
  visual overlay was not run in this batch, but rasterization uses the same rslearn UTM
  reprojection path validated for `eurocrops`.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.lem_brazil
```
Idempotent (skips already-written `{sample_id}.tif`).
