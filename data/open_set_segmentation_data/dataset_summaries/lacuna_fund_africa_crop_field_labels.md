# Lacuna Fund Africa Crop Field Labels

- **Slug:** `lacuna_fund_africa_crop_field_labels`
- **Status:** completed
- **Task type:** classification (dense 3-class segmentation)
- **Num samples:** 1000
- **Label type:** polygons (`field_boundary` family)
- **License:** CC-BY-4.0 (labels)

## Source

"A region-wide, multi-year set of crop field boundary labels for Africa" (Estes et al.,
2024, arXiv:2412.18483), funded by the Lacuna Fund, led by Farmerline with Spatial
Collective and the Agricultural Impacts Research Group at Clark University.

- GitHub: https://github.com/agroimpacts/lacunalabels
- Data: public Registry of Open Data on AWS bucket `s3://africa-field-boundary-labels`
  (region `us-west-2`, **unsigned / no credential**); also Zenodo record 11060871.

~825k crop-field boundary polygons, **manually digitized by visual interpretation of Planet
NICFI basemaps**, covering continental Africa for imagery months spanning **2017–2023**.

## Access method

Downloaded label-only artifacts (no credential) to `raw/{slug}/`:
- `mapped_fields_final.parquet` (80 MB) — 825,395 field polygons in WGS84, columns
  `fid, name, assignment_id, completion_time, category`.
- `label_catalog_allclasses.csv` — per-assignment metadata: chip-center lon/lat (`x`,`y`)
  and `image_date` (the labelled Planet basemap month, YYYY-MM-15).

`(name, assignment_id)` identifies one labelled ~1.2 km Planet chip. We joined polygons to
the catalog on this key to recover each assignment's center and imagery year. The Planet
image chips and the pre-baked 3-class label rasters were **not** downloaded (pretraining
supplies its own imagery).

## Class mapping

3-class dense segmentation (mirrors the sibling `ai4boundaries` dataset for consistency;
manifest classes were `crop field boundary` / `non-field`, refined into interior vs boundary):

| id | name | definition |
|----|------|------------|
| 0 | non-field | Background within a labelled chip that an annotator examined and did not delineate as a crop field. |
| 1 | crop field interior | Interior of a digitized crop-field polygon (categories annualcropland/fallow/treecrop). |
| 2 | crop field boundary | Parcel-outline pixel of a crop field (boundary wins over interior). |

`category` distribution in the source: annualcropland 824,651; fallow 452; treecrop 233;
unsure2 36; unsure1 22; cloudshadow 1. The three field categories are folded into the field
classes; the ~59 unsure/cloudshadow polygons are written as **nodata/ignore (255)** so they
count as neither field nor background. `nodata_value = 255`.

## Observability at 10 m

Median field ~0.5 ha (~4938 m² ≈ 50 px at 10 m; 5th pct ~9 px), so fields are well resolved.
Field boundaries (~1–2 px at 10 m) are exactly the signal this dataset was built to expose —
same rationale accepted for `ai4boundaries`. Accepted.

## Processing

- **One 64×64, 10 m, local-UTM tile per labelling assignment**, centered on the chip center
  (catalog `x`,`y`). 64 px = 640 m stays inside the ~1.2 km labelled chip footprint (field
  extent per chip: median ~610 m, 90th pct ~800 m), so background pixels are genuinely
  examined non-field land rather than un-labelled area. Polygons reprojected to the tile's
  UTM pixel grid and rasterized; boundaries rasterized with `all_touched=True` (no extra
  dilation — tests showed dilation over-consumed interiors on dense small-field chips).
- Candidates: assignments with ≥1 field polygon AND a valid catalog center + `image_date`
  (36,626 candidates). 2,115 field assignments were dropped for missing catalog center/date.
- **Tiles-per-class balanced** selection (`sampling.select_tiles_per_class`), ≤1000/class,
  rarest-first, ≤25k total. Because essentially every chip contains all three classes, the
  selection settles at exactly **1000 tiles** (each contributes to all of classes 0/1/2 →
  1000/1000/1000). This is the spec-default 1000/class classification cap; it undersamples a
  very large source, but follows the spec and matches `ai4boundaries`.

## Time range & change handling

Seasonal crop labels → a **1-year window anchored on the labelled imagery year** (`image_date`).
All imagery months fall in 2017–2023 (Sentinel era, post-2016). Selected-sample year spread:
2017:134, 2018:164, 2019:132, 2020:161, 2021:170, 2022:140, 2023:99. Not a change dataset
(`change_time = null`).

## Verification

- 1000 `.tif` + 1000 `.json`. Spot-checked tiles: single band, uint8, 64×64, UTM CRS
  (e.g. EPSG:32733), 10 m resolution, nodata 255, values in {0,1,2}; JSONs carry a 1-year
  `time_range` and `change_time=null`; metadata class ids cover all raster values.
- Georeferencing sanity check: tile centers reproject back to their source assignment's
  catalog lon/lat to sub-pixel precision (Δlat ≤ 0.0001°). Sampled locations span Angola,
  Nigeria, Zimbabwe — genuinely continent-wide. Alignment is exact by construction (labels
  rasterized directly onto the georeferenced UTM grid via rslearn).

## Caveats

- 3-class (interior/boundary) split is a refinement of the manifest's 2-class field/non-field;
  documented above. Boundary class relies on the polygon outlines, ~1–2 px at 10 m.
- Sample count capped at 1000 by tiles-per-class balancing (all classes co-occur per tile);
  a much larger, spatially diverse set exists in the source if a higher cap is ever desired.
- "Background" (0) is examined-but-undelineated land within a chip; like other field-boundary
  benchmarks it is not a guaranteed-pure negative if an annotator missed a field.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.lacuna_fund_africa_crop_field_labels
```
Idempotent (skips existing `{id}.tif`); a `raw/{slug}/scan_cache.pkl` caches the rasterized
candidate scan. Raw labels are re-downloadable unsigned from
`s3://africa-field-boundary-labels/{mapped_fields_final.parquet,label-catalog-filtered.csv}`
(the per-assignment catalog with image dates is `data/interim/label_catalog_allclasses.csv`
in the GitHub repo).
```
```
