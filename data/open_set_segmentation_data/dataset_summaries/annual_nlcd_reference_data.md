# Annual NLCD Reference Data

- **Slug:** `annual_nlcd_reference_data`
- **Status:** completed
- **Task type:** classification (sparse point segmentation)
- **Samples:** 15,174 (point table, `points.json`)

## Source

USGS ScienceBase, *Annual National Land Cover Database (NLCD) Collection 1.0 Reference
Data Product* — DOI [10.5066/P13EDMAF](https://doi.org/10.5066/P13EDMAF), item
`6813a71bd4be023163051775`. License **CC0-1.0**.

An independent, **manually interpreted reference dataset of 8,360 30 m x 30 m plots**
across the conterminous US, each with an annual land-cover label for every year
**1984-2023** (analyst interpretation, two-phase stratified sample). This is the manual
*reference* data behind Annual NLCD, preferred over the derived map product (per the
manifest note).

### Access method

ScienceBase's own zip download is **CAPTCHA-gated** (`requestDownload` page) and its S3
object (`prod-is-usgs-sb-prod-content`) returns 403 to unsigned requests. The identical
1.17 GB release zip is served, unauthenticated, from the **MRLC direct-download mirror**:
`https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/Annual_NLCD_CONUSV1_Ref_Data_Release.zip`.
Downloaded there; CSVs + shapefile `.prj` unzipped into `raw/annual_nlcd_reference_data/extracted/`.

Files used:
- `NLCD2023_Full8360_AnnualAttributes.csv` — per (plotid, image_year) attributes incl.
  `primary_landcover_code` (the label).
- `Plot_Coordinates_List_Simple_and_Stratified.csv` — plotid, x, y (plot-center coords).
- `lcnext_8360_final.prj` — CRS (CONUS Albers, WGS84 datum).

## Processing

- **Point-only dataset** (each label is one 30 m plot pixel) → written as one dataset-wide
  `points.json` (spec §2a), **not** per-point GeoTIFFs.
- **One sample per (plot, year).** Kept only **Sentinel-era years 2016-2023** (66,880 of
  334,393 plot-years); each sample gets a **1-year `time_range`** anchored on `image_year`.
  `change_time` is null — the label is the annual land-cover *state*, not a change event.
- **Coordinates**: plot `x`/`y` are CONUS Albers Conic Equal Area on the **WGS84 datum**
  (std parallels 29.5/45.5, central meridian -96, lat-of-origin 23). Reprojected to WGS84
  lon/lat with pyproj (verified: all points fall within CONUS, lon -124.6..-67.0,
  lat 24.9..49.2). Pretraining snaps the point onto the S2 grid.

### Class mapping

Label = `primary_landcover_code` (standard NLCD level-2 legend), remapped to contiguous
0-based uint8 ids (255 = nodata). All 16 NLCD level-2 classes appear in 2016-2023:

| id | NLCD code | name | selected |
|----|-----------|------|----------|
| 0  | 11 | Open Water | 1000 |
| 1  | 12 | Perennial Ice/Snow | 361 |
| 2  | 21 | Developed, Open Space | 1000 |
| 3  | 22 | Developed, Low Intensity | 1000 |
| 4  | 23 | Developed, Medium Intensity | 1000 |
| 5  | 24 | Developed, High Intensity | 1000 |
| 6  | 31 | Barren Land | 1000 |
| 7  | 41 | Deciduous Forest | 1000 |
| 8  | 42 | Evergreen Forest | 1000 |
| 9  | 43 | Mixed Forest | 813 |
| 10 | 52 | Shrub/Scrub | 1000 |
| 11 | 71 | Grassland/Herbaceous | 1000 |
| 12 | 81 | Pasture/Hay | 1000 |
| 13 | 82 | Cultivated Crops | 1000 |
| 14 | 90 | Woody Wetlands | 1000 |
| 15 | 95 | Emergent Herbaceous Wetlands | 1000 |

**Total: 15,174.** Balanced to <=1000 samples per class (`balance_by_class`, 25k cap; with
16 classes the effective cap is 1000/class). Two classes fall short of 1000 because they
have fewer plot-years in 2016-2023: Perennial Ice/Snow (361) and Mixed Forest (813). All
source selection strategies (simple + stratified) are used; no train/val/test filtering.

### Notes / caveats

- Only `primary_landcover_code` is used; `alternate_landcover_code`, `change_process`, and
  the sub-class fields are ignored. A plot can contribute up to 8 samples (one per year
  2016-2023), each with its own year window, so nearby years of the same plot may repeat a
  location — acceptable for pretraining (temporal diversity), and class balancing caps the
  total per class.
- The `primary_cover`/`primary_landcover_label` free-text labels differ slightly from the
  numeric-code legend; we use the authoritative numeric `primary_landcover_code`.
- Years 1984-2015 (267,513 plot-years) are dropped as pre-Sentinel.

## Reproduce

```
# 1. download + unzip (already staged under raw/annual_nlcd_reference_data/)
curl -sL "https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/Annual_NLCD_CONUSV1_Ref_Data_Release.zip" \
  -o Annual_NLCD_CONUSV1_Ref_Data_Release.zip
unzip -o -j Annual_NLCD_CONUSV1_Ref_Data_Release.zip \
  "*/NLCD2023_Full8360_AnnualAttributes.csv" \
  "*/Plot_Coordinates_List_Simple_and_Stratified.csv" \
  "*/lcnext_8360_final.prj" -d extracted
# 2. build points.json + metadata.json
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.annual_nlcd_reference_data
```
